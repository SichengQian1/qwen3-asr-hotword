#!/usr/bin/env python3
"""Prepare reusable English G2P or run sealed English–Arabic labeling/cache/training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Set
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.experiment_a import (
    ctc_minimum_input_length,
    estimate_qwen_lengths,
    read_audio_metadata,
)
from qwen_hotword.training.stage_progress import stage_progress

if TYPE_CHECKING:
    from qwen_hotword.training.sharded_ctc import DiskFeatureCache

VOCAB = ROOT / "configs/phonemes/en_ar_mfa_g2p_vocab.v1.0.json"
MAPPING = ROOT / "configs/phonemes/en_ar_mfa_g2p_mapping.v1.0.tsv"
VOCAB_SHA = "888881b84d0b2d2a1a2591baa1e160be82395fd21e6753246c80f9b9a12b55df"
MAPPING_SHA = "69814cb775c51af2dd491188c77ecd6867b1f2652d9aa83b3a17fa338f7e594a"
FOUNDATION = ROOT / "configs/mfa_v03_foundation.workzone.json"
ASSETS = ROOT / "models/mfa/hf_g2p_20261029_v03"
AUDITS = {
    "en_egy": "7987bde0",
    "en_us_donor": "7987bde0",
    "ar_egy": "5bda8943",
    "ar_msa": "7987bde0",
}
DONOR_SHA = "cb85a548f41f1b21e682d09a90264829296aae01381a60aa603bbcf66ba67a5b"
VALIDATION_HOURS = {"en_egy": 4, "en_us_donor": 0, "ar_egy": 2, "ar_msa": 2}


def inventory_paths(root: Path) -> dict[str, Path]:
    return {
        group: root / "outputs" / f"en_ar_training_source_audit_{run}" / "source_inventory.jsonl"
        for group, run in AUDITS.items()
    }


def eligible(row: dict[str, Any]) -> bool:
    return bool(row.get("duration_seconds")) and not (
        set(row["issues"]) - {"assistant_punctuation_or_spacing_only"}
    )


def prepare_english(root: Path, inventories: dict[str, Path], output: Path) -> Path:
    from qwen_hotword.training.mfa_round_common import read, words
    from qwen_hotword.training.mfa_round_labels import generate

    source = read(root / "configs/en_ar_training_sources.workzone.json")
    with stage_progress("Verify original US English training donor"):
        pin(root / source["us_donor"], DONOR_SHA)
    inventory = inventories["en_egy"]
    with stage_progress("Record English source inventory identity"):
        inventory_sha = sha(inventory)
    wanted: set[str] = set()
    eligible_records: Counter[str] = Counter()
    with stage_progress("Scan previously audited English text") as tick:
        for index, row in enumerate(iter_jsonl(inventory), 1):
            if row["group"].startswith("en_") and eligible(row):
                tokenized, issues = words(row["text"])
                if not issues:
                    wanted.update(tokenized)
                    eligible_records[row["group"]] += 1
            tick(index)
    if not wanted or set(eligible_records) != {"en_egy", "en_us_donor"}:
        raise ValueError("both English source groups must supply text")
    foundation = {
        **read(FOUNDATION),
        "vocab": str(VOCAB),
        "vocab_sha256": VOCAB_SHA,
    }
    worker_config = output / "english_g2p_config.json"
    write_json(worker_config, foundation)
    generated = generate(
        root,
        {
            "foundation_config": str(worker_config),
            "assets_dir": str(ASSETS),
            "g2p_words_per_part": 2000,
            "g2p_workers": 4,
        },
        output,
        {"en": wanted},
    )["en"]
    with stage_progress("Recheck English source inventory"):
        pin(inventory, inventory_sha)
    destination = output / "english_g2p.json"
    summary = {
        "status": "english_g2p_completed",
        "vocab_sha256": VOCAB_SHA,
        "donor_sha256": DONOR_SHA,
        "inventory_sha256": inventory_sha,
        "unique_words": len(generated),
        "eligible_records": eligible_records,
        "g2p_status_counts": Counter(v["status"] for v in generated.values()),
    }
    write_json(destination, {**summary, "words": generated})
    write_json(output / "english_g2p_report.json", summary)
    return destination


def romanizations(path: Path) -> Iterator[dict[str, Any]]:
    for row in iter_jsonl(path):
        if (
            row["group"] not in {"ar_egy", "ar_msa"}
            or not isinstance(row["source"], str)
            or type(row["row"]) is not int
            or row["row"] < 1
            or not isinstance(row["text"], str)
            or not isinstance(row["romanized"], str)
            or not row["romanized"].isascii()
            or not row["romanized"].split()
        ):
            raise ValueError("invalid Arabic romanization record")
        yield row


def arabic_g2p(archive: Path, source: Path, output: Path) -> None:
    import pynini  # type: ignore[import-not-found]
    from pynini.lib import rewrite  # type: ignore[import-not-found]

    spec = json.loads(VOCAB.read_text(encoding="utf-8"))["models"]["ar"]
    pin(archive, spec["archive_sha256"])
    with zipfile.ZipFile(archive) as zipped:
        for filename, expected in spec["files_sha256"].items():
            members = [member for member in zipped.namelist() if Path(member).name == filename]
            if len(members) != 1:
                raise ValueError(f"archive must contain one {filename}")
            content = zipped.read(members[0])
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Arabic archive member changed: {filename}")
            (output / filename).write_bytes(content)
    fst = pynini.Fst.read(str(output / "model.fst"))
    phones = pynini.SymbolTable.read_text(str(output / "phones.sym"))
    alphabet = {chr(a.ilabel) for state in fst.states() for a in fst.arcs(state) if a.ilabel}
    with MAPPING.open(encoding="utf-8") as mapping:
        ids = {
            row["source_unit"]: int(row["token_id"])
            for row in csv.DictReader(mapping, delimiter="\t")
            if row["language"] == "ar" and row["token_id"]
        }
    memo: dict[str, list[int]] = {}
    target = output / "arabic_labels.jsonl"
    with target.open("x", encoding="utf-8") as handle, stage_progress("Arabic MFA Top1") as tick:
        for index, row in enumerate(romanizations(source), 1):
            tokens = []
            for word in row["romanized"].split():
                if word not in memo:
                    memo[word] = []
                    if set(word) <= alphabet:
                        try:
                            raw = rewrite.top_rewrite(
                                word, fst, input_token_type="utf8", output_token_type=phones
                            )
                        except rewrite.Error:
                            pass
                        else:
                            units = raw.split()
                            if all(unit in ids for unit in units):
                                memo[word] = [ids[unit] for unit in units]
                tokens.append(memo[word])
            result = (
                [token for word_tokens in tokens for token in word_tokens] if all(tokens) else []
            )
            handle.write(json.dumps({**row, "ids": result}, ensure_ascii=False) + "\n")
            tick(index)


def source_records(inventories: dict[str, Path]) -> Iterator[dict[str, Any]]:
    for group, path in inventories.items():
        with stage_progress(f"Build manifest candidates {group}") as tick:
            for index, row in enumerate(iter_jsonl(path), 1):
                if row["group"] == group:
                    yield row
                tick(index)


def freeze(
    output: Path,
    inventories: dict[str, Path],
    english: Path | None,
    arabic: Path | None,
    train_hours: float,
    *,
    protected_audio: Set[str] = frozenset(),
) -> dict[str, Any]:
    from qwen_hotword.training.mfa_round_common import read, words
    from qwen_hotword.training.mfa_round_labels import top1_tokens

    en = {}
    if english is not None:
        en = read(english)
        if (
            en["status"] != "english_g2p_completed"
            or en["vocab_sha256"] != VOCAB_SHA
            or en["donor_sha256"] != DONOR_SHA
            or en["inventory_sha256"] != sha(inventories["en_egy"])
        ):
            raise ValueError("English G2P is not from this source inventory and vocabulary")
    ar: dict[tuple[str, int], dict[str, Any]] = {}
    if arabic is not None:
        for row in iter_jsonl(arabic):
            key = (row["source"], row["row"])
            if key in ar:
                raise ValueError("duplicate source/row in Arabic romanization input")
            ar[key] = row
    languages = [lang for lang, path in (("en", english), ("ar", arabic)) if path is not None]
    if not languages:
        raise ValueError("at least one language is required")
    selected_sources = {
        group: path for group, path in inventories.items() if group[:2] in languages
    }
    validation_hours = {group: VALIDATION_HOURS[group] for group in selected_sources}
    hours: dict[tuple[str, str], float] = defaultdict(float)
    counts: Counter[tuple[str, str]] = Counter()
    frame_totals: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    seen = set(protected_audio)
    with ExitStack() as stack:
        files = {
            split: stack.enter_context((output / f"{split}.jsonl").open("x", encoding="utf-8"))
            for split in ("train", "validation")
        }
        for row in source_records(selected_sources):
            group = row["group"]
            lang = group[:2]
            val_goal = VALIDATION_HOURS[group]
            train_key = ("train", lang)
            val_key = ("validation", group)
            if hours[train_key] >= train_hours and hours[val_key] >= val_goal:
                continue
            if not eligible(row):
                rejections[group + ":source_issue"] += 1
                continue
            if lang == "en":
                tokenized, issues = words(row["text"])
                token_ids, missing = top1_tokens(tokenized, en["words"])
                if issues or missing:
                    token_ids = []
                if any(type(t) is not int or not 2 <= t < 72 for t in token_ids):
                    raise ValueError("invalid English token IDs")
            else:
                label = ar.get((row["source"], row["row"]))
                if label is None:
                    rejections[group + ":no_romanization"] += 1
                    continue
                if label["text"] != row["text"] or label["group"] != group:
                    raise ValueError(
                        f"Arabic romanization text/source identity mismatch: {label['row']}"
                    )
                native_words, issues = words(row["text"])
                token_ids = (
                    label["ids"]
                    if not issues and len(native_words) == len(label["romanized"].split())
                    else []
                )
                if any(type(t) is not int or not 72 <= t < 114 for t in token_ids):
                    raise ValueError("invalid Arabic token IDs")
            if not token_ids:
                rejections[group + ":incomplete_label"] += 1
                continue
            path = Path(row["audio_path"]).resolve(strict=True)
            fingerprint = sha(path)
            if fingerprint in seen:
                rejections[group + ":duplicate_audio"] += 1
                continue
            # Partition by audio bytes even after a validation target is full.
            heldout = int(fingerprint[:12], 16) % 100 < 3
            split = "validation" if heldout and val_goal else "train"
            key = val_key if split == "validation" else train_key
            if hours[key] >= (val_goal if split == "validation" else train_hours):
                continue
            metadata = read_audio_metadata(path)
            if not math.isclose(metadata.duration_seconds, row["duration_seconds"], abs_tol=1e-9):
                raise ValueError(f"audio duration changed since source audit: {path}")
            _, encoder_frames = estimate_qwen_lengths(metadata)
            minimum = ctc_minimum_input_length(token_ids)
            if minimum > 2 * encoder_frames:
                rejections[group + ":estimated_ctc_length"] += 1
                continue
            seconds = metadata.duration_seconds
            seen.add(fingerprint)
            hours[key] += seconds / 3600
            counts[(split, group)] += 1
            frame_totals[split] += encoder_frames
            manifest_row = {
                "id": f"en-ar-v1-{fingerprint}",
                "audio_path": str(path),
                "text": row["text"],
                "language": lang,
                "source_group": group,
                "source": row["source"],
                "source_row": row["row"],
                "duration_seconds": seconds,
                "audio_sha256": fingerprint,
                "phoneme_token_ids": token_ids,
                "label_length": len(token_ids),
                "ctc_minimum_input_length": minimum,
                "ctc_time_upsampling_factor": 2,
                "split": split,
                "experiment": "full-ctc-v1",
            }
            files[split].write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
    capacity = all(hours[("train", lang)] >= train_hours for lang in languages) and all(
        hours[("validation", group)] >= goal for group, goal in validation_hours.items()
    )
    return {
        "status": "labels_complete" if capacity else "insufficient_labeled_hours",
        "languages": languages,
        "vocab_sha256": VOCAB_SHA,
        "train_target_hours_per_language": train_hours,
        "validation_target_hours": validation_hours,
        "hours": {"/".join(key): value for key, value in hours.items()},
        "rows_by_group": {"/".join(key): value for key, value in counts.items()},
        "estimated_encoder_frames": frame_totals,
        "rejections": rejections,
        "train_manifest_sha256": sha(output / "train.jsonl"),
        "validation_manifest_sha256": sha(output / "validation.jsonl"),
        "arabic_label_policy": (
            "externally_supplied_romanizations_official_mfa_top1" if arabic is not None else None
        ),
        "split_identity": "exact_file_sha256",
        "speaker_disjoint_verified": False,
    }


def wait_for_english(path: Path, pid: int) -> None:
    with stage_progress(f"Wait for running English MFA PID {pid}"):
        while not path.is_file():
            try:
                os.kill(pid, 0)
            except ProcessLookupError as error:
                if path.is_file():
                    return
                raise ValueError(f"English MFA exited without publishing {path}") from error
            proc = Path(f"/proc/{pid}")
            stat = proc / "stat"
            if stat.is_file():
                try:
                    state = stat.read_text().rsplit(")", 1)[-1].split()[0]
                    command = (proc / "cmdline").read_bytes().split(b"\0")
                except FileNotFoundError:
                    continue
                if path.is_file():
                    return
                if state == "Z":
                    raise ValueError(f"English MFA exited without publishing {path}")
                if b"prepare-en" not in command:
                    raise ValueError(f"PID {pid} is no longer the English MFA process")
            time.sleep(5)


def frozen_english_audio(output: Path, train_hours: float) -> set[str]:
    from qwen_hotword.training.mfa_round_common import read

    report = read(output / "label_report.json")
    if (
        report.get("status") != "labels_complete"
        or report.get("languages") != ["en"]
        or report.get("vocab_sha256") != VOCAB_SHA
        or report.get("train_target_hours_per_language") != train_hours
    ):
        raise ValueError(f"incompatible frozen English labels: {output}")
    seen: set[str] = set()
    with stage_progress("Verify frozen English manifests and protected audio"):
        for split in ("train", "validation"):
            manifest = output / f"{split}.jsonl"
            pin(manifest, report[f"{split}_manifest_sha256"])
            for row in iter_jsonl(manifest):
                identity = row["audio_sha256"]
                if row["language"] != "en" or row["split"] != split or identity in seen:
                    raise ValueError("invalid or duplicate frozen English audio identity")
                seen.add(identity)
    return seen


def cache_stage(output: Path, gpu: int, report: dict[str, Any]) -> None:
    cache = output / "cache"
    if not cache.exists():
        estimated = sum(report["estimated_encoder_frames"].values()) * 1024 * 2
        minimum_free = math.ceil(1.15 * estimated) + 2 * 1024**3
        free = shutil.disk_usage(output).free
        if free < minimum_free:
            raise ValueError(
                f"insufficient space for feature cache at {output}: "
                f"{free / 1024**3:.1f} GiB available, need about {minimum_free / 1024**3:.1f} GiB"
            )
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/cache_full_training_features.py"),
        "--config",
        str(ROOT / "configs/480h_cache.workzone.yaml"),
        "--train-manifest",
        str(output / "train.jsonl"),
        "--validation-manifest",
        str(output / "validation.jsonl"),
        "--vocab",
        str(VOCAB),
        "--output-dir",
        str(cache),
        "--prevalidated-audio-paths",
    ]
    for split in ("train", "validation"):
        count = sum(
            n for group, n in report["rows_by_group"].items() if group.startswith(split + "/")
        )
        command.extend([f"--expected-{split}-samples", str(count)])
    with stage_progress("Cache full frozen Qwen Encoder features"):
        subprocess.run(
            command,
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        )


def combine_caches(parts: list[DiskFeatureCache], output: Path) -> DiskFeatureCache:
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if any(
        p.split != first.split
        or p.encoder_frozen_parameters != first.encoder_frozen_parameters
        or p.ctc_time_upsampling_factor != first.ctc_time_upsampling_factor
        or not p.sha256_verified
        for p in parts
    ):
        raise ValueError("incompatible feature cache views")
    shards = tuple(shard for part in parts for shard in part.shards)
    ids = [sample for shard in shards for sample in shard.sample_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate audio across the combined feature caches")
    return replace(
        first,
        root=output / "cache_views" / first.split,
        sample_count=len(ids),
        shard_count=len(shards),
        feature_bytes=sum(shard.feature_bytes for shard in shards),
        fingerprint=hashlib.sha256(
            "".join(part.fingerprint for part in parts).encode()
        ).hexdigest(),
        shards=shards,
    )


def train_stage(output: Path, gpu: int, english: Path | None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    from qwen_hotword.phonemes.coverage import load_phoneme_vocab
    from qwen_hotword.training.ctc_diagnostics import load_validation_sample_groups
    from qwen_hotword.training.mfa_round_common import read
    from qwen_hotword.training.sharded_ctc import (
        exclusive_training_run,
        load_disk_feature_cache,
        train_sharded_ctc_head,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable on the requested physical GPU")
    roots = [output] if english is None else [english, output]
    views = {}
    groups: dict[str, str] = {}
    encoder_identity: dict[str, Any] | None = None
    with stage_progress("Verify feature shard SHA256 and Encoder identity before training"):
        for split in ("train", "validation"):
            parts = []
            for root in roots:
                cache = root / "cache" / split
                manifest = root / f"{split}.jsonl"
                config = read(cache / "cache_config.json")
                current = {
                    k: config[k] for k in ("model", "tap_module", "hidden_size", "feature_dtype")
                }
                if encoder_identity is not None and current != encoder_identity:
                    raise ValueError(
                        "English and Arabic features have different Encoder identities"
                    )
                encoder_identity = current
                part = load_disk_feature_cache(
                    cache, expected_split=split, source_manifest_path=manifest, vocab_path=VOCAB
                )
                parts.append(part)
                if split == "validation":
                    groups.update(
                        load_validation_sample_groups(manifest, part, group_column="language")
                    )
            views[split] = combine_caches(parts, output)
    train_ids = {value for shard in views["train"].shards for value in shard.sample_ids}
    if set(groups.values()) != {"en", "ar"} or train_ids.intersection(groups):
        raise ValueError("combined train/validation isolation or bilingual coverage failed")
    write_json(
        output / "cache_sources.json",
        {
            "reused_english_output": str(english) if english else None,
            "train_fingerprint": views["train"].fingerprint,
            "validation_fingerprint": views["validation"].fingerprint,
            "encoder_identity": encoder_identity,
        },
    )
    head = output / "head"
    with exclusive_training_run(head), stage_progress("Train new 114-class Head"):
        train_sharded_ctc_head(
            views["train"],
            views["validation"],
            load_phoneme_vocab(VOCAB),
            head,
            vocab_path=VOCAB,
            device="cuda:0",
            head_type="temporal_upsample",
            epochs=40,
            early_stopping_min_delta=0.0005,
            learning_rate=0.0003,
            minimum_learning_rate=0.000005,
            validation_sample_groups=groups,
            early_stopping_metric="validation_macro_per",
            checkpoint_selection_metric="validation_macro_per",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-en", "cache-en", "all", "_arabic-worker"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--english-g2p", type=Path)
    parser.add_argument("--english-cache", type=Path, help="completed cache-en output for all")
    parser.add_argument("--wait-for-english-pid", type=int)
    parser.add_argument("--resume", action="store_true", help="resume existing cache-en shards")
    parser.add_argument("--arabic-romanizations", type=Path)
    parser.add_argument("--arabic-model", type=Path, default=ROOT / "models/mfa/g2p/arabic_g2p.zip")
    parser.add_argument("--train-hours", type=float, default=480)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.action in {"all", "_arabic-worker"} and (
        args.arabic_romanizations is None or not args.arabic_romanizations.is_file()
    ):
        parser.error("an Arabic romanization JSONL is required")
    if args.action in {"all", "cache-en"} and (
        args.gpu is None
        or args.gpu < 0
        or not math.isfinite(args.train_hours)
        or args.train_hours <= 0
    ):
        parser.error("GPU actions require --gpu N and positive --train-hours")
    if args.resume and args.action != "cache-en":
        parser.error("--resume applies only to cache-en")
    if args.wait_for_english_pid is not None and (
        args.action != "cache-en"
        or args.english_g2p is None
        or args.wait_for_english_pid <= 0
        or args.resume
    ):
        parser.error("waiting requires new cache-en, --english-g2p and a positive PID")
    if args.english_cache is not None and (args.action != "all" or args.english_g2p is not None):
        parser.error("--english-cache is for all and replaces --english-g2p")
    try:
        pin(VOCAB, VOCAB_SHA)
        pin(MAPPING, MAPPING_SHA)
        if not args.resume:
            output.mkdir(parents=True, exist_ok=False)
        print(f"Output directory: {output}", flush=True)
        if args.action == "_arabic-worker":
            arabic_g2p(args.arabic_model, args.arabic_romanizations, output)
            return 0
        inventories = inventory_paths(ROOT)
        if args.action == "prepare-en":
            path = prepare_english(ROOT, inventories, output)
            print(json.dumps({"status": "english_g2p_completed", "english_g2p": str(path)}))
            return 0
        if args.action == "cache-en":
            if args.resume:
                frozen_english_audio(output, args.train_hours)
                report = json.loads((output / "label_report.json").read_text(encoding="utf-8"))
            else:
                if args.wait_for_english_pid is not None:
                    wait_for_english(args.english_g2p, args.wait_for_english_pid)
                english = args.english_g2p or prepare_english(ROOT, inventories, output)
                report = freeze(output, inventories, english, None, args.train_hours)
                write_json(output / "label_report.json", report)
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
            if report["status"] != "labels_complete":
                return 2
            cache_stage(output, args.gpu, report)
            print(json.dumps({"status": "english_cache_completed", "output_dir": str(output)}))
            return 0
        english_cache = args.english_cache.resolve() if args.english_cache else None
        protected: set[str] = set()
        if english_cache is not None:
            cache_report = english_cache / "cache/feature_cache_report.json"
            if json.loads(cache_report.read_text(encoding="utf-8")).get("status") != "pass":
                raise ValueError(f"incomplete English feature cache: {english_cache}")
            protected = frozen_english_audio(english_cache, args.train_hours)
        conda = shutil.which("conda")
        if conda is None:
            raise ValueError("existing aligner Conda environment required")
        arabic = output / "arabic"
        with stage_progress("Generate Arabic labels in existing aligner"):
            subprocess.run(
                [
                    conda,
                    "run",
                    "--no-capture-output",
                    "-n",
                    "aligner",
                    "python",
                    "-B",
                    __file__,
                    "_arabic-worker",
                    "--output-dir",
                    str(arabic),
                    "--arabic-romanizations",
                    str(args.arabic_romanizations.resolve()),
                    "--arabic-model",
                    str(args.arabic_model.resolve()),
                ],
                check=True,
            )
        english = None
        if english_cache is None:
            english = args.english_g2p or prepare_english(ROOT, inventories, output)
        report = freeze(
            output,
            inventories,
            english,
            arabic / "arabic_labels.jsonl",
            args.train_hours,
            protected_audio=protected,
        )
        write_json(output / "label_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report["status"] != "labels_complete":
            return 2
        cache_stage(output, args.gpu, report)
        train_stage(output, args.gpu, english_cache)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
