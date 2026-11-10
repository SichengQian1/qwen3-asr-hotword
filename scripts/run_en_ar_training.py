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
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

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
    english: Path,
    arabic: Path,
    train_hours: float,
) -> dict[str, Any]:
    from qwen_hotword.training.mfa_round_common import read, words
    from qwen_hotword.training.mfa_round_labels import top1_tokens

    en = read(english)
    if (
        en["status"] != "english_g2p_completed"
        or en["vocab_sha256"] != VOCAB_SHA
        or en["donor_sha256"] != DONOR_SHA
        or en["inventory_sha256"] != sha(inventories["en_egy"])
    ):
        raise ValueError("English G2P is not from this source inventory and vocabulary")
    ar: dict[tuple[str, int], dict[str, Any]] = {}
    for row in iter_jsonl(arabic):
        key = (row["source"], row["row"])
        if key in ar:
            raise ValueError("duplicate source/row in Arabic romanization input")
        ar[key] = row
    hours: dict[tuple[str, str], float] = defaultdict(float)
    counts: Counter[tuple[str, str]] = Counter()
    rejections: Counter[str] = Counter()
    seen: set[str] = set()
    with ExitStack() as stack:
        files = {
            split: stack.enter_context((output / f"{split}.jsonl").open("x", encoding="utf-8"))
            for split in ("train", "validation")
        }
        for row in source_records(inventories):
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
    capacity = all(hours[("train", lang)] >= train_hours for lang in ("en", "ar")) and all(
        hours[("validation", group)] >= goal for group, goal in VALIDATION_HOURS.items()
    )
    return {
        "status": "labels_complete" if capacity else "insufficient_labeled_hours",
        "train_target_hours_per_language": train_hours,
        "validation_target_hours": VALIDATION_HOURS,
        "hours": {"/".join(key): value for key, value in hours.items()},
        "rows_by_group": {"/".join(key): value for key, value in counts.items()},
        "rejections": rejections,
        "train_manifest_sha256": sha(output / "train.jsonl"),
        "validation_manifest_sha256": sha(output / "validation.jsonl"),
        "arabic_label_policy": "externally_supplied_romanizations_official_mfa_top1",
        "split_identity": "exact_file_sha256",
        "speaker_disjoint_verified": False,
    }


def gpu_stages(output: Path, gpu: int) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    cache = output / "cache"
    manifests = [
        "--train-manifest",
        str(output / "train.jsonl"),
        "--validation-manifest",
        str(output / "validation.jsonl"),
        "--vocab",
        str(VOCAB),
    ]
    for script, extra, stage in (
        (
            "cache_full_training_features.py",
            [
                "--config",
                str(ROOT / "configs/480h_cache.workzone.yaml"),
                "--output-dir",
                str(cache),
                "--prevalidated-audio-paths",
            ],
            "Cache frozen Qwen Encoder features",
        ),
        (
            "train_full_ctc.py",
            [
                "--train-cache",
                str(cache / "train"),
                "--validation-cache",
                str(cache / "validation"),
                "--output-dir",
                str(output / "head"),
                "--head-type",
                "temporal_upsample",
                "--epochs",
                "40",
                "--early-stopping-min-delta",
                "0.0005",
                "--learning-rate",
                "0.0003",
                "--minimum-learning-rate",
                "0.000005",
                "--validation-group-column",
                "language",
                "--expected-validation-groups",
                "en,ar",
                "--early-stopping-metric",
                "validation_macro_per",
                "--checkpoint-selection-metric",
                "validation_macro_per",
            ],
            "Train new 114-class Head",
        ),
    ):
        with stage_progress(stage):
            subprocess.run(
                [sys.executable, "-B", str(ROOT / "scripts" / script), *manifests, *extra],
                check=True,
                env=environment,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-en", "all", "_arabic-worker"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--english-g2p", type=Path)
    parser.add_argument("--arabic-romanizations", type=Path)
    parser.add_argument("--arabic-model", type=Path, default=ROOT / "models/mfa/g2p/arabic_g2p.zip")
    parser.add_argument("--train-hours", type=float, default=480)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.action != "prepare-en":
        if args.arabic_romanizations is None or not args.arabic_romanizations.is_file():
            parser.error("an Arabic romanization JSONL is required")
        if args.action == "all" and (
            args.gpu is None
            or args.gpu < 0
            or not math.isfinite(args.train_hours)
            or args.train_hours <= 0
        ):
            parser.error("all requires --gpu N and positive --train-hours")
    try:
        pin(VOCAB, VOCAB_SHA)
        pin(MAPPING, MAPPING_SHA)
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
        english = args.english_g2p or prepare_english(ROOT, inventories, output)
        report = freeze(
            output, inventories, english, arabic / "arabic_labels.jsonl", args.train_hours
        )
        write_json(output / "label_report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report["status"] != "labels_complete":
            return 2
        gpu_stages(output, args.gpu)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
