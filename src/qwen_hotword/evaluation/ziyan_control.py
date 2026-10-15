"""Controlled English non-target pruning and reference-conditioned <=4-target subset."""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation import ziyan_control_replay as replay
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.ziyan_assets import NAME, bundle_from_table
from qwen_hotword.evaluation.ziyan_identity import digest, inspect, read_json, verify_seal
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _select, _validated_source_gate
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.wave_retrieval import _row_digest
from qwen_hotword.phonemes.coverage import load_phoneme_vocab


def prune_table(
    table: dict[str, Any], groups: dict[str, list[str]], targets: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    words = [w for values in groups.values() for w in values]
    if (
        not words
        or len(words) != len(set(words))
        or any(not w.isascii() or not w.isalpha() or w != w.lower() for w in words)
        or set(words).intersection(targets)
    ):
        raise ValueError("removals must be distinct English non-target words")
    result = copy.deepcopy(table)
    removed, found, kept = [], set(), []
    for entry in result["entries"]:
        variants = []
        for variant in entry["variants"]:
            if variant["language"] == "en" and variant["normalized"] in words:
                if variant.get("role") != "filler":
                    raise ValueError("refusing removal of a non-filler English variant")
                removed.append(variant)
                found.add(variant["normalized"])
            else:
                variants.append(variant)
        if variants:
            entry["variants"] = variants
            entry["languages"] = [v["language"] for v in variants]
            entry["surface_forms"] = sorted({v["surface"] for v in variants})
            entry["surface"] = variants[0]["surface"]
            kept.append(entry)
    if found != set(words):
        raise ValueError(
            f"requested removals missing from English index: {sorted(set(words) - found)}"
        )
    result["entries"] = kept
    result["scope"] = "ziyan_control_english_non_target_pruning"
    old_surfaces = {e["normalized"] for e in table["entries"]}
    new_surfaces = {e["normalized"] for e in kept}

    def english_words(value: dict[str, Any]) -> set[str]:
        return {
            v["normalized"]
            for e in value["entries"]
            for v in e["variants"]
            if v["language"] == "en"
        }

    return result, {
        "groups": groups,
        "removed_english_words": sorted(found),
        "removed_variant_count": len(removed),
        "removed_variants": removed,
        "old_surface_count": len(old_surfaces),
        "new_surface_count": len(new_surfaces),
        "old_english_surface_count": len(english_words(table)),
        "new_english_surface_count": len(english_words(result)),
        "deleted_surfaces": sorted(old_surfaces - new_surfaces),
        "removed_english_surfaces_retained_in_other_languages": sorted(found & new_surfaces),
        "es_pt_variants_preserved": True,
        "remaining_pronunciations_unchanged": True,
    }


def select_records(
    records: list[dict[str, Any]], targets: list[str], maximum: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
        raise ValueError("maximum targets must be a nonnegative integer")
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in records:
        expected = [w for w in targets if strict_phrase_match(record["reference_text"], w)]
        item = {**record, "target_words": expected, "target_count": len(expected)}
        (included if len(expected) <= maximum else excluded).append(item)
    if not included:
        raise ValueError("reference-conditioned subset is empty")
    counts = [r["target_count"] for r in included]
    return (
        included,
        excluded,
        {
            "samples": len(included),
            "positive_samples": sum(n > 0 for n in counts),
            "negative_samples": counts.count(0),
            "target_sample_pairs": sum(counts),
            "excluded_samples": len(excluded),
            "maximum_targets_per_audio": maximum,
            "target_count_histogram": {str(k): v for k, v in sorted(Counter(counts).items())},
            "active_target_words_in_subset": sorted(
                {w for r in included for w in r["target_words"]}
            ),
            "targets_absent_from_subset_still_indexed": sorted(
                set(targets) - {w for r in included for w in r["target_words"]}
            ),
            "reference_conditioned_selection": True,
            "predictions_used_for_selection": False,
            "zero_target_samples_retained": True,
            "deduplicated": False,
            "topk_capacity_upper_bound": {
                f"top{k}": sum(min(k, n) for n in counts) / sum(counts) if sum(counts) else None
                for k in (5, 7)
            },
        },
    )


def _runtime(table: dict[str, Any], path: Path, vocab: Any, targets: list[str]) -> dict[str, Any]:
    bundle, variants, mapping = bundle_from_table(table, path, vocab)
    ids = {e.normalized: e.hotword_id for e in bundle.entries}
    if not set(targets) <= ids.keys():
        raise ValueError("pruned index does not preserve every frozen target")
    return {
        "bundle": bundle,
        "variants": variants,
        "mapping": mapping,
        "target_ids": [ids[w] for w in targets],
    }


def _check_ctc(row: dict[str, Any], vocab: Any) -> None:
    ids, probabilities = row["decoded_token_ids"], row["decoded_confidences"]
    if (
        not isinstance(ids, list)
        or not isinstance(probabilities, list)
        or len(ids) != len(probabilities)
        or any(type(t) is not int or not 0 < t < len(vocab.tokens) for t in ids)
        or any(
            not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities
        )
        or row["decoded_tokens"] != [vocab.tokens[t] for t in ids]
    ):
        raise ValueError(f"invalid saved CTC sequence: {row['sample_id']}")


def _contract(
    root: Path, config: Path, cfg: dict[str, Any], base: dict[str, Any]
) -> dict[str, Any]:
    files = [config, root / cfg["baseline_config"]]
    files += [root / base[k] for k in ("dataset", "targets", "vocab", "checkpoint")]
    files += [root / base[k] / "sha256.txt" for k in ("assets", "output")]
    source_plan = read_json(root / base["output"] / "run_config.json")
    for relative, expected in source_plan["code_sha256"].items():
        path = (
            root / relative
            if relative.startswith("scripts/")
            else root / "src/qwen_hotword" / relative
        )
        if digest(path) != expected:
            raise ValueError(f"original retrieval code changed: {relative}")
        files.append(path)
    files += list(Path(__file__).parent.glob("ziyan_control*.py"))
    files += [
        Path(__file__).parents[3] / "scripts/run_ziyan_control.py",
        Path(__file__).parent / "ziyan_identity.py",
        Path(__file__).parent / "retrieval_case_trace.py",
        Path(__file__).parents[1] / "training/edit_distance.py",
    ]
    return {
        "configuration": cfg,
        "input_and_code_sha256": {str(p.resolve()): digest(p) for p in files},
    }


def _package(output: Path) -> None:
    names = [
        "report.json",
        "selection_report.json",
        "removal_report.json",
        "config.json",
        "run_config.json",
        "source_identity/summary.json",
        "english_words.txt",
        "removed_english_words.txt",
        "targets.txt",
        "paired_changes.jsonl",
        *(f"delivery/{NAME}_top{k}.json" for k in (5, 7)),
    ]
    returned = output / "return"
    returned.mkdir(exist_ok=True)
    for name in names:
        target = returned / name
        target.parent.mkdir(exist_ok=True, parents=True)
        if target.exists():
            if digest(target) != digest(output / name):
                raise ValueError("existing return copy differs")
        else:
            shutil.copyfile(output / name, target)
    seal(returned)
    archive = output / "return_bundle.zip"
    members = {p.relative_to(returned).as_posix(): p for p in returned.rglob("*") if p.is_file()}
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist()) != set(members) or any(
                z.read(name) != path.read_bytes() for name, path in members.items()
            ):
                raise ValueError("existing return archive differs")
    else:
        temporary = output / f"return_bundle_{uuid.uuid4().hex}.zip.tmp"
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as z:
            for name, path in sorted(members.items()):
                z.write(path, name)
        temporary.rename(archive)


def run(root: Path, config: Path, output: Path, *, resume: bool = False) -> dict[str, Any]:
    root, config, output = (p.resolve() for p in (root, config, output))
    cfg = read_json(config)
    base_config = root / cfg["baseline_config"]
    base = read_json(base_config)
    assets, source = (root / base[k] for k in ("assets", "output"))
    for original in (assets, source):
        if output.is_relative_to(original.resolve()) or original.resolve().is_relative_to(output):
            raise ValueError("new output must be separate from both original output directories")
    if output.exists() and not resume:
        raise FileExistsError(f"existing output refused: {output}; identical runs require --resume")
    if cfg.get("keep_zero_target_samples") is not True:
        raise ValueError("this controlled experiment must retain zero-target samples")
    contract = _contract(root, config, cfg, base)
    if output.exists():
        if (
            not (output / "run_config.json").is_file()
            or read_json(output / "run_config.json") != contract
        ):
            raise ValueError("resume configuration/source/code differs; use a new output directory")
    else:
        output.mkdir(parents=True)
        external._write_json(output / "run_config.json", contract)
    started = time.monotonic()
    print(
        "[1/4] Verify original assets, saved rows, JSON, Head/vocab and retrieval code", flush=True
    )
    for directory in (assets, source):
        verify_seal(directory)
    verified = output / "source_identity"
    if verified.exists():
        verify_seal(verified)
        if read_json(verified / "summary.json")["status"] != "identity_verified":
            raise ValueError(
                "incomplete original identity verification; use a new output directory"
            )
    else:
        inspect(root, base_config, assets, source, verified)
    if (output / "sha256.txt").exists():
        verify_seal(output)
        completed: dict[str, Any] = read_json(output / "report.json")
        return completed
    source_plan = read_json(source / "run_config.json")
    detector_config = source_plan["group"]["config"]
    gate = _validated_source_gate(detector_config)
    if detector_config["retrieval"]["backend"] != "anchor_guided":
        raise ValueError("only the original Anchor-guided backend can be replayed")
    original_table = read_json(assets / "keyword_table.json")
    if digest(assets / "keyword_table.json") != cfg["source_table_sha256"]:
        raise ValueError("unexpected source table identity")
    targets = read_json(assets / "targets.json")
    table, removal = prune_table(original_table, cfg["remove_english_non_targets"], targets)
    included, excluded, selection = select_records(
        source_plan["records"], targets, cfg["maximum_targets_per_audio"]
    )
    if any(selection[k] != v for k, v in cfg["expected_selection"].items()):
        raise ValueError("reference-conditioned selection differs from the frozen expected counts")
    audio_hashes = [source_plan["inputs"][r["audio_path"]]["sha256"] for r in included]
    selection["duplicate_extra_records"] = len(audio_hashes) - len(set(audio_hashes))
    print(
        f"[2/4] Keep {len(included)}/{len(source_plan['records'])} audio, "
        f"{selection['target_sample_pairs']} target pairs; remove "
        f"{len(removal['removed_english_words'])} English non-target words",
        flush=True,
    )
    for name, value in (
        ("config.json", cfg),
        ("keyword_table.json", table),
        ("selection_report.json", selection),
        ("removal_report.json", removal),
    ):
        external._write_or_verify_json(output / name, value)
    for name, values in (
        ("subset_manifest.jsonl", included),
        ("excluded_manifest.jsonl", excluded),
    ):
        external._write_or_verify_jsonl(output / name, values)
    selected_ids = {r["sample_id"] for r in included}
    raw_json = read_json(root / base["dataset"])
    external._write_or_verify_jsonl(
        output / "subset_source_records.jsonl",
        [r for r in raw_json if Path(r["audios"][0]).stem in selected_ids],
    )
    vocab = load_phoneme_vocab(root / base["vocab"])
    old = _runtime(original_table, assets / "keyword_table.json", vocab, targets)
    new = _runtime(table, output / "keyword_table.json", vocab, targets)
    for runtime in (old, new):
        replay.prepare_runtime(runtime, detector_config)
    english = sorted(
        {v["normalized"] for e in table["entries"] for v in e["variants"] if v["language"] == "en"}
    )
    for name, words in (
        ("english_words.txt", english),
        ("targets.txt", targets),
        ("removed_english_words.txt", removal["removed_english_words"]),
    ):
        text = "\n".join(words) + "\n"
        path = output / name
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"existing generated word list differs: {path}")
        else:
            external._atomic_write_text(path, text)
    source_rows = [
        json.loads(s) for s in (source / NAME / "retrieval_details.jsonl").read_text().splitlines()
    ]
    selected = [r for r in source_rows if r["sample_id"] in selected_ids]
    if len(selected) != len(included):
        raise ValueError("selected source rows differ")
    shards = output / "sample_shards"
    shards.mkdir(exist_ok=True)
    old_rows, new_rows, changes = [], [], []
    print("[3/4] CPU: reproduce ORIGINAL index, then query REBUILT pruned index", flush=True)
    for number, row in enumerate(selected, 1):
        _check_ctc(row, vocab)
        path = external._shard_path(
            shards,
            external.ExternalAudioRecord(
                row["sample_id"],
                row["source"],
                Path(row["audio_path"]),
                row["reference_text"],
                row["language"],
            ),
        )
        if path.exists():
            pair = read_json(path)
            if (
                pair.get("row_sha256") != _row_digest(pair)
                or pair.get("source_row_sha256") != row["row_sha256"]
            ):
                raise ValueError("corrupt/changed paired replay shard")
        else:
            reproduced = replay.retrieve(row, old, detector_config)
            replay.verify_reproduction(row, reproduced)
            before = replay.make_row(row, reproduced, old)
            after = replay.make_row(row, replay.retrieve(row, new, detector_config), new)
            pair = {
                "source_row_sha256": row["row_sha256"],
                "baseline_reproduced": True,
                "old": before,
                "new": after,
                "change": replay.paired_change(before, after, gate, set(old["target_ids"])),
            }
            pair["row_sha256"] = _row_digest(pair)
            external._write_json(path, pair)
        old_rows.append(pair["old"])
        new_rows.append(pair["new"])
        changes.append(pair["change"])
        print(
            f"  {number}/{len(selected)} [{number / len(selected):.1%}] "
            f"elapsed {time.monotonic() - started:.1f}s {row['sample_id']}",
            flush=True,
        )
    return _finish(
        root,
        config,
        cfg,
        base,
        output,
        contract,
        selection,
        removal,
        old,
        new,
        old_rows,
        new_rows,
        changes,
        gate,
        started,
    )


def _finish(
    root: Path,
    config: Path,
    cfg: dict[str, Any],
    base: dict[str, Any],
    output: Path,
    contract: dict[str, Any],
    selection: dict[str, Any],
    removal: dict[str, Any],
    old: dict[str, Any],
    new: dict[str, Any],
    old_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    gate: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    print("[4/4] Export paired metrics, Top5/Top7 deliveries and compact return bundle", flush=True)
    (output / "delivery").mkdir(exist_ok=True)
    (output / NAME).mkdir(exist_ok=True)
    for k in (5, 7):
        delivery = {
            r["sample_id"]: [
                {"word": m["surface"], "phoneme": m["matched_pronunciation"]}
                for m in _select(r["raw_ranked_matches"], gate=gate, top_k=k)
            ]
            for r in new_rows
        }
        external._write_or_verify_json(output / "delivery" / f"{NAME}_top{k}.json", delivery)
    external._write_or_verify_jsonl(output / NAME / "retrieval_details.jsonl", new_rows)
    external._write_or_verify_jsonl(output / "paired_changes.jsonl", changes)
    before, after = replay.metrics(old_rows, old, gate), replay.metrics(new_rows, new, gate)
    by_count = {}
    for count in range(cfg["maximum_targets_per_audio"] + 1):
        ids = {r["sample_id"] for r in changes if r["target_count"] == count}
        if ids:
            by_count[str(count)] = {
                "old": replay.metrics([r for r in old_rows if r["sample_id"] in ids], old, gate),
                "new": replay.metrics([r for r in new_rows if r["sample_id"] in ids], new, gate),
            }
    report = {
        "status": "completed",
        "output_dir": str(output),
        "selection": selection,
        "removal": {k: v for k, v in removal.items() if k != "removed_variants"},
        "baseline_original_full_set_metrics": read_json(root / base["output"] / "report.json")[
            "metrics"
        ],
        "baseline_same_subset_metrics": before,
        "pruned_same_subset_metrics": after,
        "paired_delta": {
            f"top{k}": {
                "target_hits_delta": after[f"top{k}"]["gated_target_hits"]
                - before[f"top{k}"]["gated_target_hits"],
                "gained_target_pairs": sum(len(r[f"top{k}"]["gained_target_ids"]) for r in changes),
                "lost_target_pairs": sum(len(r[f"top{k}"]["lost_target_ids"]) for r in changes),
                "reference_absent_output_delta": after[f"top{k}"][
                    "selected_words_absent_from_transcript"
                ]
                - before[f"top{k}"]["selected_words_absent_from_transcript"],
            }
            for k in (5, 7)
        },
        "by_target_count": by_count,
        "index_sizes": {
            "old_surfaces": len(old["bundle"].entries),
            "new_surfaces": len(new["bundle"].entries),
            "old_indexed_pronunciations": len(old["variants"]),
            "new_indexed_pronunciations": len(new["variants"]),
        },
        "source_head_sha256": base["checkpoint_sha256"],
        "source_vocab_sha256": base["vocab_sha256"],
        "new_table_sha256": digest(output / "keyword_table.json"),
        "gate_and_anchor_parameters": {
            k: read_json(root / base["output"] / "run_config.json")["group"]["config"][k]
            for k in ("gate", "retrieval")
        },
        "baseline_reproduced_for_all_selected_samples": True,
        "ctc_sequences_and_confidences_unchanged": True,
        "index_rebuilt_before_top64_query": True,
        "qwen_model_loaded": False,
        "audio_read": False,
        "qwen_decoder_used": False,
        "asr_wer_measured": False,
        "ctc_per_measured": False,
        "cpu_elapsed_seconds": time.monotonic() - started,
        "limitations": [
            "Reference-conditioned <=4-target subset and exploratory table; not a blind benchmark.",
            "Only paired old/new results on the SAME subset estimate the pruning effect.",
            "Removing negative/difficult audio is not a model improvement; 0-target rows retained.",
            "Absent frozen targets stay indexed; no 300-500 target-count constraint.",
            "Only listed English fillers removed; same ES/PT surfaces may remain active.",
            "CTC IDs/confidences and recorded audio hashes reused; audio bytes/SNR not rechecked.",
            "Decoded-token positions are not acoustic time; unavailable frame intervals are null.",
            "CPU replay timings are not H200 end-to-end inference latency.",
        ],
    }
    if _contract(root, config, cfg, base) != contract:
        raise ValueError("source inputs/code changed during replay")
    for key in ("assets", "output"):
        verify_seal(root / base[key])
    if (output / "report.json").exists():
        saved: dict[str, Any] = read_json(output / "report.json")
        if {k: v for k, v in saved.items() if k != "cpu_elapsed_seconds"} != {
            k: v for k, v in report.items() if k != "cpu_elapsed_seconds"
        }:
            raise ValueError("existing report differs from resumed results")
        report = saved
    external._write_or_verify_json(output / "report.json", report)
    _package(output)
    seal(output)
    return report
