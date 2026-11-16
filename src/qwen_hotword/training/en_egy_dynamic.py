"""Pure EGY admission, resumable labeling, and capacity-driven bilingual selection.

All capacities are durations, never counts. Unknown dialect provenance is recorded;
text agreement is not treated as proof of pronunciation correctness.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.arabic_refresh import prefetch_hashes, rank, recording_key
from qwen_hotword.training.egyptian_source_audit import comparison_key
from qwen_hotword.training.en_ar_source_audit import iter_jsonl, transcript_key
from qwen_hotword.training.stage_progress import stage_progress

AGREED = {"exact", "punctuation_or_spacing_only", "arabic_spelling_normalization_equivalent"}
BUCKETS = ("open_source", "gaoshan_train", "coarse_train")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    if path.exists():
        pin(path, sha(pending))
        pending.unlink()
    else:
        pending.replace(path)


def seal(path: Path, value: Any) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError(f"sealed inputs/configuration changed: {path}; use a new output")
    else:
        write_json(path, value)


def heldout(row: dict[str, Any]) -> bool:
    embedded = str(row.get("embedded_source") or "").lower()
    return row.get("role") == "original_test" or bool(
        re.search(r"(?:^|[-_./])(test|validation|dev)(?:[-_./]|$)", embedded)
    )


def admission(row: dict[str, Any]) -> str | None:
    if heldout(row):
        return "protected_test_or_validation"
    if row["group"] not in BUCKETS or row.get("role") != "train_pool":
        return "outside_egyptian_training_pools"
    relation = row.get("text_relation")
    allowed = (
        {"arabic_spelling_normalization_equivalent"} if row["group"] == "open_source" else AGREED
    )
    if relation not in allowed:
        return "text_relation_not_selected"
    issues = set(row.get("issues", []))
    # This exact audit warning is now explicitly admitted by the user.
    issues.discard("arabic_spelling_normalization_equivalent")
    # The audit classified an empty dialect string as a conflict; it means unspecified.
    if row.get("declared_dialect") in (None, ""):
        issues.discard("conflicting_dialect_field")
    if issues:
        return "source_issues:" + ",".join(sorted(issues))
    if not row.get("assistant_text") or not row.get("audio_path"):
        return "missing_text_or_audio"
    seconds = row.get("duration_seconds")
    if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        return "invalid_duration"
    return None


def group_key(row: dict[str, Any]) -> tuple[str, str]:
    path = row["audio_path"]
    if row.get("language") == "en" or row.get("bucket", "").startswith("en_"):
        # English retains the existing file/byte-disjoint validation policy.
        return "file:" + str(path), "file_only"
    try:
        return recording_key(path), "recognized_recording"
    except ValueError:
        stem = Path(path).stem
        if match := re.fullmatch(r"([0-9a-fA-F]{32})_\d+", stem):
            return "coarse-recording:" + match[1].lower(), "recognized_recording"
        # Do not invent a speaker ID from an unverified filename convention.
        return "file:" + str(path), "file_only_unknown_recording"


def capacity(english: float, arabic: dict[str, float], ratios: dict[str, float]) -> float:
    if set(ratios) != set(BUCKETS) or not math.isclose(sum(ratios.values()), 1):
        raise ValueError("expected the three Egyptian buckets with ratios summing to one")
    if any(ratios[k] <= 0 or arabic.get(k, 0) <= 0 for k in ratios) or english <= 0:
        raise ValueError(
            "no usable training data for a required language/source; see capacity report"
        )
    return min(english, *(arabic[k] / ratios[k] for k in ratios))


def select_seconds(rows: list[dict[str, Any]], seconds: float, seed: int) -> list[dict[str, Any]]:
    """Nearest deterministic prefix; error <= one recording, no slicing or repetition."""
    ordered = sorted(rows, key=lambda r: rank(r["audio_sha256"], seed))
    selected: list[dict[str, Any]] = []
    used = 0.0
    for row in ordered:
        length = row["duration_seconds"]
        if used >= seconds:
            break
        if selected and used + length > seconds and seconds - used < used + length - seconds:
            break
        selected.append(row)
        used += length
    return selected


def manifest(row: dict[str, Any], split: str) -> dict[str, Any]:
    ids = row.get("ids", row.get("phoneme_token_ids"))
    language = "en" if row["bucket"].startswith("en_") else "ar"
    lower, upper = (2, 72) if language == "en" else (72, 167)
    if not ids or any(type(i) is not int or not lower <= i < upper for i in ids):
        raise ValueError("invalid complete label for " + row["audio_path"])
    minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
    return {
        "id": "en-ar-v1-" + row["audio_sha256"],
        "audio_path": row["audio_path"],
        "audio_sha256": row["audio_sha256"],
        "duration_seconds": row["duration_seconds"],
        "text": row["text"],
        "language": language,
        "source_group": row["bucket"],
        "source": row["source"],
        "source_row": row["row"],
        "recording_key": row["recording_key"],
        "grouping_basis": row["grouping_basis"],
        "phoneme_token_ids": ids,
        "label_length": len(ids),
        "ctc_minimum_input_length": minimum,
        "ctc_time_upsampling_factor": 2,
        "split": split,
        "experiment": "full-ctc-v1",
    }


def fingerprint(rows: list[dict[str, Any]], output: Path, workers: int) -> None:
    """Persist each completed audio hash, including across interrupted runs."""
    with sqlite3.connect(output / "audio_hashes.sqlite") as db, ThreadPoolExecutor(workers) as pool:
        db.execute(
            "CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY,size INT,mtime INT,sha TEXT)"
        )
        with stage_progress(
            "Fingerprint eligible/protected audio; durable per-file cache", total=len(rows)
        ) as tick:
            for start in range(0, len(rows), 512):
                batch = rows[start : start + 512]
                prefetch_hashes(batch, db, pool)
                for row in batch:
                    row["audio_sha256"] = db.execute(
                        "SELECT sha FROM hashes WHERE path=?", (row["audio_path"],)
                    ).fetchone()[0]
                tick(start + len(batch))


def prepare_inputs(root: Path, cfg: dict[str, Any], output: Path, source: dict[str, Any]) -> None:
    """No audio read here: audited candidates and frozen existing manifests only."""
    marker = output / "admission_report.json"
    if marker.exists():
        for name, checksum in read(marker)["files"].items():
            pin(output / name, checksum)
        return
    reasons: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    hours: Counter[str] = Counter()
    ar, protected = [], []
    with stage_progress("Admit agreed Egyptian texts; protect explicit heldout sources") as tick:
        for number, row in enumerate(iter_jsonl(root / cfg["arabic_inventory"]), 1):
            if heldout(row) and row.get("audio_path") and row.get("duration_seconds"):
                protected.append({**row, "bucket": row["group"]})
            reason = admission(row)
            if reason:
                reasons[row["group"] + "/" + reason] += 1
            else:
                ar.append(
                    {
                        **row,
                        "bucket": row["group"],
                        "group": "ar_egy",
                        "text": row["assistant_text"],
                        "audit_issues": row.get("issues", []),
                    }
                )
                counts[row["group"]] += 1
                hours[row["group"]] += row["duration_seconds"] / 3600
            tick(number)
    english = []
    with stage_progress("Collect ALL released US train and audited Egyptian English") as tick:
        for number, row in enumerate(iter_jsonl(root / cfg["english_inventory"]), 1):
            if (
                row["group"] == "en_egy"
                and row.get("duration_seconds")
                and not (set(row["issues"]) - {"assistant_punctuation_or_spacing_only"})
            ):
                english.append({**row, "bucket": "en_egy"})
            tick(number)
        pool = read(root / cfg["english_pool_config"])["pools"]["en"]
        for split in ("train", "validation", "test"):
            path = root / pool["pool"] / f"full_ctc_{split}.jsonl"
            pin(path, pool["manifest_sha256"][split])
            for number, row in enumerate(iter_jsonl(path), 1):
                value = {**row, "bucket": "en_us", "source": str(path), "row": number}
                (english if split == "train" else protected).append(value)
    en_val = []
    for lang in ("en", "ar"):
        path = Path(source["plan"]["manifests"][f"{lang}/validation"]["path"])
        for number, row in enumerate(iter_jsonl(path), 1):
            value = {
                **row,
                "bucket": "en_validation" if lang == "en" else "old_ar_validation",
                "row": row.get("source_row", number),
            }
            protected.append(value)
            if lang == "en":
                en_val.append(value)
    for name, values in (
        ("arabic_candidates.jsonl", ar),
        ("english_candidates.jsonl", english),
        ("protected.jsonl", protected),
        ("english_validation.jsonl", en_val),
    ):
        save_rows(output / name, values)
    write_json(
        marker,
        {
            "candidate_counts": counts,
            "candidate_hours": hours,
            "excluded_counts": reasons,
            "english_candidate_records": len(english),
            "dialect_verified_by_human": False,
            "label_text_policy": "assistant_original_orthography",
            "files": {
                name: sha(output / name)
                for name in (
                    "arabic_candidates.jsonl",
                    "english_candidates.jsonl",
                    "protected.jsonl",
                    "english_validation.jsonl",
                )
            },
        },
    )


def prepare_english(root: Path, cfg: dict[str, Any], output: Path) -> None:
    from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
    from qwen_hotword.training.mfa_round_common import words
    from qwen_hotword.training.mfa_round_labels import generate, top1_tokens
    from scripts.run_en_ar_training import ASSETS, FOUNDATION, VOCAB, VOCAB_SHA

    marker = output / "english_labels_report.json"
    if marker.exists():
        for name, checksum in read(marker)["files"].items():
            pin(output / name, checksum)
        return
    rows = list(iter_jsonl(output / "english_candidates.jsonl"))
    wanted = set()
    for row in rows:
        ws, issues = words(row["text"])
        if not issues:
            wanted.update(ws)
    existing = read(root / cfg["english_g2p"])
    if existing["vocab_sha256"] != VOCAB_SHA:
        raise ValueError("existing English dictionary uses other IDs")
    generated = dict(existing["words"])
    missing = wanted - generated.keys()
    if missing:
        foundation = {**read(FOUNDATION), "vocab": str(VOCAB), "vocab_sha256": VOCAB_SHA}
        conf = output / "english_foundation.json"
        write_json(conf, foundation)
        generated.update(
            generate(
                root,
                {
                    "foundation_config": str(conf),
                    "assets_dir": str(ASSETS),
                    "g2p_words_per_part": 2000,
                    "g2p_workers": 4,
                },
                output,
                {"en": missing},
            )["en"]
        )
    ready, review = [], []
    reasons: Counter[str] = Counter()
    with stage_progress("Label full English candidate pool", total=len(rows)) as tick:
        for index, row in enumerate(rows, 1):
            ws, issues = words(row["text"])
            ids, failures = top1_tokens(ws, generated)
            minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
            frames = qwen3_asr_audio_output_length(round(row["duration_seconds"] * 16000) // 160)
            issues += failures
            if not ids or minimum > 2 * frames:
                issues.append("empty_or_CTC_infeasible")
            if issues:
                reasons.update(i.split(":", 1)[0] for i in issues)
                review.append({**row, "label_issues": issues})
            else:
                ready.append({**row, "ids": ids})
            tick(index)
    save_rows(output / "english_ready.jsonl", ready)
    save_rows(output / "english_review.jsonl", review)
    write_json(
        marker,
        {
            "ready_rows": len(ready),
            "ready_hours": sum(r["duration_seconds"] for r in ready) / 3600,
            "review_rows": len(review),
            "reasons": reasons,
            "new_unique_words": len(missing),
            "files": {
                name: sha(output / name) for name in ("english_ready.jsonl", "english_review.jsonl")
            },
        },
    )


def freeze(root: Path, cfg: dict[str, Any], output: Path, workers: int) -> dict[str, Any]:
    marker = output / "prepare_report.json"
    if marker.exists():
        report = read(marker)
        for item in report["manifests"].values():
            pin(Path(item["path"]), item["sha256"])
        return dict(report)
    ar = []
    with stage_progress("Read compact ready Arabic labels; keep review cases isolated"):
        for piece in read(output / "labels/report.json")["pieces"]:
            pin(output / "labels" / piece["labels"], piece["labels_sha256"])
            for row in iter_jsonl(output / "labels" / piece["labels"]):
                ar.append(
                    {
                        k: row[k]
                        for k in (
                            "source",
                            "row",
                            "text",
                            "audio_path",
                            "duration_seconds",
                            "ids",
                            "bucket",
                        )
                    }
                )
    english = list(iter_jsonl(output / "english_ready.jsonl"))
    protected = list(iter_jsonl(output / "protected.jsonl"))
    en_val = list(iter_jsonl(output / "english_validation.jsonl"))
    all_rows = [*protected, *en_val, *english, *ar]
    fingerprint(all_rows, output, workers)
    protected_sha = {r["audio_sha256"] for r in protected}
    protected_groups = {group_key(r)[0] for r in protected}
    # Conflicting labels for byte-identical audio quarantine the WHOLE group.
    label_by_sha: dict[str, set[str]] = defaultdict(set)
    for row in [*english, *ar]:
        text_key = (
            transcript_key(row["text"]).casefold()
            if row["bucket"].startswith("en_")
            else comparison_key(row["text"])
        )
        label_by_sha[row["audio_sha256"]].add(json.dumps([text_key, row["ids"]]))
    conflicts = {key for key, labels in label_by_sha.items() if len(labels) > 1}
    seen = set()
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    # Same bytes in multiple Arabic sources belong to the highest-priority source.
    priority = {"en_egy": 0, "en_us": 1, "open_source": 2, "gaoshan_train": 3, "coarse_train": 4}
    for row in sorted(
        [*english, *ar], key=lambda r: (priority[r["bucket"]], r["source"], r["row"])
    ):
        key, basis = group_key(row)
        digest = row["audio_sha256"]
        reason = None
        if digest in protected_sha or key in protected_groups:
            reason = "protected_holdout"
        elif digest in conflicts:
            reason = "conflicting_labels_for_same_audio"
        elif digest in seen:
            reason = "duplicate_audio"
        if reason:
            reasons[row["bucket"] + "/" + reason] += 1
            continue
        seen.add(digest)
        row.update(recording_key=key, grouping_basis=basis)
        if row["bucket"].startswith("en_"):
            split = "train"
        else:
            hash_bucket = int(rank(key, cfg["seed"])[:12], 16) % 100
            split = (
                "test"
                if hash_bucket < cfg["test_percent"]
                else (
                    "validation"
                    if hash_bucket < cfg["test_percent"] + cfg["validation_percent"]
                    else "train"
                )
            )
        pools[split + "/" + row["bucket"]].append(row)
    arabic_capacity = {b: sum(r["duration_seconds"] for r in pools["train/" + b]) for b in BUCKETS}
    en = [*pools["train/en_egy"], *pools["train/en_us"]]
    english_capacity = sum(r["duration_seconds"] for r in en)
    capacity_report = {
        "english_hours": english_capacity / 3600,
        "arabic_hours": {k: v / 3600 for k, v in arabic_capacity.items()},
        "excluded_counts": reasons,
        "conflicting_audio_groups": len(conflicts),
        "minimum_training_hours": None,
    }
    write_json(output / "capacity_report.json", capacity_report)
    seconds = capacity(english_capacity, arabic_capacity, cfg["ratios"])
    selected = []
    # Retain Egyptian-accent English first, then US supplementation.
    egy = pools["train/en_egy"]
    first = select_seconds(egy, min(seconds, sum(r["duration_seconds"] for r in egy)), cfg["seed"])
    selected.extend(first)
    rest = seconds - sum(r["duration_seconds"] for r in first)
    if rest > 0:
        selected.extend(select_seconds(pools["train/en_us"], rest, cfg["seed"]))
    for bucket in BUCKETS:
        selected.extend(
            select_seconds(pools["train/" + bucket], seconds * cfg["ratios"][bucket], cfg["seed"])
        )
    # Validation uses only heldout EGY records, spreading the cap across available buckets.
    ar_val = []
    for bucket in BUCKETS:
        available = pools["validation/" + bucket]
        ar_val.extend(
            select_seconds(
                available,
                cfg["arabic_validation_hours"] * 3600 * cfg["ratios"][bucket],
                cfg["seed"],
            )
        )
    if not ar_val or not en_val:
        raise ValueError("empty validation language; see capacity report (no hour threshold)")
    val_seen = set()
    validation = list(ar_val)
    for row in en_val:
        if row["audio_sha256"] not in val_seen:
            val_seen.add(row["audio_sha256"])
            key, basis = group_key(row)
            row.update(recording_key=key, grouping_basis=basis)
            validation.append(row)
    test = [r for bucket in BUCKETS for r in pools["test/" + bucket]]
    # Keep inherited samples in original cache order. Sorting random content hashes would
    # reopen old 512-sample feature shards for nearly every sample (enormous repeated IO).
    old_plan = read(root / cfg["source_run"] / "plan.json")
    inherited_order: dict[str, int] = {}
    for item in old_plan["manifests"].values():
        for row in iter_jsonl(Path(item["path"])):
            inherited_order.setdefault(row["id"], len(inherited_order))
    manifests = {}
    stats = {}
    split_hashes = {}
    for split, values in (("train", selected), ("validation", validation), ("test", test)):
        records = [manifest(r, split) for r in values]
        records.sort(key=lambda r: (inherited_order.get(r["id"], len(inherited_order)), r["id"]))
        path = output / f"{split}.jsonl"
        save_rows(path, records)
        manifests[split] = {"path": str(path), "sha256": sha(path), "samples": len(records)}
        split_hashes[split] = {r["audio_sha256"] for r in records}
        stats[split] = {
            k: {
                "samples": sum(r["source_group"] == k for r in records),
                "hours": sum(r["duration_seconds"] for r in records if r["source_group"] == k)
                / 3600,
            }
            for k in sorted({r["source_group"] for r in records})
        }
        if len(split_hashes[split]) != len(records):
            raise ValueError("duplicate frozen audio IDs")
    if any(
        split_hashes[a] & split_hashes[b]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("cross-split audio-byte overlap")
    report = {
        "status": "prepared",
        "target_hours_per_language": seconds / 3600,
        "manifests": manifests,
        "by_split_source": stats,
        "capacity": capacity_report,
        "arabic_msa_used": False,
        "arabic_dialect_human_verified": False,
        "audio_byte_overlap": 0,
        "grouping_basis_counts": dict(Counter(r["grouping_basis"] for r in selected)),
        "rounding": "whole recordings, per-bucket duration error bounded by one recording",
        "human_pronunciation_review_completed": False,
        "vocab_sha256": cfg["vocab_sha256"],
    }
    write_json(marker, report)
    return report
