"""Read-only capacity bound for expansion with explicit Latin-American metadata."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_inventory import classify_spanish_accent

LATAM = {"argentinian_rioplatense_metadata", "latin_american_metadata"}
CORE = {"slr61", "common_voice_rioplatense_v26"}
AUXILIARY = "common_voice_latam_auxiliary"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified(root: Path, name: str, identities: dict[str, Any]) -> Path:
    # Verify only requested assets; never open sealed test manifests via a wildcard.
    checks: dict[str, str] = {}
    for line in (root / "sha256.txt").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip().removeprefix("*").removeprefix("./")
        if relative in checks:
            raise ValueError(f"duplicate checksum entry: {relative}")
        checks[relative] = expected
    path = root / name
    actual = _sha(path)
    if checks.get(name) != actual:
        raise ValueError(f"SHA256 mismatch or missing identity: {path}")
    identities[str(path)] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return path


def _tsv(path: Path, required: set[str]) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"missing TSV columns: {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"malformed TSV row: {path}:{reader.line_num}")
            yield row


def _duration(value: Any) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("duration must be finite and positive")
    return seconds


def _add(groups: dict[str, Any], key: str, seconds: float) -> None:
    group = groups.setdefault(key, {"records": 0, "hours": 0.0})
    group["records"] += 1
    group["hours"] += seconds / 3600


def audit_spanish_capacity(
    inventory_root: Path, pool_root: Path, *, target_hours: float = 480.0
) -> dict[str, Any]:
    """Bound additional CV capacity; do not claim candidates have usable CTC labels."""
    if not math.isfinite(target_hours) or target_hours <= 0:
        raise ValueError("target_hours must be finite and positive")
    identities: dict[str, Any] = {}
    summary = json.loads(_verified(pool_root, "split_summary.json", identities).read_text())
    if summary["status"] != "pass" or summary["test_set_used"] is not False:
        raise ValueError("existing pool must have passed with sealed test unused")
    assignments: dict[str, str] = {}
    assignment_path = _verified(pool_root, "speaker_split_assignments.tsv", identities)
    for row in _tsv(assignment_path, {"speaker_id", "split"}):
        speaker, split = row["speaker_id"], row["split"]
        if not speaker or speaker in assignments or split not in {"train", "validation", "test"}:
            raise ValueError("invalid/duplicate speaker assignment")
        assignments[speaker] = split

    totals: dict[str, Any] = {}
    scope: dict[str, Any] = {}
    existing_cv: dict[str, dict[str, Any]] = {}
    existing_ids: set[str] = set()
    existing_audio: set[str] = set()
    scope_issues: list[str] = []
    for split in ("train", "validation"):
        path = _verified(pool_root, f"full_ctc_{split}.jsonl", identities)
        if identities[str(path)]["sha256"] != summary["manifest_sha256"][split]:
            raise ValueError(f"manifest identity differs from split summary: {split}")
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                audio = str(Path(row["audio_path"]).resolve())
                if row["id"] in existing_ids or audio in existing_audio:
                    raise ValueError("duplicate/overlapping existing train/validation audio or ID")
                existing_ids.add(row["id"])
                existing_audio.add(audio)
                if row["split"] != split or assignments.get(row["speaker_id"]) != split:
                    raise ValueError("manifest split/speaker assignment mismatch")
                seconds = _duration(row["duration_seconds"])
                _add(totals, split, seconds)
                source = row["source_corpus"]
                if row["language"].lower() not in {"es", "es-ar", "es-419"}:
                    _add(scope, f"{split}::unexpected_language", seconds)
                    scope_issues.append(row["id"])
                elif source in CORE:
                    _add(scope, f"{split}::{source}::existing_curated_source_scope", seconds)
                elif source == AUXILIARY:
                    clip = Path(audio).name
                    if clip in existing_cv:
                        raise ValueError("duplicate existing CV clip basename")
                    existing_cv[clip] = row
                else:
                    _add(scope, f"{split}::unapproved_source", seconds)
                    scope_issues.append(row["id"])
        if totals[split]["records"] != summary["split_records"][split] or not math.isclose(
            totals[split]["hours"], summary["split_audio_hours"][split], abs_tol=1e-5
        ):
            raise ValueError(f"existing {split} totals differ from summary")

    cv_path = _verified(inventory_root, "common_voice_inventory.tsv", identities)
    required = {
        "audio",
        "speaker_id",
        "accent",
        "locale",
        "duration_seconds",
        "audio_status",
        "official_split",
        "metadata_text_match",
        "core_overlap_split",
    }
    decisions: dict[str, Any] = {}
    tier_totals: dict[str, Any] = {}
    new_speaker_hours: defaultdict[str, float] = defaultdict(float)
    seen_clips: set[str] = set()
    for row in _tsv(cv_path, required):
        clip = Path(row["audio"]).name
        if clip in seen_clips:
            raise ValueError(f"ambiguous duplicate CV clip: {clip}")
        seen_clips.add(clip)
        tier = classify_spanish_accent(row["accent"])
        seconds = _duration(row["duration_seconds"]) if row["audio_status"] == "ok" else 0.0
        _add(tier_totals, tier, seconds)
        current = existing_cv.pop(clip, None)
        if current is not None:
            good = (
                tier in LATAM
                and row["locale"] == "es"
                and row["audio_status"] == "ok"
                and row["metadata_text_match"] == "true"
                and row["official_split"] == "train"
                and not row["core_overlap_split"]
                and current["speaker_id"] == row["speaker_id"]
                and math.isclose(seconds, current["duration_seconds"], abs_tol=1e-3)
            )
            label = "explicit_latam_metadata" if good else "metadata_scope_issue"
            _add(
                scope,
                f"{current['split']}::{AUXILIARY}::{label}",
                _duration(current["duration_seconds"]),
            )
            if not good:
                scope_issues.append(current["id"])
            reason = f"already_in_{current['split']}"
        else:
            reason = _candidate_reason(row, tier, assignments)
        _add(decisions, reason, seconds)
        if reason == "additional_raw_candidate":
            new_speaker_hours[row["speaker_id"]] += seconds / 3600
    for current in existing_cv.values():
        _add(
            scope,
            f"{current['split']}::{AUXILIARY}::missing_inventory_join",
            _duration(current["duration_seconds"]),
        )
        scope_issues.append(current["id"])

    mls = json.loads(_verified(inventory_root, "mls_summary.json", identities).read_text())
    extra = decisions.get("additional_raw_candidate", {"records": 0, "hours": 0.0})
    upper_bound = totals["train"]["hours"] + extra["hours"]
    status = (
        "insufficient_raw_capacity" if upper_bound < target_hours else "labels_not_yet_verified"
    )
    if scope_issues:
        status = "blocked_existing_scope_audit"
    return {
        "status": status,
        "target_unique_train_hours": target_hours,
        "inputs": identities,
        "existing_pool": totals,
        "existing_latam_scope": scope,
        "scope_issue_count": len(scope_issues),
        "scope_issue_examples": scope_issues[:10],
        "cv_accent_tiers_reclassified": tier_totals,
        "cv_capacity_decisions": decisions,
        "additional_raw_candidates": extra,
        "additional_candidate_speakers": len(new_speaker_hours),
        "top5_additional_speaker_hours": sorted(new_speaker_hours.values(), reverse=True)[:5],
        "mls": {
            "inventory_hours": mls["valid_audio_hours"],
            "admitted_hours": 0,
            "reason": "no_explicit_latin_american_dialect_evidence_in_existing_inventory",
        },
        "optimistic_total_train_hours_before_label_filtering": upper_bound,
        "minimum_extra_hours_needed_even_if_all_candidates_pass": max(
            0, target_hours - upper_bound
        ),
        "gap_from_existing_ready_train_hours": max(0, target_hours - totals["train"]["hours"]),
        "limitations": [
            "Capacity is an optimistic upper bound, not a prepared training dataset.",
            "New candidate G2P/CTC labels and audio content duplicates were not audited.",
            "Speaker caps and additional quality filtering can only reduce this upper bound.",
            "Core SLR61/Rioplatense scope relies on previously approved corpus provenance.",
            "CV accents are metadata evidence, not an acoustic dialect or label certification.",
            "Unreleased rows outside the two candidate inventories are not counted.",
        ],
        "test_manifest_content_read": False,
        "test_speaker_assignment_metadata_read": True,
        "audio_read": False,
        "model_loaded": False,
        "training_dataset_created": False,
        "files_written": False,
    }


def _candidate_reason(row: dict[str, str], tier: str, assignments: dict[str, str]) -> str:
    if row["audio_status"] != "ok":
        return "exclude_invalid_audio"
    if row["core_overlap_split"]:
        return "exclude_core_overlap"
    if row["official_split"] != "train":
        return "exclude_not_official_train"
    if row["locale"] != "es" or row["metadata_text_match"] != "true":
        return "exclude_metadata_mismatch"
    if not row["speaker_id"]:
        return "exclude_missing_speaker"
    if assignments.get(row["speaker_id"]) in {"validation", "test"}:
        return "exclude_existing_holdout_speaker"
    if tier not in LATAM:
        return f"exclude_accent_{tier}"
    return "additional_raw_candidate"
