"""Explain a blocked freeze from saved fingerprints; never reopen audio or alter selections."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha, _verified


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _reasons(counts: list[int]) -> list[str]:
    train, validation, test, _ = counts
    reasons = []
    if train > 1:
        reasons.append("duplicate_train_file_bytes")
    if train and (validation or test):
        reasons.append("train_holdout_file_overlap")
    if validation and test:
        reasons.append("validation_test_protected_file_overlap")
    return reasons


def _totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"records": len(rows), "hours": sum(r["hours"] for r in rows)}
    for dimension in ("language", "source", "release"):
        groups: dict[str, Any] = {}
        for row in rows:
            key = (
                row[dimension]
                if dimension == "language"
                else f"{row['language']}::{row[dimension]}"
            )
            group = groups.setdefault(key, {"records": 0, "hours": 0.0})
            group["records"] += 1
            group["hours"] += row["hours"]
        result[f"by_{dimension}"] = groups
    return result


def inspect_conflicts(
    root: Path, *, top_n: int = 3, target_hours: float = 480
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 1 <= top_n <= 10 or not math.isfinite(target_hours) or target_hours <= 0:
        raise ValueError("invalid top_n or target_hours")
    if (root / "FAILED.txt").exists():
        raise ValueError("incomplete freeze marked FAILED")
    identities: dict[str, Any] = {}
    report = json.loads(_verified(root, "report.json", identities).read_text())
    if (
        report["status"] != "blocked_audio_conflicts"
        or report["training_manifest_ready"] is not False
    ):
        raise ValueError("requires a completed, conflict-blocked freeze")
    fingerprint = _verified(root, "audio_fingerprints.jsonl", identities)
    counts: dict[str, list[int]] = {}
    checked, byte_count = 0, 0
    for row in _rows(fingerprint):
        roles = row["roles"]
        if (
            not roles
            or len(roles) != len(set(roles))
            or set(roles) - {"train", "validation", "test"}
        ):
            raise ValueError("invalid fingerprint roles")
        digest = row["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid fingerprint SHA256")
        group = counts.setdefault(digest, [0, 0, 0, 0])
        for i, role in enumerate(("train", "validation", "test")):
            group[i] += role in roles
        group[3] += 1
        checked += 1
        byte_count += row["size_bytes"]
    unique_train = sum(value[0] > 0 for value in counts.values())
    observed: Counter[str] = Counter()
    for value in counts.values():
        observed.update(_reasons(value))
    expected = report["audio_identity_audit"]
    if (
        not observed
        or checked != expected["checked_files"]
        or byte_count != expected["bytes_read"]
        or unique_train != expected["unique_train_file_hashes"]
        or dict(observed) != expected["conflict_groups_by_reason"]
        or sum(v[0] for v in counts.values())
        != sum(s["records"] for s in report["selection"].values())
    ):
        raise ValueError("fingerprint aggregates disagree with freeze report")
    conflicts = {digest: value for digest, value in counts.items() if _reasons(value)}
    del counts
    groups: dict[str, Any] = {
        digest: {"counts": value, "train": [], "holdout_examples": []}
        for digest, value in conflicts.items()
    }
    train_paths: dict[str, str] = {}
    for row in _rows(fingerprint):
        digest = row["sha256"]
        if digest not in groups:
            continue
        path = row["audio_path"]
        if "train" in row["roles"]:
            if path in train_paths:
                raise ValueError("duplicate conflicting fingerprint path")
            train_paths[path] = digest
        heldout = sorted(set(row["roles"]) - {"train"})
        if heldout and len(groups[digest]["holdout_examples"]) < top_n:
            groups[digest]["holdout_examples"].append({"audio_path": path, "roles": heldout})
    for language in ("en", "es", "pt"):
        path = _verified(root, f"{language}.pending.jsonl", identities)
        count, hours = 0, 0.0
        for row in _rows(path):
            if row["split"] != "train" or row["balanced_language_bucket"] != language:
                raise ValueError("pending split/language differs")
            seconds = float(row["duration_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid training duration")
            count += 1
            hours += seconds / 3600
            digest = train_paths.pop(str(Path(row["audio_path"]).resolve()), None)
            if digest is None:
                continue
            groups[digest]["train"].append(
                {
                    "id": row["id"],
                    "audio_path": row["audio_path"],
                    "language": language,
                    "source": row["source_corpus"],
                    "release": row["release_source"],
                    "hours": seconds / 3600,
                    "label_signature": _digest(
                        [row["language"], row["text"], row["phoneme_token_ids"]]
                    ),
                }
            )
        previous = report["selection"][language]
        if count != previous["records"] or not math.isclose(hours, previous["hours"], abs_tol=1e-6):
            raise ValueError("pending counts/hours disagree with report")
    if train_paths or any(len(g["train"]) != g["counts"][0] for g in groups.values()):
        raise ValueError("conflicting train fingerprints missing from pending manifests")
    affected: list[dict[str, Any]] = []
    removals: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    action_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlap_roles: Counter[str] = Counter()
    overlap_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lost_unique_groups = 0
    for digest, group in sorted(groups.items()):
        train = sorted(group["train"], key=lambda r: (r["id"], r["audio_path"]))
        ntrain, nval, ntest, _ = group["counts"]
        label_mismatch = len({row["label_signature"] for row in train}) > 1
        duration_mismatch = (
            bool(train)
            and max(r["hours"] for r in train) - min(r["hours"] for r in train) > 1e-6 / 3600
        )
        if ntrain and (nval or ntest):
            action, remove = "exclude_all_train_copies_of_holdout", train
            role = "validation_and_test" if nval and ntest else "validation" if nval else "test"
            overlap_roles[role] += 1
            overlap_rows[role].extend(train)
        elif ntrain > 1 and (label_mismatch or duration_mismatch):
            action, remove = "quarantine_all_inconsistent_train_copies", train
        elif ntrain > 1:
            action, remove = "keep_one_identical_train_copy", train[1:]
        else:
            action, remove = "holdout_internal_conflict_requires_review", []
        lost_unique_groups += bool(train) and len(remove) == len(train)
        affected.extend(train)
        removals.extend(dict(row, proposed_action=action) for row in remove)
        actions[action] += 1
        action_rows[action].extend(remove)
        if len(examples[action]) < top_n:
            examples[action].append(
                {
                    "sha256": digest,
                    "reasons": _reasons(group["counts"]),
                    "train_copies": ntrain,
                    "validation_copies": nval,
                    "test_copies": ntest,
                    "training_label_mismatch": label_mismatch,
                    "training_duration_mismatch": duration_mismatch,
                    "proposed_remove_records": len(remove),
                    "proposed_keep_id": train[0]["id"]
                    if action == "keep_one_identical_train_copy"
                    else None,
                    "train_examples": [
                        {k: v for k, v in r.items() if k != "label_signature"}
                        for r in train[:top_n]
                    ],
                    "holdout_examples": group["holdout_examples"],
                }
            )
    removed = _totals(removals)
    remaining = {}
    for language, before in report["selection"].items():
        after = removed["by_language"].get(language, {"records": 0, "hours": 0.0})
        hours = before["hours"] - after["hours"]
        remaining[language] = {
            "records": before["records"] - after["records"],
            "hours": hours,
            "hours_to_reach_target": max(0.0, target_hours - hours),
        }
    projected_unique = unique_train - lost_unique_groups
    if sum(v["records"] for v in remaining.values()) != projected_unique:
        raise ValueError("proposed exclusions do not resolve train byte duplicates")
    for filename, identity in identities.items():
        if _sha(Path(filename)) != identity["sha256"]:
            raise ValueError("audit inputs changed")
    report = {
        "status": "completed",
        "scope": "read_only_conflict_diagnosis_and_proposed_exclusions",
        "freeze_dir": str(root),
        "inputs": identities,
        "conflict_groups_by_reason": dict(observed),
        "distinct_conflicting_hash_groups": len(groups),
        "train_holdout_groups_by_roles": dict(overlap_roles),
        "train_holdout_training_by_roles": {
            key: _totals(value) for key, value in overlap_rows.items()
        },
        "affected_training": _totals(affected),
        "proposed_action_groups": dict(actions),
        "proposed_removals": removed,
        "proposed_removals_by_action": {key: _totals(value) for key, value in action_rows.items()},
        "proposed_removed_ids_sha256": _digest(sorted(r["id"] for r in removals)),
        "projected_remaining": remaining,
        "projected_unique_train_file_hashes": projected_unique,
        "examples_by_action": dict(examples),
        "limitations": [
            "Removal and refill are proposals only; no files or manifests were changed.",
            "Exact train text/phone/language equality does not certify pronunciation accuracy.",
            "No holdout text or phone labels were opened or compared.",
            "Holdout protection may be a superset of released evaluation rows.",
            "Saved fingerprints describe the previous scan, not current audio bytes.",
            "All future replacement audio must pass heldout and train content deduplication.",
        ],
        "audio_read": False,
        "test_manifest_content_read": False,
        "model_loaded": False,
        "files_written": False,
        "training_ready": False,
    }
    return report, removals


def audit_conflicts(root: Path, *, top_n: int = 3, target_hours: float = 480) -> dict[str, Any]:
    report, _ = inspect_conflicts(root, top_n=top_n, target_hours=target_hours)
    return report
