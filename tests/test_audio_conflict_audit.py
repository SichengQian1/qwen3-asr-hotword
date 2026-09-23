from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.audio_conflict_audit import audit_conflicts
from qwen_hotword.training.spanish_capacity import _sha


def checks(root: Path) -> None:
    (root / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(root.iterdir()) if p.name != "sha256.txt")
    )


def setup(root: Path, *, inconsistent: str | None = None) -> None:
    pending: dict[str, list[dict[str, Any]]] = {lang: [] for lang in ("en", "es", "pt")}
    fingerprints = []
    # No audio or heldout manifest exists. All diagnosis must use only these saved files.
    for name, language, ntrain, holdout in (
        ("same", "pt", 2, []),
        ("leak", "pt", 3, ["validation"]),
        ("labels", "en", 2, []),
        ("unique", "es", 1, []),
        ("test", "pt", 0, ["test"]),
    ):
        digest = hashlib.sha256(name.encode()).hexdigest()
        for i in range(ntrain + len(holdout)):
            is_train = i < ntrain
            path = str(root / f"{name}_{i}.wav")
            fingerprints.append(
                {
                    "audio_path": path,
                    "sha256": digest,
                    "size_bytes": 10,
                    "roles": ["train"] if is_train else [holdout[i - ntrain]],
                }
            )
            if is_train:
                row = {
                    "id": f"{name}_{i}",
                    "audio_path": path,
                    "split": "train",
                    "balanced_language_bucket": language,
                    "language": language,
                    "source_corpus": "source",
                    "release_source": "original_ready",
                    "text": "same" if name != "labels" or i == 0 else "different",
                    "phoneme_token_ids": [1, 2],
                    "duration_seconds": 4.0,
                }
                if inconsistent and name == "same" and i == 1:
                    row[inconsistent] = [1, 3] if inconsistent == "phoneme_token_ids" else 5.0
                pending[language].append(row)
    (root / "audio_fingerprints.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fingerprints)
    )
    for language, rows in pending.items():
        (root / f"{language}.pending.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "status": "blocked_audio_conflicts",
        "training_manifest_ready": False,
        "selection": {
            lang: {"records": len(rows), "hours": sum(r["duration_seconds"] for r in rows) / 3600}
            for lang, rows in pending.items()
        },
        "audio_identity_audit": {
            "checked_files": 10,
            "bytes_read": 100,
            "unique_train_file_hashes": 4,
            "conflict_groups_by_reason": {
                "duplicate_train_file_bytes": 3,
                "train_holdout_file_overlap": 1,
            },
        },
    }
    (root / "report.json").write_text(json.dumps(report))
    checks(root)


def test_read_only_proposals_count_overlapping_reasons_once(tmp_path: Path) -> None:
    setup(tmp_path)
    before = {p.name: _sha(p) for p in tmp_path.iterdir()}
    result = audit_conflicts(tmp_path, target_hours=4 / 3600)
    assert result["status"] == "completed"
    assert result["distinct_conflicting_hash_groups"] == 3  # 3+1 reasons overlap.
    assert result["affected_training"]["records"] == 7
    assert result["proposed_removals"]["records"] == 6
    assert result["proposed_removals"]["hours"] == pytest.approx(24 / 3600)
    assert result["train_holdout_groups_by_roles"] == {"validation": 1}
    assert result["proposed_action_groups"] == {
        "keep_one_identical_train_copy": 1,
        "exclude_all_train_copies_of_holdout": 1,
        "quarantine_all_inconsistent_train_copies": 1,
    }
    assert result["projected_unique_train_file_hashes"] == 2
    assert result["projected_remaining"]["pt"]["records"] == 1
    assert result["projected_remaining"]["en"]["hours_to_reach_target"] == pytest.approx(4 / 3600)
    assert not result["audio_read"] and not result["files_written"]
    assert {p.name: _sha(p) for p in tmp_path.iterdir()} == before
    example = result["examples_by_action"]["keep_one_identical_train_copy"][0]
    assert example["proposed_keep_id"] == "same_0"
    assert "text" not in example["train_examples"][0]


@pytest.mark.parametrize("field", ["phoneme_token_ids", "duration_seconds"])
def test_inconsistent_labels_or_durations_quarantine_all(tmp_path: Path, field: str) -> None:
    setup(tmp_path, inconsistent=field)
    result = audit_conflicts(tmp_path)
    assert result["proposed_action_groups"]["quarantine_all_inconsistent_train_copies"] == 2
    assert result["projected_unique_train_file_hashes"] == 1
    assert result["proposed_removals"]["records"] == 7


@pytest.mark.parametrize("failure", ["hash", "aggregate", "missing_join", "failed"])
def test_invalid_saved_results_fail_closed(tmp_path: Path, failure: str) -> None:
    setup(tmp_path)
    if failure == "failed":
        (tmp_path / "FAILED.txt").write_text("incomplete")
    elif failure == "hash":
        (tmp_path / "pt.pending.jsonl").write_text("tampered")
    elif failure == "aggregate":
        path = tmp_path / "report.json"
        report = json.loads(path.read_text())
        report["audio_identity_audit"]["checked_files"] += 1
        path.write_text(json.dumps(report))
        checks(tmp_path)
    else:
        path = tmp_path / "pt.pending.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["audio_path"] = str(tmp_path / "unmatched.wav")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        checks(tmp_path)
    with pytest.raises(ValueError):
        audit_conflicts(tmp_path)
