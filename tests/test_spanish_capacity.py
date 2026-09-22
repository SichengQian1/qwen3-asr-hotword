from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.spanish_capacity import audit_spanish_capacity


def _checksum(root: Path) -> None:
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
        for p in sorted(root.iterdir())
        if p.name != "sha256.txt"
    ]
    # Prove a listed sealed test manifest need not be present or opened.
    lines.append(f"{'0' * 64}  full_ctc_test.jsonl\n")
    (root / "sha256.txt").write_text("".join(lines))


def _tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _cv(clip: str, speaker: str, accent: str = "México", **extra: str) -> dict[str, str]:
    return {
        "audio": f"/cv/{clip}.mp3",
        "speaker_id": speaker,
        "accent": accent,
        "locale": "es",
        "duration_seconds": "3600",
        "audio_status": "ok",
        "official_split": "train",
        "metadata_text_match": "true",
        "core_overlap_split": "",
        # Old inventory tiers must not override reclassification of America central.
        "accent_tier": "other_unclassified_metadata",
        **extra,
    }


def _manifest(clip: str, speaker: str, split: str, source: str) -> dict[str, Any]:
    return {
        "id": clip,
        "audio_path": f"/cv/{clip}.mp3",
        "speaker_id": speaker,
        "split": split,
        "duration_seconds": 3600,
        "source_corpus": source,
        "language": "es",
    }


@pytest.fixture
def assets(tmp_path: Path) -> tuple[Path, Path]:
    inv, pool = tmp_path / "inventory", tmp_path / "pool"
    inv.mkdir()
    pool.mkdir()
    train = [
        _manifest("old", "t", "train", "common_voice_latam_auxiliary"),
        _manifest("core", "c", "train", "slr61"),
    ]
    validation = [_manifest("val", "v", "validation", "common_voice_latam_auxiliary")]
    hashes = {}
    for split, records in (("train", train), ("validation", validation)):
        path = pool / f"full_ctc_{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    (pool / "split_summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "test_set_used": False,
                "manifest_sha256": hashes,
                "split_records": {"train": 2, "validation": 1},
                "split_audio_hours": {"train": 2, "validation": 1},
            }
        )
    )
    _tsv(
        pool / "speaker_split_assignments.tsv",
        [
            {"speaker_id": speaker, "split": split}
            for speaker, split in (
                ("t", "train"),
                ("c", "train"),
                ("v", "validation"),
                ("x", "test"),
            )
        ],
    )
    _tsv(
        inv / "common_voice_inventory.tsv",
        [
            _cv("old", "t"),
            _cv("val", "v"),
            _cv("extra", "new", "América central"),
            _cv("same_train_speaker", "t"),
            _cv("new_holdout_speaker", "v"),
            _cv("test_speaker", "x"),
            _cv("official_test", "n", official_split="test"),
            _cv("unassigned", "n", official_split="unassigned_validated"),
            _cv("peninsular", "n", "España"),
            _cv("unknown", "n", ""),
            _cv("mixed", "n", "México España"),
            _cv("core_duplicate", "c", core_overlap_split="train"),
            _cv("missing_speaker", ""),
        ],
    )
    (inv / "mls_summary.json").write_text(json.dumps({"valid_audio_hours": 900}))
    _checksum(inv)
    _checksum(pool)
    return inv, pool


def test_capacity_deduplicates_existing_and_excludes_holdout_and_nonlatam(
    assets: tuple[Path, Path],
) -> None:
    inv, pool = assets
    before = {p: p.read_bytes() for root in assets for p in root.iterdir()}
    r = audit_spanish_capacity(inv, pool, target_hours=480)
    assert r["status"] == "insufficient_raw_capacity"
    assert r["existing_pool"]["train"]["hours"] == 2
    assert r["existing_pool"]["validation"]["hours"] == 1
    assert r["additional_raw_candidates"] == {"records": 2, "hours": 2}
    assert r["optimistic_total_train_hours_before_label_filtering"] == 4
    assert r["minimum_extra_hours_needed_even_if_all_candidates_pass"] == 476
    assert r["mls"]["admitted_hours"] == 0
    assert r["scope_issue_count"] == 0
    assert not r["test_manifest_content_read"] and not r["files_written"]
    assert before == {p: p.read_bytes() for root in assets for p in root.iterdir()}


def test_enough_raw_hours_never_claims_training_ready(assets: tuple[Path, Path]) -> None:
    r = audit_spanish_capacity(*assets, target_hours=3)
    assert r["status"] == "labels_not_yet_verified"
    assert not r["training_dataset_created"]


@pytest.mark.parametrize("bad", ["España", ""])
def test_existing_validation_scope_problem_blocks_capacity_claim(
    assets: tuple[Path, Path],
    bad: str,
) -> None:
    inv, pool = assets
    path = inv / "common_voice_inventory.tsv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[1]["accent"] = bad
    _tsv(path, rows)
    _checksum(inv)
    r = audit_spanish_capacity(inv, pool)
    assert r["status"] == "blocked_existing_scope_audit"
    assert r["scope_issue_examples"] == ["val"]


def test_tampered_inventory_is_rejected(assets: tuple[Path, Path]) -> None:
    inv, pool = assets
    with (inv / "common_voice_inventory.tsv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="SHA256"):
        audit_spanish_capacity(inv, pool)


def test_duplicate_clip_under_different_paths_is_rejected(assets: tuple[Path, Path]) -> None:
    inv, pool = assets
    path = inv / "common_voice_inventory.tsv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.append({**rows[2], "audio": "/other/extra.mp3"})
    _tsv(path, rows)
    _checksum(inv)
    with pytest.raises(ValueError, match="duplicate CV clip"):
        audit_spanish_capacity(inv, pool)
