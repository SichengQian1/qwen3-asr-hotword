from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.training.full_training import (
    EXPERIMENT_NAME,
    build_full_training_splits,
)


def _ready_record(index: int, split_hash: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "training_ready": True,
        "label_status": "ready",
        "id": f"sample-{index:03d}",
        "audio_path": f"/audio/{index:03d}.wav",
        "audio_relative": f"{index:03d}.wav",
        "text": "bom dia",
        "language": "pt-BR",
        "phoneme_token_ids": [2, 3, 4],
        "label_length": 3,
        "ctc_minimum_input_length": 3,
        "estimated_ctc_input_length": 13,
        "duration_seconds": 1.0,
        "source_tsv": "/data/source.tsv",
        "row_number": index + 2,
        "split_hash": split_hash,
    }


def _write_ready_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_full_training_splits_is_complete_disjoint_and_reproducible(
    tmp_path: Path,
) -> None:
    records = [_ready_record(index, (index + 0.5) / 100) for index in range(100)]
    source = tmp_path / "ready.jsonl"
    _write_ready_manifest(source, records)

    first = build_full_training_splits(source, tmp_path / "first")
    second = build_full_training_splits(source, tmp_path / "second")

    assert first.status == "pass"
    assert first.source_records == 100
    assert first.split_records == {"train": 96, "validation": 2, "test": 2}
    assert first.split_audio_hours == pytest.approx(
        {"train": 96 / 3600, "validation": 2 / 3600, "test": 2 / 3600}
    )
    assert first.cross_split_id_overlaps == 0
    assert first.cross_split_audio_overlaps == 0
    assert first.test_set_sealed is True
    assert first.manifest_sha256 == second.manifest_sha256

    split_ids: dict[str, set[str]] = {}
    for split, path in first.manifest_paths.items():
        rows = _read_jsonl(Path(path))
        assert all(row["experiment"] == EXPERIMENT_NAME for row in rows)
        assert all(row["split"] == split for row in rows)
        split_ids[split] = {str(row["id"]) for row in rows}
    assert len(set().union(*split_ids.values())) == 100
    assert not (split_ids["train"] & split_ids["validation"])
    assert not (split_ids["train"] & split_ids["test"])
    assert not (split_ids["validation"] & split_ids["test"])


def test_build_full_training_splits_rejects_non_ready_input(tmp_path: Path) -> None:
    records = [_ready_record(index, (index + 0.5) / 100) for index in range(100)]
    records[4]["training_ready"] = False
    records[4]["label_status"] = "needs_review"
    source = tmp_path / "ready.jsonl"
    _write_ready_manifest(source, records)

    with pytest.raises(ValueError, match="not marked training-ready"):
        build_full_training_splits(source, tmp_path / "output")


def test_build_full_training_splits_refuses_accidental_overwrite(tmp_path: Path) -> None:
    records = [_ready_record(index, (index + 0.5) / 100) for index in range(100)]
    source = tmp_path / "ready.jsonl"
    _write_ready_manifest(source, records)
    output = tmp_path / "output"
    build_full_training_splits(source, output)

    with pytest.raises(ValueError, match="already contains"):
        build_full_training_splits(source, output)
