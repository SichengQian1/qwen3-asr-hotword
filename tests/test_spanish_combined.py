from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.spanish_combined import (
    SpanishCorpusInput,
    build_spanish_temporal2x_training,
    combine_spanish_inputs,
)


def test_spanish_combined_preserves_explicit_and_assigns_unsplit_speakers(
    tmp_path: Path,
) -> None:
    explicit_rows = [
        _ready("core_train", "/audio/core_train.wav", "train", 100.0),
        _ready("core_validation", "/audio/core_validation.wav", "validation", 80.0),
        _ready("core_test", "/audio/core_test.wav", "test", 70.0),
    ]
    explicit = _write_corpus(
        tmp_path,
        "core",
        explicit_rows,
        [],
        {
            "/audio/core_train.wav": ("shared_train", "train"),
            "/audio/core_validation.wav": ("core_validation", "validation"),
            "/audio/core_test.wav": ("core_test", "test"),
        },
    )
    slr_ready = [
        _ready(f"slr_{index}", f"/audio/slr_{index}.wav", "unsplit", duration)
        for index, duration in enumerate((1000.0, 900.0, 800.0, 70.0, 60.0, 50.0))
    ]
    recovered = _review(
        "slr_recovered",
        "/audio/slr_recovered.wav",
        "unsplit",
        40.0,
        reasons=["ctc_length_infeasible"],
        estimated=2,
        minimum=3,
    )
    blocked = _review(
        "slr_blocked",
        "/audio/slr_blocked.wav",
        "unsplit",
        30.0,
        reasons=["dictionary_missing"],
        estimated=2,
        minimum=3,
    )
    slr_metadata = {
        f"/audio/slr_{index}.wav": (f"slr_speaker_{index}", "unsplit")
        for index in range(6)
    }
    slr_metadata["/audio/slr_recovered.wav"] = ("slr_speaker_0", "unsplit")
    slr_metadata["/audio/slr_blocked.wav"] = ("slr_blocked", "unsplit")
    slr = _write_corpus(
        tmp_path,
        "slr",
        slr_ready,
        [recovered, blocked],
        slr_metadata,
    )

    output = tmp_path / "combined"
    summary = build_spanish_temporal2x_training(
        [explicit, slr],
        output,
        speaker_split_seed=17,
    )

    rows_by_split = {
        split: _read_jsonl(output / f"full_ctc_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    assert {row["id"] for row in rows_by_split["train"]} >= {"core_train"}
    assert {row["id"] for row in rows_by_split["validation"]} >= {
        "core_validation"
    }
    assert {row["id"] for row in rows_by_split["test"]} >= {"core_test"}
    all_rows = [row for rows in rows_by_split.values() for row in rows]
    assert "slr_recovered" in {row["id"] for row in all_rows}
    assert "slr_blocked" not in {row["id"] for row in all_rows}
    assert len(all_rows) == 10
    assert summary["recovered_records"] == 1
    assert summary["review_not_released_records"] == 1
    assert summary["input_manifest_records"] == 11
    assert summary["released_records"] == 10
    assert summary["test_set_content_processed_for_mechanical_copy"] is True
    assert summary["cross_split_speaker_overlaps"] == 0
    assert summary["unsplit_speakers_assigned"] == 6
    assert all(summary["unsplit_assignment"][split]["speakers"] for split in rows_by_split)
    assert all(row["ctc_time_upsampling_factor"] == 2 for row in all_rows)
    assert all(row["language"] == "es" for row in all_rows)
    assert (output / "sha256.txt").is_file()

    speaker_splits: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            speaker_splits.setdefault(row["speaker_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in speaker_splits.values())


def test_spanish_combined_rejects_explicit_speaker_leakage(tmp_path: Path) -> None:
    corpus = _write_corpus(
        tmp_path,
        "core",
        [
            _ready("train", "/audio/train.wav", "train", 10.0),
            _ready("test", "/audio/test.wav", "test", 10.0),
        ],
        [],
        {
            "/audio/train.wav": ("same_speaker", "train"),
            "/audio/test.wav": ("same_speaker", "test"),
        },
    )

    with pytest.raises(ValueError, match="multiple explicit splits"):
        build_spanish_temporal2x_training([corpus], tmp_path / "output")


def test_combine_spanish_inputs_requires_matching_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must match exactly"):
        combine_spanish_inputs(
            [f"core={tmp_path / 'manifest'}"],
            [f"other={tmp_path / 'source.tsv'}"],
        )


def _write_corpus(
    root: Path,
    name: str,
    ready: list[dict[str, Any]],
    review: list[dict[str, Any]],
    metadata: dict[str, tuple[str, str]],
) -> SpanishCorpusInput:
    manifest_dir = root / name / "manifest"
    manifest_dir.mkdir(parents=True)
    _write_jsonl(manifest_dir / "train_ready.jsonl", ready)
    _write_jsonl(manifest_dir / "needs_review.jsonl", review)
    (manifest_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "ready_records": len(ready),
                "review_records": len(review),
                "source_records": len(ready) + len(review),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_tsv = root / name / "source.tsv"
    with source_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audio", "speaker_id", "source_split"),
            delimiter="\t",
        )
        writer.writeheader()
        for audio, (speaker, split) in metadata.items():
            writer.writerow(
                {
                    "audio": audio,
                    "speaker_id": speaker,
                    "source_split": split,
                }
            )
    return SpanishCorpusInput(name, manifest_dir, source_tsv)


def _ready(
    sample_id: str,
    audio: str,
    split: str,
    duration: float,
) -> dict[str, Any]:
    return _base_record(sample_id, audio, split, duration) | {
        "training_ready": True,
        "label_status": "ready",
        "issues": [],
        "estimated_ctc_input_length": 2,
        "ctc_minimum_input_length": 1,
    }


def _review(
    sample_id: str,
    audio: str,
    split: str,
    duration: float,
    *,
    reasons: list[str],
    estimated: int,
    minimum: int,
) -> dict[str, Any]:
    return _base_record(sample_id, audio, split, duration) | {
        "training_ready": False,
        "label_status": "needs_review",
        "issues": [{"reason": reason} for reason in reasons],
        "estimated_ctc_input_length": estimated,
        "ctc_minimum_input_length": minimum,
    }


def _base_record(
    sample_id: str,
    audio: str,
    split: str,
    duration: float,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "audio_path": audio,
        "audio_relative": audio,
        "text": sample_id,
        "language": "es",
        "dataset": "fixture",
        "phoneme_token_ids": [1],
        "label_length": 1,
        "duration_seconds": duration,
        "source_tsv": "fixture.tsv",
        "row_number": 2,
        "split": split,
        "split_hash": 0.1,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
