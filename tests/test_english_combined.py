from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.english_combined import (
    build_english_us_temporal2x_training,
)


def test_english_combined_filters_non_us_and_assigns_complete_speakers(
    tmp_path: Path,
) -> None:
    ready = [
        _ready(
            f"us_{index}",
            f"/audio/US_101_F_{index:04d}_{index}.wav",
            1000.0 - index * 100.0,
        )
        for index in range(6)
    ]
    ready.extend(
        (
            _ready("au", "/audio/AU_201_F_0001_1.wav", 30.0),
            _ready("cn", "/audio/CN_301_M_0001_1.wav", 40.0),
        )
    )
    review = [
        _review(
            "recovered",
            "/audio/us_101_F_0000_99.wav",
            50.0,
            reasons=["ctc_length_infeasible"],
        ),
        _review(
            "blocked",
            "/audio/US_101_F_0000_100.wav",
            60.0,
            reasons=["dictionary_missing"],
        ),
    ]
    manifest_dir, inventory, audit = _write_inputs(tmp_path, ready, review)

    output = tmp_path / "combined"
    summary = build_english_us_temporal2x_training(
        manifest_dir,
        inventory,
        audit,
        output,
        speaker_split_seed=17,
    )

    rows_by_split = {
        split: _read_jsonl(output / f"full_ctc_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    assert len(rows) == 7
    assert {row["id"] for row in rows} >= {"recovered"}
    assert not ({"au", "cn", "blocked"} & {row["id"] for row in rows})
    assert all(row["speaker_id"].split("_", 1)[0].casefold() == "us" for row in rows)
    assert all(row["language"] == "en-US" for row in rows)
    assert all(row["dataset_version"] == "english-us-temporal2x-combined-v1" for row in rows)
    assert summary["original_ready_records"] == 6
    assert summary["recovered_records"] == 1
    assert summary["review_not_released_records"] == 1
    assert summary["excluded_by_speaker_filter_records"] == 2
    assert summary["input_manifest_records"] == 10
    assert summary["cross_split_speaker_overlaps"] == 0
    assert summary["test_set_sealed"] is True
    assert all(summary["split_records"][split] for split in rows_by_split)
    assert (output / "sha256.txt").is_file()

    speaker_splits: dict[str, set[str]] = {}
    for split, split_rows in rows_by_split.items():
        for row in split_rows:
            speaker_splits.setdefault(str(row["speaker_id"]), set()).add(split)
    assert all(len(splits) == 1 for splits in speaker_splits.values())


def test_english_combined_requires_passing_speaker_audit(tmp_path: Path) -> None:
    ready = [_ready("us", "/audio/US_101_F_0001_1.wav", 10.0)]
    manifest_dir, inventory, audit = _write_inputs(tmp_path, ready, [])
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["status"] = "warn"
    audit.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="status is not pass"):
        build_english_us_temporal2x_training(
            manifest_dir,
            inventory,
            audit,
            tmp_path / "output",
        )


def _write_inputs(
    root: Path,
    ready: list[dict[str, Any]],
    review: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    manifest_dir = root / "manifest"
    manifest_dir.mkdir()
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
    rows = ready + review
    inventory = root / "speaker_inventory.tsv"
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audio", "speaker_id", "source_split"),
            delimiter="\t",
        )
        writer.writeheader()
        counts: dict[str, int] = {}
        for row in rows:
            audio = str(row["audio_path"])
            speaker = Path(audio).stem.rpartition("_")[0]
            component = speaker.partition("_")[0]
            counts[component] = counts.get(component, 0) + 1
            writer.writerow(
                {
                    "audio": audio,
                    "speaker_id": speaker,
                    "source_split": "unsplit",
                }
            )
    audit = root / "summary.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "source_records": len(rows),
                "parsed_records": len(rows),
                "manifest_records": len(rows),
                "parse_failure_records": 0,
                "manifest_missing_records": 0,
                "manifest_extra_records": 0,
                "duplicate_speaker_utterance_keys": 0,
                "speaker_inventory_path": str(inventory),
                "first_component_counts": counts,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_dir, inventory, audit


def _ready(sample_id: str, audio: str, duration: float) -> dict[str, Any]:
    return _base_record(sample_id, audio, duration) | {
        "training_ready": True,
        "label_status": "ready",
        "issues": [],
        "estimated_ctc_input_length": 2,
        "ctc_minimum_input_length": 1,
    }


def _review(
    sample_id: str,
    audio: str,
    duration: float,
    *,
    reasons: list[str],
) -> dict[str, Any]:
    return _base_record(sample_id, audio, duration) | {
        "training_ready": False,
        "label_status": "needs_review",
        "issues": [{"reason": reason} for reason in reasons],
        "estimated_ctc_input_length": 2,
        "ctc_minimum_input_length": 3,
    }


def _base_record(sample_id: str, audio: str, duration: float) -> dict[str, Any]:
    return {
        "id": sample_id,
        "audio_path": audio,
        "audio_relative": audio,
        "text": sample_id,
        "language": "en-US",
        "dataset": "swift_us_english",
        "phoneme_token_ids": [1],
        "label_length": 1,
        "duration_seconds": duration,
        "source_tsv": "fixture.tsv",
        "row_number": 2,
        "split": "unsplit",
        "split_hash": 0.1,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
