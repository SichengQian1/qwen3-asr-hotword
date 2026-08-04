from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from qwen_hotword.training.ctc_overfit import (
    collapse_ctc_ids,
    edit_distance,
    load_experiment_records,
)


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)


def _write_manifest(path: Path, audio_path: Path, **overrides: object) -> None:
    record: dict[str, object] = {
        "experiment": "A",
        "split": "train",
        "id": "sample-1",
        "audio_path": str(audio_path),
        "text": "bom dia",
        "language": "pt-BR",
        "phoneme_token_ids": [2, 3, 4],
        "label_length": 3,
        "ctc_minimum_input_length": 3,
    }
    record.update(overrides)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_load_experiment_records_validates_training_contract(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    manifest_path = tmp_path / "experiment.jsonl"
    _write_wav(audio_path)
    _write_manifest(manifest_path, audio_path)

    records = load_experiment_records(manifest_path, num_classes=10)

    assert len(records) == 1
    assert records[0].sample_id == "sample-1"
    assert records[0].token_ids == (2, 3, 4)
    assert records[0].ctc_time_upsampling_factor == 1


def test_load_experiment_records_reads_temporal_upsampling_contract(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    manifest_path = tmp_path / "experiment.jsonl"
    _write_wav(audio_path)
    _write_manifest(manifest_path, audio_path, ctc_time_upsampling_factor=2)

    records = load_experiment_records(manifest_path, num_classes=10)

    assert records[0].ctc_time_upsampling_factor == 2


def test_load_experiment_records_accepts_expected_experiment_b_split(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    manifest_path = tmp_path / "experiment.jsonl"
    _write_wav(audio_path)
    _write_manifest(
        manifest_path,
        audio_path,
        experiment="B",
        split="validation",
    )

    records = load_experiment_records(
        manifest_path,
        num_classes=10,
        expected_experiment="B",
        expected_split="validation",
    )

    assert len(records) == 1


@pytest.mark.parametrize(  # type: ignore[misc]
    ("overrides", "message"),
    [
        ({"phoneme_token_ids": [0, 2, 3]}, "blank or out-of-range"),
        ({"phoneme_token_ids": [2, 10, 3]}, "blank or out-of-range"),
        ({"label_length": 2}, "label_length"),
        ({"ctc_minimum_input_length": 2}, "CTC minimum"),
        ({"experiment": "B"}, "not Experiment A"),
    ],
)
def test_load_experiment_records_rejects_invalid_rows(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    audio_path = tmp_path / "audio.wav"
    manifest_path = tmp_path / "experiment.jsonl"
    _write_wav(audio_path)
    _write_manifest(manifest_path, audio_path, **overrides)

    with pytest.raises(ValueError, match=message):
        load_experiment_records(manifest_path, num_classes=10)


def test_collapse_ctc_ids_removes_repeats_and_blanks_in_ctc_order() -> None:
    assert collapse_ctc_ids([0, 2, 2, 0, 2, 3, 3, 0]) == [2, 2, 3]


def test_edit_distance_counts_insertions_deletions_and_substitutions() -> None:
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert edit_distance([1, 2, 3], [1, 2]) == 1
    assert edit_distance([1, 2], [1, 2, 3]) == 1
