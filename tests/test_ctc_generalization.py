from __future__ import annotations

from pathlib import Path

import pytest

from qwen_hotword.training.ctc_generalization import (
    GeneralizationEpochMetrics,
    validate_disjoint_records,
    validation_rank,
)
from qwen_hotword.training.ctc_overfit import EpochMetrics, ExperimentRecord


def _record(sample_id: str, audio_path: Path) -> ExperimentRecord:
    return ExperimentRecord(
        sample_id=sample_id,
        audio_path=audio_path,
        text="bom dia",
        language="pt-BR",
        token_ids=(2, 3),
        ctc_minimum_input_length=2,
    )


def _epoch(epoch: int, validation_per: float, validation_loss: float) -> GeneralizationEpochMetrics:
    train = EpochMetrics(epoch, 1.0, 0.5, 5, 10)
    validation = EpochMetrics(
        epoch,
        validation_loss,
        validation_per,
        int(validation_per * 100),
        100,
    )
    return GeneralizationEpochMetrics(epoch, 1e-3, train, validation)


def test_validate_disjoint_records_accepts_distinct_splits(tmp_path: Path) -> None:
    validate_disjoint_records(
        [_record("train-1", tmp_path / "train.wav")],
        [_record("validation-1", tmp_path / "validation.wav")],
    )


@pytest.mark.parametrize("duplicate", ["id", "audio"])
def test_validate_disjoint_records_rejects_overlap(tmp_path: Path, duplicate: str) -> None:
    train = _record("sample-1", tmp_path / "train.wav")
    validation = _record(
        "sample-1" if duplicate == "id" else "sample-2",
        tmp_path / "train.wav" if duplicate == "audio" else tmp_path / "validation.wav",
    )

    with pytest.raises(ValueError, match="share"):
        validate_disjoint_records([train], [validation])


def test_validation_rank_prefers_per_then_loss() -> None:
    low_per = _epoch(1, 0.20, 3.0)
    high_per = _epoch(2, 0.21, 1.0)
    same_per_lower_loss = _epoch(3, 0.20, 2.0)

    assert validation_rank(low_per) < validation_rank(high_per)
    assert validation_rank(same_per_lower_loss) < validation_rank(low_per)
