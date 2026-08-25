from __future__ import annotations

from qwen_hotword.training.ctc_diagnostics import (
    DetailedErrorAccumulator,
    ErrorAccumulator,
    _accumulate_errors,
    ctc_pressure_bucket,
)
from qwen_hotword.training.edit_distance import sequence_edit_distance, sequence_editops


def test_ctc_pressure_bucket_uses_effective_alignment_length() -> None:
    assert ctc_pressure_bucket((2, 3), input_length=8) == "minimum_ratio_le_0_50"
    assert (
        ctc_pressure_bucket((2, 2, 3), input_length=6)
        == "minimum_ratio_gt_0_50_le_0_75"
    )
    assert (
        ctc_pressure_bucket((2, 2, 3), input_length=5)
        == "minimum_ratio_gt_0_75_le_0_90"
    )
    assert ctc_pressure_bucket((2, 2, 3), input_length=4) == "minimum_ratio_gt_0_90"


def test_error_accumulator_separates_error_types() -> None:
    detailed = DetailedErrorAccumulator()
    bucket = ErrorAccumulator()
    group = ErrorAccumulator()
    _accumulate_errors(
        detailed,
        bucket,
        reference=(2, 3),
        hypothesis=(2, 6),
        raw_prediction=[0, 2, 0, 6],
        blank_id=0,
        group_accumulator=group,
    )
    _accumulate_errors(
        detailed,
        bucket,
        reference=(2, 3, 4),
        hypothesis=(2, 4),
        raw_prediction=[0, 2, 0, 4],
        blank_id=0,
    )
    _accumulate_errors(
        detailed,
        bucket,
        reference=(2, 4),
        hypothesis=(2, 7, 4),
        raw_prediction=[0, 2, 7, 0, 4],
        blank_id=0,
    )

    assert detailed.totals.substitutions == 1
    assert detailed.totals.deletions == 1
    assert detailed.totals.insertions == 1
    assert detailed.totals.blank_frames == 6
    assert detailed.deleted_tokens[3] == 1
    assert detailed.inserted_tokens[7] == 1
    assert detailed.substituted_tokens[(3, 6)] == 1
    assert bucket.errors == 3
    assert group.sample_count == 1
    assert group.errors == 1


def test_edit_distance_matches_operation_count() -> None:
    reference = (2, 3, 4)
    hypothesis = (2, 6, 4, 7)
    operations = sequence_editops(reference, hypothesis)
    assert sequence_edit_distance(reference, hypothesis) == len(operations) == 2
