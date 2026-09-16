from __future__ import annotations

from qwen_hotword.training.portuguese_ctc_diagnostics import (
    CtcSampleLengths,
    PortugueseValidationMetadata,
    build_pressure_tertiles,
    build_top_error_rows,
    compare_checkpoint_reports,
)


def _metrics(errors: int, substitutions: int, deletions: int, insertions: int) -> dict[str, object]:
    return {
        "sample_count": 2,
        "reference_tokens": 20,
        "hypothesis_tokens": 20 - deletions + insertions,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "input_frames": 50,
        "blank_frames": 10,
        "errors": errors,
        "phoneme_error_rate": errors / 20,
        "hypothesis_reference_length_ratio": (20 - deletions + insertions) / 20,
        "blank_frame_ratio": 0.2,
    }


def _report(
    first_errors: int,
    second_errors: int,
    total: dict[str, object],
) -> dict[str, object]:
    return {
        "validation": total,
        "validation_by_dimension": {
            "source_corpus": {"noah": total},
            "release_source": {"original_ready": total},
        },
        "sample_metrics": [
            {"sample_id": "pt-1", **_metrics(first_errors, first_errors, 0, 0)},
            {"sample_id": "pt-2", **_metrics(second_errors, second_errors, 0, 0)},
        ],
    }


def test_pressure_tertiles_are_stable_and_cover_every_sample() -> None:
    lengths = {
        f"pt-{index}": CtcSampleLengths(1, 10, ratio, 1, ratio)
        for index, ratio in enumerate((0.10, 0.20, 0.30, 0.40, 0.50, 0.60))
    }

    groups, summary = build_pressure_tertiles(lengths)

    assert groups == {
        "pt-0": "low",
        "pt-1": "low",
        "pt-2": "medium",
        "pt-3": "medium",
        "pt-4": "high",
        "pt-5": "high",
    }
    assert summary["groups"]["low"]["maximum_ratio"] == 0.20
    assert summary["groups"]["high"]["minimum_ratio"] == 0.50


def test_checkpoint_comparison_reports_paired_and_group_deltas() -> None:
    multilingual_total = _metrics(5, 2, 2, 1)
    portuguese_total = _metrics(3, 1, 1, 1)
    multilingual = _report(1, 4, multilingual_total)
    portuguese = _report(2, 1, portuguese_total)

    comparison = compare_checkpoint_reports(multilingual, portuguese)

    assert comparison["paired_sample_counts"] == {
        "multilingual_better": 1,
        "multilingual_worse": 1,
        "tied": 0,
    }
    assert comparison["paired_total_error_delta"] == 2
    assert comparison["overall"]["delta_phoneme_error_rate"] == 0.1
    assert comparison["overall"]["delta_deletion_rate"] == 0.05
    assert comparison["by_dimension"]["source_corpus"]["noah"]["delta_phoneme_error_rate"] == 0.1


def test_top_error_rows_add_only_actual_regressions_beyond_high_per_samples() -> None:
    multilingual = _report(3, 1, _metrics(4, 4, 0, 0))
    portuguese = _report(1, 2, _metrics(3, 3, 0, 0))
    metadata = {
        sample_id: PortugueseValidationMetadata(
            sample_id=sample_id,
            source_corpus="noah",
            release_source="original_ready",
            text=sample_id,
            audio_path=f"/audio/{sample_id}.wav",
            duration_seconds=1.0,
        )
        for sample_id in ("pt-1", "pt-2")
    }
    lengths = {sample_id: CtcSampleLengths(20, 50, 0.4, 20, 0.4) for sample_id in ("pt-1", "pt-2")}

    rows = build_top_error_rows(
        {"multilingual": multilingual, "portuguese": portuguese},
        metadata,
        lengths,
        limit=1,
    )

    assert [row["sample_id"] for row in rows] == ["pt-1"]
    assert rows[0]["multilingual_minus_portuguese_errors"] == 2
