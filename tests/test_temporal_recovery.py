from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.training.temporal_recovery import (
    CorpusAuditInput,
    audit_temporal_recovery_corpus,
    parse_corpus_spec,
    run_temporal_recovery_audit,
)


def _issue(reason: str) -> dict[str, object]:
    return {"reason": reason, "detail": None}


def _row(
    record_id: str,
    issues: list[str],
    *,
    estimated: int | None = None,
    minimum: int | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "id": record_id,
        "training_ready": False,
        "label_status": "needs_review",
        "duration_seconds": 3600.0,
        "issues": [_issue(reason) for reason in issues],
    }
    if estimated is not None:
        row["estimated_ctc_input_length"] = estimated
    if minimum is not None:
        row["ctc_minimum_input_length"] = minimum
    return row


def _manifest_dir(tmp_path: Path, name: str = "corpus") -> Path:
    manifest = tmp_path / name
    manifest.mkdir()
    (manifest / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "source_records": 9,
                "ready_records": 3,
                "review_records": 6,
                "total_audio_hours": 9.0,
                "ready_audio_hours": 3.0,
            }
        ),
        encoding="utf-8",
    )
    # Deliberately not JSON. The audit must never read ready record content.
    (manifest / "train_ready.jsonl").write_text(
        "SEALED READY CONTENT MUST NOT BE READ\n",
        encoding="utf-8",
    )
    rows = [
        _row("recommended", ["ctc_length_infeasible"], estimated=5, minimum=8),
        _row("high-pressure", ["ctc_length_infeasible"], estimated=5, minimum=10),
        _row("still-infeasible", ["ctc_length_infeasible"], estimated=4, minimum=9),
        _row(
            "time-and-dictionary",
            ["ctc_length_infeasible", "dictionary_missing"],
            estimated=5,
            minimum=8,
        ),
        _row("dictionary", ["dictionary_missing"], estimated=8, minimum=4),
        _row("empty-target", ["empty_ctc_target"]),
    ]
    (manifest / "needs_review.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_temporal_recovery_audit_partitions_review_and_issue_intersections(
    tmp_path: Path,
) -> None:
    manifest = _manifest_dir(tmp_path)

    report = audit_temporal_recovery_corpus(
        CorpusAuditInput("fixture", manifest),
        print_progress=False,
    )

    assert report["ready_manifest_content_read"] is False
    assert report["sealed_test_content_read"] is False
    assert report["original"]["ready"]["records"] == 3
    assert report["original"]["ready"]["hours"] == 3.0
    assert report["original"]["review"]["records"] == 6
    assert report["original"]["review"]["hours"] == 6.0

    temporal = report["temporal_2x_results"]
    assert temporal["pure_temporal_original_review"]["records"] == 3
    assert temporal["recoverable_total"]["records"] == 2
    assert temporal["recommended_ratio_le_limit"]["records"] == 1
    assert temporal["high_pressure_deferred"]["records"] == 1
    assert temporal["still_infeasible"]["records"] == 1
    assert temporal["blocked_by_other_issues"]["records"] == 3
    assert temporal["blocked_by_other_issues_with_temporal_issue"]["records"] == 1
    assert temporal["blocked_by_other_issues_without_temporal_issue"]["records"] == 2

    buckets = report["effective_ratio_buckets"]["pure_temporal_review"]
    assert buckets["ratio_0_75_to_0_90"]["records"] == 1
    assert buckets["ratio_0_90_to_1_00"]["records"] == 1
    assert buckets["ratio_gt_1_00"]["records"] == 1
    issues = report["issue_analysis"]
    assert issues["reason_totals"]["ctc_length_infeasible"]["records"] == 4
    assert issues["reason_totals"]["dictionary_missing"]["records"] == 2
    assert (
        issues["exact_issue_intersections"]["ctc_length_infeasible + dictionary_missing"]["records"]
        == 1
    )
    assert (
        issues["pairwise_issue_intersections"]["ctc_length_infeasible & dictionary_missing"][
            "records"
        ]
        == 1
    )


def test_temporal_recovery_group_writes_atomic_reports_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    first = _manifest_dir(tmp_path, "first")
    second = _manifest_dir(tmp_path, "second")
    output = tmp_path / "audit"
    corpora = [
        CorpusAuditInput("first", first),
        CorpusAuditInput("second", second),
    ]

    summary = run_temporal_recovery_audit(
        corpora,
        output,
        print_progress=False,
    )

    assert summary["corpus_order"] == ["first", "second"]
    assert summary["aggregate"]["recommended_ratio_le_limit"]["records"] == 2
    assert (output / "first.json").is_file()
    assert (output / "second.json").is_file()
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_temporal_recovery_audit(
            corpora,
            output,
            print_progress=False,
        )


def test_temporal_recovery_rejects_bad_specs_and_inconsistent_temporal_rows(
    tmp_path: Path,
) -> None:
    parsed = parse_corpus_spec("noah_finance=outputs/finance")
    assert parsed.name == "noah_finance"
    assert parsed.manifest_dir == Path("outputs/finance")
    with pytest.raises(ValueError, match="NAME=MANIFEST_DIR"):
        parse_corpus_spec("missing-separator")
    with pytest.raises(ValueError, match="corpus name"):
        parse_corpus_spec("Bad Name=outputs/example")

    manifest = _manifest_dir(tmp_path)
    bad = _row(
        "bad",
        ["ctc_length_infeasible"],
        estimated=10,
        minimum=8,
    )
    (manifest / "needs_review.jsonl").write_text(
        json.dumps(bad) + "\n",
        encoding="utf-8",
    )
    summary_path = manifest / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(source_records=4, review_records=1)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent with original lengths"):
        audit_temporal_recovery_corpus(
            CorpusAuditInput("fixture", manifest),
            print_progress=False,
        )
