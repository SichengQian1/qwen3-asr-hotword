from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.capacity_history import build_hotword_capacity_history


def _write_stage(
    path: Path,
    *,
    mode: str,
    rows: list[dict[str, object]],
    gc_policy: str | None = None,
) -> None:
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps(
            {"mode": mode, "gc_policy": gc_policy, "test_set_used": False}
        ),
        encoding="utf-8",
    )
    (path / "query_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _base_row(case_id: str, expected: list[str]) -> dict[str, object]:
    return {
        "profile": "representative",
        "size": 4000,
        "case_id": case_id,
        "is_final": True,
        "expected_hotword_ids": expected,
    }


def test_capacity_history_normalizes_existing_full_scan_and_legacy_anchor(
    tmp_path: Path,
) -> None:
    full = tmp_path / "full"
    positive = {
        **_base_row("positive", ["right"]),
        "raw_top5_ids": ["wrong", "right"],
        "raw_top7_ids": ["wrong", "right", "x"],
        "raw_top10_ids": ["wrong", "right", "x", "y"],
        "operating_ids": ["right"],
        "retrieval_seconds": 1.0,
    }
    negative = {
        **_base_row("negative", []),
        "raw_top5_ids": ["wrong"],
        "raw_top7_ids": ["wrong"],
        "raw_top10_ids": ["wrong"],
        "operating_ids": [],
        "retrieval_seconds": 3.0,
    }
    _write_stage(full, mode="full_scan", rows=[positive, negative])

    anchor = tmp_path / "anchor"
    anchor_positive = {
        **_base_row("positive", ["right"]),
        "candidate_ids_at_64": ["wrong", "right", "x", "y", "z", "a", "b", "c"],
        "reference_raw_top5_ids": ["right", "wrong"],
        "anchor_retrieval_seconds": 0.01,
        "full_scan_reference_seconds": 1.0,
    }
    anchor_negative = {
        **_base_row("negative", []),
        "candidate_ids_at_64": ["wrong", "x", "y", "z", "a", "b", "c", "d"],
        "reference_raw_top5_ids": ["wrong"],
        "anchor_retrieval_seconds": 0.03,
        "full_scan_reference_seconds": 3.0,
    }
    _write_stage(
        anchor,
        mode="anchor",
        rows=[anchor_positive, anchor_negative],
        gc_policy="defer_during_anchor_pass",
    )

    output = tmp_path / "history"
    report = build_hotword_capacity_history(
        stages=(("baseline", full), ("anchor_v3", anchor)),
        output_dir=output,
        sizes=(4000,),
    )

    assert report["status"] == "pass"
    rows = report["history"]
    assert isinstance(rows, list)
    baseline = next(row for row in rows if row["stage"] == "baseline")
    assert baseline["raw_recall_at_5"] == 1.0
    assert baseline["raw_precision_at_5"] == pytest.approx(1 / 3)
    assert baseline["operating_precision_at_5"] == 1.0
    assert baseline["negative_case_false_positive_rate"] == 0.0
    anchor_row = next(row for row in rows if row["ranking"] == "anchor_shortlist")
    assert anchor_row["raw_recall_at_5"] == 1.0
    assert anchor_row["raw_precision_at_5"] == pytest.approx(0.1)
    assert anchor_row["raw_recall_at_7"] == 1.0
    assert anchor_row["gc_policy"] == "defer_during_anchor_pass"
    reference = next(row for row in rows if row["ranking"] == "full_scan_reference")
    assert reference["raw_recall_at_5"] == 1.0
    assert reference["raw_recall_at_7"] is None
    assert reference["operating_recall_at_5"] is None
    assert anchor_row["latency_p50"] == pytest.approx(0.02)
    assert (output / "optimization_history.tsv").is_file()
    assert (output / "sha256.txt").is_file()


def test_capacity_history_rejects_test_set_stage(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "summary.json").write_text(
        json.dumps({"test_set_used": True}), encoding="utf-8"
    )
    (stage / "query_results.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="refuses test-set"):
        build_hotword_capacity_history(
            stages=(("bad", stage),), output_dir=tmp_path / "output"
        )
