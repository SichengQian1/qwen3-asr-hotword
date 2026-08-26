from __future__ import annotations

import json
from pathlib import Path

from qwen_hotword.inference.streaming_gate_suite import (
    STREAMING_GATE_PROFILES,
    build_streaming_gate_suite_report,
)


def test_gate_suite_combines_same_selection_and_preserves_latency(tmp_path: Path) -> None:
    root = tmp_path / "suite"
    for profile in STREAMING_GATE_PROFILES:
        run = root / profile.output_subdir
        run.mkdir(parents=True, exist_ok=True)
        summary_path = run / "summary.json"
        latency_path = run / "latency_summary.json"
        results_path = run / "sample_results.jsonl"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {"groups": {}}
        )
        summary["groups"][profile.group] = {
            "hotword_exact_recall": 0.9 if profile.group != "C" else 0.5,
            "sample_hotword_hit_rate": 0.8,
            "wer": 0.1,
            "cer": 0.05,
            "negative_hotword_hallucination_rate": 0.0,
            "mean_inference_seconds": 1.0,
        }
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        latency = (
            json.loads(latency_path.read_text(encoding="utf-8"))
            if latency_path.is_file()
            else {"groups": {}}
        )
        latency["groups"][profile.group] = {
            "compute": {"retrieval_over_50ms_rate": 0.0}
        }
        latency_path.write_text(json.dumps(latency), encoding="utf-8")
        existing = results_path.read_text(encoding="utf-8") if results_path.is_file() else ""
        results_path.write_text(
            existing
            + json.dumps(
                {
                    "case_id": "case-1",
                    "sample_id": "sample-1",
                    "reference_text": "hello",
                    "experiment_group": profile.group,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    report = build_streaming_gate_suite_report(root)
    assert report["status"] == "pass"
    assert report["sample_count"] == 1
    assert report["profiles"]["balanced"]["gate"]["top_k"] == 7
    assert (
        report["comparisons_vs_no_rag"]["recall_first"]["hotword_exact_recall"]
        == 0.4
    )
