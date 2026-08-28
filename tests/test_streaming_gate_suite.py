from __future__ import annotations

import json
from pathlib import Path

from qwen_hotword.inference.streaming_gate_suite import (
    STREAMING_GATE_PROFILES,
    build_streaming_gate_suite_report,
    completed_profile_run,
    suite_resume_config_matches,
    write_streaming_gate_suite_report,
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
            "correct_prompt_injected_hotwords": 1 if profile.group != "C" else 0,
            "final_hotword_recall": 0.9 if profile.group != "C" else 0.5,
            "final_hotword_precision": 0.8 if profile.group != "C" else None,
            "correct_prompt_adoption_rate": 0.85 if profile.group != "C" else None,
            "wrong_prompt_filter_rate": 0.75 if profile.group != "C" else None,
            "wrong_prompt_landing_rate": 0.25 if profile.group != "C" else None,
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
                    "expected_hotword_ids": ["hot"],
                    "injected_hotword_ids": (
                        ["hot", "extra"]
                        if profile.name == "recall_first"
                        else ["hot"]
                    ),
                    "prediction": (
                        "hello extra"
                        if profile.name == "recall_first"
                        else "hello"
                    ),
                }
            )
            + "\n",
            encoding="utf-8",
        )

    report = build_streaming_gate_suite_report(root)
    assert report["status"] == "pass"
    assert report["sample_count"] == 1
    assert report["profiles"]["balanced"]["gate"]["top_k"] == 7
    top5 = report["profiles"]["recall_first_top5"]["gate"]
    assert top5 == {
        "threshold": 0.75,
        "top_k": 5,
        "maximum_edit_ratio": 0.35,
        "posterior_weight": 0.25,
        "minimum_posterior_confidence": 0.5,
        "minimum_top1_margin": 0.0,
    }
    assert (
        report["comparisons_vs_no_rag"]["recall_first"]["hotword_exact_recall"]
        == 0.4
    )
    assert (
        report["comparisons_vs_no_rag"]["recall_first"]["final_hotword_recall"]
        == 0.4
    )
    assert report["comparisons_vs_no_rag"]["recall_first"]["final_hotword_precision"] is None
    assert report["comparisons_vs_no_rag"]["recall_first_top5"]["prompt_hotword_recall"] == 1.0
    isolation = report["topk_isolation"]
    assert isolation["only_changed_parameter"] == "top_k"
    assert isolation["shared_gate"]["threshold"] == 0.75
    assert isolation["sample_comparison"]["changed_case_count"] == 1

    written = write_streaming_gate_suite_report(root)
    assert written["topk_isolation"] == isolation
    assert (root / "topk_isolation_summary.json").is_file()
    case_rows = [
        json.loads(line)
        for line in (root / "topk_isolation_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert case_rows[0]["additional_top7_injected_hotword_ids"] == ["extra"]
    assert case_rows[0]["prediction_changed"] is True


def test_gate_suite_resume_accepts_only_additive_top5_profile() -> None:
    current_profiles = [profile.__dict__ for profile in STREAMING_GATE_PROFILES]
    current = {"status": "running", "profiles": current_profiles, "shared": "same"}
    legacy = {
        "status": "pass",
        "profiles": [
            profile
            for profile in current_profiles
            if profile["name"] != "recall_first_top5"
        ],
        "shared": "same",
    }
    assert suite_resume_config_matches(legacy, current)
    assert suite_resume_config_matches({**current, "status": "pass"}, current)
    assert not suite_resume_config_matches({**legacy, "shared": "different"}, current)


def test_completed_profile_run_requires_passed_outputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    for name in (
        "run_config.json",
        "latency_summary.json",
        "sample_results.jsonl",
        "sha256.txt",
    ):
        (run / name).write_text("{}\n", encoding="utf-8")
    (run / "summary.json").write_text(
        json.dumps({"status": "pass", "groups": {"D": {}}}),
        encoding="utf-8",
    )
    assert completed_profile_run(run, ("D",))
    assert not completed_profile_run(run, ("C", "E"))
