from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_hotword.inference.checkpoint_regression import (
    compare_streaming_checkpoint_suites,
)
from qwen_hotword.inference.streaming_gate_suite import STREAMING_GATE_PROFILES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_hashes(path: Path, names: list[str]) -> None:
    (path / "sha256.txt").write_text(
        "".join(f"{_sha256(path / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _quality(value: float, *, control: bool = False) -> dict[str, object]:
    return {
        "expected_hotwords": 2,
        "correct_prompt_injected_hotwords": 0 if control else int(value * 2),
        "prompt_hotword_recall": 0.0 if control else value,
        "correct_prompt_adopted_hotwords": 0 if control else int(value * 2),
        "correct_prompt_adoption_rate": None if control else value,
        "wrong_injected_hotwords": 0,
        "wrong_prompt_written_hotwords": 0,
        "wrong_prompt_landing_rate": None,
        "final_hotword_recall": value,
        "final_hotword_precision": None if control else 1.0,
        "sample_hotword_hit_rate": value,
        "wer": 0.1,
        "cer": 0.05,
        "negative_hotword_hallucination_rate": 0.0,
    }


def _write_suite(path: Path, checkpoint_name: str, checkpoint_sha: str, d_value: float) -> None:
    path.mkdir()
    ctc_report = path / "ctc_report.json"
    ctc_report.write_text(
        json.dumps({"checkpoint_sha256": checkpoint_sha}), encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "profiles": [profile.__dict__ for profile in STREAMING_GATE_PROFILES],
        "anchor": {"shortlist_size": 64},
        "inputs": {
            "model": "/models/Qwen3-ASR-1.7B",
            "validation_manifest": "validation.jsonl",
            "ctc_checkpoint": checkpoint_name,
            "ctc_report": str(ctc_report),
        },
        "runtime": {"language": "Portuguese", "gpu_memory_utilization": 0.15},
        "max_samples": 0,
    }
    (path / "suite_config.json").write_text(json.dumps(config), encoding="utf-8")
    profiles: dict[str, object] = {}
    for profile in STREAMING_GATE_PROFILES:
        quality_value = (
            0.5
            if profile.name == "no_rag"
            else (0.75 if profile.name == "oracle" else d_value)
        )
        profiles[profile.name] = {
            "gate": None if profile.group in {"C", "E"} else {"top_k": profile.top_k},
            "quality": _quality(
                quality_value,
                control=profile.name == "no_rag",
            ),
        }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "sample_count": 1,
        "profiles": profiles,
    }
    (path / "suite_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    grouped_subdirs: dict[str, list[object]] = {}
    for profile in STREAMING_GATE_PROFILES:
        grouped_subdirs.setdefault(profile.output_subdir, []).append(profile)
    for subdir, grouped_profiles in grouped_subdirs.items():
        run = path / subdir
        run.mkdir()
        run_config = {
            "schema_version": 2,
            "git_commit": "commit-" + checkpoint_name,
            "inputs": {
                "validation": {"sha256": "c" * 64},
                "checkpoint": {"path": checkpoint_name, "sha256": checkpoint_sha},
            },
            "offline_control": {
                "report": {"path": str(ctc_report), "sha256": _sha256(ctc_report)},
                "control_mode": "selection_only",
            },
            "runtime": "same",
        }
        (run / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")
        rows = []
        for profile in grouped_profiles:
            rows.append(
                {
                    "case_id": "case-1",
                    "sample_id": "sample-1",
                    "reference_text": "hello",
                    "experiment_group": profile.group,
                    "expected_hotword_ids": ["hot"],
                    "injected_hotword_ids": (
                        ["hot"] if profile.group == "D" and d_value > 0.8 else []
                    ),
                    "prediction": (
                        "hello hot"
                        if profile.group == "D" and d_value > 0.8
                        else "hello"
                    ),
                }
            )
        (run / "sample_results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run / "summary.json").write_text("{}\n", encoding="utf-8")
        (run / "latency_summary.json").write_text("{}\n", encoding="utf-8")
        _write_hashes(
            run,
            ["run_config.json", "sample_results.jsonl", "summary.json", "latency_summary.json"],
        )
    _write_hashes(path, ["suite_config.json", "suite_summary.json", "ctc_report.json"])


def test_streaming_checkpoint_regression_verifies_identity_and_deltas(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_suite(baseline, "pt.pt", "a" * 64, 0.8)
    _write_suite(candidate, "multilingual.pt", "b" * 64, 0.9)

    result = compare_streaming_checkpoint_suites(
        baseline,
        candidate,
        tmp_path / "comparison",
    )

    assert result["status"] == "pass"
    assert result["identity_checks"]["checkpoint_changed"] is True
    assert result["control_stability"]["all_control_quality_equal"] is True
    delta = result["quality"]["recall_first"]["delta_candidate_minus_baseline"]
    assert delta["final_hotword_recall"] == pytest.approx(0.1)
    assert result["case_comparison"]["changed_profile_case_rows"] == 4
    assert (tmp_path / "comparison" / "sha256.txt").is_file()


def test_streaming_checkpoint_regression_rejects_non_checkpoint_change(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_suite(baseline, "pt.pt", "a" * 64, 0.8)
    _write_suite(candidate, "multilingual.pt", "b" * 64, 0.9)
    config_path = candidate / "suite_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["runtime"]["language"] = "Spanish"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    _write_hashes(
        candidate,
        ["suite_config.json", "suite_summary.json", "ctc_report.json"],
    )

    with pytest.raises(ValueError, match="differ beyond"):
        compare_streaming_checkpoint_suites(baseline, candidate, tmp_path / "comparison")
