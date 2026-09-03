from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.inference.streaming_calibrated_gate_suite import (
    BASELINE_GATE,
    CALIBRATED_GATE_PROFILES,
    build_calibrated_gate_suite_report,
    calibrated_suite_resume_config_matches,
    profile_dicts,
    validate_calibrated_gate_preflight,
    write_calibrated_gate_suite_report,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_hashes(root: Path, names: list[str]) -> None:
    (root / "sha256.txt").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def _quality(*, injected: int, recall: float, precision: float) -> dict[str, object]:
    return {
        "expected_hotwords": 2,
        "correct_prompt_injected_hotwords": injected,
        "correct_prompt_adopted_hotwords": injected,
        "correct_prompt_adoption_rate": 1.0,
        "wrong_injected_hotwords": 1,
        "wrong_prompt_written_hotwords": 0,
        "wrong_prompt_landing_rate": 0.0,
        "final_hotword_recall": recall,
        "final_hotword_precision": precision,
        "sample_hotword_hit_rate": recall,
        "wer": 0.1,
        "cer": 0.05,
        "negative_hotword_hallucination_rate": 0.0,
        "mean_inference_seconds": 0.5,
    }


def _child_config(
    *,
    checkpoint_sha: str,
    offline_report_sha: str,
    threshold: float,
    top_k: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "git_commit": "allowed-to-change",
        "gpu_memory_utilization": 0.18,
        "inputs": {
            "validation": {"sha256": "v" * 64},
            "checkpoint": {"path": "ctc.pt", "sha256": checkpoint_sha},
        },
        "groups": ["D"],
        "threshold": threshold,
        "top_k": top_k,
        "maximum_edit_ratio": 0.35,
        "posterior_weight": 0.25,
        "minimum_posterior_confidence": 0.0,
        "minimum_top1_margin": 0.0,
        "offline_control": {
            "report": {"path": "selection.json", "sha256": offline_report_sha},
            "control_mode": "selection_only",
            "current_retrieval_config_not_compared": {
                "threshold": threshold,
                "top_k": top_k,
            },
        },
    }


def _write_child(
    root: Path,
    *,
    config: dict[str, object],
    quality: dict[str, object],
    injected: list[str],
    prediction: str,
) -> None:
    root.mkdir(parents=True)
    _write_json(root / "run_config.json", config)
    _write_json(root / "summary.json", {"status": "pass", "groups": {"D": quality}})
    _write_json(
        root / "latency_summary.json",
        {"groups": {"D": {"compute": {"retrieval_over_50ms_rate": 0.0}}}},
    )
    (root / "sample_results.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "sample_id": "sample-1",
                "reference_text": "hello hot one hot two",
                "experiment_group": "D",
                "primary_group": "multi_positive",
                "expected_hotword_ids": ["hot-1", "hot-2"],
                "injected_hotword_ids": injected,
                "prediction": prediction,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("child\n", encoding="utf-8")
    _write_hashes(
        root,
        [
            "run_config.json",
            "summary.json",
            "latency_summary.json",
            "sample_results.jsonl",
            "README.md",
        ],
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    checkpoint = tmp_path / "ctc.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _sha256(checkpoint)
    full_rank = tmp_path / "full-rank"
    full_rank.mkdir()
    ctc_report = full_rank / "ctc_report.json"
    _write_json(ctc_report, {"checkpoint_sha256": checkpoint_sha})
    _write_hashes(full_rank, ["ctc_report.json"])

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    calibration_summary = calibration / "candidate_summary.json"
    _write_json(
        calibration_summary,
        {
            "status": "guarded_recall_gain_candidate_available",
            "ranked_matches_complete": True,
            "non_exact_point_count": 0,
            "recommended_candidates": [
                {
                    "role": "precision_guarded",
                    "config": {
                        "threshold": 0.86,
                        "top_k": 7,
                        "maximum_edit_ratio": 0.35,
                        "minimum_posterior_confidence": 0.0,
                        "minimum_top1_margin": 0.0,
                    },
                },
                {
                    "role": "f1_with_fpr_guard",
                    "config": {
                        "threshold": 0.83,
                        "top_k": 7,
                        "maximum_edit_ratio": 0.35,
                        "minimum_posterior_confidence": 0.0,
                        "minimum_top1_margin": 0.0,
                    },
                },
            ],
        },
    )
    _write_json(
        calibration / "calibration_config.json",
        {"candidate_checkpoint_sha256": checkpoint_sha},
    )
    _write_hashes(
        calibration,
        ["candidate_summary.json", "calibration_config.json"],
    )

    offline_report_sha = "o" * 64
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_json(
        baseline / "suite_config.json",
        {
            "status": "pass",
            "test_set_used": False,
            "profiles": [
                {
                    "name": "conservative",
                    "output_subdir": "conservative",
                }
            ],
        },
    )
    baseline_quality = _quality(injected=1, recall=0.5, precision=0.8)
    _write_json(
        baseline / "suite_summary.json",
        {
            "status": "pass",
            "test_set_used": False,
            "sample_count": 1,
            "profiles": {
                "conservative": {
                    "gate": BASELINE_GATE,
                    "quality": baseline_quality,
                }
            },
        },
    )
    _write_child(
        baseline / "conservative",
        config=_child_config(
            checkpoint_sha=checkpoint_sha,
            offline_report_sha=offline_report_sha,
            threshold=0.86,
            top_k=5,
        ),
        quality=baseline_quality,
        injected=["hot-1"],
        prediction="hello hot one",
    )
    _write_hashes(baseline, ["suite_config.json", "suite_summary.json"])

    output = tmp_path / "output"
    output.mkdir()
    preflight = validate_calibrated_gate_preflight(
        baseline_suite_dir=baseline,
        calibration_summary_path=calibration_summary,
        ctc_report_path=ctc_report,
        ctc_checkpoint_path=checkpoint,
    )
    _write_json(
        output / "suite_config.json",
        {
            "status": "pass",
            "test_set_used": False,
            "preflight": preflight,
        },
    )
    for index, profile in enumerate(CALIBRATED_GATE_PROFILES, start=1):
        _write_child(
            output / profile.output_subdir,
            config=_child_config(
                checkpoint_sha=checkpoint_sha,
                offline_report_sha=offline_report_sha,
                threshold=profile.threshold,
                top_k=profile.top_k,
            ),
            quality=_quality(injected=2, recall=0.5 + index * 0.1, precision=0.8),
            injected=["hot-1", "hot-2", "wrong"],
            prediction="hello hot one hot two",
        )
    return {
        "baseline": baseline,
        "output": output,
        "checkpoint": checkpoint,
        "ctc_report": ctc_report,
        "calibration_summary": calibration_summary,
    }


def test_preflight_and_report_bind_exact_candidates(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report, cases = build_calibrated_gate_suite_report(paths["output"], paths["baseline"])

    assert report["status"] == "pass"
    assert report["sample_count"] == 1
    assert report["identity_checks"]["same_ctc_checkpoint_as_d5_baseline"] is True
    assert report["identity_checks"]["same_d_sample_selection"] is True
    precision = report["candidates"]["precision_guarded_top7"]
    assert precision["gate"]["top_k"] == 7
    assert precision["quality"]["prompt_hotword_recall"] == 1.0
    assert precision["delta_from_d5_baseline"]["prompt_hotword_recall"] == 0.5
    assert report["comparisons"]["topk_isolation"]["only_gate_change"] == "top_k_5_to_7"
    assert len(cases) == 2

    written = write_calibrated_gate_suite_report(paths["output"], paths["baseline"])
    assert written == report
    assert (paths["output"] / "sha256.txt").is_file()


def test_preflight_rejects_unsealed_calibration_candidate(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    calibration = json.loads(paths["calibration_summary"].read_text(encoding="utf-8"))
    calibration["recommended_candidates"][0]["config"]["threshold"] = 0.85
    _write_json(paths["calibration_summary"], calibration)
    _write_hashes(
        paths["calibration_summary"].parent,
        ["candidate_summary.json", "calibration_config.json"],
    )

    with pytest.raises(ValueError, match="sealed choice"):
        validate_calibrated_gate_preflight(
            baseline_suite_dir=paths["baseline"],
            calibration_summary_path=paths["calibration_summary"],
            ctc_report_path=paths["ctc_report"],
            ctc_checkpoint_path=paths["checkpoint"],
        )


def test_resume_requires_identical_config() -> None:
    current = {"status": "running", "profiles": profile_dicts(), "shared": "same"}
    assert calibrated_suite_resume_config_matches({**current, "status": "pass"}, current)
    assert not calibrated_suite_resume_config_matches({**current, "shared": "different"}, current)
