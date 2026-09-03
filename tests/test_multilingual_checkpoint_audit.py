from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.training.multilingual_checkpoint_audit import (
    EXPECTED_SELECTION,
    audit_multilingual_ctc_checkpoint,
)


def _group_metrics(values: dict[str, float]) -> dict[str, dict[str, object]]:
    return {
        group: {
            "sample_count": 2,
            "phoneme_errors": int(per * 100),
            "reference_phonemes": 100,
            "phoneme_error_rate": per,
        }
        for group, per in values.items()
    }


def _write_training_run(path: Path) -> None:
    path.mkdir()
    initial = {"en": 0.30, "es": 0.20, "pt": 0.40}
    best = {"en": 0.10, "es": 0.08, "pt": 0.15}
    final = {"en": 0.11, "es": 0.09, "pt": 0.16}
    metrics = [
        {
            "epoch": 1,
            "learning_rate": 0.0003,
            "train": {"phoneme_error_rate": 0.20},
            "validation": {"phoneme_error_rate": 0.14},
            "validation_by_group": _group_metrics(best),
            "validation_macro_phoneme_error_rate": sum(best.values()) / 3,
            "epoch_seconds": 1.0,
        },
        {
            "epoch": 2,
            "learning_rate": 0.0003,
            "train": {"phoneme_error_rate": 0.18},
            "validation": {"phoneme_error_rate": 0.15},
            "validation_by_group": _group_metrics(final),
            "validation_macro_phoneme_error_rate": sum(final.values()) / 3,
            "epoch_seconds": 1.0,
        },
    ]
    (path / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8"
    )
    for name in ("ctc_head_best.pt", "ctc_head_latest.pt", "training_state_latest.pt"):
        (path / name).write_bytes(name.encode("utf-8"))
    report = {
        "status": "completed",
        "test_set_used": False,
        "cache_sha256_verified": True,
        "checkpoint_selection_metric": "validation_macro_per",
        "selection_metric": EXPECTED_SELECTION,
        "validation_groups": ["en", "es", "pt"],
        "head_config": {
            "head_type": "temporal_upsample",
            "input_dimension": 1024,
            "num_classes": 90,
            "hidden_dimension": 512,
            "kernel_size": 5,
            "dropout": 0.1,
            "time_upsampling_factor": 2,
        },
        "epochs_requested": 30,
        "epochs_completed": 2,
        "resumed_from_epoch": 1,
        "early_stopped": False,
        "best_epoch": 1,
        "final_learning_rate": 0.0003,
        "initial_validation_phoneme_error_rate": 0.31,
        "initial_validation_macro_phoneme_error_rate": sum(initial.values()) / 3,
        "initial_validation_by_group": _group_metrics(initial),
        "best_validation_phoneme_error_rate": 0.14,
        "best_validation_macro_phoneme_error_rate": sum(best.values()) / 3,
        "best_validation_worst_group": "pt",
        "best_validation_worst_group_phoneme_error_rate": 0.15,
        "best_validation_by_group": _group_metrics(best),
        "final_validation_phoneme_error_rate": 0.15,
        "final_validation_macro_phoneme_error_rate": sum(final.values()) / 3,
        "final_validation_worst_group": "pt",
        "final_validation_worst_group_phoneme_error_rate": 0.16,
        "final_validation_by_group": _group_metrics(final),
    }
    (path / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_checkpoint_audit_freezes_identity_and_metrics(tmp_path: Path) -> None:
    training = tmp_path / "training"
    _write_training_run(training)

    result = audit_multilingual_ctc_checkpoint(training, tmp_path / "audit")

    assert result["status"] == "pass"
    assert result["test_set_used"] is False
    assert result["training_progress"]["best_epoch"] == 1
    assert result["best_validation"]["by_group_per"] == {
        "en": 0.10,
        "es": 0.08,
        "pt": 0.15,
    }
    assert len(result["best_checkpoint"]["sha256"]) == 64
    assert (tmp_path / "audit" / "sha256.txt").is_file()


def test_checkpoint_audit_refuses_existing_output_and_non_macro_run(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    _write_training_run(training)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit_multilingual_ctc_checkpoint(training, existing)

    report_path = training / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint_selection_metric"] = "validation_per"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_macro_per"):
        audit_multilingual_ctc_checkpoint(training, tmp_path / "invalid")
