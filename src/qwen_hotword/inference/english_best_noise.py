"""Bind the completed training best Head and evaluate paired English noisy audio."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference.english_head_preview import prepare_preview
from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.spanish_capacity import _sha


def load_completed_best(
    root: Path, cfg: dict[str, Any], vocab: PhonemeVocab, inputs: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    import torch

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
        ctc_head_config,
    )

    training = (root / cfg["training_run"]).resolve()
    marker_path = _checked(training / "training_completed.json", inputs)
    marker = _json(marker_path)
    report_path = _checked(training / "head/report.json", inputs, marker["report_sha256"])
    _checked(training / "plan.json", inputs, marker["plan_sha256"])
    report = _json(report_path)
    head = (root / cfg["checkpoint"]).resolve()
    if (
        report.get("status") != "completed"
        or report.get("test_set_used") is not False
        or marker.get("test_set_used") is not False
        or report.get("cache_sha256_verified") is not True
        or report.get("checkpoint_selection_metric") != "validation_macro_per"
        or report["best_epoch"] != cfg["expected_epoch"]
        or report["epochs_completed"] != cfg["expected_epochs_completed"]
        or report["epochs_requested"] != cfg["expected_epochs_requested"]
        or (
            not report["early_stopped"] and report["epochs_completed"] != report["epochs_requested"]
        )
        or report["validation_groups"] != ["en", "es", "pt"]
        or Path(report["best_checkpoint_path"]).resolve() != head
        or head != training / "head/ctc_head_best.pt"
    ):
        raise ValueError("completed formal training best identity differs")
    _checked(head, inputs, marker["checkpoint_sha256"])
    payload = torch.load(head, map_location="cpu", weights_only=True)
    if tuple(payload.get("vocab_tokens", ())) != tuple(vocab.tokens):
        raise ValueError("completed best vocabulary differs")
    metrics = payload["epoch_metrics"]
    by_group = report["best_validation_by_group"]
    if (
        metrics["epoch"] != report["best_epoch"]
        or metrics["loss"] != report["best_validation_loss"]
        or metrics["phoneme_error_rate"] != report["best_validation_phoneme_error_rate"]
        or metrics["phoneme_errors"] != sum(g["phoneme_errors"] for g in by_group.values())
        or metrics["reference_phonemes"] != sum(g["reference_phonemes"] for g in by_group.values())
        or set(by_group) != {"en", "es", "pt"}
        or not math.isclose(
            sum(g["phoneme_error_rate"] for g in by_group.values()) / 3,
            report["best_validation_macro_phoneme_error_rate"],
            abs_tol=1e-12,
        )
    ):
        raise ValueError("best checkpoint metrics differ from completed training report")
    head_module = build_ctc_head_from_checkpoint(payload)
    if (
        not isinstance(head_module, TemporalUpsampleCtcHead)
        or head_module.time_upsampling_factor != 2
    ):
        raise ValueError("completed best must be a Temporal 2x Head")
    head_module.load_state_dict(payload["state_dict"], strict=True)
    head_config = ctc_head_config(head_module)
    if head_config != report["head_config"]:
        raise ValueError("completed best architecture differs from training report")
    return head, {
        "sha256": marker["checkpoint_sha256"],
        "epoch_metrics": metrics,
        "head_config": head_config,
        "checkpoint_selection_metric": "validation_macro_per",
        "best_validation_by_group": by_group,
        "best_validation_macro_phoneme_error_rate": report[
            "best_validation_macro_phoneme_error_rate"
        ],
        "training_report_sha256": marker["report_sha256"],
        "training_plan_sha256": marker["plan_sha256"],
    }


def prepare_noise(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _json(config)
    audio_root = (root / cfg["audio_root"]).resolve()
    if not audio_root.is_dir():
        raise FileNotFoundError(f"noisy audio root is unavailable: {audio_root}")
    plan, runtime = prepare_preview(
        root, config, checkpoint_loader=load_completed_best, noisy_audio_root=audio_root
    )
    plan["code_sha256"]["inference/english_best_noise.py"] = _sha(Path(__file__))
    return plan, runtime
