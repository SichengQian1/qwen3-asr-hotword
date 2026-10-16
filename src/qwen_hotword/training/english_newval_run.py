"""New validation run: isolated smoke, then a fresh Head for up to 40 epochs, no pilot."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from qwen_hotword.training.english_newval import pin, prepare, save, small_bundle, verify
from qwen_hotword.training.english_newval_cache import build_cache, build_smoke_cache
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress


def head_command(output: Path, plan: dict[str, Any], smoke: bool) -> list[str]:
    options = dict(plan["training_options"])
    if smoke:
        options["minimum-epochs"] = 1
    command = [
        sys.executable,
        "-B",
        "scripts/train_full_ctc.py",
        "--train-cache",
        str(output / "smoke_cache/train") if smoke else plan["train_cache"],
        "--validation-cache",
        str(output / ("smoke_cache/validation" if smoke else "cache/validation")),
        "--train-manifest",
        str(output / "smoke_train.jsonl") if smoke else plan["train_manifest"],
        "--validation-manifest",
        str(output / ("smoke_validation.jsonl" if smoke else "full_ctc_validation.jsonl")),
        "--vocab",
        plan["vocab"],
        "--output-dir",
        str(output / ("smoke_head" if smoke else "head")),
        "--device",
        "cuda:0",
        "--epochs",
        str(1 if smoke else plan["epochs"]),
    ]
    for name, value in options.items():
        command.extend([f"--{name}", str(value)])
    return command


def run_head(
    output: Path, plan: dict[str, Any], gpu: str, *, smoke: bool, resume: bool
) -> dict[str, Any]:
    folder = output / ("smoke_head" if smoke else "head")
    report_path = folder / "report.json"
    marker = output / ("smoke_completed.json" if smoke else "training_completed.json")
    if marker.exists():
        identity = json.loads(marker.read_text())
        pin(report_path, {}, identity["report_sha256"])
        pin(output / "plan.json", {}, identity["plan_sha256"])
        report: dict[str, Any] = json.loads(report_path.read_text())
        pin(Path(report["best_checkpoint_path"]), {}, identity["checkpoint_sha256"])
        return report
    if not report_path.exists():
        state = folder / "training_state_latest.pt"
        if (
            folder.exists()
            and not state.exists()
            and any(p.name != ".ctc_training.lock" for p in folder.iterdir())
        ):
            raise ValueError("partial Head output has no resumable state; preserve and inspect")
        if state.exists() and not resume and not smoke:
            raise FileExistsError("Head state exists; use --resume")
        command = head_command(output, plan, smoke)
        if state.exists():
            command.append("--resume")
        print("Running: " + " ".join(command), flush=True)
        with stage_progress("Smoke Head 1 epoch" if smoke else "Head up to 40 epochs; no pilot"):
            subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), check=True)
    # A completed report may precede our marker if interrupted just after the child exited.
    # Validate/publish it, rather than resume an already early-stopped run for another epoch.
    report = json.loads(report_path.read_text())
    if (
        report["status"] != "completed"
        or report["cache_sha256_verified"] is not True
        or report["test_set_used"] is not False
        or not math.isfinite(report["final_validation_phoneme_error_rate"])
        or not math.isfinite(report["final_validation_loss"])
        or report["epochs_requested"] != (1 if smoke else plan["epochs"])
        or report["validation_groups"] != ["en", "es", "pt"]
        or (smoke and report["epochs_completed"] != 1)
    ):
        raise ValueError("completed finite Head report with verified caches required")
    if not smoke and not report["early_stopped"] and report["epochs_completed"] != plan["epochs"]:
        raise ValueError("formal run stopped without reaching epoch cap or early stopping")
    save(
        marker,
        {
            "report_sha256": _sha(report_path),
            "plan_sha256": _sha(output / "plan.json"),
            "checkpoint_sha256": _sha(Path(report["best_checkpoint_path"])),
            "test_set_used": False,
        },
    )
    return report


def old_english_control(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Separate after-training old-domain diagnostic; never passed to selection/scheduler."""
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab
    from qwen_hotword.training.ctc_diagnostics import (
        diagnose_ctc_checkpoint,
        load_validation_sample_groups,
    )
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

    checkpoint = output / "head/ctc_head_best.pt"
    destination = output / "old_domain_control.json"
    if destination.exists():
        result: dict[str, Any] = json.loads(destination.read_text())
        pin(checkpoint, {}, result["checkpoint_sha256"])
        return result
    with stage_progress("After training: old-domain diagnostic only; no checkpoint selection"):
        vocab = load_phoneme_vocab(plan["vocab"])
        cache = load_disk_feature_cache(
            plan["legacy_validation_cache"],
            expected_split="validation",
            source_manifest_path=plan["legacy_validation_manifest"],
            vocab_path=plan["vocab"],
        )
        groups = load_validation_sample_groups(
            plan["legacy_validation_manifest"],
            cache,
            group_column="balanced_language_bucket",
            expected_groups=("en", "es", "pt"),
        )
        result = {
            "purpose": "old_domain_control_only_not_used_for_training_decisions",
            "checkpoint_sha256": _sha(checkpoint),
            "test_set_used": False,
            "diagnostics": diagnose_ctc_checkpoint(
                checkpoint,
                cache,
                vocab,
                device="cuda:0",
                sample_groups=groups,
            ),
        }
        save(destination, result)
    return result


def bundle(output: Path) -> str:
    path = output / "return_bundle.zip"
    # Additive bundles by milestone: an already delivered bundle is never replaced.
    if path.exists():
        path = output / f"return_bundle_{_sha(output / 'head/report.json')[:12]}.zip"
    if not path.exists():
        small_bundle(
            path,
            output,
            [
                "report.json",
                "plan.json",
                "cache_report.json",
                "smoke_head/report.json",
                "head/report.json",
                "head/metrics.jsonl",
                "old_domain_control.json",
                "training_completed.json",
            ],
        )
    return str(path)


def execute(
    root: Path, cfg: dict[str, Any], output: Path, stage: str, gpu: str, *, resume: bool = False
) -> dict[str, Any]:
    if stage == "prepare":
        return prepare(root, cfg, output)
    if stage not in {"cache", "smoke", "train", "run"} or not gpu.isdecimal():
        raise ValueError("choose prepare/cache/smoke/train/run and an explicit physical GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    with exclusive_feature_cache_run(output.parent / f".{output.name}.orchestration"):
        if not output.exists() and stage == "run":
            prepare(root, cfg, output)
        plan = verify(output)
        if plan["configuration"] != cfg:
            raise ValueError("configuration changed; use a new output directory")
        if stage in {"run", "cache"}:
            build_cache(output, plan)
        elif not (output / "cache_report.json").exists():
            raise ValueError("complete new validation cache first")
        result: dict[str, Any] = {"status": f"{stage}_completed", "output_dir": str(output)}
        if stage in {"run", "smoke"}:
            build_smoke_cache(output, plan)
            result["smoke_report"] = run_head(output, plan, gpu, smoke=True, resume=True)
        if stage in {"run", "train"}:
            if not (output / "smoke_completed.json").exists():
                raise ValueError("complete new-validation smoke first")
            run_head(output, plan, gpu, smoke=True, resume=True)
            result["head_report"] = run_head(output, plan, gpu, smoke=False, resume=resume)
            old_english_control(output, plan)
            result["return_file"] = bundle(output)
        verify(output)
        return result


def _sha(path: Path) -> str:
    from qwen_hotword.training.spanish_capacity import _sha as sha

    return sha(path)
