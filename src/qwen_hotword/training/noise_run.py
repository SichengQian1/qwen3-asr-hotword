"""Work-zone orchestration: prepare, two full-data warm starts, paired retrieval evaluation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.english_newval import pin
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.noise_cache import build_noise_cache, compose_views, publish
from qwen_hotword.training.noise_prepare import prepare, verify_plan
from qwen_hotword.training.sharded_ctc import exclusive_training_run, train_sharded_ctc_head
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def train(
    output: Path, plan: dict[str, Any], ratio: int, *, resume: bool, device: str = "cuda:0"
) -> dict[str, Any]:
    folder = output / f"head_{ratio}"
    marker = folder / "completed.json"
    if marker.exists():
        identity = json.loads(marker.read_text())
        pin(output / "plan.json", {}, identity["plan_sha256"])
        pin(Path(plan["initial_head"]), {}, identity["initial_head_sha256"])
        pin(folder / "report.json", {}, identity["report_sha256"])
        pin(folder / "ctc_head_best.pt", {}, identity["checkpoint_sha256"])
        result: dict[str, Any] = json.loads((folder / "report.json").read_text())
        return result
    if folder.exists() and not resume:
        raise FileExistsError("Head output exists; use --resume for this same experiment")
    if (
        folder.exists()
        and not (folder / "training_state_latest.pt").exists()
        and any(p.name != ".ctc_training.lock" for p in folder.iterdir())
    ):
        raise ValueError("partial Head output has no resumable state; preserve and inspect")
    train_cache, validation_cache, groups = compose_views(output, plan, ratio)
    options = plan["configuration"]["training"]
    head_config = plan["head_config"]
    with exclusive_training_run(folder):
        # A finished trainer report can precede the orchestration marker after interruption.
        if (folder / "report.json").exists():
            result = json.loads((folder / "report.json").read_text())
        else:
            result = train_sharded_ctc_head(
                train_cache,
                validation_cache,
                load_phoneme_vocab(plan["vocab"]),
                folder,
                vocab_path=plan["vocab"],
                device=device,
                epochs=options["epochs"],
                minimum_epochs=options["epochs"],
                early_stopping_patience=options["epochs"] + 1,
                early_stopping_min_delta=0.0005,
                early_stopping_metric="validation_macro_per",
                checkpoint_selection_metric="validation_macro_per",
                validation_sample_groups=groups,
                train_batch_size=options["batch_size"],
                learning_rate=options["learning_rate"],
                weight_decay=0.0001,
                max_gradient_norm=5.0,
                scheduler_patience=options["epochs"] + 1,
                scheduler_factor=0.5,
                minimum_learning_rate=5e-6,
                seed=plan["configuration"]["seed"],
                log_every_shards=25,
                resume=resume,
                head_type=head_config["head_type"],
                head_hidden_dimension=head_config["hidden_dimension"],
                head_kernel_size=head_config["kernel_size"],
                head_dropout=head_config["dropout"],
                head_time_upsampling_factor=head_config["time_upsampling_factor"],
                initial_head_checkpoint=plan["initial_head"],
            ).to_dict()
        if (
            result["status"] != "completed"
            or result["epochs_completed"] != options["epochs"]
            or not result["cache_sha256_verified"]
            or result["test_set_used"]
        ):
            raise ValueError("incomplete or incompatible noise Head training report")
        expected_initial = {
            "path": str(Path(plan["initial_head"]).resolve()),
            "sha256": plan["configuration"]["initial_head_sha256"],
        }
        if json.loads((folder / "initial_head.json").read_text()) != expected_initial:
            raise ValueError("noise training did not warm start from the fixed baseline")
        publish(
            marker,
            {
                "plan_sha256": _sha(output / "plan.json"),
                "initial_head_sha256": expected_initial["sha256"],
                "report_sha256": _sha(folder / "report.json"),
                "checkpoint_sha256": _sha(folder / "ctc_head_best.pt"),
                "optimizer_reset_for_new_experiment": True,
                "noise_percent": ratio,
            },
        )
    return result


def bundle(output: Path) -> Path:
    """Small reviewable results only: no tensors, model weights, WAVs or full manifests."""
    names = ["prepared.json", "noise_inventory.json"]
    for name in ("cache_completed.json", "evaluation/report.json"):
        if (output / name).is_file():
            names.append(name)
    for ratio in (30, 50):
        for name in ("report.json", "metrics.jsonl", "initial_head.json", "completed.json"):
            relative = f"head_{ratio}/{name}"
            if (output / relative).is_file():
                names.append(relative)
    # Keep the large cache-file identity listing in the work zone; return its identity/count.
    if "cache_completed.json" in names:
        names.remove("cache_completed.json")
        cache = json.loads((output / "cache_completed.json").read_text())
        publish(
            output / "cache_summary_return.json",
            {
                "status": cache["status"],
                "parts": cache["parts"],
                "cache_marker_sha256": _sha(output / "cache_completed.json"),
                "identity_files": len(cache["sha256"]),
                "mixing": cache["mixing"],
                "test_set_used": False,
            },
        )
        names.append("cache_summary_return.json")
    payload = {name: (output / name).read_bytes() for name in names}
    checksums = "".join(f"{_sha(output / name)}  {name}\n" for name in sorted(names))
    final = output / "return_bundle.zip"
    temporary = output / ".return_bundle.pending.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in payload.items():
            archive.writestr(name, content)
        archive.writestr("sha256.txt", checksums)
    temporary.replace(final)  # Only this explicitly regenerable compact return bundle is replaced.
    return final


def run(
    root: Path, config: dict[str, Any], output: Path, stage: str, *, ratio: int | None, resume: bool
) -> dict[str, Any]:
    lock = output.parent / f".{output.name}.noise-orchestration"
    with exclusive_feature_cache_run(lock):
        if stage in {"prepare", "all"} and not output.exists():
            plan = prepare(root, config, output)
        else:
            if stage == "prepare" and not resume:
                raise FileExistsError("preparation exists; --resume verifies it")
            plan = verify_plan(output, config)
        result: dict[str, Any] = {"status": "prepared", "output": str(output)}
        if stage in {"cache", "train", "all"}:
            with stage_progress("Build/reuse shared 10dB noise features"):
                cache = build_noise_cache(output, plan)
                result = {"status": cache["status"], "parts": cache["parts"]}
            import torch

            torch.cuda.empty_cache()
        if stage in {"train", "all"}:
            for percent in (ratio,) if ratio else (30, 50):
                with stage_progress(f"Warm-start full-data noise Head {percent}%"):
                    report = train(output, plan, percent, resume=resume)
                result[str(percent)] = {
                    "best_epoch": report["best_epoch"],
                    "macro_per": report["best_validation_macro_phoneme_error_rate"],
                    "checkpoint": report["best_checkpoint_path"],
                }
            result["status"] = "training_completed"
        if stage in {"evaluate", "all"}:
            from qwen_hotword.training.noise_evaluation import evaluate

            result = evaluate(root, output, plan, resume=resume)
        result["return_bundle"] = str(bundle(output))
        return result
