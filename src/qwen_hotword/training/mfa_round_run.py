"""One fresh 100-class Head: prepare -> mixed cache -> train -> retrieval -> ASR."""

from __future__ import annotations

import json
import traceback
import zipfile
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.mfa_round_cache import build, compose
from qwen_hotword.training.mfa_round_common import contract, read
from qwen_hotword.training.mfa_round_labels import prepare
from qwen_hotword.training.sharded_ctc import exclusive_training_run, train_sharded_ctc_head
from qwen_hotword.training.stage_progress import stage_progress


def status(output: Path) -> dict[str, Any]:
    """Quick progress snapshot; no Tensor reads, hashing, GPU or inference."""
    value: dict[str, Any] = {
        "status": "progress_snapshot",
        "output": str(output),
        "metadata_only": True,
        "gpu_used": False,
    }
    if (output / "report.json").exists():
        value["run"] = read(output / "report.json")
    value["g2p_word_parts"] = {
        "requested": len(list((output / "g2p").glob("*/part-*/request.jsonl"))),
        "completed": len(list((output / "g2p").glob("*/part-*/predictions.completed.json"))),
    }
    value["cache_parts"] = {
        "requested": len(read(output / "prepared.json")["parts"])
        if (output / "prepared.json").exists()
        else None,
        "completed": len(list((output / "parts").glob("*/*/features/cache_summary.json"))),
    }
    metrics = output / "head/metrics.jsonl"
    if metrics.exists():
        lines = [s for s in metrics.read_text().splitlines() if s.strip()]
        for line in reversed(lines):
            try:
                value["last_completed_epoch_metrics"] = json.loads(line)
                break
            except json.JSONDecodeError:
                value["partial_metrics_line_ignored"] = True
    value["training_complete"] = (output / "training_completed.json").exists()
    value["retrieval_complete"] = (output / "retrieval/report.json").exists()
    value["asr_complete"] = (output / "asr/report.json").exists()
    return value


def train(
    root: Path,
    cfg: dict[str, Any],
    output: Path,
    prepared: dict[str, Any],
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    marker = output / "training_completed.json"
    if marker.exists():
        value = read(marker)
        pin(output / "head/ctc_head_best.pt", value["checkpoint_sha256"])
        pin(output / "head/report.json", value["report_sha256"])
        return value
    local = {**cfg, "absolute_vocab": str(root / cfg["vocab"])}
    train_cache, validation_cache, groups = compose(output, local, prepared)
    head, options, folder = prepared["head_config"], cfg["training"], output / "head"
    if (folder / "initial_head.json").exists():
        raise ValueError("this experiment must NOT initialize from any old Head")
    with exclusive_training_run(folder):
        with stage_progress("Train NEW random 100-class Head; frozen Encoder; 70% noisy inputs"):
            if (folder / "report.json").exists():
                result = read(folder / "report.json")
            else:
                result = train_sharded_ctc_head(
                    train_cache,
                    validation_cache,
                    load_phoneme_vocab(root / cfg["vocab"]),
                    folder,
                    vocab_path=root / cfg["vocab"],
                    device=device,
                    epochs=options["epochs"],
                    minimum_epochs=5,
                    early_stopping_patience=6,
                    early_stopping_min_delta=0.0005,
                    early_stopping_metric="validation_macro_per",
                    checkpoint_selection_metric="validation_macro_per",
                    validation_sample_groups=groups,
                    train_batch_size=options["batch_size"],
                    learning_rate=options["learning_rate"],
                    weight_decay=0.0001,
                    max_gradient_norm=5.0,
                    scheduler_patience=2,
                    scheduler_factor=0.5,
                    minimum_learning_rate=5e-6,
                    seed=cfg["training_seed"],
                    log_every_shards=25,
                    resume=(folder / "training_state_latest.pt").exists(),
                    head_type=head["head_type"],
                    head_hidden_dimension=head["hidden_dimension"],
                    head_kernel_size=head["kernel_size"],
                    head_dropout=head["dropout"],
                    head_time_upsampling_factor=head["time_upsampling_factor"],
                    initial_head_checkpoint=None,
                ).to_dict()
        if (
            result["status"] != "completed"
            or result["num_classes"] != 100
            or result["test_set_used"]
            or not result["cache_sha256_verified"]
            or result["best_epoch"] < 1
            or result["train_sample_count"] != train_cache.sample_count
        ):
            raise ValueError("incomplete or incompatible new-Head training report")
        value = {
            "status": "fresh_head_training_completed",
            "num_classes": 100,
            "head_initialization": "random",
            "old_head_loaded": False,
            "checkpoint_sha256": sha(folder / "ctc_head_best.pt"),
            "report_sha256": sha(folder / "report.json"),
            "best_epoch": result["best_epoch"],
            "epochs_completed": result["epochs_completed"],
            "test_set_used_for_training": False,
        }
        write_json(marker, value)
        return value


def bundle(output: Path) -> Path:
    selected = [
        "contract.json",
        "identity.json",
        "report.json",
        "label_report.json",
        "hotword_report.json",
        "training_completed.json",
        "head/report.json",
        "head/metrics.jsonl",
        "retrieval/report.json",
        "asr/report.json",
        "asr/runtime.json",
        "evaluation_inputs.json",
        "logs/runtime_probe.json",
        "logs/runtime_probe.log",
    ]
    if (output / "report.json").exists() and read(output / "report.json")["status"] == "failed":
        selected.append("error_traceback.txt")
    selected.extend(str(p.relative_to(output)) for p in (output / "parts").rglob("infeasible.json"))
    for name in ("prepared.json", "cache_completed.json"):
        if (output / name).exists():
            value = read(output / name)
            # Full file manifests remain in the work zone. Compact counts and identities return.
            write_json(
                output / (name.removesuffix(".json") + "_summary.json"),
                {
                    **{k: v for k, v in value.items() if k not in {"files", "parts"}},
                    "parts": len(value.get("parts", [])),
                    "identity_files": len(value.get("files", {})),
                    "full_marker_sha256": sha(output / name),
                },
            )
            selected.append(name.removesuffix(".json") + "_summary.json")
    path = output / "return_bundle.zip"
    temporary = output / "return_bundle.pending.zip"
    checksums = []
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in selected:
            source = output / name
            if source.is_file():
                archive.write(source, name)
                checksums.append(f"{sha(source)}  {name}\n")
        archive.writestr("sha256.txt", "".join(checksums))
    temporary.replace(path)
    return path


def run(root: Path, cfg: dict[str, Any], output: Path, action: str) -> dict[str, Any]:
    stages = ["prepare", "cache", "train", "evaluate", "asr"] if action == "all" else [action]
    with exclusive_feature_cache_run(output.parent / ("." + output.name + ".orchestration")):
        contract(root, cfg, output)
        report: dict[str, Any] = {
            "status": "running",
            "output": str(output),
            "stages": {},
            "noise_percent": 70,
            "snr_db": 10,
            "num_classes": 100,
            "head_initialization": "random",
            "old_labels_reused": False,
            "encoder_frozen": True,
            "human_review_required": False,
        }
        try:
            for stage in stages:
                report["active_stage"] = stage
                write_json(output / "report.json", report)
                if stage == "prepare":
                    result = prepare(root, cfg, output)
                    report["stages"][stage] = {"status": result["status"], "noise": result["noise"]}
                elif stage == "cache":
                    result = build(root, cfg, output, read(output / "prepared.json"))
                    report["stages"][stage] = {
                        "status": result["status"],
                        "backends": result["backends"],
                    }
                elif stage == "train":
                    build(root, cfg, output, read(output / "prepared.json"))
                    report["stages"][stage] = train(
                        root, cfg, output, read(output / "prepared.json")
                    )
                elif stage == "evaluate":
                    from qwen_hotword.evaluation.mfa_round import evaluate

                    result = evaluate(root, cfg, output)
                    report["stages"][stage] = {"status": result["status"], "rows": result["rows"]}
                elif stage == "asr":
                    from qwen_hotword.evaluation.mfa_round import asr

                    result = asr(root, cfg, output)
                    report["stages"][stage] = {"status": result["status"], "rows": result["rows"]}
            report["status"] = "completed"
            report["active_stage"] = None
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            (output / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(output / "report.json", report)
        report["return_bundle"] = str(bundle(output))
        write_json(output / "report.json", report)
        bundle(output)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
