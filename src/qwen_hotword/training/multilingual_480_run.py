"""Pinned data preparation and explicit H200 smoke/cache stages; no automatic formal training."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.config import load_workzone_config
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha, _verified


def _pin(path: Path, expected: str, identities: dict[str, str]) -> None:
    actual = _sha(path)
    if actual != expected:
        raise ValueError(f"SHA256 differs: {path}")
    identities[str(path.resolve())] = actual


def _save(path: Path, data: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def scan_manifest(
    path: Path, split: str, *, per_group: int
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str], set[str]]:
    counts: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    smoke: list[dict[str, Any]] = []
    paths: set[str] = set()
    ids: set[str] = set()
    frames = 0
    hours: dict[str, float] = defaultdict(float)
    for row in _rows(path):
        lang = row["balanced_language_bucket"]
        audio = str(Path(row["audio_path"]).resolve())
        if (
            row["split"] != split
            or row["experiment"] != "full-ctc-v1"
            or lang not in {"en", "es", "pt"}
            or row["id"] in ids
            or audio in paths
            or row.get("ctc_time_upsampling_factor") != 2
        ):
            raise ValueError("invalid split/group/time factor or duplicate manifest identity")
        duration = float(row["duration_seconds"])
        estimate = row["estimated_ctc_input_length"]
        if (
            not math.isfinite(duration)
            or duration <= 0
            or type(estimate) is not int
            or estimate <= 0
        ):
            raise ValueError("invalid duration/frame estimate")
        release = row["release_source"].replace("temporal_2x_recovered", "temporal_2x_recovery")
        if release not in {"original_ready", "temporal_2x_recovery"}:
            raise ValueError("unexpected release group")
        ids.add(row["id"])
        paths.add(audio)
        counts[lang] += 1
        hours[lang] += duration / 3600
        frames += estimate
        group = f"{lang}::{release}"
        if groups[group] < per_group:
            smoke.append(row)
            groups[group] += 1
    if set(counts) != {"en", "es", "pt"}:
        raise ValueError("all three languages required")
    return (
        {
            "records": sum(counts.values()),
            "by_language": dict(counts),
            "hours_by_language": dict(hours),
            "estimated_base_frames": frames,
            "smoke_groups": dict(groups),
        },
        smoke,
        paths,
        ids,
    )


def prepare(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Read metadata only, pin inputs and create a fresh experiment contract."""
    if output.exists():
        raise FileExistsError(f"prepare requires a new directory: {output}")
    identities: dict[str, str] = {}
    freeze = Path(config["freeze_dir"])
    checked: dict[str, Any] = {}
    report = json.loads(_verified(freeze, "report.json", checked).read_text())
    if (
        report.get("status") != "completed"
        or report.get("training_manifest_ready") is not True
        or report["audio_identity_audit"]["conflict_groups_by_reason"]
        or report["artifacts"]["combined"]["sha256"] != config["train_sha256"]
    ):
        raise ValueError("frozen train was not released without conflicts")
    train = _verified(freeze, "full_ctc_train.jsonl", checked)
    validation, vocab = Path(config["validation_manifest"]), Path(config["vocab"])
    _pin(train, config["train_sha256"], identities)
    _pin(validation, config["validation_sha256"], identities)
    _pin(vocab, config["vocab_sha256"], identities)
    zone_path = Path(config["workzone_config"])
    zone = load_workzone_config(zone_path, require_existing_model=True)
    if (
        zone.model.dtype != "bfloat16"
        or zone.model.device != "cuda:0"
        or not (zone.model.local_files_only)
    ):
        raise ValueError("requires local-only bf16 Qwen on logical cuda:0")
    identities[str(zone_path.resolve())] = _sha(zone_path)
    baseline_path = Path(config["baseline_validation_cache"]) / "cache_config.json"
    baseline = json.loads(baseline_path.read_text())
    identities[str(baseline_path.resolve())] = _sha(baseline_path)
    if (
        baseline["source_manifest"]["sha256"] != config["validation_sha256"]
        or baseline["vocab"]["sha256"] != config["vocab_sha256"]
        or baseline["model"]["dtype"] != zone.model.dtype
        or Path(baseline["model"]["path"]).resolve() != zone.model.path.resolve()
        or baseline["tap_module"] != "thinker.audio_tower.ln_post"
        or baseline["ctc_time_upsampling_factor"] != 2
    ):
        raise ValueError("legacy cache model/validation/vocab contract differs")
    for name, key in (
        ("config.json", "config_sha256"),
        ("model.safetensors.index.json", "weight_index_sha256"),
    ):
        _pin(zone.model.path / name, baseline["model"][key], identities)
    print("Checking frozen manifests and choosing small deterministic smoke subsets", flush=True)
    train_stats, small_train, train_paths, train_ids = scan_manifest(train, "train", per_group=16)
    val_stats, small_val, val_paths, val_ids = scan_manifest(validation, "validation", per_group=8)
    if train_paths & val_paths or train_ids & val_ids:
        raise ValueError("train/legacy validation overlap")
    if train_stats["by_language"] != config["train_counts"] or (
        val_stats["records"] != config["validation_count"]
    ):
        raise ValueError("unexpected frozen sample counts")
    # Confirm the old fixed validation was protected by the completed byte audit.
    fingerprints = _verified(freeze, "audio_fingerprints.jsonl", checked)
    remaining = set(val_paths)
    for row in _rows(fingerprints):
        if "validation" in row["roles"]:
            remaining.discard(row["audio_path"])
    if remaining:
        raise ValueError("legacy validation contains audio outside audited protection")
    for filename, identity in checked.items():
        identities[str(Path(filename).resolve())] = identity["sha256"]
    # 1024 bf16 base-frame values; 25% storage margin plus 10 GiB for metadata/smoke/runs.
    raw_bytes = (train_stats["estimated_base_frames"] + val_stats["estimated_base_frames"]) * 2048
    plan = {
        "status": "prepared",
        "config": config,
        "input_sha256": identities,
        "train_manifest": str(train.resolve()),
        "validation_manifest": str(validation.resolve()),
        "train": train_stats,
        "validation": val_stats,
        "estimated_feature_bytes": raw_bytes,
        "recommended_free_bytes": math.ceil(raw_bytes * 1.25) + 10 * 1024**3,
        "smoke_policy": "first 16 train / 8 validation per language-release in frozen order",
        "training_policy": "fresh Head; all_samples_once; pilot 5 epochs, formal cap 30",
        "scheduler_metric": "validation_loss (unchanged from baseline)",
        "test_set_used": False,
        "model_loaded": False,
        "training_started": False,
        "limitations": [
            "Frame/storage estimates are not actual encoder lengths or a runtime estimate.",
            "Model config/index identity is checked; full weight bytes are not rehashed.",
            "Smoke PER is mechanics-only, not a new quality baseline.",
            "Equal epochs give the larger dataset about 3.2x baseline audio exposure.",
        ],
    }
    for filename, digest in identities.items():
        _pin(Path(filename), digest, {})
    output.mkdir(parents=True)
    for split, rows in (("train", small_train), ("validation", small_val)):
        path = output / f"smoke_{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        plan["input_sha256"][str(path.resolve())] = _sha(path)
    _save(output / "run_plan.json", plan)
    return plan


def commands(config: dict[str, Any], output: Path, stage: str) -> list[list[str]]:
    if stage not in {"smoke", "cache", "pilot", "formal"}:
        raise ValueError("unknown stage")
    smoke = stage == "smoke"
    cache = output / ("smoke_cache" if smoke else "cache")
    train = (
        output / "smoke_train.jsonl"
        if smoke
        else Path(config["freeze_dir"]) / ("full_ctc_train.jsonl")
    )
    validation = output / "smoke_validation.jsonl" if smoke else Path(config["validation_manifest"])
    if stage in {"smoke", "cache"}:
        result = [
            [
                sys.executable,
                "-B",
                "scripts/cache_full_training_features.py",
                "--config",
                config["workzone_config"],
                "--train-manifest",
                str(train),
                "--validation-manifest",
                str(validation),
                "--vocab",
                config["vocab"],
                "--output-dir",
                str(cache),
                "--encoder-batch-size",
                str(config["encoder_batch_size"]),
                "--samples-per-shard",
                str(config["samples_per_shard"]),
            ]
        ]
        if not smoke:
            return result
    else:
        result = []
    options = dict(config["training_options"])
    if smoke:
        options["minimum-epochs"] = 1
    train_cmd = [
        sys.executable,
        "-B",
        "scripts/train_full_ctc.py",
        "--train-cache",
        str(cache / "train"),
        "--validation-cache",
        str(cache / "validation"),
        "--train-manifest",
        str(train),
        "--validation-manifest",
        str(validation),
        "--vocab",
        config["vocab"],
        "--output-dir",
        str(output / ("smoke_head" if smoke else "head")),
        "--device",
        "cuda:0",
        "--epochs",
        str(1 if smoke else 5 if stage == "pilot" else 30),
    ]
    for name, value in options.items():
        train_cmd.extend([f"--{name}", str(value)])
    if stage == "formal":
        train_cmd.append("--resume")
    result.append(train_cmd)
    return result


def verify_smoke(output: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((output / "smoke_head/report.json").read_text())
    if report["status"] != "completed" or report["epochs_completed"] != 1:
        raise ValueError("one-epoch smoke has not completed")
    for key in ("final_validation_loss", "final_validation_phoneme_error_rate"):
        if not math.isfinite(report[key]):
            raise ValueError("non-finite smoke metrics")
    return report


def run_stage(config: dict[str, Any], output: Path, stage: str, gpu: str) -> dict[str, Any]:
    if stage not in {"smoke", "cache"}:
        raise ValueError("only smoke and cache execute here; formal training is a separate step")
    if not gpu.isdecimal():
        raise ValueError("specify exactly one physical GPU integer")
    # Lock separate from child cache/training locks, serialize the complete staged workflow.
    with exclusive_feature_cache_run(output.parent / f".{output.name}.orchestration"):
        if not output.exists():
            prepare(config, output)
        plan = json.loads((output / "run_plan.json").read_text())
        if plan["config"] != config:
            raise ValueError("run configuration changed; use a new run directory")
        for filename, digest in plan["input_sha256"].items():
            _pin(Path(filename), digest, {})
        if stage == "cache":
            verify_smoke(output)
            stored = sum(p.stat().st_size for p in (output / "cache").rglob("*") if p.is_file())
            required = max(0, plan["recommended_free_bytes"] - stored)
            if shutil.disk_usage(output).free < required:
                raise ValueError(
                    f"insufficient cache disk space: need {required / 1024**3:.1f} GiB"
                )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        for cmd in commands(config, output, stage):
            if cmd[2].endswith("train_full_ctc.py"):
                if (output / "smoke_head/report.json").exists():
                    verify_smoke(output)
                    continue
                if (output / "smoke_head/training_state_latest.pt").exists():
                    cmd.append("--resume")
            print("Running: " + " ".join(cmd), flush=True)
            subprocess.run(cmd, env=env, check=True)
        for filename, digest in plan["input_sha256"].items():
            _pin(Path(filename), digest, {})
        if stage == "smoke":
            smoke_report = verify_smoke(output)
            return {
                "status": "smoke_completed",
                "output_dir": str(output),
                "run_plan_sha256": _sha(output / "run_plan.json"),
                "train_manifest_sha256": config["train_sha256"],
                "validation_manifest_sha256": config["validation_sha256"],
                "recommended_free_gib": plan["recommended_free_bytes"] / 1024**3,
                "head_report": smoke_report,
                "formal_training_started": False,
            }
        summary = json.loads((output / "cache/feature_cache_report.json").read_text())
        if summary["status"] != "pass" or (
            summary["train"]["sample_count"] != plan["train"]["records"]
            or summary["validation"]["sample_count"] != plan["validation"]["records"]
        ):
            raise ValueError("cache count/status mismatch")
        return {
            "status": "cache_completed",
            "output_dir": str(output),
            "run_plan_sha256": _sha(output / "run_plan.json"),
            "train_manifest_sha256": config["train_sha256"],
            "validation_manifest_sha256": config["validation_sha256"],
            "cache": summary,
            "formal_training_started": False,
        }
