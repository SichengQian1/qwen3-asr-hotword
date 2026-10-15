"""Path-only GPU preflight; retain the sealed preparation and historical training contract."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.config import load_workzone_config
from qwen_hotword.training import multilingual_480_run as historical
from qwen_hotword.training.ctc_overfit import canonical_audio_path
from qwen_hotword.training.english_refresh_fast import POLICY
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress


def scan_manifest(
    path: Path, split: str, *, per_group: int, expected: int
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str], set[str]]:
    """Same deterministic smoke order and metadata checks, without per-WAV filesystem I/O."""
    counts: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    hours: dict[str, float] = defaultdict(float)
    smoke: list[dict[str, Any]] = []
    paths: set[str] = set()
    ids: set[str] = set()
    frames = 0
    with stage_progress(f"Scan {split} manifest (metadata only)", total=expected) as progress:
        for number, row in enumerate(_rows(path), 1):
            lang = row["balanced_language_bucket"]
            audio = str(canonical_audio_path(row["audio_path"]))
            if (
                row["split"] != split
                or row["experiment"] != "full-ctc-v1"
                or lang not in {"en", "es", "pt"}
                or row["id"] in ids
                or audio in paths
                or row.get("ctc_time_upsampling_factor") != 2
            ):
                raise ValueError("invalid split/group/time factor or duplicate manifest identity")
            duration, estimate = float(row["duration_seconds"]), row["estimated_ctc_input_length"]
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
            progress(number)
        if sum(counts.values()) != expected or set(counts) != {"en", "es", "pt"}:
            raise ValueError(f"unexpected frozen {split} sample counts/languages")
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
    """Publish a compatible plan only after metadata/identity checks pass; never scan WAVs."""
    if output.exists():
        raise FileExistsError(f"prepare requires a new training directory: {output}")
    if config.get("audio_isolation_policy") != POLICY:
        raise ValueError("explicit path-only preparation required")
    identities: dict[str, str] = {}
    checked: dict[str, Any] = {}
    freeze = Path(config["freeze_dir"])
    with stage_progress("Verify sealed manifests and model metadata"):
        report = json.loads(_verified(freeze, "report.json", checked).read_text())
        if (
            report.get("status") != "completed"
            or report.get("training_manifest_ready") is not True
            or report.get("audio_isolation_policy") != POLICY
            or report.get("path_identity_audit", {}).get("status") != "pass"
            or report["artifacts"]["combined"]["sha256"] != config["train_sha256"]
        ):
            raise ValueError("frozen path-only training manifest not released")
        train = _verified(freeze, "full_ctc_train.jsonl", checked)
        validation, vocab = Path(config["validation_manifest"]), Path(config["vocab"])
        for path, digest in (
            (train, config["train_sha256"]),
            (validation, config["validation_sha256"]),
            (vocab, config["vocab_sha256"]),
        ):
            historical._pin(path, digest, identities)
        zone_path = Path(config["workzone_config"])
        zone = load_workzone_config(zone_path, require_existing_model=True)
        if (
            zone.model.dtype != "bfloat16"
            or zone.model.device != "cuda:0"
            or not zone.model.local_files_only
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
            historical._pin(zone.model.path / name, baseline["model"][key], identities)
    train_stats, small_train, train_paths, train_ids = scan_manifest(
        train, "train", per_group=16, expected=sum(config["train_counts"].values())
    )
    val_stats, small_val, val_paths, val_ids = scan_manifest(
        validation, "validation", per_group=8, expected=config["validation_count"]
    )
    if train_paths & val_paths or train_ids & val_ids:
        raise ValueError("train/legacy validation overlap")
    if train_stats["by_language"] != config["train_counts"]:
        raise ValueError("unexpected frozen sample counts")
    with stage_progress("Verify protected validation paths and input SHA256"):
        # In this policy these are retained protection metadata, not a new byte audit.
        fingerprints = _verified(freeze, "audio_fingerprints.jsonl", checked)
        remaining = set(val_paths)
        for row in _rows(fingerprints):
            if "validation" in row["roles"]:
                remaining.discard(str(canonical_audio_path(row["audio_path"])))
        if remaining:
            raise ValueError("legacy validation contains audio outside protected paths")
        for filename, identity in checked.items():
            identities[str(Path(filename).resolve())] = identity["sha256"]
        for filename, digest in identities.items():
            historical._pin(Path(filename), digest, {})
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
        "preflight_policy": "sealed canonical paths; no per-audio stat/readlink before decoding",
        "limitations": [
            "Path-only isolation; different paths with identical audio content remain undetected.",
            "Audio readability and actual CTC lengths are checked during extraction.",
            "Frame/storage estimates are not actual encoder lengths or a runtime estimate.",
            "Model config/index identity is checked; full weight bytes are not rehashed.",
            "Smoke PER is mechanics-only, not a new quality baseline.",
        ],
    }
    # Stage separately so Ctrl-C cannot leave an apparently reusable half-written run plan.
    pending = output.parent / f".{output.name}.plan-{uuid.uuid4().hex[:10]}"
    pending.mkdir()
    for split, rows in (("train", small_train), ("validation", small_val)):
        filename = f"smoke_{split}.jsonl"
        with (pending / filename).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        identities[str(output / filename)] = _sha(pending / filename)
    historical._save(pending / "run_plan.json", plan)
    pending.rename(output)
    print(f"Smoke plan ready: train={len(small_train)}, validation={len(small_val)}", flush=True)
    return plan


def commands(
    config: dict[str, Any], output: Path, stage: str, plan: dict[str, Any]
) -> list[list[str]]:
    result = historical.commands(config, output, stage)
    counts = {
        split: sum(plan[split]["smoke_groups"].values())
        if stage == "smoke"
        else plan[split]["records"]
        for split in ("train", "validation")
    }
    result[0].extend(
        [
            "--prevalidated-audio-paths",
            "--expected-train-samples",
            str(counts["train"]),
            "--expected-validation-samples",
            str(counts["validation"]),
        ]
    )
    return result


def run_stage(config: dict[str, Any], output: Path, stage: str, gpu: str) -> dict[str, Any]:
    if stage not in {"smoke", "cache"} or not gpu.isdecimal():
        raise ValueError("choose smoke/cache and exactly one physical GPU integer")
    if config.get("audio_isolation_policy") != POLICY:
        raise ValueError("explicit path-only preparation required")
    with exclusive_feature_cache_run(output.parent / f".{output.name}.orchestration"):
        if not output.exists():
            prepare(config, output)
        plan = json.loads((output / "run_plan.json").read_text())
        if plan["config"] != config:
            raise ValueError("run configuration changed; use a new run directory")
        with stage_progress("Check run-plan input SHA256"):
            for filename, digest in plan["input_sha256"].items():
                historical._pin(Path(filename), digest, {})
        if stage == "cache":
            historical.verify_smoke(output)
            with stage_progress("Check available cache storage"):
                stored = sum(p.stat().st_size for p in (output / "cache").rglob("*") if p.is_file())
                required = max(0, plan["recommended_free_bytes"] - stored)
                if shutil.disk_usage(output).free < required:
                    raise ValueError(
                        f"insufficient cache disk space: need {required / 1024**3:.1f} GiB"
                    )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        for cmd in commands(config, output, stage, plan):
            training = cmd[2].endswith("train_full_ctc.py")
            if training:
                if (output / "smoke_head/report.json").exists():
                    historical.verify_smoke(output)
                    continue
                if (output / "smoke_head/training_state_latest.pt").exists():
                    cmd.append("--resume")
            print("Running: " + " ".join(cmd), flush=True)
            label = (
                "Smoke Head: one epoch"
                if training
                else f"{stage} Encoder: see child/shard progress"
            )
            with stage_progress(label):
                subprocess.run(cmd, env=env, check=True)
        with stage_progress("Recheck inputs after execution"):
            for filename, digest in plan["input_sha256"].items():
                historical._pin(Path(filename), digest, {})
        result = {
            "status": f"{stage}_completed",
            "output_dir": str(output),
            "run_plan_sha256": _sha(output / "run_plan.json"),
            "train_manifest_sha256": config["train_sha256"],
            "validation_manifest_sha256": config["validation_sha256"],
            "formal_training_started": False,
        }
        if stage == "smoke":
            result.update(
                head_report=historical.verify_smoke(output),
                recommended_free_gib=plan["recommended_free_bytes"] / 1024**3,
            )
        else:
            summary = json.loads((output / "cache/feature_cache_report.json").read_text())
            if summary["status"] != "pass" or any(
                summary[split]["sample_count"] != plan[split]["records"]
                for split in ("train", "validation")
            ):
                raise ValueError("cache count/status mismatch")
            result["cache"] = summary
        return result
