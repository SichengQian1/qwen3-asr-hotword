#!/usr/bin/env python3
"""Recompute both language caches and train a paired Head on the merged r15 backbone."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.phonemes.mfa_v03_assets import pin, write_json
from qwen_hotword.training.en_ar_backbone_refresh import prepare, verify_plan
from qwen_hotword.training.stage_progress import stage_progress
from scripts.label_arabic_native import exclusive_job
from scripts.run_en_ar_training import NATIVE_VOCAB, NATIVE_VOCAB_SHA, VOCAB, VOCAB_SHA, train_stage

CONFIG = ROOT / "configs/en_ar_merged_r15.workzone.json"


def wait_for_manifests(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    paths = [
        ROOT / config[key] / "label_report.json" for key in ("baseline_english", "baseline_arabic")
    ]
    with stage_progress("Wait for sealed baseline English/Arabic manifests (no GPU use)"):
        last_pending = None
        while True:
            pending = [str(path) for path in paths if not path.is_file()]
            for path in paths:
                if (
                    path.is_file()
                    and json.loads(path.read_text()).get("status") != "labels_complete"
                ):
                    raise ValueError(f"baseline selection did not complete successfully: {path}")
            if not pending:
                return
            if pending != last_pending:
                print("Waiting for baseline manifest seal: " + ", ".join(pending), flush=True)
                last_pending = pending
            time.sleep(5)


def wait_for_gpu(gpu: int, minimum_gib: float) -> None:
    with stage_progress(f"Wait for physical GPU {gpu}: at least {minimum_gib:g} GiB free"):
        while True:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            free = float(result.stdout.strip()) / 1024
            print(f"GPU {gpu}: free={free:.1f} GiB; required={minimum_gib:g} GiB", flush=True)
            if free >= minimum_gib:
                return
            time.sleep(5)


def verify_cache_backbone(destination: Path, plan: dict[str, Any]) -> None:
    expected = {
        "path": plan["model"]["view"],
        "dtype": "bfloat16",
        "config_sha256": plan["model"]["files"]["config.json"]["sha256"],
        "weight_index_sha256": plan["model"]["index_sha256"],
    }
    for split in ("train", "validation"):
        metadata = json.loads((destination / "cache" / split / "cache_config.json").read_text())
        if metadata.get("model") != expected:
            raise ValueError(f"cache uses a different model: {destination}/{split}")


def cache_language(
    output: Path, plan: dict[str, Any], language: str, gpu: int, minimum_gib: float
) -> None:
    destination = output / "english" if language == "en" else output
    cache = destination / "cache"
    cache_report = cache / "feature_cache_report.json"
    if cache_report.is_file() and json.loads(cache_report.read_text()).get("status") == "pass":
        verify_cache_backbone(destination, plan)
        print(
            f"Completed NEW-model {language} cache found; shard SHA verified before Head training",
            flush=True,
        )
        return
    wait_for_gpu(gpu, minimum_gib)
    command = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/cache_full_training_features.py"),
        "--config",
        plan["cache_config"]["path"],
        "--train-manifest",
        str(destination / "train.jsonl"),
        "--validation-manifest",
        str(destination / "validation.jsonl"),
        "--vocab",
        plan["vocabularies"][language]["path"],
        "--output-dir",
        str(cache),
        "--encoder-batch-size",
        "8",
        "--samples-per-shard",
        "512",
        "--prevalidated-audio-paths",
        "--expected-train-samples",
        str(plan["manifests"][f"{language}/train"]["samples"]),
        "--expected-validation-samples",
        str(plan["manifests"][f"{language}/validation"]["samples"]),
    ]
    with stage_progress(f"Recompute {language} Encoder cache with MERGED model (see shard counts)"):
        subprocess.run(
            command,
            check=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
        )
    verify_cache_backbone(destination, plan)


def execute(output: Path, plan: dict[str, Any], gpu: int, minimum_gib: float) -> dict[str, Any]:
    for language in ("en", "ar"):
        cache_language(output, plan, language, gpu, minimum_gib)
        verify_plan(output, hash_weights=False)
    head = output / "head"
    completed = head / "report.json"
    if not completed.is_file() or json.loads(completed.read_text()).get("status") != "completed":
        state = head / "training_state_latest.pt"
        if not state.is_file() and any(
            (head / name).exists() for name in ("metrics.jsonl", "ctc_head_best.pt")
        ):
            raise ValueError(
                "Head files exist without resumable training state; preserve this output for review"
            )
        wait_for_gpu(gpu, minimum_gib)
        train_stage(
            output, gpu, output / "english", vocab_path=NATIVE_VOCAB, resume=state.is_file()
        )
    verify_plan(output, hash_weights=False)
    sources_path = output / "cache_sources.json"
    sources = json.loads(sources_path.read_text())
    sources.update(
        {
            "baseline_hidden_states_reused": False,
            "english_cache_origin": "recomputed_with_new_merged_encoder",
            "arabic_cache_origin": "recomputed_with_new_merged_encoder",
            "merged_model": plan["model"]["source"],
            "merged_weight_sha256": plan["model"]["files"]["model.safetensors"]["sha256"],
        }
    )
    write_json(sources_path, sources)
    report = {
        "status": "completed",
        "output_dir": str(output),
        "physical_gpu": gpu,
        "model_source": plan["model"]["source"],
        "merged_weight_sha256": plan["model"]["files"]["model.safetensors"]["sha256"],
        "paired_manifest_sha256": {
            key: value["sha256"] for key, value in plan["manifests"].items()
        },
        "baseline_hidden_states_reused": False,
        "english_encoder_recomputed": True,
        "arabic_encoder_recomputed": True,
        "head": json.loads(completed.read_text()),
        "test_set_used": False,
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "run" and (args.gpu is None or args.gpu < 0):
        parser.error("run requires --gpu N")
    if args.action == "prepare" and args.resume:
        parser.error("--resume applies to run after a complete plan.json exists")
    output = args.output_dir.resolve()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        with exclusive_job(output):
            pin(VOCAB, VOCAB_SHA)
            pin(NATIVE_VOCAB, NATIVE_VOCAB_SHA)
            if args.resume:
                plan = verify_plan(output)
                if plan["config"]["path"] != str(args.config.resolve()):
                    raise ValueError("resume configuration path changed")
            else:
                if output.exists():
                    raise FileExistsError(
                        "output directory exists; use --resume with a complete plan.json"
                    )
                if args.action == "run":
                    wait_for_manifests(args.config.resolve())
                plan = prepare(
                    ROOT,
                    args.config.resolve(),
                    output,
                    legacy_vocab=VOCAB,
                    native_vocab=NATIVE_VOCAB,
                )
            for key, value in plan["manifests"].items():
                print(
                    f"Frozen baseline {key}: {value['samples']:,} rows, "
                    f"{value['hours']:.6f}h, SHA={value['sha256']}",
                    flush=True,
                )
            if args.action == "prepare":
                print(
                    json.dumps(
                        {
                            "status": "prepared",
                            "output_dir": str(output),
                            "model": plan["model"],
                            "manifests": plan["manifests"],
                            "minimum_cache_free_gib": plan["minimum_cache_free_gib"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
            config = json.loads(args.config.read_text())
            report = execute(output, plan, args.gpu, config["minimum_gpu_free_gib"])
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            json.dumps(
                {"status": "failed", "error": str(error), "output_dir": str(output)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
