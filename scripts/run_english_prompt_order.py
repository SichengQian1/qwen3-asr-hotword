#!/usr/bin/env python3
"""Freeze/audit/run the 60-audio final-ASR prompt experiment on the H200 host."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation.prompt_order import preparation_report
from qwen_hotword.inference.prompt_order_experiment import audit, prepare, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "audit", "run"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/english_prompt_order.workzone.json"
    )
    parser.add_argument("--gpu", help="physical GPU index; mandatory for run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.action != "run":
        parser.error("--resume is only valid for run")
    if args.action == "run" and (args.gpu is None or not args.gpu.isdigit()):
        parser.error("run requires --gpu with one physical GPU index")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    try:
        config = args.config.resolve()
        if args.action == "prepare":
            result = prepare(ROOT, config)
        elif args.action == "audit":
            plan, output = audit(ROOT, config)
            result = {**preparation_report(plan, output), "status": "audit_pass"}
        else:
            report = run(ROOT, config, resume=args.resume)
            result = {
                k: report[k]
                for k in (
                    "status",
                    "output_dir",
                    "sample_count",
                    "main_completed_calls",
                    "extra_completed_repeat_calls",
                    "request_attempts_started",
                    "return_files",
                )
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "insufficient_samples" else 0
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
