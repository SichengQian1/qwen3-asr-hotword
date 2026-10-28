#!/usr/bin/env python3
"""Audit boundary-frame differences, reuse noise caches, and resume the same experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "train", "evaluate", "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/noise_finetune.workzone.json"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--ratio", type=int, choices=(30, 50))
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    if args.stage != "check" and (args.gpu is None or args.gpu < 0):
        parser.error("training/evaluation requires --gpu N for your allocated physical H200")
    if args.ratio is not None and args.stage != "train":
        parser.error("--ratio is supported only for train")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.training.noise_recovery import recover

    try:
        config = json.loads(args.config.read_text())
        run = (args.run_dir or args.root / config["output"]).resolve()
        result = recover(args.root.resolve(), run, config, args.stage, ratio=args.ratio)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "recovery_failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
