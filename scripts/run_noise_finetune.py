#!/usr/bin/env python3
"""Full three-language 480h/language MUSAN 10dB warm-start experiments."""

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
    parser.add_argument("stage", choices=("prepare", "cache", "train", "evaluate", "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/noise_finetune.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--ratio", type=int, choices=(30, 50), help="train only one arm; default both"
    )
    parser.add_argument(
        "--gpu", type=int, help="allocated physical H200, exposed as logical cuda:0"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage != "prepare" and (args.gpu is None or args.gpu < 0):
        parser.error("GPU stages require --gpu N for an allocated physical GPU")
    if args.ratio and args.stage != "train":
        parser.error("--ratio is supported only by the train stage")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.training.noise_run import run

    try:
        config = json.loads(args.config.read_text())
        output = (args.output_dir or args.root / config["output"]).resolve()
        result = run(
            args.root.resolve(), config, output, args.stage, ratio=args.ratio, resume=args.resume
        )
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
