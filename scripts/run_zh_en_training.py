#!/usr/bin/env python3
"""Full-data Chinese/US/UK 183-class Head, 70% 10 dB noise, through validation PER."""

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
    parser.add_argument("action", choices=("all", "prepare", "cache", "train", "status"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/zh_en_noise70.workzone.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gpu", type=int, help="allocated physical GPU; required for cache/train/all"
    )
    args = parser.parse_args()
    if args.action in {"all", "cache", "train"} and (args.gpu is None or args.gpu < 0):
        parser.error("GPU stages require an allocated --gpu N")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from qwen_hotword.training.zh_en_common import read
    from qwen_hotword.training.zh_en_run import run, status

    cfg = read(args.config)
    output = (args.output_dir or ROOT / cfg["output"]).resolve()
    if args.action == "status":
        print(json.dumps(status(output), ensure_ascii=False, indent=2))
        return 0
    try:
        result = run(ROOT, output, cfg, args.action)
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
