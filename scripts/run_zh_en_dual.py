#!/usr/bin/env python3
"""Reuse prior labels, add Chinese data, compare two-H200 concurrency, train through PER."""

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
    parser.add_argument(
        "action",
        choices=("all", "prepare", "cache", "train", "status", "gpu-worker", "train-worker"),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/zh_en_noise70_dual.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus", type=int, nargs=2, metavar=("GPU_A", "GPU_B"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--worker-id", type=int)
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from qwen_hotword.training.zh_en_common import read

    if args.action == "gpu-worker":
        if args.spec is None or args.worker_id is None:
            parser.error("internal worker requires spec and worker ID")
        from qwen_hotword.training.paired_gpu_worker import worker

        worker(args.spec, args.worker_id)
        return 0
    if args.action == "train-worker":
        if args.output_dir is None:
            parser.error("internal training worker requires output directory")
        output = args.output_dir.resolve()
        cfg = read(output / "config.json")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu_ids"][0])
        from qwen_hotword.training.zh_en_run import train

        train(ROOT, output, cfg, read(ROOT / cfg["foundation_config"]))
        return 0
    cfg = read(args.config)
    if args.gpus is not None:
        if min(args.gpus) < 0 or len(set(args.gpus)) != 2:
            parser.error("two distinct nonnegative allocated GPU IDs required")
        cfg["gpu_ids"] = args.gpus
    output = (args.output_dir or ROOT / cfg["output"]).resolve()
    from qwen_hotword.training.paired_run import run, status

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
