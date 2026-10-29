#!/usr/bin/env python3
"""New v0.3 labels -> random 100-class Head with 70% noise -> paired retrieval/ASR."""

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
        "action", choices=("prepare", "cache", "train", "evaluate", "asr", "all", "status")
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/mfa_v03_noise70.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, help="allocated physical GPU; required for GPU stages")
    args = parser.parse_args()
    if args.action not in {"prepare", "status"} and (args.gpu is None or args.gpu < 0):
        parser.error("GPU stages require --gpu N for an allocated physical GPU")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from qwen_hotword.training.mfa_round_run import run, status

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg["config_path"] = str(args.config.resolve())
    output = (args.output_dir or ROOT / cfg["output"]).resolve()
    if args.action == "status":
        print(json.dumps(status(output), ensure_ascii=False, indent=2), flush=True)
        return 0
    try:
        report = run(ROOT, cfg, output, args.action)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1
    return int(report["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
