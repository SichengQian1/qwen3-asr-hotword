#!/usr/bin/env python3
"""Run the fixed eight wave groups on a user-selected GPU and export Top5/Top7."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, help="physical GPU index, exposed as logical cuda:0")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/wave_retrieval.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only", action="store_true", help="CPU checks only; no output/model"
    )
    args = parser.parse_args()
    if not args.audit_only and (args.gpu is None or args.gpu < 0):
        parser.error("specify an unused physical GPU with --gpu N")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.inference.wave_retrieval import run_waves

    try:
        cfg = json.loads(args.config.read_text())
        output = (args.output_dir or args.root / cfg["output"]).resolve()
        report = run_waves(
            args.root.resolve(),
            args.config.resolve(),
            output,
            resume=args.resume,
            audit_only=args.audit_only,
        )
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
