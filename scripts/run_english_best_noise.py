#!/usr/bin/env python3
"""Evaluate the completed best Head on English 6000 noisy audio and the original joint 4k table."""

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
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/english_best_noise10db_4k.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.audit_only and (args.gpu is None or args.gpu < 0):
        parser.error("specify an allocated physical GPU with --gpu N")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.inference.english_best_noise import prepare_noise
    from qwen_hotword.inference.english_head_preview import run_preview

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        output = (args.output_dir or args.root / cfg["output"]).resolve()
        result = run_preview(
            args.root.resolve(),
            args.config.resolve(),
            output,
            resume=args.resume,
            audit_only=args.audit_only,
            prepare_function=prepare_noise,
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
