#!/usr/bin/env python3
"""Run all 6,000 English + 400 Spanish + 400 Portuguese causal stream tests."""

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
    parser.add_argument("--effect", choices=("same-step", "next-step"), required=True)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/multilingual_stream_contexts.workzone.json"
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
    from qwen_hotword.inference.multilingual_stream_run import run_multilingual_stream

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        output = args.output_dir or args.root / cfg["output_root"] / f"full_{args.effect}"
        summary = run_multilingual_stream(
            args.root.resolve(),
            args.config.resolve(),
            output.resolve(),
            effect=args.effect,
            resume=args.resume,
            audit_only=args.audit_only,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
