#!/usr/bin/env python3
"""Produce causal 2s English RAG context timelines for the downstream streaming interface."""

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
    parser.add_argument("stage", choices=("smoke", "full"))
    parser.add_argument("--effect", choices=("same-step", "next-step"), required=True)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/english_stream_contexts.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.audit_only and (args.gpu is None or args.gpu < 0):
        parser.error("specify an allocated physical GPU with --gpu N")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.inference.stream_context_run import run_stream_contexts

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        output = (
            args.output_dir or args.root / cfg["output_root"] / f"{args.stage}_{args.effect}"
        ).resolve()
        result = run_stream_contexts(
            args.root.resolve(),
            args.config.resolve(),
            output,
            stage=args.stage,
            effect=args.effect,
            resume=args.resume,
            audit_only=args.audit_only,
            smoke_run=(args.root / args.smoke_run).resolve() if args.smoke_run else None,
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
