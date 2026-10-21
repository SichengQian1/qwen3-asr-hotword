#!/usr/bin/env python3
"""Select clean English target-count 1–3 cases and export real epoch27 Top7 candidates."""

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
        "--config", type=Path, default=REPO / "configs/english_sample_selection.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-run", type=Path, help="Optional sealed epoch27 CLEAN offline run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and args.reuse_run is None and (args.gpu is None or args.gpu < 0):
        parser.error("specify an allocated physical GPU with --gpu N")
    if args.gpu is not None:
        if args.gpu < 0:
            parser.error("--gpu must be nonnegative")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.inference.english_sample_selection import run_selection

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        result = run_selection(
            args.root.resolve(),
            args.config.resolve(),
            (args.output_dir or args.root / cfg["output"]).resolve(),
            prepare_only=args.prepare_only,
            resume=args.resume,
            reuse_run=args.reuse_run.resolve() if args.reuse_run else None,
        )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
