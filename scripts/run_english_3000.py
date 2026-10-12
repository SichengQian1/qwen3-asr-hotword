#!/usr/bin/env python3
"""Build the English 3000-word union or evaluate three datasets with Top5/Top7."""

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
    parser.add_argument("stage", choices=("build", "audit", "run"))
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--config", type=Path, default=REPO / "configs/english_3000.workzone.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "run" and (args.gpu is None or args.gpu < 0):
        parser.error("run requires --gpu N for an unused physical GPU")
    if args.resume and args.stage != "run":
        parser.error("--resume applies only to an identical inference run")
    if args.stage == "run":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Physical GPU {args.gpu} -> logical cuda:0", flush=True)
    from qwen_hotword.evaluation.english_keywords import build_english_table
    from qwen_hotword.inference.english_retrieval import run_english_retrieval

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        root, config = args.root.resolve(), args.config.resolve()
        if args.stage == "build":
            if "audio_root" in cfg:
                raise ValueError(
                    "alternate-audio evaluation reuses the existing table; use audit/run"
                )
            report = build_english_table(root, config, root / cfg["tables"])
        else:
            report = run_english_retrieval(
                root,
                config,
                root / cfg["output"],
                resume=args.resume,
                audit_only=args.stage == "audit",
            )
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
