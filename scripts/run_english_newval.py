#!/usr/bin/env python3
"""Freeze unused-pool English validation and reuse features for a new 40-epoch Head run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cache", "smoke", "train", "run"))
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/english_countries_newval.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage != "prepare" and (args.gpu is None or not args.gpu.isdecimal()):
        parser.error("choose one allocated physical GPU using --gpu N")
    if args.resume and args.stage not in {"run", "train"}:
        parser.error("--resume applies to run/train only")
    from qwen_hotword.training.english_newval_run import execute

    try:
        root = args.root.resolve()
        os.chdir(root)
        cfg = json.loads((root / args.config).read_text())
        output = (root / (args.output_dir or cfg["output"])).resolve()
        result = execute(root, cfg, output, args.stage, args.gpu or "", resume=args.resume)
        if args.stage == "prepare":
            result = {
                "status": result["status"],
                "output_dir": result["output_dir"],
                "english_target_hours": result["validation_target_english_hours"],
                "unused_qualified_english_hours": result["unused_qualified_english"]["hours"],
                "selected_english_records": result["selected_english"]["records"],
                "selected_english_hours": result["selected_english"]["hours"],
                "selected_by_country_source": result["selected_english"]["by_dimension"]["source"],
                "validation_counts": result["validation"]["by_language"],
                "train_validation_path_overlaps": result["train_validation_path_overlaps"],
                "epochs": result["epochs"],
                "return_file": result["return_file"],
            }
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as e:
        result = {"status": "failed", "error": str(e)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
