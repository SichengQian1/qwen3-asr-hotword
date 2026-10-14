#!/usr/bin/env python3
"""Prepare new country English in full, then explicitly run the fixed trilingual CTC stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.english_refresh import prepare, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "cache", "pilot", "formal"))
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/english_countries_refresh.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = None
    if args.stage != "prepare" and (args.gpu is None or not args.gpu.isdecimal()):
        parser.error("choose one allocated physical GPU with --gpu N")
    if args.resume and args.stage not in {"prepare", "pilot"}:
        parser.error("--resume applies to prepare/pilot; other GPU stages already resume safely")
    try:
        root = args.root.resolve()
        config = json.loads((root / args.config).read_text())
        output = (root / (args.output_dir or config["output"])).resolve()
        result = (
            prepare(root, config, output, resume=args.resume, workers=args.workers, jobs=args.jobs)
            if args.stage == "prepare"
            else run_stage(root, output, args.stage, args.gpu or "", resume=args.resume)
        )
        if args.stage == "prepare":
            result = {
                k: result[k]
                for k in (
                    "status",
                    "output_dir",
                    "selection",
                    "excluded_english",
                    "fixed_languages_byte_identical",
                    "fixed_validation_unchanged",
                )
            }
            result["return_file"] = str(output / "return_bundle.zip")
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        result = {"status": "failed", "error": str(error)}
        if output is not None:
            result["output_dir"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
