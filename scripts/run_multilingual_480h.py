#!/usr/bin/env python3
"""Prepare the pinned 480h experiment, then run H200 smoke/cache explicitly."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.multilingual_480_run import commands, prepare, run_stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "cache", "show-training"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/480h_training.workzone.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", help="One allocated physical GPU, required for smoke/cache")
    args = parser.parse_args()
    os.chdir(REPO)
    config = json.loads(args.config.read_text())
    output = (args.output_dir or Path(config["run_dir"])).resolve()
    if args.stage == "show-training":
        result = {stage: commands(config, output, stage) for stage in ("pilot", "formal")}
    elif args.stage == "prepare":
        result = prepare(config, output)
    else:
        if args.gpu is None:
            parser.error("--gpu is required; choose an allocated H200")
        result = run_stage(config, output, args.stage, args.gpu)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
