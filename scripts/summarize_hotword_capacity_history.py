#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.capacity_history import build_hotword_capacity_history


def _stage(value: str) -> tuple[str, str]:
    label, separator, directory = value.partition("=")
    if not separator or not label.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("stage must use LABEL=OUTPUT_DIR")
    return label.strip(), directory.strip()


def _profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    if not profiles:
        raise argparse.ArgumentTypeError("profiles must not be empty")
    return profiles


def _sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must contain comma-separated integers") from error
    if not sizes:
        raise argparse.ArgumentTypeError("sizes must not be empty")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize full-scan, exact-AC, and Anchor capacity outputs into a "
            "step-by-step Top-5/7/10 quality and latency history."
        )
    )
    parser.add_argument("--stage", action="append", type=_stage, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", type=_profiles, default=("representative",))
    parser.add_argument("--sizes", type=_sizes)
    args = parser.parse_args()
    try:
        report = build_hotword_capacity_history(
            stages=args.stage,
            output_dir=args.output_dir,
            profiles=args.profiles,
            sizes=args.sizes,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
