#!/usr/bin/env python3
"""Freeze the reviewed EN/PT choices plus all frozen ES rows after audio identity checks."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.multilingual_480_freeze import freeze_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/multilingual_480h_freeze.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = (
        args.output_dir
        or REPO / "outputs" / f"multilingual_480h_training_v1_{uuid.uuid4().hex[:10]}"
    )
    print(f"New freeze output: {output}", flush=True)
    try:
        report = freeze_plan(json.loads(args.config.read_text()), output, workers=args.workers)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Freeze stopped; existing outputs preserved: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["training_manifest_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
