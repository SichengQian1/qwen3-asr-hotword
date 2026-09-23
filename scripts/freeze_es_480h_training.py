#!/usr/bin/env python3
"""Hash selected/heldout audio and freeze the Spanish train manifest if conflict-free."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.spanish_480_freeze import freeze_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-dir", type=Path, default=REPO / "outputs/es_480h_plan_v1_7022ece6f0"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output_dir or REPO / "outputs" / f"es_480h_training_v1_{uuid.uuid4().hex[:10]}"
    print(f"New freeze output: {output}", flush=True)
    try:
        result = freeze_plan(args.plan_dir, output, workers=args.workers)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Freeze stopped; outputs preserved: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
