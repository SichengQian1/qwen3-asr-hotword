#!/usr/bin/env python3
"""Exclude reviewed conflicts, refill original train strata, verify and release a new version."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.multilingual_480_repair import repair_training


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/480h_repair.workzone.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = (
        args.output_dir
        or REPO / "outputs" / f"multilingual_480h_training_v2_{uuid.uuid4().hex[:10]}"
    )
    print(f"New repaired training output: {output}", flush=True)
    try:
        report = repair_training(json.loads(args.config.read_text()), output, workers=args.workers)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Repair stopped; previous outputs preserved: {error}\n")
    compact = dict(report)
    compact["replacements"] = {
        lang: {k: v for k, v in data.items() if k != "strata"}
        for lang, data in report["replacements"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
