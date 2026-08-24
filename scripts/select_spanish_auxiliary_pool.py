#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_selection import select_spanish_auxiliary_pool


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic, speaker-disjoint explicit Latin-American Spanish "
            "auxiliary pool from the audited Common Voice inventory."
        )
    )
    parser.add_argument("--source-tsv", required=True)
    parser.add_argument("--inventory-tsv", required=True)
    parser.add_argument("--rioplatense-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-hours", type=float, default=170.0)
    parser.add_argument("--train-fraction", type=float, default=0.96)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument(
        "--maximum-latin-american-speaker-hours",
        type=float,
        default=2.0,
    )
    parser.add_argument("--seed", type=int, default=20_260_824)
    args = parser.parse_args()

    try:
        summary = select_spanish_auxiliary_pool(
            args.source_tsv,
            args.inventory_tsv,
            args.rioplatense_tsv,
            args.output_dir,
            target_hours=args.target_hours,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            maximum_latin_american_speaker_hours=(
                args.maximum_latin_american_speaker_hours
            ),
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
