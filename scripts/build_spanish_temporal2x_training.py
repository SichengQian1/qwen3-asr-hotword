#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_combined import (  # noqa: E402
    build_spanish_temporal2x_training,
    combine_spanish_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Spanish Temporal 2x pool while preserving explicit source "
            "splits and assigning unsplit corpora by complete speaker."
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=MANIFEST_DIR",
    )
    parser.add_argument(
        "--source-tsv",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.96)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--time-upsampling-factor", type=int, default=2)
    parser.add_argument("--release-max-effective-ratio", type=float, default=0.90)
    parser.add_argument("--speaker-split-seed", type=int, default=20_260_824)
    args = parser.parse_args()
    try:
        corpora = combine_spanish_inputs(args.corpus, args.source_tsv)
        summary = build_spanish_temporal2x_training(
            corpora,
            args.output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            time_upsampling_factor=args.time_upsampling_factor,
            release_max_effective_ratio=args.release_max_effective_ratio,
            speaker_split_seed=args.speaker_split_seed,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
