#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.combined_training import (
    build_temporal2x_combined_training,
    parse_combined_corpus_spec,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a new 96/2/2 combined Temporal 2x dataset from each corpus's "
            "original ready rows plus pure temporal recovery rows at or below the "
            "configured effective-ratio limit. Source manifests are read-only."
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=MANIFEST_DIR",
        help="Repeat once per corpus, in the desired reporting order.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.96)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--time-upsampling-factor", type=int, default=2)
    parser.add_argument("--release-max-effective-ratio", type=float, default=0.90)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    try:
        corpora = [parse_combined_corpus_spec(value) for value in args.corpus]
        summary = build_temporal2x_combined_training(
            corpora,
            args.output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            time_upsampling_factor=args.time_upsampling_factor,
            release_max_effective_ratio=args.release_max_effective_ratio,
            progress_every=args.progress_every,
            print_progress=not args.quiet_progress,
        )
    except (FileNotFoundError, OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(
            f"COMBINED TRAINING MANIFEST BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
