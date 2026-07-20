#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.full_training import build_full_training_splits


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic train/validation/test manifests from the full-corpus "
            "training-ready manifest and seal the held-out test split."
        )
    )
    parser.add_argument("--ready-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.96)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        summary = build_full_training_splits(
            args.ready_manifest,
            args.output_dir,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(
            f"FULL TRAINING SPLIT BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
