#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.balanced_multilingual import (  # noqa: E402
    build_balanced_multilingual_validation,
    parse_language_pool,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a duration-balanced en/es/pt Temporal 2x validation manifest "
            "without reading sealed test data."
        )
    )
    parser.add_argument(
        "--language-pool",
        action="append",
        required=True,
        metavar="LANGUAGE=POOL_DIR",
    )
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--target-hours", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        summary = build_balanced_multilingual_validation(
            [parse_language_pool(value) for value in args.language_pool],
            args.training_root,
            args.output_dir,
            target_hours=args.target_hours,
            seed=args.seed,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
