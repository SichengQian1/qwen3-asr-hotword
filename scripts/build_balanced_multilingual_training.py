#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.balanced_multilingual import (  # noqa: E402
    build_balanced_multilingual_training,
    parse_language_pool,
    parse_named_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a duration-balanced en/es/pt Temporal 2x train manifest "
            "without reading sealed test data."
        )
    )
    parser.add_argument(
        "--language-pool",
        action="append",
        required=True,
        metavar="LANGUAGE=POOL_DIR",
    )
    parser.add_argument(
        "--include-all-source",
        action="append",
        default=[],
        metavar="LANGUAGE=SOURCE_CORPUS",
    )
    parser.add_argument("--target-hours", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    mandatory: defaultdict[str, list[str]] = defaultdict(list)
    try:
        pools = [parse_language_pool(value) for value in args.language_pool]
        for value in args.include_all_source:
            language, source = parse_named_source(value)
            mandatory[language].append(source)
        summary = build_balanced_multilingual_training(
            pools,
            args.output_dir,
            target_hours=args.target_hours,
            seed=args.seed,
            include_all_sources=mandatory,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
