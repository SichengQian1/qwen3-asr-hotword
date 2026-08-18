#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.capacity_assets import (
    DEFAULT_SIZES,
    build_hotword_capacity_assets,
    parse_capacity_sizes,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def _sizes(value: str) -> tuple[int, ...]:
    try:
        return parse_capacity_sizes(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic nested 100-to-10k Portuguese hotword capacity "
            "assets from train-only real 1-4 word n-grams."
        )
    )
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--base-hotwords", required=True)
    parser.add_argument("--base-cases", required=True)
    parser.add_argument(
        "--selection",
        help="Optional existing formal100 sample_selection.json used to keep identical cases.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--sizes",
        type=_sizes,
        default=DEFAULT_SIZES,
        help="Strictly increasing comma-separated sizes starting at 100.",
    )
    parser.add_argument("--seed", type=int, default=20_260_818)
    parser.add_argument("--candidate-pool-multiplier", type=int, default=3)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    try:
        summary = build_hotword_capacity_assets(
            training_manifest_path=args.training_manifest,
            dictionary_path=args.dictionary,
            vocab_path=args.vocab,
            base_hotwords_path=args.base_hotwords,
            base_cases_path=args.base_cases,
            selection_path=args.selection,
            output_dir=args.output_dir,
            sizes=args.sizes,
            seed=args.seed,
            candidate_pool_multiplier=args.candidate_pool_multiplier,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
