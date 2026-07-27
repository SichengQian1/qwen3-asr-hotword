#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.simulation import (
    HotwordLengthBucket,
    build_stratified_hotword_assets,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate validation-only v2 table of 100 stratified simulated "
            "hotwords. Existing output files are never overwritten."
        )
    )
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--exclude-hotwords")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--short-count", type=int, default=30)
    parser.add_argument("--medium-count", type=int, default=40)
    parser.add_argument("--long-count", type=int, default=20)
    parser.add_argument("--very-long-count", type=int, default=10)
    parser.add_argument("--case-count", type=int, default=500)
    parser.add_argument("--active-hotwords-per-case", type=int, default=100)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--max-occurrences", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20_260_727)
    args = parser.parse_args()

    buckets = (
        HotwordLengthBucket("phonemes_4_7", 4, 7, args.short_count),
        HotwordLengthBucket("phonemes_8_12", 8, 12, args.medium_count),
        HotwordLengthBucket("phonemes_13_18", 13, 18, args.long_count),
        HotwordLengthBucket("phonemes_19_24", 19, 24, args.very_long_count),
    )
    try:
        summary = build_stratified_hotword_assets(
            args.validation_manifest,
            args.dictionary,
            args.vocab,
            args.output_dir,
            length_buckets=buckets,
            exclude_hotword_table_path=args.exclude_hotwords,
            case_count=args.case_count,
            active_hotwords_per_case=args.active_hotwords_per_case,
            positive_ratio=args.positive_ratio,
            max_occurrences=args.max_occurrences,
            seed=args.seed,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"STRATIFIED HOTWORD BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
