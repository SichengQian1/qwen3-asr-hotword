#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.simulation import build_simulated_hotword_assets

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic simulated hotword table and validation-only cases. "
            "The sealed test split is intentionally not accepted."
        )
    )
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hotword-count", type=int, default=50)
    parser.add_argument("--case-count", type=int, default=200)
    parser.add_argument("--active-hotwords-per-case", type=int, default=10)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--max-words", type=int, default=2)
    parser.add_argument("--min-phonemes", type=int, default=4)
    parser.add_argument("--max-phonemes", type=int, default=24)
    parser.add_argument("--max-occurrences", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_722)
    args = parser.parse_args()

    try:
        summary = build_simulated_hotword_assets(
            args.validation_manifest,
            args.dictionary,
            args.vocab,
            args.output_dir,
            hotword_count=args.hotword_count,
            case_count=args.case_count,
            active_hotwords_per_case=args.active_hotwords_per_case,
            positive_ratio=args.positive_ratio,
            min_words=args.min_words,
            max_words=args.max_words,
            min_phonemes=args.min_phonemes,
            max_phonemes=args.max_phonemes,
            max_occurrences=args.max_occurrences,
            seed=args.seed,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(
            f"SIMULATED HOTWORD BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
