#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.capacity_assets import parse_capacity_sizes
from qwen_hotword.hotwords.exact_capacity import benchmark_exact_hotword_capacity

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
DEFAULT_SIZES = (100, 500, 1_000, 2_000, 4_000)


def _sizes(value: str) -> tuple[int, ...]:
    try:
        return parse_capacity_sizes(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    if not profiles:
        raise argparse.ArgumentTypeError("profiles must not be empty")
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark full-current-sequence exact phoneme retrieval with an "
            "integer Aho-Corasick automaton."
        )
    )
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", type=_profiles, default=("representative",))
    parser.add_argument("--sizes", type=_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    try:
        report = benchmark_exact_hotword_capacity(
            assets_root=args.assets_root,
            replay_path=args.replay,
            vocab_path=args.vocab,
            output_dir=args.output_dir,
            profiles=args.profiles,
            sizes=args.sizes,
            warmup_queries=args.warmup_queries,
            deadline_seconds=args.deadline_ms / 1000.0,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
