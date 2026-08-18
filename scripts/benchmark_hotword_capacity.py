#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.capacity_assets import DEFAULT_SIZES, parse_capacity_sizes
from qwen_hotword.hotwords.capacity_benchmark import benchmark_hotword_capacity

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


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
            "Replay immutable CTC phoneme sequences against nested Portuguese "
            "100-to-10k hotword libraries and measure ranking quality, latency, and memory."
        )
    )
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profiles",
        type=_profiles,
        default=("representative", "hard_negative"),
    )
    parser.add_argument("--sizes", type=_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-phonemes", type=int, default=4)
    parser.add_argument("--maximum-edit-ratio", type=float, default=0.35)
    parser.add_argument("--posterior-weight", type=float, default=0.25)
    parser.add_argument("--minimum-posterior-confidence", type=float, default=0.0)
    parser.add_argument("--minimum-top1-margin", type=float, default=0.0)
    parser.add_argument("--warmup-queries", type=int, default=1)
    parser.add_argument("--stop-retrieval-p95-seconds", type=float, default=2.0)
    parser.add_argument("--continue-after-deadline-failure", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    try:
        report = benchmark_hotword_capacity(
            assets_root=args.assets_root,
            replay_path=args.replay,
            vocab_path=args.vocab,
            output_dir=args.output_dir,
            profiles=args.profiles,
            sizes=args.sizes,
            threshold=args.threshold,
            top_k=args.top_k,
            minimum_phonemes=args.minimum_phonemes,
            maximum_edit_ratio=args.maximum_edit_ratio,
            posterior_weight=args.posterior_weight,
            minimum_posterior_confidence=args.minimum_posterior_confidence,
            minimum_top1_margin=args.minimum_top1_margin,
            warmup_queries=args.warmup_queries,
            stop_retrieval_p95_seconds=args.stop_retrieval_p95_seconds,
            continue_after_deadline_failure=args.continue_after_deadline_failure,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
