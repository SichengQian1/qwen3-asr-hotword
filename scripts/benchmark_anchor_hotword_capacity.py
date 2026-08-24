#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.anchor_capacity import (
    GC_POLICIES,
    benchmark_anchor_hotword_capacity,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def _integers(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must contain comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def _profiles(value: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in value.split(",") if item.strip())
    if not profiles:
        raise argparse.ArgumentTypeError("profiles must not be empty")
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AC-union positional 2/3/4-gram Anchor shortlists against "
            "the sealed full-scan Portuguese hotword reference."
        )
    )
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiles", type=_profiles, default=("representative",))
    parser.add_argument("--sizes", type=_integers, default=(4_000,))
    parser.add_argument("--shortlist-sizes", type=_integers, default=(64, 128, 256))
    parser.add_argument("--ngram-sizes", type=_integers, default=(2, 3, 4))
    parser.add_argument("--anchors-per-entry", type=int, default=24)
    parser.add_argument("--offset-tolerance", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-phonemes", type=int, default=4)
    parser.add_argument("--maximum-edit-ratio", type=float, default=0.35)
    parser.add_argument("--posterior-weight", type=float, default=0.25)
    parser.add_argument("--minimum-posterior-confidence", type=float, default=0.0)
    parser.add_argument("--minimum-top1-margin", type=float, default=0.0)
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    parser.add_argument("--gc-policy", choices=GC_POLICIES, default="normal")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    try:
        report = benchmark_anchor_hotword_capacity(
            assets_root=args.assets_root,
            replay_path=args.replay,
            vocab_path=args.vocab,
            output_dir=args.output_dir,
            profiles=args.profiles,
            sizes=args.sizes,
            shortlist_sizes=args.shortlist_sizes,
            ngram_sizes=args.ngram_sizes,
            anchors_per_entry=args.anchors_per_entry,
            offset_tolerance=args.offset_tolerance,
            threshold=args.threshold,
            top_k=args.top_k,
            minimum_phonemes=args.minimum_phonemes,
            maximum_edit_ratio=args.maximum_edit_ratio,
            posterior_weight=args.posterior_weight,
            minimum_posterior_confidence=args.minimum_posterior_confidence,
            minimum_top1_margin=args.minimum_top1_margin,
            warmup_queries=args.warmup_queries,
            deadline_seconds=args.deadline_ms / 1000.0,
            gc_policy=args.gc_policy,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
