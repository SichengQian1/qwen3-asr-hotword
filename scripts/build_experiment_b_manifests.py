#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.experiment_b import build_experiment_b_manifests


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic clean-label Experiment B train/validation/test data."
    )
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--word-counts", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-hours", type=float, default=8.0)
    parser.add_argument("--validation-hours", type=float, default=1.0)
    parser.add_argument("--test-hours", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--minimum-word-frequency", type=int, default=100)
    parser.add_argument("--minimum-duration-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-duration-seconds", type=float, default=15.0)
    parser.add_argument("--ctc-safety-margin", type=int, default=2)
    parser.add_argument("--maximum-ctc-target-ratio", type=float, default=0.75)
    parser.add_argument("--candidate-pool-size", type=int, default=32_768)
    parser.add_argument("--review-count", type=int, default=20)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="Exclude every audio_path in this JSONL manifest; may be repeated.",
    )
    args = parser.parse_args()

    try:
        summary = build_experiment_b_manifests(
            args.tsv,
            args.audio_root,
            args.dictionary,
            args.word_counts,
            args.vocab,
            args.output_dir,
            train_hours=args.train_hours,
            validation_hours=args.validation_hours,
            test_hours=args.test_hours,
            seed=args.seed,
            language=args.language,
            minimum_word_frequency=args.minimum_word_frequency,
            minimum_duration_seconds=args.minimum_duration_seconds,
            maximum_duration_seconds=args.maximum_duration_seconds,
            ctc_safety_margin=args.ctc_safety_margin,
            maximum_ctc_target_ratio=args.maximum_ctc_target_ratio,
            candidate_pool_size=args.candidate_pool_size,
            review_count=args.review_count,
            exclusion_manifest_paths=tuple(args.exclude_manifest),
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(
            f"EXPERIMENT B MANIFEST BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
