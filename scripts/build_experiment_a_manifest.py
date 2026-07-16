#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.experiment_a import build_experiment_a_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic 128-sample clean-label CTC overfit manifest."
    )
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--minimum-duration-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-duration-seconds", type=float, default=15.0)
    parser.add_argument("--ctc-safety-margin", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=4096)
    parser.add_argument("--review-count", type=int, default=20)
    args = parser.parse_args()

    try:
        summary = build_experiment_a_manifest(
            args.tsv,
            args.audio_root,
            args.dictionary,
            args.vocab,
            args.output_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            language=args.language,
            minimum_duration_seconds=args.minimum_duration_seconds,
            maximum_duration_seconds=args.maximum_duration_seconds,
            ctc_safety_margin=args.ctc_safety_margin,
            candidate_pool_size=args.candidate_pool_size,
            review_count=args.review_count,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(
            f"EXPERIMENT A MANIFEST BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
