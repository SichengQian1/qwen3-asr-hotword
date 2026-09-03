#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.v3_operating_calibration import (
    calibrate_v3_operating_points,
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


def _floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must contain comma-separated numbers") from error
    if not values:
        raise argparse.ArgumentTypeError("value must not be empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate Portuguese v3 gates from saved case scores without CTC or Qwen "
            "inference; recommend only provably exact Top-5 replays."
        )
    )
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--case-scores", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--reference-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-ks", type=_integers, default=(1, 3, 5))
    parser.add_argument(
        "--thresholds",
        type=_floats,
        default=(
            0.70,
            0.71,
            0.72,
            0.73,
            0.74,
            0.75,
            0.76,
            0.77,
            0.78,
            0.79,
            0.80,
            0.81,
            0.82,
            0.83,
            0.84,
            0.85,
            0.86,
            0.87,
            0.88,
            0.89,
            0.90,
        ),
    )
    parser.add_argument(
        "--minimum-posterior-confidences",
        type=_floats,
        default=(0.0, 0.25, 0.50, 0.75),
    )
    args = parser.parse_args()
    try:
        summary = calibrate_v3_operating_points(
            vocab_path=args.vocab,
            hotword_path=args.hotwords,
            families_path=args.families,
            cases_path=args.cases,
            case_scores_path=args.case_scores,
            candidate_report_path=args.candidate_report,
            reference_report_path=args.reference_report,
            output_dir=args.output_dir,
            top_ks=args.top_ks,
            thresholds=args.thresholds,
            minimum_posterior_confidences=args.minimum_posterior_confidences,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
