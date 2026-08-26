#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.operating_sweep import (
    SELECTION_SCOPES,
    sweep_operating_points,
)


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
            "Replay threshold/edit/posterior/margin gates over a complete saved Anchor "
            "shortlist and select a recall-first operating point."
        )
    )
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="representative")
    parser.add_argument("--size", type=int, default=4_000)
    parser.add_argument("--window", default="full_current")
    parser.add_argument("--shortlist-size", type=int, default=64)
    parser.add_argument("--top-ks", type=_integers, default=(5, 7, 10))
    parser.add_argument(
        "--thresholds",
        type=_floats,
        default=(0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90),
    )
    parser.add_argument(
        "--maximum-edit-ratios",
        type=_floats,
        default=(0.35, 0.40, 0.45, 0.50, 0.60, 1.0),
    )
    parser.add_argument(
        "--minimum-posterior-confidences",
        type=_floats,
        default=(0.0, 0.25, 0.50, 0.75),
    )
    parser.add_argument(
        "--minimum-top1-margins",
        type=_floats,
        default=(0.0, 0.01, 0.02, 0.05),
    )
    parser.add_argument("--selection-scope", choices=SELECTION_SCOPES, default="final")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--diagnostic-precision-target", type=float, default=0.85)
    parser.add_argument("--deadline-ms", type=float, default=50.0)
    args = parser.parse_args()
    try:
        report = sweep_operating_points(
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            profile=args.profile,
            size=args.size,
            window=args.window,
            shortlist_size=args.shortlist_size,
            top_ks=args.top_ks,
            thresholds=args.thresholds,
            maximum_edit_ratios=args.maximum_edit_ratios,
            minimum_posterior_confidences=args.minimum_posterior_confidences,
            minimum_top1_margins=args.minimum_top1_margins,
            selection_scope=args.selection_scope,
            target_recall=args.target_recall,
            diagnostic_precision_target=args.diagnostic_precision_target,
            deadline_seconds=args.deadline_ms / 1000.0,
        )
    except (FileExistsError, FileNotFoundError, KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
