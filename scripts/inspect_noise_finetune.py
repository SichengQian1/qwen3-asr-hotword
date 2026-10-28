#!/usr/bin/env python3
"""Summarize noise-run progress and locate a saved-cache mismatch; CPU only."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/multilingual_noise10db_finetune_v1"
    )
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--check-replacements", action="store_true")
    parser.add_argument("--ratio", type=int, choices=(30, 50), default=30)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=16,
        help="inspect up to N shards in next-epoch training order; 0 means all",
    )
    args = parser.parse_args()
    if args.max_shards < 0:
        parser.error("--max-shards must be nonnegative")
    from qwen_hotword.training.noise_status import inspect, terminal_summary, write_diagnostic

    try:
        report = inspect(
            args.run_dir.resolve(),
            check_replacements=args.check_replacements,
            ratio=args.ratio,
            max_shards=args.max_shards,
        )
        destination = (
            args.report_dir or ROOT / "outputs" / f"noise_finetune_status_{uuid.uuid4().hex[:10]}"
        ).resolve()
        result = write_diagnostic(report, destination)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(json.dumps({"status": "inspection_failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(terminal_summary(result), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
