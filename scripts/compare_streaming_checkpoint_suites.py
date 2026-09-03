#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.checkpoint_regression import (
    compare_streaming_checkpoint_suites,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and compare two Portuguese formal100 suites whose intended "
            "treatment is the CTC checkpoint only."
        )
    )
    parser.add_argument("--baseline-suite", required=True)
    parser.add_argument("--candidate-suite", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        report = compare_streaming_checkpoint_suites(
            args.baseline_suite,
            args.candidate_suite,
            args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
