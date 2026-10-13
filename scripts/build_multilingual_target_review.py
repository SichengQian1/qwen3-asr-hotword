#!/usr/bin/env python3
"""Merge declared EN/ES/PT targets and remove explicit basic words for human review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.target_review import build_target_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/multilingual_target_review.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        report = build_target_review(args.root.resolve(), args.config.resolve(), args.output_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
