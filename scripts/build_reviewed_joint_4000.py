#!/usr/bin/env python3
"""Expand the reviewed joint vocabulary using transcript-absent EN/ES/PT train words."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.joint_fillers import build_joint_fillers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/reviewed_joint_4000_keywords.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        cfg = json.loads(args.config.read_text())
        report = build_joint_fillers(
            args.root.resolve(),
            args.config.resolve(),
            (args.output_dir or args.root / cfg["output"]).resolve(),
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
