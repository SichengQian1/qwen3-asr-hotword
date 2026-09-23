#!/usr/bin/env python3
"""Build both fixed 4000-word wave tables on mounted H200 data, using CPU only."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.wave_keywords import freeze_wave_keywords


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/wave_4000_keywords.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir or args.root / "outputs" / f"wave_4000_keywords_v1_{uuid.uuid4().hex[:10]}"
    )
    try:
        report = freeze_wave_keywords(args.root.resolve(), args.config, output.resolve())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps({k: v for k, v in report.items() if k != "inputs"}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
