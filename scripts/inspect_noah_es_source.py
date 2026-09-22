#!/usr/bin/env python3
"""Stream the new source JSON and probe a small deterministic audio sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.noah_source_inspection import inspect_noah_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/noah_es_mobile_source.workzone.json"
    )
    args = parser.parse_args()
    try:
        report = inspect_noah_source(json.loads(args.config.read_text()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"Source inspection stopped; files unchanged: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
