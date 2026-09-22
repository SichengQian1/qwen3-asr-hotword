#!/usr/bin/env python3
"""Measure all new Noah audio, deduplicate identical files, and prepare a G2P word list."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.noah_source_preparation import prepare_noah_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/noah_es_mobile_source.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = (
        args.output_dir or REPO / "outputs" / f"es_noah_mobile_source_v1_{uuid.uuid4().hex[:10]}"
    )
    try:
        report = prepare_noah_source(
            json.loads(args.config.read_text()), output, workers=args.workers
        )
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        parser.exit(1, f"Preparation stopped; partial outputs preserved: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
