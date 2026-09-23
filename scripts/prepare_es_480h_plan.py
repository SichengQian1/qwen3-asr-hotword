#!/usr/bin/env python3
"""Prepare a provisional 480h ID plan; never create a trainable manifest."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.spanish_480_plan import prepare_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/es_480h_plan.workzone.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or REPO / "outputs" / f"es_480h_plan_v1_{uuid.uuid4().hex[:10]}"
    print(f"Provisional plan output: {output}", flush=True)
    try:
        result = prepare_plan(json.loads(args.config.read_text()), output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Plan stopped; existing outputs preserved: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
