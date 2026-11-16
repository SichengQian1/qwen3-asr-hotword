#!/usr/bin/env python3
"""Inspect saved full-data durations/exclusions; no G2P, Encoder or training rerun."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.training.preparation_inspection import save_inspection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-run", type=Path, default=ROOT / "outputs/zh_en_183_noise70_scratch_v1"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--headers-per-group", type=int, default=3)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "outputs" / (
        "zh_en_preparation_inspection_" + uuid.uuid4().hex[:10]
    )
    try:
        result = save_inspection(args.source_run, output, headers_per_group=args.headers_per_group)
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
