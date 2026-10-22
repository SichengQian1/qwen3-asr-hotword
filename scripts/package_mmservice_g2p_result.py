#!/usr/bin/env python3
"""Shrink a completed G2P audit into differences and bounded text attachments."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.phonemes.mmservice_compact import package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--part-kib", type=int, default=32)
    args = parser.parse_args()
    source = args.source_run.resolve()
    output = args.output_dir or source.parent / ("mmservice_g2p_compact_" + uuid.uuid4().hex[:10])
    try:
        result = package(source, output, part_kib=args.part_kib)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        result = {"status": "failed", "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
