#!/usr/bin/env python3
"""Audit all supplied Egyptian pools and report measured audio-header hours."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.training.egyptian_source_audit import audit, read
from scripts.label_arabic_native import exclusive_job


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/egyptian_sources_audit.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        config = read(args.config)
        output = (args.output_dir or ROOT / config["output_dir"]).resolve()
        with exclusive_job(output):
            result = audit(ROOT, args.config.resolve(), output, args.workers, args.resume)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0 if result["status"] == "checked" else 2
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
