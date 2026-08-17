#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.streaming_boundary import build_streaming_boundary_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build reproducible streaming boundary cases from forced-aligned or "
            "manually confirmed hotword timestamps. Audio is not rewritten."
        )
    )
    parser.add_argument("--source-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--minimum-hotword-start-sec", type=float, default=4.0)
    parser.add_argument(
        "--allow-incomplete-coverage",
        action="store_true",
        help="Write a warn summary instead of failing when a required category is absent.",
    )
    args = parser.parse_args()
    try:
        summary = build_streaming_boundary_manifest(
            args.source_spec,
            args.output_dir,
            chunk_size_sec=args.chunk_size_sec,
            minimum_hotword_start_sec=args.minimum_hotword_start_sec,
            require_complete_coverage=not args.allow_incomplete_coverage,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
