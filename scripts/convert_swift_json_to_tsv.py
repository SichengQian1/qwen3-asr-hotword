#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.swift_json import convert_swift_json_to_tsv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ms-swift ASR JSON dataset into the canonical audio/text TSV "
            "used by the MFA G2P and CTC manifest builders."
        )
    )
    parser.add_argument("--input", required=True, help="Path to a Swift JSON list file.")
    parser.add_argument("--output-tsv", required=True, help="Destination audio/text TSV.")
    parser.add_argument("--expected-language", default=None)
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Also check whether referenced audio files exist.",
    )
    args = parser.parse_args()

    try:
        summary = convert_swift_json_to_tsv(
            args.input,
            args.output_tsv,
            expected_language=args.expected_language,
            check_audio=args.check_audio,
        )
    except (FileNotFoundError, OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(
            f"SWIFT JSON CONVERSION FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.written_records > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
