#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_sources import (
    convert_common_voice_rioplatense_to_tsv,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Common Voice Rioplatense train/dev/test TSVs to one canonical TSV "
            "while preserving official splits, client IDs, accents, and votes."
        )
    )
    parser.add_argument(
        "--corpus-root",
        required=True,
        help="Extracted es-Rioplatense directory containing clips/ and split TSVs.",
    )
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument(
        "--scan-audio-inventory",
        action="store_true",
        help="Inventory all MP3 files below clips/ and report unreferenced files.",
    )
    args = parser.parse_args()

    try:
        summary = convert_common_voice_rioplatense_to_tsv(
            args.corpus_root,
            args.output_tsv,
            check_audio=args.check_audio,
            scan_audio_inventory=args.scan_audio_inventory,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(
            f"COMMON VOICE RIOPLATENSE CONVERSION FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
