#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_sources import convert_slr61_argentinian_to_tsv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the headerless SLR61 Argentinian Spanish indices to a canonical "
            "metadata-preserving TSV. Peninsular es-es weather audio is excluded."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Directory containing downloads/ and extracted/.",
    )
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument(
        "--scan-audio-inventory",
        action="store_true",
        help="Inventory all extracted WAV files and classify unindexed es-es weather audio.",
    )
    args = parser.parse_args()

    try:
        summary = convert_slr61_argentinian_to_tsv(
            args.source_root,
            args.output_tsv,
            check_audio=args.check_audio,
            scan_audio_inventory=args.scan_audio_inventory,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(f"SLR61 CONVERSION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
