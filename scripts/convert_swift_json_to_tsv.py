#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.swift_json import convert_swift_json_to_tsv


def _audio_prefix_rewrite(value: str) -> tuple[str, str]:
    old_prefix, separator, new_prefix = value.partition("=")
    if not separator or not old_prefix.strip() or not new_prefix.strip():
        raise argparse.ArgumentTypeError("expected OLD=NEW with two non-empty prefixes")
    return old_prefix.strip(), new_prefix.strip()


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
        "--audio-prefix-rewrite",
        action="append",
        default=[],
        type=_audio_prefix_rewrite,
        metavar="OLD=NEW",
        help=(
            "Rewrite a leading audio path prefix before checking or writing it; "
            "repeat for multiple mappings."
        ),
    )
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Also check whether referenced audio files exist.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print conversion progress every N source records; 0 disables it.",
    )
    args = parser.parse_args()

    try:
        summary = convert_swift_json_to_tsv(
            args.input,
            args.output_tsv,
            expected_language=args.expected_language,
            check_audio=args.check_audio,
            audio_prefix_rewrites=args.audio_prefix_rewrite,
            progress_every_records=args.progress_every,
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
