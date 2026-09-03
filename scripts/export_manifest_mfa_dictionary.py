#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.phonemes.manifest_dictionary import (
    export_manifest_mfa_dictionary,
    export_source_mfa_dictionary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a deterministic MFA-format dictionary from existing exact per-word "
            "labels or full-manifest source dictionaries without running G2P or reading "
            "a test split."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        action="append",
        help="Manifest retaining exact per-word pronunciations; may be repeated.",
    )
    source.add_argument(
        "--build-config",
        action="append",
        help=(
            "Full-manifest build_config.json whose dictionary.path should be reused; "
            "may be repeated."
        ),
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        default=[],
        help="Additional existing MFA dictionary used with --build-config.",
    )
    parser.add_argument("--language", choices=("en", "es", "pt"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        if args.manifest:
            if args.dictionary:
                parser.error("--dictionary can only be used with --build-config")
            summary = export_manifest_mfa_dictionary(
                args.manifest,
                args.output_dir,
                language=args.language,
            )
        else:
            summary = export_source_mfa_dictionary(
                args.dictionary,
                args.build_config,
                args.output_dir,
                language=args.language,
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
