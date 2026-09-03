#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.phonemes.manifest_dictionary import export_manifest_mfa_dictionary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a deterministic MFA-format dictionary from existing exact train and "
            "validation manifest labels without running G2P or reading a test split."
        )
    )
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--language", choices=("en", "es", "pt"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        summary = export_manifest_mfa_dictionary(
            args.manifest,
            args.output_dir,
            language=args.language,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
