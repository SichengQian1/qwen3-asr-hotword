#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.english_speaker_inventory import (  # noqa: E402
    audit_swift_english_speakers,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit speaker prefixes inferred from Swift US-English WAV basenames "
            "and join them to an existing full manifest."
        )
    )
    parser.add_argument("--source-tsv", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        summary = audit_swift_english_speakers(
            args.source_tsv,
            args.manifest_dir,
            args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
