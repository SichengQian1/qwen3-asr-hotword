#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.mfa_audit import audit_mfa_dictionary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit an MFA G2P dictionary against its input words and CTC vocabulary."
    )
    parser.add_argument("--words", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--word-counts",
        help="Optional word/count TSV used to compute corpus-frequency-weighted impact.",
    )
    args = parser.parse_args()

    try:
        summary = audit_mfa_dictionary(
            args.words,
            args.dictionary,
            args.vocab,
            args.output_dir,
            word_counts_path=args.word_counts,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(f"MFA DICTIONARY AUDIT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
