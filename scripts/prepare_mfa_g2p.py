#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.g2p_prep import prepare_mfa_wordlist


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a normalized unique-word list for MFA G2P."
    )
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional row cap; 0 processes the complete TSV.",
    )
    parser.add_argument(
        "--minimum-word-count",
        type=int,
        default=1,
        help="Minimum corpus count for inclusion in words.txt (default: 1).",
    )
    args = parser.parse_args()

    try:
        summary = prepare_mfa_wordlist(
            args.tsv,
            args.output_dir,
            text_column=args.text_column,
            max_records=args.max_records,
            minimum_word_count=args.minimum_word_count,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(f"MFA G2P PREPARATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    report_path = Path(args.output_dir).expanduser() / "summary.json"
    rendered = json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
