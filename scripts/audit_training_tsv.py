#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.data_audit import audit_training_tsv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a training TSV and resolve relative audio paths."
    )
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--max-records",
        type=int,
        default=1000,
        help="Rows to inspect; 0 scans the complete TSV (default: 1000).",
    )
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    try:
        audit = audit_training_tsv(
            args.tsv,
            args.audio_root,
            audio_column=args.audio_column,
            text_column=args.text_column,
            max_records=args.max_records,
            sample_count=args.sample_count,
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        print(f"TRAINING TSV AUDIT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    report = audit.to_dict()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if audit.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
