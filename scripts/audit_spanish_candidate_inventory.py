#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_inventory import audit_spanish_candidate_inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit MLS/Common Voice Spanish candidate hours, join original Common Voice "
            "accent/split metadata, and detect overlap with the Rioplatense core corpus."
        )
    )
    parser.add_argument("--mls-tsv", required=True)
    parser.add_argument("--common-voice-tsv", required=True)
    parser.add_argument(
        "--common-voice-root",
        required=True,
        help="Original Common Voice es directory containing validated/train/dev/test TSVs.",
    )
    parser.add_argument("--rioplatense-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    try:
        summary = audit_spanish_candidate_inventory(
            args.mls_tsv,
            args.common_voice_tsv,
            args.common_voice_root,
            args.rioplatense_tsv,
            args.output_dir,
            workers=args.workers,
            progress_every=args.progress_every,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"pass", "warn"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
