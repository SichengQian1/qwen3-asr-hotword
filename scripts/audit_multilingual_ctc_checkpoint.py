#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.multilingual_checkpoint_audit import (
    audit_multilingual_ctc_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and summarize a completed en/es/pt Macro-PER CTC checkpoint "
            "without loading Qwen or reading the sealed test set."
        )
    )
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-groups", default="en,es,pt")
    args = parser.parse_args()
    groups = tuple(value.strip() for value in args.expected_groups.split(",") if value.strip())
    try:
        report = audit_multilingual_ctc_checkpoint(
            args.training_dir,
            args.output_dir,
            expected_groups=groups,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
