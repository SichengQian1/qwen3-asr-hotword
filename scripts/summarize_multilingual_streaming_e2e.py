#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.multilingual_e2e_summary import (
    summarize_multilingual_streaming_e2e,
)


def _language_run(value: str) -> tuple[str, str]:
    language, separator, path = value.partition("=")
    if not separator or language not in {"en", "es", "pt"} or not path:
        raise argparse.ArgumentTypeError("language run must use en=PATH, es=PATH, or pt=PATH")
    return language, path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and summarize English/Spanish/Portuguese 4k C/D/E streaming runs."
    )
    parser.add_argument(
        "--language-run",
        action="append",
        required=True,
        type=_language_run,
        metavar="LANG=PATH",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    runs = dict(args.language_run)
    if len(runs) != len(args.language_run):
        parser.error("duplicate --language-run language")
    try:
        report = summarize_multilingual_streaming_e2e(runs, args.output_dir)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
