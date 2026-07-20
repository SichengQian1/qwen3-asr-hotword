#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.full_manifest import build_full_training_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable full-corpus CTC manifest. Every TSV row is retained in "
            "either train_ready.jsonl or needs_review.jsonl."
        )
    )
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="pt-BR")
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--shard-size", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    try:
        summary = build_full_training_manifest(
            args.tsv,
            args.audio_root,
            args.dictionary,
            args.vocab,
            args.output_dir,
            language=args.language,
            audio_column=args.audio_column,
            text_column=args.text_column,
            shard_size=args.shard_size,
            workers=args.workers,
            resume=not args.no_resume,
        )
    except (FileNotFoundError, OSError, RuntimeError, UnicodeError, ValueError) as error:
        print(
            f"FULL MANIFEST BUILD FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
