#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.external_keyword_retrieval import (
    parse_source_spec,
    run_external_keyword_retrieval,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete-audio Portuguese CTC Anchor retrieval over external FLAC/WAV sets."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--keyword-bias", required=True)
    parser.add_argument("--keyword-set", default="hard_k266")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=AUDIO_DIR,TRANSCRIPTS",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    try:
        sources = tuple(parse_source_spec(value) for value in args.source)
        report = run_external_keyword_retrieval(
            model_path=args.model,
            checkpoint_path=args.ctc_checkpoint,
            vocab_path=args.vocab,
            keyword_bias_path=args.keyword_bias,
            sources=sources,
            output_dir=args.output_dir,
            keyword_set=args.keyword_set,
            device=args.device,
            dtype=args.dtype,
            audit_only=args.audit_only,
            resume=args.resume,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
