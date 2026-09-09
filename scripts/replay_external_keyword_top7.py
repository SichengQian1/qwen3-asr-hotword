#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.external_keyword_topk_replay import (
    replay_external_keyword_top7,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly replay threshold 0.75 / posterior 0.5 / Top-7 from a completed "
            "external Anchor Top-5 run."
        )
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--keyword-bias", required=True)
    parser.add_argument("--keyword-set", default="hard_k266")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        report = replay_external_keyword_top7(
            source_run=args.source_run,
            vocab_path=args.vocab,
            keyword_bias_path=args.keyword_bias,
            output_dir=args.output_dir,
            keyword_set=args.keyword_set,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
