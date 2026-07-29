#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.hotword_prompt import DEFAULT_PT_BR_PROMPT_TEMPLATE
from qwen_hotword.inference.prompt_smoke import (
    DEFAULT_SELECTION_SEED,
    run_prompt_smoke,
)
from qwen_hotword.modeling.qwen_backbone import ModelValidationError

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only Qwen3-ASR-1.7B hotword prompt smoke test: "
            "baseline, oracle prompt, and negative prompt control."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-count", type=int, default=30)
    parser.add_argument("--negative-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--language", default="Portuguese")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_PT_BR_PROMPT_TEMPLATE,
        help="Fixed template containing exactly one {hotwords} placeholder.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable per-inference progress lines.",
    )
    args = parser.parse_args()

    try:
        report = run_prompt_smoke(
            model_path=args.model,
            validation_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            hotword_table_path=args.hotwords,
            cases_path=args.cases,
            output_dir=args.output_dir,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            seed=args.seed,
            prompt_template=args.prompt_template,
            language=args.language,
            dtype=args.dtype,
            device=args.device,
            print_progress=not args.quiet_progress,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        ModelValidationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
