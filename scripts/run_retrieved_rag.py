#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.hotword_prompt import DEFAULT_PT_BR_PROMPT_TEMPLATE
from qwen_hotword.inference.retrieved_rag import (
    DEFAULT_SELECTION_SEED,
    run_retrieved_rag,
)
from qwen_hotword.modeling.qwen_backbone import ModelValidationError

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a validation-only end-to-end Retrieved RAG smoke test: fixed "
            "CTC candidate selection, Qwen3-ASR prompt injection, baseline, and "
            "oracle comparison."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--ctc-case-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-count", type=int, default=60)
    parser.add_argument("--negative-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--maximum-edit-ratio", type=float, default=0.35)
    parser.add_argument(
        "--minimum-posterior-confidence",
        type=float,
        default=0.0,
    )
    parser.add_argument("--minimum-top1-margin", type=float, default=0.0)
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
        report = run_retrieved_rag(
            model_path=args.model,
            validation_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            hotword_table_path=args.hotwords,
            cases_path=args.cases,
            ctc_case_scores_path=args.ctc_case_scores,
            output_dir=args.output_dir,
            positive_count=args.positive_count,
            negative_count=args.negative_count,
            seed=args.seed,
            threshold=args.threshold,
            top_k=args.top_k,
            maximum_edit_ratio=args.maximum_edit_ratio,
            minimum_posterior_confidence=args.minimum_posterior_confidence,
            minimum_top1_margin=args.minimum_top1_margin,
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
