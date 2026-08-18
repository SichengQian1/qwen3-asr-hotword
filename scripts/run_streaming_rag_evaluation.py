#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.hotword_prompt import DEFAULT_PT_BR_PROMPT_TEMPLATE
from qwen_hotword.inference.streaming_rag import run_streaming_rag_evaluation

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the sealed Qwen3-ASR 2-second streaming end-to-end hotword "
            "evaluation with offline A/B import and streaming C/D/E groups."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--offline-rag-dir", required=True)
    parser.add_argument(
        "--offline-format",
        choices=("retrieved_v2", "multi_nested_v3"),
        default="retrieved_v2",
    )
    parser.add_argument("--hotword-families")
    parser.add_argument("--ctc-report")
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--groups",
        default="A,B,C,D,E",
        help="Comma-separated subset of A,B,C,D,E.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--boundary-manifest")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--unfixed-chunk-num", type=int, default=2)
    parser.add_argument("--unfixed-token-num", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval-mode",
        choices=("operating", "forced_topk"),
        default="operating",
        help=(
            "Use thresholded Operating candidates or raw ranked Top-K candidates. "
            "The mode must match the offline control report."
        ),
    )
    parser.add_argument("--maximum-edit-ratio", type=float, default=0.35)
    parser.add_argument("--posterior-weight", type=float, default=0.25)
    parser.add_argument("--minimum-posterior-confidence", type=float, default=0.0)
    parser.add_argument("--minimum-top1-margin", type=float, default=0.0)
    parser.add_argument("--prompt-template", default=DEFAULT_PT_BR_PROMPT_TEMPLATE)
    parser.add_argument("--language", default="Portuguese")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Must match the offline report; omitted means inherit its recorded value.",
    )
    parser.add_argument("--seed", type=int, default=20_260_817)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    groups = tuple(item.strip().upper() for item in args.groups.split(",") if item.strip())
    try:
        report = run_streaming_rag_evaluation(
            model_path=args.model,
            validation_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            hotword_table_path=args.hotwords,
            cases_path=args.cases,
            offline_rag_dir=args.offline_rag_dir,
            offline_format=args.offline_format,
            hotword_families_path=args.hotword_families,
            ctc_report_path=args.ctc_report,
            ctc_checkpoint_path=args.ctc_checkpoint,
            output_dir=args.output_dir,
            groups=groups,
            max_samples=args.max_samples,
            boundary_manifest_path=args.boundary_manifest,
            chunk_size_sec=args.chunk_size_sec,
            unfixed_chunk_num=args.unfixed_chunk_num,
            unfixed_token_num=args.unfixed_token_num,
            threshold=args.threshold,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            maximum_edit_ratio=args.maximum_edit_ratio,
            posterior_weight=args.posterior_weight,
            minimum_posterior_confidence=args.minimum_posterior_confidence,
            minimum_top1_margin=args.minimum_top1_margin,
            prompt_template=args.prompt_template,
            language=args.language,
            dtype=args.dtype,
            device=args.device,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            resume=args.resume,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
