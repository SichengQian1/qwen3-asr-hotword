#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.evaluation import evaluate_hotword_scoring
from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.hotwords.simulation import load_simulated_cases
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def _parse_thresholds(value: str) -> tuple[float, ...]:
    try:
        thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from error
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be within [0, 1]")
    return thresholds


def _parse_ranking_ks(value: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("ranking ks must be comma-separated integers") from error
    if not ks or any(k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise argparse.ArgumentTypeError("ranking ks must be unique positive integers")
    return ks


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score simulated hotwords with the 2x temporal CTC Head on validation "
            "features and sweep false-positive control thresholds."
        )
    )
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--ranking-ks",
        type=_parse_ranking_ks,
        default=_parse_ranking_ks("1,3,5"),
    )
    parser.add_argument("--minimum-phonemes", type=int, default=4)
    parser.add_argument("--maximum-edit-ratio", type=float, default=0.35)
    parser.add_argument("--posterior-weight", type=float, default=0.25)
    parser.add_argument("--minimum-posterior-confidence", type=float, default=0.0)
    parser.add_argument("--minimum-top1-margin", type=float, default=0.03)
    parser.add_argument("--target-precision", type=float, default=0.90)
    parser.add_argument(
        "--maximum-negative-case-false-positive-rate",
        type=float,
        default=0.03,
    )
    parser.add_argument("--skip-cache-sha256-verification", action="store_true")
    args = parser.parse_args()

    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA hotword evaluation was requested but CUDA is unavailable")
        vocab = load_phoneme_vocab(args.vocab)
        hotwords = load_hotword_table(args.hotwords, vocab=vocab, blank_id=0)
        cases = load_simulated_cases(args.cases)
        cache = load_disk_feature_cache(
            args.validation_cache,
            expected_split="validation",
            source_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            verify_sha256=not args.skip_cache_sha256_verification,
        )
        report = evaluate_hotword_scoring(
            args.checkpoint,
            cache,
            vocab,
            hotwords,
            cases,
            args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            thresholds=args.thresholds,
            top_k=args.top_k,
            minimum_phonemes=args.minimum_phonemes,
            maximum_edit_ratio=args.maximum_edit_ratio,
            posterior_weight=args.posterior_weight,
            minimum_posterior_confidence=args.minimum_posterior_confidence,
            minimum_top1_margin=args.minimum_top1_margin,
            target_precision=args.target_precision,
            maximum_negative_case_false_positive_rate=(
                args.maximum_negative_case_false_positive_rate
            ),
            ranking_ks=args.ranking_ks,
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"HOTWORD SCORING FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
