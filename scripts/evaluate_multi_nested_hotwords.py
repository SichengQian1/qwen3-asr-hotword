#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.multi_nested import (
    load_hotword_families,
    load_multi_nested_cases,
    score_multi_nested_cases,
)
from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score v3 multi/compound/nested hotwords using only the existing validation "
            "feature cache and fixed Temporal 2x CTC operating point."
        )
    )
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument(
        "--cache-source-manifest",
        help=(
            "Manifest used to build the validation cache. Defaults to "
            "--validation-manifest; set this when scoring a single-language subset "
            "from a combined multilingual cache."
        ),
    )
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--asset-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--saved-ranked-matches",
        type=int,
        default=5,
        help=(
            "Number of pre-Operating ranked matches saved per case. Use 100 in a new "
            "output directory for exact offline gate calibration."
        ),
    )
    parser.add_argument("--skip-cache-sha256-verification", action="store_true")
    args = parser.parse_args()
    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA v3 evaluation was requested but CUDA is unavailable")
        vocab = load_phoneme_vocab(args.vocab)
        hotwords = load_hotword_table(args.hotwords, vocab=vocab, blank_id=0)
        families = load_hotword_families(args.families)
        cases = load_multi_nested_cases(args.cases)
        cache = load_disk_feature_cache(
            args.validation_cache,
            expected_split="validation",
            source_manifest_path=args.cache_source_manifest or args.validation_manifest,
            vocab_path=args.vocab,
            verify_sha256=not args.skip_cache_sha256_verification,
        )
        report = score_multi_nested_cases(
            args.checkpoint,
            cache,
            vocab,
            hotwords,
            families,
            cases,
            args.output_dir,
            device=args.device,
            manifest_path=args.validation_manifest,
            dictionary_path=args.dictionary,
            vocab_path=args.vocab,
            hotword_table_path=args.hotwords,
            families_path=args.families,
            cases_path=args.cases,
            asset_summary_path=args.asset_summary,
            batch_size=args.batch_size,
            saved_ranked_matches=args.saved_ranked_matches,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"V3 CTC EVALUATION FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
