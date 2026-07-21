#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_diagnostics import diagnose_ctc_checkpoint
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose frozen-encoder CTC checkpoints on the validation cache. "
            "The sealed test split is intentionally not accepted."
        )
    )
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--skip-cache-sha256-verification", action="store_true")
    args = parser.parse_args()

    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostics were requested but CUDA is unavailable")
        vocab = load_phoneme_vocab(args.vocab)
        cache = load_disk_feature_cache(
            args.validation_cache,
            expected_split="validation",
            source_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            verify_sha256=not args.skip_cache_sha256_verification,
        )
        reports = [
            diagnose_ctc_checkpoint(
                checkpoint,
                cache,
                vocab,
                device=args.device,
                batch_size=args.batch_size,
            )
            for checkpoint in args.checkpoint
        ]
        result = {
            "purpose": "frozen_encoder_ctc_validation_diagnostics",
            "validation_samples": cache.sample_count,
            "test_set_used": False,
            "checkpoints": reports,
            "status": "pass",
        }
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"CTC DIAGNOSTICS FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
