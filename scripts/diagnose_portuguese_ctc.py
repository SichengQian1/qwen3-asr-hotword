#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.portuguese_ctc_diagnostics import (
    diagnose_portuguese_ctc_checkpoints,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multilingual and Portuguese-specific CTC Heads on the exact same "
            "Portuguese balanced-validation samples, stratified by source, release, and "
            "reference-token pressure."
        )
    )
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--multilingual-checkpoint", required=True)
    parser.add_argument("--portuguese-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA diagnostics were requested but CUDA is unavailable")
        report = diagnose_portuguese_ctc_checkpoints(
            validation_cache_path=args.validation_cache,
            validation_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            multilingual_checkpoint_path=args.multilingual_checkpoint,
            portuguese_checkpoint_path=args.portuguese_checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(f"{type(error).__name__}: {error}")
    comparison = report.get("comparison")
    if not isinstance(comparison, dict):
        parser.error("diagnostic report has no comparison object")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(Path(args.output_dir).expanduser()),
                "selection": report["selection"],
                "overall_comparison": comparison["overall"],
                "paired_sample_counts": comparison["paired_sample_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
