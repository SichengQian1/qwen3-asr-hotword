#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.config import ConfigError, load_workzone_config
from qwen_hotword.modeling.qwen_backbone import ModelValidationError, load_asr_model
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_overfit import (
    extract_frozen_features,
    load_experiment_records,
    train_cached_ctc_head,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overfit a frozen-Qwen linear CTC head on Experiment A."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--target-per", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20_260_716)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    try:
        config = load_workzone_config(args.config, require_existing_model=True)
        vocab = load_phoneme_vocab(args.vocab)
        if not vocab.tokens or vocab.tokens[0] != "<blank>":
            raise ValueError("CTC vocabulary must place <blank> at token ID 0")
        records = load_experiment_records(
            args.manifest,
            num_classes=len(vocab.tokens),
            blank_id=0,
        )
        print(
            json.dumps(
                {
                    "stage": "manifest_validation",
                    "status": "pass",
                    "samples": len(records),
                    "num_classes": len(vocab.tokens),
                },
                indent=2,
            ),
            flush=True,
        )
        wrapper = load_asr_model(config.model)
        cached, frozen_parameters, extraction_seconds = extract_frozen_features(
            records,
            wrapper,
            encoder_batch_size=args.encoder_batch_size,
        )
        del wrapper

        import torch

        torch.cuda.empty_cache()
        report = train_cached_ctc_head(
            cached,
            vocab,
            args.output_dir,
            manifest_path=args.manifest,
            vocab_path=args.vocab,
            device=config.model.device,
            encoder_frozen_parameters=frozen_parameters,
            feature_extraction_seconds=extraction_seconds,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            encoder_batch_size=args.encoder_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_gradient_norm=args.max_gradient_norm,
            target_phoneme_error_rate=args.target_per,
            seed=args.seed,
            log_every=args.log_every,
        )
    except (
        ConfigError,
        FileNotFoundError,
        KeyError,
        ModelValidationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"EXPERIMENT A TRAINING FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
