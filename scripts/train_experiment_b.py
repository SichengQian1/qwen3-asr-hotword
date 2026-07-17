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
from qwen_hotword.training.ctc_generalization import (
    train_cached_ctc_with_validation,
    validate_disjoint_records,
)
from qwen_hotword.training.ctc_overfit import (
    extract_frozen_features,
    load_experiment_records,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh frozen-Qwen linear CTC head on Experiment B and select "
            "checkpoints using validation PER. The held-out test manifest is intentionally "
            "not accepted by this command."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    try:
        config = load_workzone_config(args.config, require_existing_model=True)
        vocab = load_phoneme_vocab(args.vocab)
        if not vocab.tokens or vocab.tokens[0] != "<blank>":
            raise ValueError("CTC vocabulary must place <blank> at token ID 0")

        train_records = load_experiment_records(
            args.train_manifest,
            num_classes=len(vocab.tokens),
            blank_id=0,
            expected_experiment="B",
            expected_split="train",
        )
        validation_records = load_experiment_records(
            args.validation_manifest,
            num_classes=len(vocab.tokens),
            blank_id=0,
            expected_experiment="B",
            expected_split="validation",
        )
        validate_disjoint_records(train_records, validation_records)
        print(
            json.dumps(
                {
                    "stage": "manifest_validation",
                    "status": "pass",
                    "train_samples": len(train_records),
                    "validation_samples": len(validation_records),
                    "cross_split_audio_overlaps": 0,
                    "num_classes": len(vocab.tokens),
                    "head_initialization": "fresh_random",
                    "test_set_used": False,
                },
                indent=2,
            ),
            flush=True,
        )

        wrapper = load_asr_model(config.model)
        train_cached, frozen_parameters, train_extraction_seconds = (
            extract_frozen_features(
                train_records,
                wrapper,
                encoder_batch_size=args.encoder_batch_size,
                progress_every_batches=100,
                progress_label="Experiment B train features",
            )
        )
        validation_cached, validation_frozen_parameters, validation_extraction_seconds = (
            extract_frozen_features(
                validation_records,
                wrapper,
                encoder_batch_size=args.encoder_batch_size,
                progress_every_batches=25,
                progress_label="Experiment B validation features",
            )
        )
        if validation_frozen_parameters != frozen_parameters:
            raise RuntimeError("audio encoder parameter count changed between split extraction")
        del wrapper

        import torch

        torch.cuda.empty_cache()
        report = train_cached_ctc_with_validation(
            train_cached,
            validation_cached,
            vocab,
            args.output_dir,
            train_manifest_path=args.train_manifest,
            validation_manifest_path=args.validation_manifest,
            vocab_path=args.vocab,
            device=config.model.device,
            encoder_frozen_parameters=frozen_parameters,
            train_feature_extraction_seconds=train_extraction_seconds,
            validation_feature_extraction_seconds=validation_extraction_seconds,
            epochs=args.epochs,
            minimum_epochs=args.minimum_epochs,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            train_batch_size=args.train_batch_size,
            encoder_batch_size=args.encoder_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_gradient_norm=args.max_gradient_norm,
            scheduler_patience=args.scheduler_patience,
            scheduler_factor=args.scheduler_factor,
            minimum_learning_rate=args.minimum_learning_rate,
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
            f"EXPERIMENT B TRAINING FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
