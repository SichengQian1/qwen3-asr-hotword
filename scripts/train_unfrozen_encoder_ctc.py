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
from qwen_hotword.training.sharded_ctc import exclusive_training_run
from qwen_hotword.training.unfrozen_encoder_ctc import train_unfrozen_encoder_ctc

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a control CTC branch while recomputing Qwen3-ASR audio encoder states "
            "and selectively unfreezing encoder parameters. The sealed test manifest is "
            "intentionally not accepted."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minimum-epochs", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("validation_loss", "validation_per"),
        default="validation_loss",
    )
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--minimum-encoder-learning-rate", type=float, default=1e-7)
    parser.add_argument("--unfreeze-last-encoder-layers", type=int, default=1)
    parser.add_argument("--no-train-ln-post", action="store_true")
    parser.add_argument("--unfreeze-all-encoder", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--log-every-batches", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    try:
        config = load_workzone_config(args.config, require_existing_model=True)
        vocab = load_phoneme_vocab(args.vocab)
        if not vocab.tokens or vocab.tokens[0] != "<blank>":
            raise ValueError("CTC vocabulary must place <blank> at token ID 0")
        with exclusive_training_run(args.output_dir):
            wrapper = load_asr_model(config.model)
            report = train_unfrozen_encoder_ctc(
                wrapper,
                args.train_manifest,
                args.validation_manifest,
                vocab,
                args.output_dir,
                vocab_path=args.vocab,
                device=config.model.device,
                epochs=args.epochs,
                minimum_epochs=args.minimum_epochs,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                early_stopping_metric=args.early_stopping_metric,
                train_batch_size=args.train_batch_size,
                validation_batch_size=args.validation_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                head_learning_rate=args.head_learning_rate,
                encoder_learning_rate=args.encoder_learning_rate,
                weight_decay=args.weight_decay,
                max_gradient_norm=args.max_gradient_norm,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                minimum_head_learning_rate=args.minimum_head_learning_rate,
                minimum_encoder_learning_rate=args.minimum_encoder_learning_rate,
                seed=args.seed,
                log_every_batches=args.log_every_batches,
                resume=args.resume,
                unfreeze_last_encoder_layers=args.unfreeze_last_encoder_layers,
                train_ln_post=not args.no_train_ln_post,
                unfreeze_all_encoder=args.unfreeze_all_encoder,
                max_train_samples=args.max_train_samples,
                max_validation_samples=args.max_validation_samples,
            )
    except (
        ConfigError,
        FileNotFoundError,
        KeyError,
        ModelValidationError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
    ) as error:
        print(
            f"UNFROZEN ENCODER CTC TRAINING FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

