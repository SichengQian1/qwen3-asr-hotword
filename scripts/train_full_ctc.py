#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_diagnostics import load_validation_sample_groups
from qwen_hotword.training.sharded_ctc import (
    exclusive_training_run,
    load_disk_feature_cache,
    train_sharded_ctc_head,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a formal CTC Head from completed train/validation feature "
            "shards. The sealed test manifest is intentionally not accepted."
        )
    )
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument(
        "--early-stopping-metric",
        choices=("validation_loss", "validation_per", "validation_macro_per"),
        default="validation_loss",
        help=(
            "Metric used to count stale epochs. Macro PER requires validation groups."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-metric",
        choices=("validation_per", "validation_macro_per"),
        default="validation_per",
        help=(
            "Select best checkpoints by aggregate PER/loss or by Macro PER, "
            "worst-group PER, then aggregate loss."
        ),
    )
    parser.add_argument(
        "--validation-group-column",
        help="Optional validation manifest field used for per-group PER metrics.",
    )
    parser.add_argument(
        "--expected-validation-groups",
        default="",
        help="Optional comma-separated exact group set, for example en,es,pt.",
    )
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20_260_720)
    parser.add_argument("--log-every-shards", type=int, default=25)
    parser.add_argument(
        "--head-type",
        choices=("linear", "temporal_upsample"),
        default="linear",
        help="Keep linear for the baseline; use temporal_upsample for the new Head experiment.",
    )
    parser.add_argument("--head-hidden-dimension", type=int, default=512)
    parser.add_argument("--head-kernel-size", type=int, default=5)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--head-time-upsampling-factor", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-cache-sha256-verification", action="store_true")
    args = parser.parse_args()

    try:
        with exclusive_training_run(args.output_dir):
            import torch

            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA training was requested but CUDA is unavailable")
            vocab = load_phoneme_vocab(args.vocab)
            verify_sha256 = not args.skip_cache_sha256_verification
            train_cache = load_disk_feature_cache(
                args.train_cache,
                expected_split="train",
                source_manifest_path=args.train_manifest,
                vocab_path=args.vocab,
                verify_sha256=verify_sha256,
            )
            validation_cache = load_disk_feature_cache(
                args.validation_cache,
                expected_split="validation",
                source_manifest_path=args.validation_manifest,
                vocab_path=args.vocab,
                verify_sha256=verify_sha256,
            )
            expected_validation_groups = tuple(
                value.strip()
                for value in args.expected_validation_groups.split(",")
                if value.strip()
            )
            if expected_validation_groups and not args.validation_group_column:
                raise ValueError(
                    "--expected-validation-groups requires --validation-group-column"
                )
            if (
                args.early_stopping_metric == "validation_macro_per"
                or args.checkpoint_selection_metric == "validation_macro_per"
            ) and not args.validation_group_column:
                raise ValueError(
                    "Macro validation metrics require --validation-group-column"
                )
            validation_sample_groups = (
                load_validation_sample_groups(
                    args.validation_manifest,
                    validation_cache,
                    group_column=args.validation_group_column,
                    expected_groups=expected_validation_groups,
                )
                if args.validation_group_column
                else None
            )
            print(
                json.dumps(
                    {
                        "stage": "feature_cache_validation",
                        "status": "pass",
                        "train_samples": train_cache.sample_count,
                        "validation_samples": validation_cache.sample_count,
                        "train_shards": train_cache.shard_count,
                        "validation_shards": validation_cache.shard_count,
                        "cache_sha256_verified": verify_sha256,
                        "validation_group_column": args.validation_group_column,
                        "validation_groups": list(expected_validation_groups),
                        "device": args.device,
                        "test_set_used": False,
                    },
                    indent=2,
                ),
                flush=True,
            )
            report = train_sharded_ctc_head(
                train_cache,
                validation_cache,
                vocab,
                args.output_dir,
                vocab_path=args.vocab,
                device=args.device,
                epochs=args.epochs,
                minimum_epochs=args.minimum_epochs,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                early_stopping_metric=args.early_stopping_metric,
                checkpoint_selection_metric=args.checkpoint_selection_metric,
                validation_sample_groups=validation_sample_groups,
                train_batch_size=args.train_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_gradient_norm=args.max_gradient_norm,
                scheduler_patience=args.scheduler_patience,
                scheduler_factor=args.scheduler_factor,
                minimum_learning_rate=args.minimum_learning_rate,
                seed=args.seed,
                log_every_shards=args.log_every_shards,
                resume=args.resume,
                head_type=args.head_type,
                head_hidden_dimension=args.head_hidden_dimension,
                head_kernel_size=args.head_kernel_size,
                head_dropout=args.head_dropout,
                head_time_upsampling_factor=args.head_time_upsampling_factor,
            )
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"FULL CTC TRAINING FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
