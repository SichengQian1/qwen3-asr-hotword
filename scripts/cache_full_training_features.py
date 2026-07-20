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
from qwen_hotword.training.ctc_generalization import validate_disjoint_records
from qwen_hotword.training.ctc_overfit import load_experiment_records
from qwen_hotword.training.feature_cache import (
    cache_feature_split,
    exclusive_feature_cache_run,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache frozen Qwen3-ASR ln_post features for the formal train and "
            "validation splits. The sealed test manifest is intentionally not accepted."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--samples-per-shard", type=int, default=512)
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
            expected_experiment="full-ctc-v1",
            expected_split="train",
        )
        validation_records = load_experiment_records(
            args.validation_manifest,
            num_classes=len(vocab.tokens),
            blank_id=0,
            expected_experiment="full-ctc-v1",
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
                    "test_set_used": False,
                    "gpu_processes": 1,
                    "logical_device": config.model.device,
                },
                indent=2,
            ),
            flush=True,
        )

        destination = Path(args.output_dir).expanduser()
        with exclusive_feature_cache_run(destination):
            wrapper = load_asr_model(config.model)
            train_summary = cache_feature_split(
                train_records,
                wrapper,
                destination / "train",
                split="train",
                source_manifest_path=args.train_manifest,
                model_path=config.model.path,
                model_dtype=config.model.dtype,
                vocab_path=args.vocab,
                encoder_batch_size=args.encoder_batch_size,
                samples_per_shard=args.samples_per_shard,
            )
            validation_summary = cache_feature_split(
                validation_records,
                wrapper,
                destination / "validation",
                split="validation",
                source_manifest_path=args.validation_manifest,
                model_path=config.model.path,
                model_dtype=config.model.dtype,
                vocab_path=args.vocab,
                encoder_batch_size=args.encoder_batch_size,
                samples_per_shard=args.samples_per_shard,
            )
            del wrapper

            import torch

            torch.cuda.empty_cache()
            report = {
                "schema_version": 1,
                "purpose": "full_ctc_v1_single_gpu_feature_cache",
                "gpu_processes": 1,
                "logical_device": config.model.device,
                "test_set_used": False,
                "train": train_summary.to_dict(),
                "validation": validation_summary.to_dict(),
                "status": "pass",
            }
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "feature_cache_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
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
            f"FULL FEATURE CACHE FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
