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
from qwen_hotword.training.ctc_overfit import load_experiment_records
from qwen_hotword.training.sealed_test import evaluate_sealed_ctc_test

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the one-time sealed formal test PER evaluation with the already-fixed "
            "2x temporal best CTC Head. This command does not write a test feature cache."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--records-per-chunk", type=int, default=512)
    parser.add_argument(
        "--acknowledge-sealed-test-evaluation",
        action="store_true",
        help=(
            "Required acknowledgement that this consumes the sealed test split once "
            "and must not be used for checkpoint selection or tuning."
        ),
    )
    args = parser.parse_args()

    if not args.acknowledge_sealed_test_evaluation:
        print(
            "SEALED TEST EVALUATION REFUSED: pass "
            "--acknowledge-sealed-test-evaluation to confirm the one-time test read",
            file=sys.stderr,
        )
        return 2

    try:
        import torch

        config = load_workzone_config(args.config, require_existing_model=True)
        if config.model.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA sealed-test evaluation was requested but unavailable")
        output = Path(args.output).expanduser()
        if output.exists():
            raise FileExistsError(
                f"sealed-test report already exists and will not be overwritten: {output}"
            )
        vocab = load_phoneme_vocab(args.vocab)
        if not vocab.tokens or vocab.tokens[0] != "<blank>":
            raise ValueError("CTC vocabulary must place <blank> at token ID 0")
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.name != "ctc_head_best.pt":
            raise ValueError("sealed test requires the fixed ctc_head_best.pt checkpoint")
        test_records = load_experiment_records(
            args.test_manifest,
            num_classes=len(vocab.tokens),
            blank_id=0,
            expected_experiment="full-ctc-v1",
            expected_split="test",
        )
        print(
            json.dumps(
                {
                    "stage": "sealed_test_manifest_validation",
                    "status": "pass",
                    "test_samples": len(test_records),
                    "checkpoint": str(checkpoint),
                    "test_set_used": True,
                    "checkpoint_selection_or_tuning_permitted": False,
                    "feature_cache_written": False,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        wrapper = load_asr_model(config.model)
        report = evaluate_sealed_ctc_test(
            test_records,
            wrapper,
            checkpoint,
            vocab,
            output,
            test_manifest_path=args.test_manifest,
            vocab_path=args.vocab,
            model_path=config.model.path,
            device=config.model.device,
            encoder_batch_size=args.encoder_batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            records_per_chunk=args.records_per_chunk,
        )
        del wrapper
        torch.cuda.empty_cache()
    except (
        ConfigError,
        FileExistsError,
        FileNotFoundError,
        KeyError,
        ModelValidationError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
    ) as error:
        print(
            f"SEALED TEST EVALUATION FAILED: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
