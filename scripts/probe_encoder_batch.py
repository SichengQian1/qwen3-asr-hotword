#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.config import ConfigError, load_workzone_config
from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post
from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_lengths
from qwen_hotword.modeling.qwen_backbone import ModelValidationError, load_asr_model


def synthetic_waveform(seconds: float, sample_rate: int = 16_000) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for the encoder batch probe") from error

    sample_count = int(seconds * sample_rate)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    return (0.01 * np.sin(2.0 * math.pi * 440.0 * time)).astype(np.float32)


def build_audio_prompt(processor: Any, language: str) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return prompt + f"language {language}<asr_text>"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a padded batch from real Qwen3-ASR ln_post features."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--seconds",
        nargs="+",
        type=float,
        default=[1.0, 2.0],
        help="Synthetic sample durations used as one batch (default: 1.0 2.0).",
    )
    args = parser.parse_args()
    if any(seconds <= 0 for seconds in args.seconds):
        parser.error("all --seconds values must be positive")

    try:
        config = load_workzone_config(args.config, require_existing_model=True)
        wrapper = load_asr_model(config.model)
        waveforms = [synthetic_waveform(seconds) for seconds in args.seconds]
        prompts = [build_audio_prompt(wrapper.processor, "English") for _ in waveforms]
        processor_batch = wrapper.processor(
            text=prompts,
            audio=waveforms,
            return_tensors="pt",
            padding=True,
        )
        input_features = processor_batch["input_features"].to(
            device=wrapper.model.device,
            dtype=wrapper.model.dtype,
        )
        feature_attention_mask = processor_batch["feature_attention_mask"].to(
            device=wrapper.model.device
        )
        feature_lengths = feature_attention_mask.sum(dim=1).long().cpu().tolist()
        encoder_batch = extract_padded_ln_post(
            wrapper.model.thinker.audio_tower,
            input_features,
            feature_attention_mask,
            no_grad=True,
        )
        input_lengths = encoder_batch.input_lengths.cpu().tolist()
        expected_lengths = qwen3_asr_audio_output_lengths(feature_lengths)
        valid_mask_counts = encoder_batch.attention_mask.sum(dim=1).cpu().tolist()
        errors: list[str] = []
        if input_lengths != expected_lengths:
            errors.append(
                f"encoder lengths differ: actual={input_lengths}, expected={expected_lengths}"
            )
        if valid_mask_counts != input_lengths:
            errors.append(
                f"padding mask counts differ: mask={valid_mask_counts}, lengths={input_lengths}"
            )
        report = {
            "input_seconds": args.seconds,
            "processor": {
                "input_features_shape": list(input_features.shape),
                "feature_attention_mask_shape": list(feature_attention_mask.shape),
                "feature_lengths": feature_lengths,
            },
            "encoder_batch": {
                "hidden_states_shape": list(encoder_batch.hidden_states.shape),
                "input_lengths": input_lengths,
                "attention_mask_shape": list(encoder_batch.attention_mask.shape),
                "attention_mask_valid_counts": valid_mask_counts,
                "dtype": str(encoder_batch.hidden_states.dtype),
                "device": str(encoder_batch.hidden_states.device),
            },
            "backbone_gradients_enabled": False,
            "errors": errors,
            "status": "pass" if not errors else "fail",
        }
    except (
        ConfigError,
        KeyError,
        ModelValidationError,
        RuntimeError,
        OSError,
        ValueError,
    ) as error:
        print(f"ENCODER BATCH PROBE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
