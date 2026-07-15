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
from qwen_hotword.modeling.ctc_tap import (
    tensor_shape,
    tensor_values,
    validate_packed_ctc_tap,
)
from qwen_hotword.modeling.qwen_backbone import ModelValidationError, load_asr_model


def synthetic_tone(seconds: float, sample_rate: int = 16_000) -> tuple[object, int]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for the CTC tap probe") from error

    sample_count = int(seconds * sample_rate)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = (0.01 * np.sin(2.0 * math.pi * 440.0 * time)).astype(np.float32)
    return waveform, sample_rate


def tensor_metadata(value: Any) -> dict[str, Any]:
    return {
        "shape": tensor_shape(value),
        "dtype": str(getattr(value, "dtype", "unknown")),
        "device": str(getattr(value, "device", "unknown")),
    }


def positional_or_keyword(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    position: int,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


class TapRecorder:
    def __init__(self) -> None:
        self.audio_calls: list[dict[str, Any]] = []
        self.tap_calls: list[dict[str, Any]] = []

    def record_audio_input(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        input_features = positional_or_keyword(args, kwargs, "input_features", 0)
        feature_lens = positional_or_keyword(args, kwargs, "feature_lens", 1)
        aftercnn_lens = positional_or_keyword(args, kwargs, "aftercnn_lens", 2)
        self.audio_calls.append(
            {
                "input_features": tensor_metadata(input_features),
                "feature_lengths": tensor_values(feature_lens),
                "provided_aftercnn_lengths": tensor_values(aftercnn_lens),
            }
        )

    def record_tap_output(
        self,
        _module: Any,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        self.tap_calls.append(
            {
                "input": tensor_metadata(inputs[0]),
                "output": tensor_metadata(output),
            }
        )

    def build_report(self, expected_hidden_size: int = 1024) -> dict[str, Any]:
        errors: list[str] = []
        if len(self.audio_calls) != len(self.tap_calls):
            errors.append(
                "audio tower and ln_post hook call counts differ: "
                f"audio_tower={len(self.audio_calls)}, ln_post={len(self.tap_calls)}"
            )

        calls: list[dict[str, Any]] = []
        for index, (audio_call, tap_call) in enumerate(
            zip(self.audio_calls, self.tap_calls, strict=False)
        ):
            output_lengths, call_errors = validate_packed_ctc_tap(
                feature_lengths=audio_call["feature_lengths"],
                tap_shape=tap_call["output"]["shape"],
                expected_hidden_size=expected_hidden_size,
            )
            errors.extend(f"call {index}: {error}" for error in call_errors)
            calls.append(
                {
                    "call_index": index,
                    **audio_call,
                    "expected_ctc_lengths": output_lengths,
                    "expected_packed_rows": sum(output_lengths),
                    "ln_post": tap_call,
                }
            )

        if not calls:
            errors.append("ln_post was not executed")
        return {
            "tap_module": "thinker.audio_tower.ln_post",
            "tap_format": "packed_2d",
            "calls": calls,
            "errors": errors,
            "status": "pass" if not errors else "fail",
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture the real Qwen3-ASR ln_post tensor and packed lengths."
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
        audio_tower = wrapper.model.thinker.audio_tower
        recorder = TapRecorder()
        audio_handle = audio_tower.register_forward_pre_hook(
            recorder.record_audio_input,
            with_kwargs=True,
        )
        tap_handle = audio_tower.ln_post.register_forward_hook(recorder.record_tap_output)
        audios = [synthetic_tone(seconds) for seconds in args.seconds]
        try:
            results = wrapper.transcribe(
                audio=audios,
                language=["English"] * len(audios),
            )
        finally:
            tap_handle.remove()
            audio_handle.remove()
        report = recorder.build_report()
        report["input_seconds"] = args.seconds
        report["result_count"] = len(results)
    except (ConfigError, ModelValidationError, RuntimeError, OSError, ValueError) as error:
        print(f"CTC TAP PROBE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
