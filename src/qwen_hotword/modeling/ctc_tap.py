from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def qwen3_asr_audio_output_length(feature_length: int) -> int:
    """Return the packed audio-encoder length used by Qwen3-ASR."""
    if feature_length < 0:
        raise ValueError("feature_length must be non-negative")
    remainder = feature_length % 100
    half_length = (remainder - 1) // 2 + 1
    remainder_output = ((half_length - 1) // 2 + 1 - 1) // 2 + 1
    return remainder_output + (feature_length // 100) * 13


def qwen3_asr_audio_output_lengths(feature_lengths: Iterable[int]) -> list[int]:
    return [qwen3_asr_audio_output_length(int(length)) for length in feature_lengths]


def tensor_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"value does not expose a tensor shape: {type(value).__name__}")
    return [int(dimension) for dimension in shape]


def tensor_values(value: Any) -> list[int]:
    if value is None:
        return []
    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    tolist = getattr(converted, "tolist", None)
    if callable(tolist):
        converted = tolist()
    if isinstance(converted, Sequence) and not isinstance(converted, str | bytes):
        return [int(item) for item in converted]
    return [int(converted)]


def validate_packed_ctc_tap(
    *,
    feature_lengths: Sequence[int],
    tap_shape: Sequence[int],
    expected_hidden_size: int = 1024,
) -> tuple[list[int], list[str]]:
    output_lengths = qwen3_asr_audio_output_lengths(feature_lengths)
    errors: list[str] = []
    if len(tap_shape) != 2:
        errors.append(f"ln_post output must be rank 2; got shape={list(tap_shape)}")
        return output_lengths, errors
    expected_rows = sum(output_lengths)
    if int(tap_shape[0]) != expected_rows:
        errors.append(
            f"packed row count mismatch: actual={tap_shape[0]}, expected={expected_rows}"
        )
    if int(tap_shape[1]) != expected_hidden_size:
        errors.append(
            f"hidden size mismatch: actual={tap_shape[1]}, expected={expected_hidden_size}"
        )
    return output_lengths, errors
