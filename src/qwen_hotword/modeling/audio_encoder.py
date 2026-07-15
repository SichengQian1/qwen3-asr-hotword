from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class PaddedEncoderBatch:
    """Padded ln_post states plus the true time length of each sample."""

    hidden_states: torch.Tensor
    input_lengths: torch.Tensor
    attention_mask: torch.Tensor


def extract_padded_ln_post(
    audio_tower: Any,
    input_features: torch.Tensor,
    feature_attention_mask: torch.Tensor,
    *,
    no_grad: bool = True,
    expected_hidden_size: int = 1024,
) -> PaddedEncoderBatch:
    """Run Qwen3-ASR's encoder per sample and pad packed ln_post outputs."""
    if input_features.ndim != 3:
        raise ValueError(
            "input_features must have shape [batch, mel_bins, feature_time]; "
            f"got {list(input_features.shape)}"
        )
    if feature_attention_mask.ndim != 2:
        raise ValueError(
            "feature_attention_mask must have shape [batch, feature_time]; "
            f"got {list(feature_attention_mask.shape)}"
        )
    if input_features.shape[0] != feature_attention_mask.shape[0]:
        raise ValueError("input_features and feature_attention_mask batch sizes differ")
    if input_features.shape[2] != feature_attention_mask.shape[1]:
        raise ValueError("input feature time and feature_attention_mask time differ")

    feature_lengths = feature_attention_mask.sum(dim=1).to(dtype=torch.long)
    if bool(torch.any(feature_lengths <= 0).item()):
        raise ValueError("every audio sample must contain at least one valid feature frame")

    captured: list[torch.Tensor] = []

    def capture_ln_post(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured.append(output)

    handle = audio_tower.ln_post.register_forward_hook(capture_ln_post)
    context = torch.no_grad() if no_grad else nullcontext()
    try:
        with context:
            for input_feature, feature_length in zip(
                input_features,
                feature_lengths,
                strict=True,
            ):
                previous_calls = len(captured)
                audio_tower(
                    input_feature[:, :feature_length],
                    feature_lens=feature_length.unsqueeze(0),
                )
                if len(captured) != previous_calls + 1:
                    raise RuntimeError("ln_post must run exactly once for each audio sample")
    finally:
        handle.remove()

    for sample_index, hidden_state in enumerate(captured):
        if hidden_state.ndim != 2 or hidden_state.shape[1] != expected_hidden_size:
            raise RuntimeError(
                f"sample {sample_index} produced unexpected ln_post shape: "
                f"{list(hidden_state.shape)}"
            )

    input_lengths = torch.tensor(
        [hidden_state.shape[0] for hidden_state in captured],
        dtype=torch.long,
        device=input_features.device,
    )
    hidden_states = pad_sequence(captured, batch_first=True, padding_value=0.0)
    time_indices = torch.arange(hidden_states.shape[1], device=input_features.device)
    attention_mask = time_indices.unsqueeze(0) < input_lengths.unsqueeze(1)
    return PaddedEncoderBatch(
        hidden_states=hidden_states,
        input_lengths=input_lengths,
        attention_mask=attention_mask,
    )
