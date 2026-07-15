from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CtcComputation:
    logits: torch.Tensor
    log_probs: torch.Tensor
    loss: torch.Tensor


class LinearCtcHead(nn.Module):  # type: ignore[misc]
    """Minimal baseline projection from Qwen audio states to phoneme classes."""

    def __init__(self, input_dimension: int, num_classes: int) -> None:
        super().__init__()
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must contain blank and at least one label")
        self.projection = nn.Linear(input_dimension, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, time, hidden]; "
                f"got {list(hidden_states.shape)}"
            )
        return self.projection(hidden_states)


def compute_ctc(
    head: LinearCtcHead,
    hidden_states: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_id: int = 0,
) -> CtcComputation:
    """Project encoder states and compute numerically stable phoneme CTC loss."""
    logits = head(hidden_states)
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape [batch, target_time]; got {targets.shape}")
    batch_size, max_input_time, num_classes = logits.shape
    if input_lengths.shape != (batch_size,):
        raise ValueError("input_lengths must contain one value per batch sample")
    if targets.shape[0] != batch_size or target_lengths.shape != (batch_size,):
        raise ValueError("targets and target_lengths must match the logits batch size")
    if blank_id < 0 or blank_id >= num_classes:
        raise ValueError(f"blank_id={blank_id} is outside {num_classes} classes")
    if bool(torch.any(input_lengths <= 0).item()):
        raise ValueError("all input lengths must be positive")
    if bool(torch.any(input_lengths > max_input_time).item()):
        raise ValueError("an input length exceeds the padded logits time dimension")
    if bool(torch.any(target_lengths <= 0).item()):
        raise ValueError("all target lengths must be positive")
    if bool(torch.any(target_lengths > targets.shape[1]).item()):
        raise ValueError("a target length exceeds the padded target dimension")
    if bool(torch.any(target_lengths > input_lengths).item()):
        raise ValueError("a CTC target is longer than its encoder input")

    # CTC accumulation is more stable and broadly supported in float32.
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1).contiguous()
    loss = nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    return CtcComputation(logits=logits, log_probs=log_probs, loss=loss)
