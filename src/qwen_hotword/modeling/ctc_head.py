from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class CtcComputation:
    logits: torch.Tensor
    log_probs: torch.Tensor
    loss: torch.Tensor
    input_lengths: torch.Tensor


class LinearCtcHead(nn.Module):  # type: ignore[misc]
    head_type = "linear"

    def __init__(self, input_dimension: int, num_classes: int) -> None:
        super().__init__()
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must contain blank and at least one label")
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.time_upsampling_factor = 1
        self.projection = nn.Linear(input_dimension, num_classes)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_lengths
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, D]")
        if hidden_states.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Expected hidden dimension {self.input_dimension}, got {hidden_states.shape[-1]}"
            )
        return self.projection(hidden_states)


class TemporalUpsampleCtcHead(nn.Module):  # type: ignore[misc]
    """CTC Head with exact temporal upsampling and lightweight local context."""

    head_type = "temporal_upsample"

    def __init__(
        self,
        input_dimension: int,
        num_classes: int,
        *,
        hidden_dimension: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.1,
        time_upsampling_factor: int = 2,
    ) -> None:
        super().__init__()
        if input_dimension <= 0 or num_classes <= 1 or hidden_dimension <= 0:
            raise ValueError("Head dimensions must be positive and num_classes must exceed one")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if time_upsampling_factor <= 0:
            raise ValueError("time_upsampling_factor must be positive")

        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.hidden_dimension = hidden_dimension
        self.kernel_size = kernel_size
        self.dropout_probability = dropout
        self.time_upsampling_factor = time_upsampling_factor

        self.input_norm = nn.LayerNorm(input_dimension)
        self.input_projection = nn.Conv1d(input_dimension, hidden_dimension, kernel_size=1)
        self.depthwise_convolution = nn.Conv1d(
            hidden_dimension,
            hidden_dimension,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dimension,
        )
        self.context_projection = nn.Conv1d(hidden_dimension, hidden_dimension, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dimension, num_classes)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths * self.time_upsampling_factor

    @staticmethod
    def _mask_padded_steps(
        values: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if lengths is None:
            return values
        steps = torch.arange(values.shape[1], device=values.device)
        mask = steps.unsqueeze(0) < lengths.to(values.device).unsqueeze(1)
        return values * mask.unsqueeze(-1).to(values.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, T, D]")
        if hidden_states.shape[-1] != self.input_dimension:
            raise ValueError(
                f"Expected hidden dimension {self.input_dimension}, got {hidden_states.shape[-1]}"
            )
        if input_lengths is not None:
            if input_lengths.ndim != 1 or input_lengths.shape[0] != hidden_states.shape[0]:
                raise ValueError("input_lengths must have shape [B]")
            if torch.any(input_lengths <= 0) or torch.any(input_lengths > hidden_states.shape[1]):
                raise ValueError("input_lengths must be within the hidden-state time dimension")

        values = self.input_norm(hidden_states)
        values = torch.repeat_interleave(
            values,
            repeats=self.time_upsampling_factor,
            dim=1,
        )
        output_lengths = self.output_lengths(input_lengths) if input_lengths is not None else None
        values = self._mask_padded_steps(values, output_lengths)

        values = self.input_projection(values.transpose(1, 2)).transpose(1, 2)
        values = self.dropout(self.activation(values))
        values = self._mask_padded_steps(values, output_lengths)
        residual = values
        values = self.depthwise_convolution(values.transpose(1, 2))
        values = self.context_projection(values).transpose(1, 2)
        values = self.dropout(self.activation(values + residual))
        values = self._mask_padded_steps(values, output_lengths)
        return self.output_projection(values)


CtcHead = LinearCtcHead | TemporalUpsampleCtcHead


def build_ctc_head(
    *,
    head_type: str,
    input_dimension: int,
    num_classes: int,
    hidden_dimension: int = 512,
    kernel_size: int = 5,
    dropout: float = 0.1,
    time_upsampling_factor: int = 2,
) -> CtcHead:
    normalized = head_type.strip().lower()
    if normalized in {"linear", "linearctchead"}:
        return LinearCtcHead(input_dimension, num_classes)
    if normalized in {"temporal_upsample", "temporalupsamplectchead"}:
        return TemporalUpsampleCtcHead(
            input_dimension,
            num_classes,
            hidden_dimension=hidden_dimension,
            kernel_size=kernel_size,
            dropout=dropout,
            time_upsampling_factor=time_upsampling_factor,
        )
    raise ValueError(f"Unsupported CTC head type: {head_type}")


def ctc_head_config(head: CtcHead) -> dict[str, Any]:
    config: dict[str, Any] = {
        "head_type": head.head_type,
        "input_dimension": head.input_dimension,
        "num_classes": head.num_classes,
        "time_upsampling_factor": head.time_upsampling_factor,
    }
    if isinstance(head, TemporalUpsampleCtcHead):
        config.update(
            {
                "hidden_dimension": head.hidden_dimension,
                "kernel_size": head.kernel_size,
                "dropout": head.dropout_probability,
            }
        )
    return config


def build_ctc_head_from_checkpoint(payload: Mapping[str, Any]) -> CtcHead:
    config = payload.get("head_config")
    if isinstance(config, Mapping):
        return build_ctc_head(
            head_type=str(config.get("head_type", "linear")),
            input_dimension=int(config["input_dimension"]),
            num_classes=int(config["num_classes"]),
            hidden_dimension=int(config.get("hidden_dimension", 512)),
            kernel_size=int(config.get("kernel_size", 5)),
            dropout=float(config.get("dropout", 0.1)),
            time_upsampling_factor=int(config.get("time_upsampling_factor", 2)),
        )
    return build_ctc_head(
        head_type=str(payload.get("head_type", "linear")),
        input_dimension=int(payload["input_dimension"]),
        num_classes=int(payload["num_classes"]),
    )


def compute_ctc(
    head: CtcHead,
    hidden_states: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_id: int = 0,
) -> CtcComputation:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, T, D]")
    if input_lengths.ndim != 1 or input_lengths.shape[0] != hidden_states.shape[0]:
        raise ValueError("input_lengths must have shape [B]")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > hidden_states.shape[1]):
        raise ValueError("input_lengths must be within the hidden-state time dimension")

    logits = head(hidden_states, input_lengths=input_lengths)
    effective_input_lengths = head.output_lengths(input_lengths)
    if logits.ndim != 3 or logits.shape[0] != hidden_states.shape[0]:
        raise ValueError("CTC Head logits must have shape [B, T_out, C]")
    if logits.shape[-1] != head.num_classes:
        raise ValueError("CTC Head logits class dimension does not match the Head config")
    if blank_id < 0 or blank_id >= head.num_classes:
        raise ValueError(f"blank_id={blank_id} is outside {head.num_classes} classes")
    if torch.any(effective_input_lengths <= 0) or torch.any(
        effective_input_lengths > logits.shape[1]
    ):
        raise ValueError("effective CTC input lengths must fit within the Head output")
    if targets.ndim not in {1, 2}:
        raise ValueError("targets must be a packed [N] tensor or padded [B, S] tensor")
    if targets.ndim == 2 and targets.shape[0] != hidden_states.shape[0]:
        raise ValueError("padded targets must have shape [B, S]")
    if target_lengths.ndim != 1 or target_lengths.shape[0] != hidden_states.shape[0]:
        raise ValueError("target_lengths must have shape [B]")
    if torch.any(target_lengths <= 0):
        raise ValueError("target_lengths must be positive")
    if targets.ndim == 2 and torch.any(target_lengths > targets.shape[1]):
        raise ValueError("a target length exceeds the padded target dimension")
    if targets.ndim == 1 and int(target_lengths.sum().item()) != targets.numel():
        raise ValueError("packed targets must contain exactly sum(target_lengths) labels")
    if torch.any(target_lengths > effective_input_lengths):
        raise ValueError("target_lengths cannot exceed effective CTC input lengths")

    # CTC accumulation is more stable and broadly supported in float32.
    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1).contiguous()
    loss = nn.functional.ctc_loss(
        log_probs,
        targets,
        effective_input_lengths.cpu(),
        target_lengths.cpu(),
        blank=blank_id,
        reduction="mean",
        zero_infinity=True,
    )
    return CtcComputation(
        logits=logits,
        log_probs=log_probs,
        loss=loss,
        input_lengths=effective_input_lengths,
    )
