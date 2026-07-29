from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.training.edit_distance import sequence_edit_distance


@dataclass(frozen=True)
class DecodedPhoneme:
    token_id: int
    confidence: float
    start_step: int
    end_step: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HotwordMatch:
    hotword_id: str
    surface: str
    language: str
    score: float
    edit_similarity: float
    edit_distance: int
    edit_ratio: float
    posterior_confidence: float
    decoded_start: int
    decoded_end: int
    start_step: int | None
    end_step: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HotwordScoringConfig:
    score_threshold: float = 0.75
    top_k: int = 3
    minimum_phonemes: int = 4
    maximum_edit_ratio: float = 0.35
    posterior_weight: float = 0.25
    minimum_posterior_confidence: float = 0.0
    minimum_top1_margin: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if self.top_k <= 0 or self.minimum_phonemes <= 0:
            raise ValueError("top_k and minimum_phonemes must be positive")
        for name, value in (
            ("maximum_edit_ratio", self.maximum_edit_ratio),
            ("posterior_weight", self.posterior_weight),
            ("minimum_posterior_confidence", self.minimum_posterior_confidence),
            ("minimum_top1_margin", self.minimum_top1_margin),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class HotwordScoringResult:
    effective_time_steps: int
    decoded_token_ids: tuple[int, ...]
    decoded_confidences: tuple[float, ...]
    ranked_matches: tuple[HotwordMatch, ...]
    selected_matches: tuple[HotwordMatch, ...]
    suppressed_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "effective_time_steps": self.effective_time_steps,
            "decoded_token_ids": list(self.decoded_token_ids),
            "decoded_confidences": list(self.decoded_confidences),
            "ranked_matches": [match.to_dict() for match in self.ranked_matches],
            "selected_matches": [match.to_dict() for match in self.selected_matches],
            "suppressed_reason": self.suppressed_reason,
        }


def decode_ctc_posterior(
    logits: Any,
    *,
    input_length: int,
    blank_id: int = 0,
) -> tuple[DecodedPhoneme, ...]:
    import torch

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("logits must be a [T, C] tensor")
    if input_length <= 0 or input_length > logits.shape[0]:
        raise ValueError("input_length must fit within the effective Head time axis")
    if blank_id < 0 or blank_id >= logits.shape[1]:
        raise ValueError("blank_id is outside the logits class dimension")
    probabilities = logits[:input_length].float().softmax(dim=-1)
    confidences, token_ids = probabilities.max(dim=-1)
    raw_ids = [int(value) for value in token_ids.detach().cpu().tolist()]
    raw_confidences = [float(value) for value in confidences.detach().cpu().tolist()]

    decoded: list[DecodedPhoneme] = []
    start = 0
    while start < input_length:
        token_id = raw_ids[start]
        end = start + 1
        while end < input_length and raw_ids[end] == token_id:
            end += 1
        if token_id != blank_id:
            decoded.append(
                DecodedPhoneme(
                    token_id=token_id,
                    confidence=sum(raw_confidences[start:end]) / (end - start),
                    start_step=start,
                    end_step=end,
                )
            )
        start = end
    return tuple(decoded)


def score_hotwords(
    logits: Any,
    *,
    input_length: int,
    hotwords: list[HotwordEntry] | tuple[HotwordEntry, ...],
    config: HotwordScoringConfig | None = None,
    blank_id: int = 0,
) -> HotwordScoringResult:
    scoring_config = config or HotwordScoringConfig()
    scoring_config.validate()
    decoded = decode_ctc_posterior(
        logits,
        input_length=input_length,
        blank_id=blank_id,
    )
    matches = [
        _score_candidate(candidate, decoded, scoring_config)
        for candidate in hotwords
        if len(candidate.token_ids) >= scoring_config.minimum_phonemes
    ]
    matches.sort(
        key=lambda item: (
            -item.score,
            item.edit_ratio,
            -item.posterior_confidence,
            item.hotword_id,
        )
    )
    qualified = [
        match
        for match in matches
        if match.score >= scoring_config.score_threshold
        and match.edit_ratio <= scoring_config.maximum_edit_ratio
        and match.posterior_confidence
        >= scoring_config.minimum_posterior_confidence
    ]
    suppressed_reason: str | None = None
    if not qualified:
        suppressed_reason = "below_threshold"
        selected: list[HotwordMatch] = []
    elif (
        len(qualified) > 1
        and qualified[0].score - qualified[1].score
        < scoring_config.minimum_top1_margin
    ):
        suppressed_reason = "ambiguous_top_matches"
        selected = []
    else:
        selected = qualified[: scoring_config.top_k]
    return HotwordScoringResult(
        effective_time_steps=input_length,
        decoded_token_ids=tuple(item.token_id for item in decoded),
        decoded_confidences=tuple(item.confidence for item in decoded),
        ranked_matches=tuple(matches),
        selected_matches=tuple(selected),
        suppressed_reason=suppressed_reason,
    )


def _score_candidate(
    candidate: HotwordEntry,
    decoded: tuple[DecodedPhoneme, ...],
    config: HotwordScoringConfig,
) -> HotwordMatch:
    target = candidate.token_ids
    if not decoded:
        return HotwordMatch(
            hotword_id=candidate.hotword_id,
            surface=candidate.surface,
            language=candidate.language,
            score=0.0,
            edit_similarity=0.0,
            edit_distance=len(target),
            edit_ratio=1.0,
            posterior_confidence=0.0,
            decoded_start=0,
            decoded_end=0,
            start_step=None,
            end_step=None,
        )
    decoded_ids = tuple(item.token_id for item in decoded)
    maximum_delta = max(2, math.ceil(len(target) * 0.5))
    maximum_width = min(len(decoded), len(target) + maximum_delta)
    minimum_width = min(maximum_width, max(1, len(target) - maximum_delta))
    best: tuple[float, int, float, int, int, int] | None = None
    for width in range(minimum_width, maximum_width + 1):
        for start in range(0, len(decoded) - width + 1):
            end = start + width
            distance = sequence_edit_distance(target, decoded_ids[start:end])
            denominator = max(len(target), width)
            edit_ratio = distance / denominator
            posterior = sum(item.confidence for item in decoded[start:end]) / width
            key = (
                edit_ratio,
                distance,
                -posterior,
                abs(width - len(target)),
                start,
                end,
            )
            if best is None or key < best:
                best = key
    if best is None:
        raise RuntimeError("hotword local matcher failed to inspect a decoded window")
    edit_ratio, distance, negative_posterior, _, start, end = best
    posterior = -negative_posterior
    similarity = 1.0 - edit_ratio
    confidence_factor = (
        1.0 - config.posterior_weight + config.posterior_weight * posterior
    )
    score = max(0.0, min(1.0, similarity * confidence_factor))
    return HotwordMatch(
        hotword_id=candidate.hotword_id,
        surface=candidate.surface,
        language=candidate.language,
        score=score,
        edit_similarity=similarity,
        edit_distance=distance,
        edit_ratio=edit_ratio,
        posterior_confidence=posterior,
        decoded_start=start,
        decoded_end=end,
        start_step=decoded[start].start_step,
        end_step=decoded[end - 1].end_step,
    )
