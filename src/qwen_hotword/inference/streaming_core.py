from __future__ import annotations

import difflib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from qwen_hotword.inference.hotword_prompt import (
    build_hotword_prompt,
    strict_phrase_match,
)


@dataclass(frozen=True)
class StreamChunk:
    chunk_id: int
    start_sample: int
    end_sample: int
    sample_rate: int
    is_tail_flush: bool

    @property
    def start_sec(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_sec(self) -> float:
        return self.end_sample / self.sample_rate


@dataclass(frozen=True)
class StreamingCandidate:
    hotword_id: str
    surface: str
    score: float
    edit_ratio: float
    posterior_confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RollbackSnapshot:
    fixed_prefix: str
    rollback_token_ids: tuple[int, ...]
    rollback_text: str
    unfixed_text: str


@dataclass(frozen=True)
class HotwordTiming:
    hotword_id: str
    start_sec: float
    end_sec: float
    timing_source: str


@dataclass(frozen=True)
class StreamingSample:
    case_id: str
    sample_id: str
    reference_text: str
    language: str
    expected_hotword_ids: tuple[str, ...]
    expected_surfaces: tuple[str, ...]
    active_hotword_ids: tuple[str, ...]
    timings: tuple[HotwordTiming, ...] = ()
    boundary_bucket: str | None = None
    audio_path: str = ""
    leading_silence_sec: float = 0.0


class TokenizerLike(Protocol):
    def encode(self, text: str, **kwargs: Any) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str: ...


class StreamingBackend(Protocol):
    tokenizer: TokenizerLike

    def init_streaming_state(
        self,
        *,
        context: str,
        language: str,
        unfixed_chunk_num: int,
        unfixed_token_num: int,
        chunk_size_sec: float,
    ) -> Any: ...

    def streaming_transcribe(self, audio: Any, state: Any) -> None: ...

    def finish_streaming_transcribe(self, state: Any) -> None: ...


CtcDetector = Callable[[Any, tuple[str, ...]], Sequence[StreamingCandidate]]


def schedule_stream_chunks(
    total_samples: int,
    *,
    sample_rate: int = 16_000,
    chunk_size_sec: float = 2.0,
) -> tuple[StreamChunk, ...]:
    if total_samples < 0:
        raise ValueError("total_samples must not be negative")
    if sample_rate <= 0 or chunk_size_sec <= 0:
        raise ValueError("sample_rate and chunk_size_sec must be positive")
    chunk_samples = int(round(sample_rate * chunk_size_sec))
    if chunk_samples <= 0:
        raise ValueError("chunk size rounds to zero samples")
    chunks: list[StreamChunk] = []
    start = 0
    while start + chunk_samples <= total_samples:
        end = start + chunk_samples
        chunks.append(
            StreamChunk(
                chunk_id=len(chunks),
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
                is_tail_flush=False,
            )
        )
        start = end
    if start < total_samples:
        chunks.append(
            StreamChunk(
                chunk_id=len(chunks),
                start_sample=start,
                end_sample=total_samples,
                sample_rate=sample_rate,
                is_tail_flush=True,
            )
        )
    return tuple(chunks)


def tokenizer_rollback(
    tokenizer: TokenizerLike,
    text: str,
    *,
    token_count: int,
    minimum_fixed_tokens: int = 0,
) -> RollbackSnapshot:
    if token_count < 0 or minimum_fixed_tokens < 0:
        raise ValueError("rollback counts must not be negative")
    token_ids = tuple(int(value) for value in tokenizer.encode(text))
    fixed_count = max(minimum_fixed_tokens, len(token_ids) - token_count)
    fixed_count = min(fixed_count, len(token_ids))
    fixed_ids = token_ids[:fixed_count]
    rollback_ids = token_ids[fixed_count:]
    fixed = str(tokenizer.decode(fixed_ids)) if fixed_ids else ""
    rollback = str(tokenizer.decode(rollback_ids)) if rollback_ids else ""
    return RollbackSnapshot(
        fixed_prefix=fixed,
        rollback_token_ids=rollback_ids,
        rollback_text=rollback,
        unfixed_text=rollback,
    )


def official_prefix_snapshot(
    tokenizer: TokenizerLike,
    raw_decoded: str,
    *,
    chunk_id: int,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    is_tail_flush: bool,
) -> RollbackSnapshot:
    if chunk_id < unfixed_chunk_num:
        token_ids = tuple(int(value) for value in tokenizer.encode(raw_decoded))
        return RollbackSnapshot("", token_ids, raw_decoded, raw_decoded)
    # Qwen's current finish path keeps at least one fixed token, unlike normal chunks.
    minimum_fixed = 1 if is_tail_flush else 0
    snapshot = tokenizer_rollback(
        tokenizer,
        raw_decoded,
        token_count=unfixed_token_num,
        minimum_fixed_tokens=minimum_fixed,
    )
    if is_tail_flush:
        return snapshot
    token_count = unfixed_token_num
    while "\ufffd" in snapshot.fixed_prefix and token_count < len(
        tuple(tokenizer.encode(raw_decoded))
    ):
        token_count += 1
        snapshot = tokenizer_rollback(
            tokenizer,
            raw_decoded,
            token_count=token_count,
        )
    return snapshot


def refresh_streaming_prompt(
    backend: StreamingBackend,
    state: Any,
    *,
    prompt: str,
    language: str,
    unfixed_chunk_num: int,
    unfixed_token_num: int,
    chunk_size_sec: float,
) -> None:
    """Refresh only prompt metadata through the public state initializer.

    Qwen currently has no public per-step context setter.  We deliberately avoid
    calling its private prompt builder: a temporary official state supplies the
    version-specific prompt fields, which are then copied to the live state.
    """
    temporary = backend.init_streaming_state(
        context=prompt,
        language=language,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
    )
    required_fields = ("prompt_raw", "context", "force_language")
    missing = [name for name in required_fields if not hasattr(temporary, name)]
    live_missing = [name for name in required_fields if not hasattr(state, name)]
    if missing or live_missing:
        raise RuntimeError(
            "installed qwen-asr streaming state cannot refresh context safely: "
            f"temporary_missing={missing}, live_missing={live_missing}"
        )
    for name in required_fields:
        setattr(state, name, getattr(temporary, name))


def boundary_bucket(
    *,
    hotword_start_sec: float,
    hotword_end_sec: float,
    audio_duration_sec: float,
    chunk_size_sec: float = 2.0,
    tolerance_sec: float = 0.05,
) -> str:
    if not 0 <= hotword_start_sec < hotword_end_sec <= audio_duration_sec + tolerance_sec:
        raise ValueError("hotword timing must fit within the audio duration")
    start_chunk = int(hotword_start_sec // chunk_size_sec)
    adjusted_end = max(hotword_start_sec, hotword_end_sec - 1e-9)
    end_chunk = int(adjusted_end // chunk_size_sec)
    final_full_boundary = math.floor(audio_duration_sec / chunk_size_sec) * chunk_size_sec
    if (
        audio_duration_sec % chunk_size_sec > tolerance_sec
        and hotword_start_sec >= final_full_boundary
    ):
        return "tail_flush"
    if end_chunk > start_chunk:
        return "long_multi_chunk" if end_chunk - start_chunk > 1 else "cross_boundary"
    phase_start = hotword_start_sec % chunk_size_sec
    phase_end = hotword_end_sec % chunk_size_sec
    if abs(phase_end) <= tolerance_sec or abs(phase_end - chunk_size_sec) <= tolerance_sec:
        return "boundary_before"
    if phase_start <= tolerance_sec:
        return "boundary_after"
    return "chunk_middle"


def _partial_change(previous: str, current: str) -> dict[str, object]:
    matcher = difflib.SequenceMatcher(a=previous, b=current, autojunk=False)
    operations = [
        {"tag": tag, "previous": previous[i1:i2], "current": current[j1:j2]}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]
    return {"changed": previous != current, "operations": operations}


def run_streaming_sample(
    *,
    backend: StreamingBackend,
    waveform: Any,
    sample: StreamingSample,
    group: str,
    hotword_surfaces: Mapping[str, str],
    ctc_detector: CtcDetector | None,
    prompt_template: str,
    sample_rate: int = 16_000,
    chunk_size_sec: float = 2.0,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    prompt_effect_policy: str = "same_step_state_refresh",
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if group not in {"C", "D", "E"}:
        raise ValueError("streaming group must be C, D, or E")
    if prompt_effect_policy != "same_step_state_refresh":
        raise ValueError("only explicit same_step_state_refresh is implemented")
    chunks = schedule_stream_chunks(
        len(waveform),
        sample_rate=sample_rate,
        chunk_size_sec=chunk_size_sec,
    )
    state = backend.init_streaming_state(
        context="",
        language=sample.language,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        chunk_size_sec=chunk_size_sec,
    )
    timeline: list[dict[str, object]] = []
    previous_text = ""
    accumulated_ids: list[str] = []
    started = time.monotonic()
    for chunk in chunks:
        cumulative = waveform[: chunk.end_sample]
        raw_before = str(getattr(state, "_raw_decoded", getattr(state, "text", "")))
        rollback = official_prefix_snapshot(
            backend.tokenizer,
            raw_before,
            chunk_id=chunk.chunk_id,
            unfixed_chunk_num=unfixed_chunk_num,
            unfixed_token_num=unfixed_token_num,
            is_tail_flush=chunk.is_tail_flush,
        )
        candidates: tuple[StreamingCandidate, ...] = ()
        if group == "D":
            if ctc_detector is None:
                raise ValueError("group D requires a causal CTC detector")
            candidates = tuple(ctc_detector(cumulative, sample.active_hotword_ids))
        elif group == "E":
            candidates = tuple(
                StreamingCandidate(
                    hotword_id=hotword_id,
                    surface=hotword_surfaces[hotword_id],
                    score=1.0,
                    edit_ratio=0.0,
                    posterior_confidence=1.0,
                )
                for hotword_id in sample.expected_hotword_ids
            )
        injected_ids = tuple(candidate.hotword_id for candidate in candidates)
        injected_surfaces = tuple(candidate.surface for candidate in candidates)
        prompt = build_hotword_prompt(injected_surfaces, template=prompt_template)
        if group in {"D", "E"}:
            refresh_streaming_prompt(
                backend,
                state,
                prompt=prompt,
                language=sample.language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )
        if chunk.is_tail_flush:
            backend.streaming_transcribe(waveform[chunk.start_sample : chunk.end_sample], state)
            backend.finish_streaming_transcribe(state)
        else:
            backend.streaming_transcribe(waveform[chunk.start_sample : chunk.end_sample], state)
        current_text = str(getattr(state, "text", ""))
        raw_after = str(getattr(state, "_raw_decoded", current_text))
        after = tokenizer_rollback(
            backend.tokenizer,
            raw_after,
            token_count=unfixed_token_num,
        )
        accumulated_ids.extend(
            hotword_id for hotword_id in injected_ids if hotword_id not in accumulated_ids
        )
        expected_status = {
            hotword_id: {
                "ctc_detected": hotword_id in accumulated_ids if group == "D" else None,
                "transcribed_correctly": strict_phrase_match(current_text, surface),
                "in_fixed_prefix": strict_phrase_match(rollback.fixed_prefix, surface),
                "in_next_fixed_prefix": strict_phrase_match(after.fixed_prefix, surface),
            }
            for hotword_id, surface in zip(
                sample.expected_hotword_ids,
                sample.expected_surfaces,
                strict=True,
            )
        }
        timeline.append(
            {
                "case_id": sample.case_id,
                "sample_id": sample.sample_id,
                "experiment_group": group,
                "chunk_id": chunk.chunk_id,
                "new_audio_start_sec": chunk.start_sec,
                "new_audio_end_sec": chunk.end_sec,
                "cumulative_audio_sec": chunk.end_sec,
                "is_tail_flush": chunk.is_tail_flush,
                "ctc_input_strategy": "cumulative_audio",
                "ctc_top_k": [candidate.to_dict() for candidate in candidates],
                "injected_hotword_ids": list(injected_ids),
                "injected_hotwords": list(injected_surfaces),
                "actual_prompt": prompt,
                "prompt_effect_policy": prompt_effect_policy,
                "prompt_effective_chunk": chunk.chunk_id if prompt else None,
                "fixed_prefix": rollback.fixed_prefix,
                "next_fixed_prefix": after.fixed_prefix,
                "rollback_token_ids": list(rollback.rollback_token_ids),
                "rollback_text": rollback.rollback_text,
                "unfixed_text": after.unfixed_text,
                "partial_transcript": current_text,
                "raw_decoded": raw_after,
                "expected_hotword_status": expected_status,
                "partial_change": _partial_change(previous_text, current_text),
            }
        )
        previous_text = current_text
    if not chunks or not chunks[-1].is_tail_flush:
        backend.finish_streaming_transcribe(state)
    elapsed = time.monotonic() - started
    final_text = str(getattr(state, "text", ""))
    result = summarize_streaming_sample(
        sample=sample,
        group=group,
        final_text=final_text,
        timeline=timeline,
        inference_seconds=elapsed,
    )
    return result, tuple(timeline)


def summarize_streaming_sample(
    *,
    sample: StreamingSample,
    group: str,
    final_text: str,
    timeline: Sequence[Mapping[str, Any]],
    inference_seconds: float,
) -> dict[str, object]:
    timing_by_id = {timing.hotword_id: timing for timing in sample.timings}
    hotword_metrics: list[dict[str, object]] = []
    for hotword_id, surface in zip(
        sample.expected_hotword_ids,
        sample.expected_surfaces,
        strict=True,
    ):
        detected_steps = [
            row
            for row in timeline
            if any(candidate["hotword_id"] == hotword_id for candidate in row["ctc_top_k"])
        ]
        correct_steps = [
            row for row in timeline if strict_phrase_match(str(row["partial_transcript"]), surface)
        ]
        stable_step = None
        for index, row in enumerate(timeline):
            if strict_phrase_match(str(row["partial_transcript"]), surface) and all(
                strict_phrase_match(str(later["partial_transcript"]), surface)
                for later in timeline[index:]
            ):
                stable_step = row
                break
        timing = timing_by_id.get(hotword_id)
        end_sec = timing.end_sec if timing else None
        timing_bucket = sample.boundary_bucket
        if timing is not None and timeline:
            timing_bucket = boundary_bucket(
                hotword_start_sec=timing.start_sec,
                hotword_end_sec=timing.end_sec,
                audio_duration_sec=float(timeline[-1]["cumulative_audio_sec"]),
            )
        first_detect_sec = (
            float(detected_steps[0]["cumulative_audio_sec"]) if detected_steps else None
        )
        first_correct_sec = (
            float(correct_steps[0]["cumulative_audio_sec"]) if correct_steps else None
        )
        stable_sec = float(stable_step["cumulative_audio_sec"]) if stable_step else None
        detect_row = detected_steps[0] if detected_steps else None
        mutable_at_detect = None
        correction_window_evidence = "not_detected"
        if detect_row is not None:
            mutable_at_detect = _reference_span_is_mutable(
                reference_text=sample.reference_text,
                fixed_prefix=str(detect_row["fixed_prefix"]),
                surface=surface,
            )
            correction_window_evidence = (
                "reference_word_alignment_to_fixed_prefix"
                if mutable_at_detect is not None
                else "reference_alignment_inconclusive"
            )
        injected_steps = [row for row in timeline if hotword_id in row["injected_hotword_ids"]]
        first_injected_chunk = int(injected_steps[0]["chunk_id"]) if injected_steps else None
        first_correct_chunk = int(correct_steps[0]["chunk_id"]) if correct_steps else None
        hotword_metrics.append(
            {
                "hotword_id": hotword_id,
                "surface": surface,
                "boundary_bucket": timing_bucket,
                "timing_source": timing.timing_source if timing else None,
                "acoustic_start_sec": timing.start_sec if timing else None,
                "acoustic_end_sec": end_sec,
                "ctc_first_detect_sec": first_detect_sec,
                "first_correct_sec": first_correct_sec,
                "stabilization_sec": stable_sec,
                "ctc_first_detect_latency_sec": _latency(first_detect_sec, end_sec),
                "first_correct_latency_sec": _latency(first_correct_sec, end_sec),
                "stabilization_latency_sec": _latency(stable_sec, end_sec),
                "mutable_at_first_detect": mutable_at_detect,
                "correction_window_evidence": correction_window_evidence,
                "first_injected_chunk": first_injected_chunk,
                "first_correct_chunk": first_correct_chunk,
                "chunks_from_injection_to_first_correct": (
                    first_correct_chunk - first_injected_chunk
                    if first_correct_chunk is not None and first_injected_chunk is not None
                    else None
                ),
                "final_correct": strict_phrase_match(final_text, surface),
                "appearance_then_disappearance_count": _disappearance_count(timeline, surface),
                "correct_then_wrong_then_correct_count": _correct_wrong_correct_count(
                    timeline, surface
                ),
            }
        )
    return {
        "case_id": sample.case_id,
        "sample_id": sample.sample_id,
        "experiment_group": group,
        "reference_text": sample.reference_text,
        "prediction": final_text,
        "expected_hotword_ids": list(sample.expected_hotword_ids),
        "expected_hotwords": list(sample.expected_surfaces),
        "matched_expected_hotword_ids": [
            hotword_id
            for hotword_id, surface in zip(
                sample.expected_hotword_ids,
                sample.expected_surfaces,
                strict=True,
            )
            if strict_phrase_match(final_text, surface)
        ],
        "injected_hotword_ids": sorted(
            {str(hotword_id) for row in timeline for hotword_id in row["injected_hotword_ids"]}
        ),
        "injected_hotwords": sorted(
            {str(surface) for row in timeline for surface in row["injected_hotwords"]}
        ),
        "boundary_bucket": sample.boundary_bucket,
        "hotword_metrics": hotword_metrics,
        "partial_modification_count": sum(
            bool(row["partial_change"]["changed"]) for row in timeline
        ),
        "fixed_boundary_modification_count": sum(
            bool(row["fixed_prefix"]) and bool(row["partial_change"]["changed"]) for row in timeline
        ),
        "inference_seconds": inference_seconds,
        "failure_reason": classify_streaming_failure(
            group=group,
            expected=bool(sample.expected_hotword_ids),
            final_correct=all(item["final_correct"] for item in hotword_metrics),
            hotword_metrics=hotword_metrics,
            timeline=timeline,
        ),
    }


def classify_streaming_failure(
    *,
    group: str,
    expected: bool,
    final_correct: bool,
    hotword_metrics: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
) -> str | None:
    if not expected:
        if any(row["injected_hotword_ids"] for row in timeline):
            return "wrong_hotword_injected"
        return None
    if final_correct:
        if any(
            item.get("stabilization_latency_sec") is not None
            and float(item["stabilization_latency_sec"]) > 2.0
            for item in hotword_metrics
        ):
            return "final_correct_high_latency"
        return None
    if group == "D":
        if any(item.get("ctc_first_detect_sec") is None for item in hotword_metrics):
            if any(item.get("boundary_bucket") == "cross_boundary" for item in hotword_metrics):
                return "boundary_specific_ctc_miss"
            return "ctc_never_detected"
        if any(item.get("mutable_at_first_detect") is False for item in hotword_metrics):
            return "ctc_detected_too_late_already_fixed"
        return "ctc_detected_in_unfixed_window_but_not_corrected"
    if group == "E":
        return "prompt_injected_but_decoder_failed"
    if timeline and bool(timeline[-1].get("is_tail_flush")):
        return "tail_flush_failure"
    return "streaming_baseline_regression" if group == "C" else "unknown_requires_review"


def _latency(event_sec: float | None, end_sec: float | None) -> float | None:
    if event_sec is None or end_sec is None:
        return None
    return event_sec - end_sec


def _disappearance_count(timeline: Sequence[Mapping[str, Any]], surface: str) -> int:
    states = [strict_phrase_match(str(row["partial_transcript"]), surface) for row in timeline]
    return sum(
        previous and not current for previous, current in zip(states, states[1:], strict=False)
    )


def _correct_wrong_correct_count(timeline: Sequence[Mapping[str, Any]], surface: str) -> int:
    states = [strict_phrase_match(str(row["partial_transcript"]), surface) for row in timeline]
    return sum(
        states[index - 1] and not states[index] and states[index + 1]
        for index in range(1, len(states) - 1)
    )


def _reference_span_is_mutable(
    *,
    reference_text: str,
    fixed_prefix: str,
    surface: str,
) -> bool | None:
    from qwen_hotword.inference.hotword_prompt import normalize_match_words

    reference = normalize_match_words(reference_text)
    phrase = normalize_match_words(surface)
    fixed = normalize_match_words(fixed_prefix)
    if not phrase:
        return None
    starts = [
        start
        for start in range(len(reference) - len(phrase) + 1)
        if reference[start : start + len(phrase)] == phrase
    ]
    if len(starts) != 1:
        return None
    if not fixed:
        return True
    matcher = difflib.SequenceMatcher(a=reference, b=fixed, autojunk=False)
    frontier = max(
        (
            block.a + block.size
            for block in matcher.get_matching_blocks()
            if block.b + block.size <= len(fixed)
        ),
        default=0,
    )
    phrase_end = starts[0] + len(phrase)
    return phrase_end > frontier
