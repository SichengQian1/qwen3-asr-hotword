from __future__ import annotations

from dataclasses import dataclass

import pytest

from qwen_hotword.inference.streaming_core import (
    HotwordTiming,
    StreamingCandidate,
    StreamingSample,
    boundary_bucket,
    official_prefix_snapshot,
    run_streaming_sample,
    schedule_stream_chunks,
    tokenizer_rollback,
)


class PieceTokenizer:
    def __init__(self) -> None:
        self.pieces: list[str] = []

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        pieces = text.split("|") if text else []
        ids = []
        for piece in pieces:
            if piece not in self.pieces:
                self.pieces.append(piece)
            ids.append(self.pieces.index(piece))
        return ids

    def decode(self, token_ids: list[int] | tuple[int, ...], **_kwargs: object) -> str:
        return "|".join(self.pieces[token_id] for token_id in token_ids)


@dataclass
class FakeState:
    prompt_raw: str
    context: str
    force_language: str
    chunk_id: int = 0
    text: str = ""
    _raw_decoded: str = ""
    buffered: list[float] | None = None


class FakeBackend:
    def __init__(self, outputs: list[str]) -> None:
        self.tokenizer = PieceTokenizer()
        self.outputs = outputs
        self.contexts: list[str] = []
        self.languages: list[str] = []
        self.finish_calls = 0

    def init_streaming_state(
        self,
        *,
        context: str,
        language: str,
        unfixed_chunk_num: int,
        unfixed_token_num: int,
        chunk_size_sec: float,
    ) -> FakeState:
        del unfixed_chunk_num, unfixed_token_num, chunk_size_sec
        self.languages.append(language)
        return FakeState(
            prompt_raw=f"PROMPT:{context}",
            context=context,
            force_language=language,
            buffered=[],
        )

    def streaming_transcribe(self, audio: list[float], state: FakeState) -> None:
        assert state.buffered is not None
        state.buffered.extend(audio)
        if len(audio) == 32_000:
            self._decode(state)

    def finish_streaming_transcribe(self, state: FakeState) -> None:
        self.finish_calls += 1
        assert state.buffered is not None
        if state.buffered:
            self._decode(state)

    def _decode(self, state: FakeState) -> None:
        self.contexts.append(state.context)
        output = self.outputs[state.chunk_id]
        state.text = output
        state._raw_decoded = output
        state.chunk_id += 1
        state.buffered = []


def _sample(*, expected: bool = True, timing: bool = False) -> StreamingSample:
    return StreamingSample(
        case_id="case-1",
        sample_id="sample-1",
        reference_text="hello hot word",
        language="English",
        expected_hotword_ids=("hot",) if expected else (),
        expected_surfaces=("hot word",) if expected else (),
        active_hotword_ids=("hot", "wrong"),
        timings=(HotwordTiming("hot", 3.7, 4.2, "manual_confirmed"),)
        if timing and expected
        else (),
        boundary_bucket="cross_boundary" if timing else None,
    )


def test_chunk_scheduler_covers_empty_short_exact_and_tail() -> None:
    assert schedule_stream_chunks(0) == ()
    short = schedule_stream_chunks(8_000)
    assert [(item.start_sample, item.end_sample, item.is_tail_flush) for item in short] == [
        (0, 8_000, True)
    ]
    exact = schedule_stream_chunks(32_000)
    assert [(item.start_sample, item.end_sample, item.is_tail_flush) for item in exact] == [
        (0, 32_000, False)
    ]
    longer = schedule_stream_chunks(70_000)
    assert [item.end_sample for item in longer] == [32_000, 64_000, 70_000]
    assert [item.is_tail_flush for item in longer] == [False, False, True]


def test_tokenizer_rollback_is_token_not_character_based() -> None:
    tokenizer = PieceTokenizer()
    text = "Buenos|días|señor|Qwen|阿根廷|💡"
    snapshot = tokenizer_rollback(tokenizer, text, token_count=5)
    assert snapshot.fixed_prefix == "Buenos"
    assert snapshot.rollback_text == "días|señor|Qwen|阿根廷|💡"
    assert len(snapshot.rollback_token_ids) == 5


def test_first_two_chunks_have_no_prefix_then_rollback_five_tokens() -> None:
    tokenizer = PieceTokenizer()
    text = "a|b|c|d|e|f|g"
    first = official_prefix_snapshot(
        tokenizer,
        text,
        chunk_id=1,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        is_tail_flush=False,
    )
    third = official_prefix_snapshot(
        tokenizer,
        text,
        chunk_id=2,
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        is_tail_flush=False,
    )
    assert first.fixed_prefix == ""
    assert first.rollback_token_ids == tuple(range(7))
    assert third.fixed_prefix == "a|b"
    assert third.rollback_text == "c|d|e|f|g"


def test_tail_flush_runs_and_current_ctc_candidate_affects_same_step() -> None:
    backend = FakeBackend(["hello", "hello", "hello hot word"])
    detector_calls: list[int] = []

    def detector(audio: list[float], active: tuple[str, ...]) -> list[StreamingCandidate]:
        detector_calls.append(len(audio))
        assert active == ("hot", "wrong")
        if len(audio) > 64_000:
            return [StreamingCandidate("hot", "hot word", 0.9, 0.0, 0.9)]
        return []

    result, timeline = run_streaming_sample(
        backend=backend,
        waveform=[0.0] * 72_000,
        sample=_sample(timing=True),
        group="D",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=detector,
        prompt_template="Reference only: {hotwords}",
    )
    assert detector_calls == [32_000, 64_000, 72_000]
    assert backend.finish_calls == 1
    assert timeline[-1]["is_tail_flush"] is True
    assert timeline[-1]["prompt_effective_chunk"] == 2
    assert backend.contexts[-1] == "Reference only: hot word"
    assert result["matched_expected_hotword_ids"] == ["hot"]


def test_chunk_timeline_records_detector_retrieval_and_end_to_end_timing() -> None:
    class TimedDetector:
        def __init__(self) -> None:
            self.last_timing: dict[str, object] = {}

        def __call__(
            self, _audio: list[float], _active: tuple[str, ...]
        ) -> list[StreamingCandidate]:
            self.last_timing = {
                "ctc_encoder_seconds": 0.02,
                "retrieval_seconds": 0.06,
                "retrieval_backend": "anchor_guided",
            }
            return [StreamingCandidate("hot", "hot word", 0.9, 0.0, 0.9)]

    result, timeline = run_streaming_sample(
        backend=FakeBackend(["hello hot word"]),
        waveform=[0.0] * 16_000,
        sample=_sample(),
        group="D",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=TimedDetector(),
        prompt_template="Reference only: {hotwords}",
    )
    timing = timeline[0]["compute_timing"]
    assert isinstance(timing, dict)
    assert timing["retrieval_backend"] == "anchor_guided"
    assert timing["retrieval_seconds"] == pytest.approx(0.06)
    assert timing["retrieval_over_50ms"] is True
    assert timing["step_total_seconds"] >= 0
    totals = result["compute_timing_totals"]
    assert isinstance(totals, dict)
    assert totals["retrieval_seconds"] == pytest.approx(0.06)
    assert result["real_time_factor"] is not None


def test_explicit_asr_language_overrides_manifest_language() -> None:
    backend = FakeBackend(["olá"])
    sample = StreamingSample(
        case_id="case-pt",
        sample_id="sample-pt",
        reference_text="olá",
        language="pt-BR",
        expected_hotword_ids=(),
        expected_surfaces=(),
        active_hotword_ids=("hot",),
    )
    run_streaming_sample(
        backend=backend,
        waveform=[0.0] * 32_000,
        sample=sample,
        group="C",
        hotword_surfaces={"hot": "palavra"},
        ctc_detector=None,
        prompt_template="Reference only: {hotwords}",
        asr_language="Portuguese",
    )
    assert backend.languages == ["Portuguese"]


def test_empty_audio_finishes_without_creating_a_fake_chunk() -> None:
    backend = FakeBackend([])
    result, timeline = run_streaming_sample(
        backend=backend,
        waveform=[],
        sample=_sample(expected=False),
        group="C",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=None,
        prompt_template="Reference only: {hotwords}",
    )
    assert timeline == ()
    assert backend.finish_calls == 1
    assert result["prediction"] == ""


def test_group_c_never_runs_ctc_and_group_e_is_oracle_only() -> None:
    def forbidden(_audio: list[float], _active: tuple[str, ...]) -> list[StreamingCandidate]:
        raise AssertionError("future/CTC data leaked into a non-CTC group")

    c_backend = FakeBackend(["hello"])
    _, c_timeline = run_streaming_sample(
        backend=c_backend,
        waveform=[0.0] * 16_000,
        sample=_sample(),
        group="C",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=forbidden,
        prompt_template="Reference only: {hotwords}",
    )
    assert c_timeline[0]["ctc_top_k"] == []
    assert c_timeline[0]["injected_hotword_ids"] == []

    e_backend = FakeBackend(["hello hot word"])
    _, e_timeline = run_streaming_sample(
        backend=e_backend,
        waveform=[0.0] * 16_000,
        sample=_sample(),
        group="E",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=forbidden,
        prompt_template="Reference only: {hotwords}",
    )
    assert e_timeline[0]["injected_hotword_ids"] == ["hot"]
    assert "wrong" not in e_timeline[0]["injected_hotword_ids"]


@pytest.mark.parametrize(
    ("start", "end", "duration", "expected"),
    [
        (0.5, 1.0, 6.0, "chunk_middle"),
        (1.5, 2.0, 6.0, "boundary_before"),
        (1.8, 2.2, 6.0, "cross_boundary"),
        (2.0, 2.4, 6.0, "boundary_after"),
        (1.5, 4.2, 6.0, "long_multi_chunk"),
        (4.2, 4.8, 5.0, "tail_flush"),
    ],
)
def test_boundary_buckets(start: float, end: float, duration: float, expected: str) -> None:
    assert (
        boundary_bucket(
            hotword_start_sec=start,
            hotword_end_sec=end,
            audio_duration_sec=duration,
        )
        == expected
    )


def test_cross_boundary_failure_uses_timeline_evidence() -> None:
    backend = FakeBackend(["hello", "hello", "hello"])

    def detector(_audio: list[float], _active: tuple[str, ...]) -> list[StreamingCandidate]:
        return []

    result, _ = run_streaming_sample(
        backend=backend,
        waveform=[0.0] * 72_000,
        sample=_sample(timing=True),
        group="D",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=detector,
        prompt_template="Reference only: {hotwords}",
    )
    assert result["failure_reason"] == "boundary_specific_ctc_miss"


def test_correct_before_injection_is_not_negative_correction_latency() -> None:
    backend = FakeBackend(["hello hot word", "hello hot word", "hello hot word"])

    def detector(audio: list[float], _active: tuple[str, ...]) -> list[StreamingCandidate]:
        if len(audio) > 64_000:
            return [StreamingCandidate("hot", "hot word", 0.9, 0.0, 0.9)]
        return []

    result, _ = run_streaming_sample(
        backend=backend,
        waveform=[0.0] * 72_000,
        sample=_sample(),
        group="D",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=detector,
        prompt_template="Reference only: {hotwords}",
    )
    metric = result["hotword_metrics"][0]
    assert metric["correct_before_first_injection"] is True
    assert metric["chunks_from_injection_to_first_correct"] is None


def test_tail_failure_requires_confirmed_tail_hotword_timing() -> None:
    without_timing, _ = run_streaming_sample(
        backend=FakeBackend(["hello", "hello", "hello"]),
        waveform=[0.0] * 72_000,
        sample=_sample(),
        group="C",
        hotword_surfaces={"hot": "hot word", "wrong": "cold word"},
        ctc_detector=None,
        prompt_template="Reference only: {hotwords}",
    )
    assert without_timing["failure_reason"] == "streaming_baseline_regression"

    tail_sample = StreamingSample(
        case_id="tail-case",
        sample_id="tail-sample",
        reference_text="hello hot word",
        language="English",
        expected_hotword_ids=("hot",),
        expected_surfaces=("hot word",),
        active_hotword_ids=("hot",),
        timings=(HotwordTiming("hot", 4.2, 4.8, "manual_confirmed"),),
        boundary_bucket="tail_flush",
    )
    with_timing, _ = run_streaming_sample(
        backend=FakeBackend(["hello", "hello", "hello"]),
        waveform=[0.0] * 80_000,
        sample=tail_sample,
        group="C",
        hotword_surfaces={"hot": "hot word"},
        ctc_detector=None,
        prompt_template="Reference only: {hotwords}",
    )
    assert with_timing["failure_reason"] == "tail_flush_failure"
