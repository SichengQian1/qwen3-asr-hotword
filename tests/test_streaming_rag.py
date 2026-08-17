from __future__ import annotations

from pathlib import Path

import pytest

from qwen_hotword.inference.streaming_core import StreamingSample
from qwen_hotword.inference.streaming_rag import (
    _build_latency_summary,
    _build_summary,
    _collect_shards,
    _prepare_output,
    _write_shard,
)


def _sample() -> StreamingSample:
    return StreamingSample(
        case_id="case-1",
        sample_id="sample-1",
        reference_text="hello hot word",
        language="English",
        expected_hotword_ids=("hot",),
        expected_surfaces=("hot word",),
        active_hotword_ids=("hot",),
        audio_path="/ignored/audio.wav",
    )


def test_sample_shards_are_resumable_and_collect_deterministically(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = {"schema_version": 1, "chunk_size_sec": 2.0}
    _prepare_output(output, config, resume=False)
    sample = _sample()
    result = {
        "case_id": sample.case_id,
        "sample_id": sample.sample_id,
        "experiment_group": "C",
        "reference_text": sample.reference_text,
        "prediction": "hello",
        "expected_hotword_ids": ["hot"],
        "expected_hotwords": ["hot word"],
        "matched_expected_hotword_ids": [],
        "injected_hotword_ids": [],
        "injected_hotwords": [],
        "boundary_bucket": None,
        "hotword_metrics": [],
        "partial_modification_count": 1,
        "inference_seconds": 0.5,
        "failure_reason": "unknown_requires_review",
    }
    timeline = (
        {
            "case_id": sample.case_id,
            "experiment_group": "C",
            "chunk_id": 0,
        },
    )
    shard_dir = output / "sample_shards"
    shard_dir.mkdir()
    _write_shard(shard_dir, "C", sample.case_id, result, timeline)

    first = _collect_shards(shard_dir, ("C",), (sample,))
    _prepare_output(output, config, resume=True)
    second = _collect_shards(shard_dir, ("C",), (sample,))
    assert first == second
    with pytest.raises(ValueError, match="config differs"):
        _prepare_output(output, {"schema_version": 2}, resume=True)


def test_summary_and_latency_keep_missing_timestamps_explicit() -> None:
    rows = [
        {
            "case_id": "case-1",
            "sample_id": "sample-1",
            "experiment_group": "D",
            "reference_text": "hello hot word",
            "prediction": "hello hot word",
            "expected_hotword_ids": ["hot"],
            "expected_hotwords": ["hot word"],
            "matched_expected_hotword_ids": ["hot"],
            "injected_hotword_ids": ["hot"],
            "injected_hotwords": ["hot word"],
            "boundary_bucket": None,
            "hotword_metrics": [
                {
                    "ctc_first_detect_latency_sec": None,
                    "first_correct_latency_sec": None,
                    "stabilization_latency_sec": None,
                    "chunks_from_injection_to_first_correct": 0,
                    "mutable_at_first_detect": True,
                }
            ],
            "partial_modification_count": 1,
            "inference_seconds": 0.5,
            "failure_reason": None,
        }
    ]
    summary = _build_summary(rows)
    assert summary["groups"]["D"]["hotword_exact_recall"] == 1.0
    latency = _build_latency_summary(rows)
    assert latency["groups"]["D"]["ctc_first_detect_latency_sec"]["count"] == 0
    assert latency["groups"]["D"]["chunks_from_injection_to_first_correct"]["median"] == 0.0
