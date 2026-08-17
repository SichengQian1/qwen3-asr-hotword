from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.inference import streaming_boundary
from qwen_hotword.inference.streaming_boundary import build_streaming_boundary_manifest


def _write_spec(path: Path, audio: Path, *, complete: bool) -> None:
    common = {
        "audio_path": str(audio),
        "reference_text": "reference text",
        "language": "Portuguese",
        "active_hotword_ids": ["hot-1", "hot-2"],
    }
    rows = [
        {
            **common,
            "case_id": "single",
            "sample_id": "sample-single",
            "expected_hotword_ids": ["hot-1"],
            "coverage_tags": ["multiword_phrase"],
            "hotword_timings": [
                {
                    "hotword_id": "hot-1",
                    "start_sec": 4.0,
                    "end_sec": 4.5,
                    "timing_source": "manual_confirmed",
                }
            ],
        }
    ]
    if complete:
        rows.extend(
            [
                {
                    **common,
                    "case_id": "long",
                    "sample_id": "sample-long",
                    "expected_hotword_ids": ["hot-1"],
                    "coverage_tags": [],
                    "hotword_timings": [
                        {
                            "hotword_id": "hot-1",
                            "start_sec": 0.5,
                            "end_sec": 3.0,
                            "timing_source": "forced_alignment",
                        }
                    ],
                },
                {
                    **common,
                    "case_id": "multiple",
                    "sample_id": "sample-multiple",
                    "expected_hotword_ids": ["hot-1", "hot-2"],
                    "coverage_tags": [],
                    "hotword_timings": [
                        {
                            "hotword_id": "hot-1",
                            "start_sec": 1.0,
                            "end_sec": 1.3,
                            "timing_source": "manual_confirmed",
                        },
                        {
                            "hotword_id": "hot-2",
                            "start_sec": 3.0,
                            "end_sec": 3.4,
                            "timing_source": "manual_confirmed",
                        },
                    ],
                },
                {
                    **common,
                    "case_id": "negative",
                    "sample_id": "sample-negative",
                    "expected_hotword_ids": [],
                    "coverage_tags": [],
                    "hotword_timings": [],
                },
            ]
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_boundary_builder_is_dynamic_reproducible_and_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"not read because duration is mocked")
    source = tmp_path / "source.jsonl"
    _write_spec(source, audio, complete=True)
    monkeypatch.setattr(streaming_boundary, "_audio_duration", lambda _path: 5.0)

    summary = build_streaming_boundary_manifest(source, tmp_path / "output")
    assert summary["status"] == "pass"
    assert summary["original_audio_overwritten"] is False
    assert audio.read_bytes() == b"not read because duration is mocked"
    coverage = summary["coverage_counts"]
    assert isinstance(coverage, dict)
    assert set(streaming_boundary.REQUIRED_COVERAGE).issubset(coverage)

    manifest = (tmp_path / "output" / "boundary_cases.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in manifest.splitlines()]
    assert all(row["audio_path"] == str(audio) for row in rows)
    assert all(row["leading_silence_sec"] >= 0 for row in rows)
    assert any(row["boundary_bucket"] == "tail_flush" for row in rows)
    assert any(timing["start_sec"] >= 4.0 for row in rows for timing in row["hotword_timings"])


def test_boundary_builder_rejects_estimated_timing_and_incomplete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"x")
    source = tmp_path / "source.jsonl"
    _write_spec(source, audio, complete=False)
    monkeypatch.setattr(streaming_boundary, "_audio_duration", lambda _path: 5.0)
    with pytest.raises(ValueError, match="required coverage"):
        build_streaming_boundary_manifest(source, tmp_path / "incomplete")

    raw = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    raw["hotword_timings"][0]["timing_source"] = "estimated"
    source.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timing_source"):
        build_streaming_boundary_manifest(
            source,
            tmp_path / "estimated",
            require_complete_coverage=False,
        )
