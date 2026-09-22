from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.noah_source_inspection import (
    inspect_noah_source,
    iter_json_array,
    resolve_audio_path,
)


def _record(audio: Path, text: str = "Olá español") -> dict[str, Any]:
    response = "language Spanish<asr_text>" + text
    return {
        "audios": [str(audio)],
        "language": "Spanish",
        "response": response,
        "messages": [
            {"role": "user", "content": "<audio>"},
            {"role": "assistant", "content": response},
        ],
    }


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 1600)


def _config(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = tmp_path / "source.json"
    path.write_text(json.dumps(rows, ensure_ascii=False))
    return {
        "input": str(path),
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples_per_corpus": 2,
        "seed": 7,
        "audio_prefix_rewrites": {},
    }


def test_stream_single_line_array_across_tiny_chunks(tmp_path: Path) -> None:
    rows = [_record(tmp_path / "á.wav", 'ñ [é] \\"quoted\\"') for _ in range(3)]
    path = tmp_path / "source.json"
    path.write_text("\ufeff " + json.dumps(rows, ensure_ascii=False) + "\n")
    assert list(iter_json_array(path, chunk_size=3)) == rows


@pytest.mark.parametrize("text", ["[{},]", "[{}", "[{}] garbage", "{}", "[1]", "[{} {}]"])
def test_rejects_malformed_or_unsupported_json(tmp_path: Path, text: str) -> None:
    path = tmp_path / "source.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(iter_json_array(path, chunk_size=2))


def test_explicit_mapping_is_used_even_when_original_exists(tmp_path: Path) -> None:
    original, mapped = tmp_path / "old", tmp_path / "new"
    target = mapped / "a.wav"
    _wav(target)
    prefixes = {str(original) + "/": str(mapped) + "/"}
    assert resolve_audio_path(str(original / "a.wav"), prefixes) == (
        target,
        "explicit_prefix_rewrite",
    )
    _wav(original / "a.wav")
    assert resolve_audio_path(str(original / "a.wav"), prefixes) == (
        target,
        "explicit_prefix_rewrite",
    )


def test_deterministic_per_source_sample_never_certifies_total_hours(tmp_path: Path) -> None:
    rows = []
    for corpus in ("first", "second"):
        for index in range(5):
            audio = tmp_path / "noah_esZ00" / corpus / f"{index}.wav"
            _wav(audio)
            rows.append(_record(audio))
    config = _config(tmp_path, rows)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = inspect_noah_source(config)
    assert result == inspect_noah_source(config)
    assert result["status"] == "inspection_completed"
    assert result["record_count"] == 10
    assert set(result["audio_probe"]) == {"first", "second"}
    assert all(group["readable_samples"] == 2 for group in result["audio_probe"].values())
    assert result["full_corpus_measured_hours"] is None
    assert (
        not result["training_ready"] and not result["latin_american_scope_independently_verified"]
    )
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_flags_duplicate_text_conflicts_and_multiple_audio_without_silent_drop(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "noah_esZ00" / "corpus" / "a.wav"
    _wav(audio)
    multi = _record(audio)
    multi["audios"] = [str(audio), str(audio)]
    mismatch = _record(tmp_path / "missing.wav")
    mismatch["messages"][1]["content"] = "different"
    result = inspect_noah_source(
        _config(tmp_path, [_record(audio), _record(audio, "otro texto"), multi, mismatch])
    )
    assert result["record_count"] == 4
    assert result["issue_counts"]["duplicate_audio_path"] == 1
    assert result["issue_counts"]["duplicate_audio_conflicting_text"] == 1
    assert result["issue_counts"]["not_exactly_one_audio_path"] == 1
    assert result["issue_counts"]["assistant_response_not_identical"] == 1
    assert result["status"] == "inspection_has_flags"


def test_wrong_sha_stops_before_audio_reads(tmp_path: Path) -> None:
    config = _config(tmp_path, [_record(tmp_path / "missing.wav")])
    config["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        inspect_noah_source(config)
