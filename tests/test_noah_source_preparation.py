from __future__ import annotations

import csv
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.noah_source_preparation import prepare_noah_source


def _audio(path: Path, frames: int = 1600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * frames)
    return path


def _row(audio: Path, text: str) -> dict[str, Any]:
    response = "language Spanish<asr_text>" + text
    return {
        "audios": [str(audio)],
        "language": "Spanish",
        "response": response,
        "messages": [{"role": "assistant", "content": response}],
    }


def _config(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / "input.json"
    path.write_text(json.dumps(rows, ensure_ascii=False))
    return {
        "input": str(path),
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "audio_prefix_rewrites": {},
    }


def test_quarantines_all_conflicts_and_preserves_complete_partition(tmp_path: Path) -> None:
    batch = tmp_path / "noah_esZ00" / "batch" / "category" / "G0001"
    a, b = _audio(batch / "a.wav"), _audio(batch / "b.wav")
    clean = _audio(batch / "clean.wav", 3200)
    config = _config(
        tmp_path,
        [
            _row(a, "hola"),
            _row(b, "hola"),
            _row(a, "otra transcripción"),
            _row(clean, '¿Cómo estás? "Bien".'),
            _row(batch / "missing.wav", "texto"),
        ],
    )
    before = Path(config["input"]).read_bytes()
    output = tmp_path / "output"
    r = prepare_noah_source(config, output, workers=2)
    assert r["source_record_count"] == 5
    assert r["candidate_records"] == 1
    assert r["review_records"] == 4
    assert r["conflicting_file_hash_groups"] == 1
    assert r["review_by_reason"]["identical_file_conflicting_transcripts"]["records"] == 3
    assert r["candidate_hours"] == pytest.approx(0.2 / 3600)
    assert r["wordlist"]["records_seen"] == 1
    assert not r["training_ready"] and not r["existing_holdout_overlap_checked"]
    with (output / "source.tsv").open() as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["text"] == '¿Cómo estás? "Bien".'
    assert row["speaker_id"] == "" and row["directory_group_hint"] == "G0001"
    assert row["source_split"] == "unsplit"
    assert Path(config["input"]).read_bytes() == before
    for line in (output / "sha256.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_same_file_same_transcript_only_counted_once(tmp_path: Path) -> None:
    batch = tmp_path / "noah_esZ00" / "batch"
    a, b = _audio(batch / "a.wav"), _audio(batch / "b.wav")
    r = prepare_noah_source(
        _config(tmp_path, [_row(a, "hola"), _row(b, "hola")]), tmp_path / "output", workers=1
    )
    assert r["candidate_records"] == 1
    assert r["review_by_reason"]["duplicate_file_same_transcript"]["records"] == 1
    assert r["candidate_hours"] == pytest.approx(0.1 / 3600)


def test_no_candidates_does_not_prepare_wordlist(tmp_path: Path) -> None:
    r = prepare_noah_source(
        _config(tmp_path, [_row(tmp_path / "missing.wav", "hola")]), tmp_path / "output"
    )
    assert r["status"] == "no_candidates" and r["wordlist"] is None
    assert r["review_records"] == r["source_record_count"] == 1


def test_existing_directory_and_wrong_sha_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, [])
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "user.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        prepare_noah_source(config, output)
    assert sentinel.read_text() == "preserve"
    config["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        prepare_noah_source(config, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_parse_failure_preserves_partial_output_with_failed_report(tmp_path: Path) -> None:
    source = tmp_path / "broken.json"
    source.write_text('[{"audios": []},')
    config = {
        "input": str(source),
        "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "audio_prefix_rewrites": {},
    }
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        prepare_noah_source(config, output)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["partial_outputs_preserved"]
    assert (output / "inventory.jsonl").exists()
