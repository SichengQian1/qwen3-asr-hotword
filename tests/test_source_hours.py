from __future__ import annotations

import io
import json
import tarfile
import wave
from pathlib import Path

import pytest

from qwen_hotword.evaluation.source_hours import (
    archive_hours,
    chinese_hours,
    english_hours,
    header_seconds,
    totals,
)


def wav_bytes(seconds: int = 1, channels: int = 1) -> bytes:
    handle = io.BytesIO()
    with wave.open(handle, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0" * (8000 * seconds * channels * 2))
    return handle.getvalue()


def test_header_duration_does_not_multiply_by_channels() -> None:
    assert header_seconds(io.BytesIO(wav_bytes(2, 2))) == 2
    assert header_seconds(io.BytesIO(wav_bytes(3))) == 3
    with pytest.raises(ValueError):
        header_seconds(io.BytesIO(wav_bytes(0)))


def test_english_counts_raw_rows_and_reports_duplicate_paths(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(wav_bytes(2))
    source = tmp_path / "source.json"
    # Invalid training transcript still belongs to raw source duration inventory.
    source.write_text(json.dumps([{"audios": [str(audio)], "response": ""}] * 2))
    result = english_hours(source, {}, 2)
    assert result["status"] == "complete"
    assert result["rows"] == 2 and result["known_seconds"] == 4
    assert result["duplicate_path_references"] == 1
    assert result["unique_readable_path_seconds"] == 2


def test_english_maps_old_paths_and_does_not_hide_missing(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(wav_bytes(1))
    source = tmp_path / "source.jsonl"
    rows = [{"audios": ["/old/a.wav"]}, {"audios": [str(tmp_path / "missing.wav")]}, {"audios": []}]
    source.write_text("\n".join(json.dumps(row) for row in rows))
    result = english_hours(source, {"/old": str(tmp_path)}, 2)
    assert result["known_seconds"] == 1 and result["readable_references"] == 1
    assert result["status"] == "incomplete" and result["failed_references"] == 2
    assert result["invalid_rows"] == 1


def make_tar(path: Path, compressed: bool = False) -> None:
    with tarfile.open(path, "w:gz" if compressed else "w") as archive:
        for name, payload in [
            ("test/a.wav", wav_bytes(2, 2)),
            ("b.wav", wav_bytes(1)),
            ("b.txt", b"text"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


@pytest.mark.parametrize("compressed", [False, True])
def test_shards_include_holdouts_and_audio_without_labels(tmp_path: Path, compressed: bool) -> None:
    archive = tmp_path / ("a.tar.gz" if compressed else "a.tar")
    make_tar(archive, compressed)
    result = archive_hours(archive)
    assert result["status"] == "complete"
    assert result["audio_references"] == 2 and result["known_seconds"] == 3
    assert list(tmp_path.iterdir()) == [archive]


def test_four_sources_sum_is_not_nominal_directory_hours(tmp_path: Path) -> None:
    results = {}
    for n in range(4):
        root = tmp_path / f"{n}_1000H_shards"
        root.mkdir()
        make_tar(root / "a.tar")
        results[str(n)] = chinese_hours(root, 2)
    result = totals(results, list(results))
    assert result["complete"] is True
    assert result["known_hours"] == pytest.approx(12 / 3600)


def test_bad_tar_and_unexpected_loose_audio_report_incomplete(tmp_path: Path) -> None:
    bad = tmp_path / "bad.tar"
    bad.write_bytes(b"broken")
    result = archive_hours(bad)
    assert result["status"] == "incomplete" and "scan_error" in result
    bad.unlink()
    make_tar(tmp_path / "ok.tar")
    (tmp_path / "extra.wav").write_bytes(wav_bytes())
    result = chinese_hours(tmp_path, 2)
    assert result["status"] == "incomplete" and result["loose_audio_files_not_counted"] == 1
    assert totals({"source": result}, ["source"])["complete"] is False
