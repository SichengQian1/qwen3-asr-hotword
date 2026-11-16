from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training import verified_audio_hashes as audio
from qwen_hotword.training.en_egy_dynamic import read, save_rows, select_seconds
from scripts import run_en_egy_480 as recovery


def record(path, duration, **extra):
    return {"audio_path": str(path), "duration_seconds": duration, "ids": [72, 73], **extra}


def test_legacy_microsecond_rounding_accepts_actual_header_and_persists_it(tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    actual = 3.1234375
    header = Mock(return_value=SimpleNamespace(duration_seconds=actual))
    monkeypatch.setattr(audio, "read_audio_metadata", header)
    # An already completed old hash must not trigger another full WAV read.
    stat = path.stat()
    with sqlite3.connect(tmp_path / "audio_hashes.sqlite") as db:
        db.execute("CREATE TABLE hashes(path TEXT PRIMARY KEY,size INT,mtime INT,sha TEXT)")
        db.execute(
            "INSERT INTO hashes VALUES(?,?,?,?)",
            (str(path), stat.st_size, stat.st_mtime_ns, sha(path)),
        )
    monkeypatch.setattr(audio, "sha", Mock(side_effect=AssertionError("must reuse old hash")))
    row = record(path, round(actual, 6))
    audio.fingerprint([row], tmp_path, 2)
    assert row["duration_seconds"] == actual
    assert "audio_review_issue" not in row
    assert read(tmp_path / "audio_duration_report.json")["row_counts"] == {"rounding_only": 1}
    second = record(path, round(actual, 6))
    audio.fingerprint([second], tmp_path, 2)
    assert header.call_count == 1
    assert second["duration_seconds"] == actual


@pytest.mark.parametrize("delta", [0.000001, 0.001, 1.0])
def test_real_duration_mismatch_is_recorded_and_quarantined_not_relaxed(
    tmp_path, monkeypatch, delta
):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"audio")
    monkeypatch.setattr(
        audio, "read_audio_metadata", lambda _: SimpleNamespace(duration_seconds=5 + delta)
    )
    row = record(path, 5.0)
    audio.fingerprint([row], tmp_path, 1)
    assert row["audio_review_issue"] == "audio_duration_mismatch"
    assert row["audio_sha256"] == sha(path)  # retain protection even for rejected recordings
    example = read(tmp_path / "audio_duration_report.json")["examples"][0]
    assert example["recorded_seconds"] == 5.0 and example["actual_seconds"] == 5 + delta


def test_same_duration_changed_bytes_cannot_reuse_old_encoder_identity(tmp_path, monkeypatch):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"changed")
    monkeypatch.setattr(
        audio, "read_audio_metadata", lambda _: SimpleNamespace(duration_seconds=5.0)
    )
    row = record(path, 5.0, audio_sha256="original")
    audio.fingerprint([row], tmp_path, 1)
    assert row["audio_review_issue"] == "audio_bytes_changed"
    assert row["expected_audio_sha256"] == "original"


def test_ceiling_never_overshoots_and_keeps_complete_recordings():
    rows = [{"audio_sha256": str(i), "duration_seconds": 7.7} for i in range(100)]
    chosen = select_seconds(rows, 40, 1, ceiling=True)
    assert 0 < sum(r["duration_seconds"] for r in chosen) <= 40
    assert len({r["audio_sha256"] for r in chosen}) == len(chosen)


def test_recovery_import_reuses_labels_and_db_without_english_g2p(tmp_path, monkeypatch):
    previous, output = tmp_path / "old", tmp_path / "new"
    previous.mkdir()
    write_json(previous / "labels/report.json", {"status": "automatic_labels_completed"})
    write_json(previous / "admission_report.json", {})
    write_json(previous / "contract.json", {})
    save_rows(previous / "protected.jsonl", [{"audio_path": "/protected.wav"}])
    helper = tmp_path / "src/qwen_hotword/training/verified_audio_hashes.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# fixture")
    base_row = {
        "source_group": "en_egy",
        "source": "old-egy.json",
        "source_row": 4,
        "phoneme_token_ids": [2, 3],
        "duration_seconds": 8.0,
        "audio_path": "/en.wav",
    }
    save_rows(tmp_path / "baseline/train.jsonl", [base_row])
    save_rows(tmp_path / "baseline/val.jsonl", [dict(base_row, audio_path="/val.wav")])
    source = {
        "identity": {},
        "plan": {
            "manifests": {
                "en/train": {"path": str(tmp_path / "baseline/train.jsonl")},
                "en/validation": {"path": str(tmp_path / "baseline/val.jsonl")},
            }
        },
    }
    monkeypatch.setattr(recovery, "completed_run", lambda _: source)
    monkeypatch.setattr(
        recovery,
        "verify_previous",
        lambda *a: {
            "configuration": {"source_run": "baseline"},
            "source_identity": {},
            "files": {},
        },
    )
    with sqlite3.connect(previous / "audio_hashes.sqlite") as db:
        db.execute("CREATE TABLE hashes(path TEXT PRIMARY KEY,size INT,mtime INT,sha TEXT)")
        db.execute("INSERT INTO hashes VALUES ('/cached.wav',1,2,'digest')")
    original_db = sha(previous / "audio_hashes.sqlite")
    cfg, result = recovery.prepare(tmp_path, previous, output)
    assert cfg["maximum_hours_per_language"] == 480
    assert (output / "labels").resolve() == (previous / "labels").resolve()
    assert read(output / "english_labels_report.json")["new_G2P_run"] is False
    assert sha(previous / "audio_hashes.sqlite") == original_db
    with sqlite3.connect(output / "audio_hashes.sqlite") as db:
        assert db.execute("SELECT sha FROM hashes").fetchone()[0] == "digest"
        db.execute("INSERT INTO hashes VALUES ('/new.wav',3,4,'new-digest')")
    assert recovery.prepare(tmp_path, previous, output) == (cfg, result)
    with sqlite3.connect(output / "audio_hashes.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM hashes").fetchone()[0] == 2
    assert sha(previous / "audio_hashes.sqlite") == original_db


def test_unreviewed_old_code_rejected_before_import(tmp_path):
    previous = tmp_path / "old"
    path = tmp_path / "src/qwen_hotword/training/en_egy_dynamic.py"
    write_json(previous / "contract.json", {"files": {str(path): "unknown-version"}})
    with pytest.raises(ValueError, match="unreviewed"):
        recovery.verify_previous(tmp_path, previous)
