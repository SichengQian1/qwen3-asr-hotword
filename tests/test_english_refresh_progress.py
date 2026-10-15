from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.training.english_refresh_progress import PreparationProgress, render


def write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_unknown_mfa_progress_never_creates_percentage_or_eta_and_does_not_write(tmp_path):
    write(tmp_path / "preparation_contract.json", {})
    (tmp_path / "source").mkdir()
    (tmp_path / "source/sha256.txt").write_text("sealed fixture")
    write(tmp_path / "words/report.json", {"unique_words": 46041})
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monitor = PreparationProgress(tmp_path)
    state = monitor.snapshot()
    assert state["phase"] == 4 and state["progress"] is None
    text = render(state, 0)
    assert "46041" in text and "ETA" not in text
    assert "0%" not in text and "100%" not in text
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_completed_shards_drive_real_label_count_and_bar(tmp_path):
    write(tmp_path / "preparation_contract.json", {})
    write(tmp_path / "source/report.json", {"source_records": 12000})
    (tmp_path / "source/sha256.txt").write_text("sealed fixture")
    write(tmp_path / "words/report.json", {"unique_words": 1000})
    write(tmp_path / "g2p_report.json", {"status": "completed"})
    write(tmp_path / "manifest/shard_reports/shard-00000.json", {"source_records": 5000})
    write(tmp_path / "manifest/shard_reports/shard-00001.json", {"source_records": 5000})
    monitor = PreparationProgress(tmp_path)
    state = monitor.snapshot()
    assert state["phase"] == 5 and state["progress"] == (10000, 12000)
    assert "10,000/12,000 (83.3%)" in render(state, 0)


def test_incremental_line_counter_handles_partial_lines_and_truncation(tmp_path):
    path = tmp_path / "rows.jsonl"
    monitor = PreparationProgress(tmp_path)
    path.write_bytes(b"one\npart")
    assert monitor.line_count(path) == 1
    with path.open("ab") as handle:
        handle.write(b"ial\nthree\n")
    assert monitor.line_count(path) == 3
    assert monitor.line_count(path) == 3
    path.write_bytes(b"new\n")
    assert monitor.line_count(path) == 1


def test_native_mfa_log_is_tailed_without_duplicate_lines_or_ansi(tmp_path):
    log = tmp_path / "mfa_attempt_one/mfa.log"
    log.parent.mkdir()
    log.write_text("loading\n\x1b[32m50%|#####\x1b[0m\r", encoding="utf-8")
    monitor = PreparationProgress(tmp_path)
    assert monitor.log_updates() == ["loading", "50%|#####"]
    assert monitor.log_updates() == []
    with log.open("a") as handle:
        handle.write("finished\n")
    assert monitor.log_updates() == ["finished"]


def test_completion_requires_original_data_prepared_report(tmp_path):
    monitor = PreparationProgress(tmp_path)
    assert monitor.snapshot()["done"] is False
    write(tmp_path / "report.json", {"status": "failed"})
    assert monitor.snapshot()["done"] is False
    write(
        tmp_path / "report.json", {"status": "data_prepared", "selection": {"en": {"hours": 480.1}}}
    )
    state = monitor.snapshot()
    assert state["done"] is True and state["phase"] == 7
    assert "Encoder cache" in state["detail"]


@pytest.fixture
def watcher():
    path = Path(__file__).resolve().parents[1] / "scripts/watch_english_refresh.py"
    spec = importlib.util.spec_from_file_location("watch_english_refresh_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_watch_never_starts_task_or_changes_completed_outputs(
    tmp_path, monkeypatch, watcher
):
    write(tmp_path / "report.json", {"status": "data_prepared"})
    before = (tmp_path / "report.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["watch", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(watcher.subprocess, "Popen", lambda *a, **k: pytest.fail("must not run"))
    assert watcher.main() == 0
    assert (tmp_path / "report.json").read_bytes() == before
    assert len(list(tmp_path.iterdir())) == 1


def test_run_forwards_to_unchanged_prepare_and_preserves_failure_code(
    tmp_path, monkeypatch, watcher
):
    observed = []

    def launch(command, **kwargs):
        observed.append(command)
        return SimpleNamespace(poll=lambda: 3)

    monkeypatch.setattr(watcher.subprocess, "Popen", launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "watch",
            "--run",
            "--resume",
            "--workers",
            "7",
            "--jobs",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert watcher.main() == 3
    command = observed[0]
    assert command[command.index("--workers") + 1] == "7"
    assert command[command.index("--jobs") + 1] == "3"
    assert "prepare" in command and command[-1] == "--resume"
    assert command[3].endswith("scripts/run_english_refresh.py")
