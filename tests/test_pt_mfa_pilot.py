from __future__ import annotations

import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.pt_mfa_pilot import inspect_alignment, select_pilot, sha256_file
from qwen_hotword.training.pt_mfa_runner import run_pilot


def _pool(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    rows = []
    for source in ("cv", "fleurs", "mls", "noah", "finance"):
        for i in range(60):
            rows.append(
                {
                    "id": f"{source}_{i}",
                    "audio_path": str(tmp_path / f"{source}_{i}.wav"),
                    "source_corpus": source,
                    "release_source": (
                        "original_ready" if i < 40 or source == "fleurs" else "temporal_2x_recovery"
                    ),
                    "duration_seconds": 1.0,
                    "text": "olá mundo",
                    "split": "validation",
                    "label_length": 5 + i,
                    "effective_ctc_input_length": 100,
                    "ctc_time_upsampling_factor": 2,
                }
            )
    path = tmp_path / "validation.jsonl"
    _write_pool(path, rows)
    return path, rows


def _write_pool(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_sampling_is_frozen_stratified_and_prediction_independent(tmp_path: Path) -> None:
    path, rows = _pool(tmp_path)
    sources = ["cv", "fleurs", "mls", "noah", "finance"]
    selected, strata = select_pilot(path, sha256_file(path), sources, 20, 20260920)
    assert len(selected) == 100
    assert Counter(r["source_corpus"] for r in selected) == dict.fromkeys(sources, 20)
    assert all(s["selected"] > 0 for s in strata)
    assert all(s["inclusion_fraction"] == s["selected"] / s["population"] for s in strata)
    for row in rows:
        row["per"] = 100
        row["prediction"] = "unused"
    _write_pool(path, list(reversed(rows)))
    again, _ = select_pilot(path, sha256_file(path), list(reversed(sources)), 20, 20260920)
    assert selected == again
    assert all("prediction" not in row and "per" not in row for row in again)
    other, _ = select_pilot(path, sha256_file(path), sources, 20, 42)
    assert {r["id"] for r in other} != {r["id"] for r in selected}


@pytest.mark.parametrize("change", ["split", "duplicate", "nan", "sha", "small"])
def test_invalid_sampling_input_is_rejected(tmp_path: Path, change: str) -> None:
    path, rows = _pool(tmp_path)
    if change == "split":
        rows[0]["split"] = "test"
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "nan":
        rows[0]["duration_seconds"] = float("nan")
    _write_pool(path, rows)
    with pytest.raises(ValueError):
        select_pilot(
            path,
            "wrong" if change == "sha" else sha256_file(path),
            ["cv"],
            100 if change == "small" else 20,
            1,
        )


def _alignment(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "start": 0,
                "end": 1,
                "tiers": {
                    "words": {"type": "interval", "entries": [[0.1, 0.9, "olá"]]},
                    "phones": {
                        "type": "interval",
                        "entries": [
                            [0.1, 0.11, "o"],
                            [0.11, 0.12, "l"],
                            [0.12, 0.5, "a"],
                            [0.5, 0.9, "sil"],
                        ],
                    },
                },
            }
        )
    )


def test_alignment_flags_exclude_silence_and_do_not_claim_label_error(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    assert inspect_alignment(path, 1)["status"] == "missing_alignment"
    _alignment(path)
    result = inspect_alignment(path, 1)
    assert result["phone_count"] == 3
    assert result["minimum_duration_phone_count"] == 2
    assert result["long_phone_count"] == 1
    assert result["flags"] == ["many_minimum_duration_phones", "long_phone"]
    assert "label_error" not in result
    data = json.loads(path.read_text())
    data["tiers"]["phones"]["entries"][0][1] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="timing"):
        inspect_alignment(path, 1)


def test_unsupported_or_incomplete_exports_do_not_pass(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text('{"tiers": []}')
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_alignment(path, 1)
    path.write_text('{"tiers": {}}')
    assert inspect_alignment(path, 1)["status"] == "incomplete_alignment"


def _config(tmp_path: Path) -> dict[str, Any]:
    path, rows = _pool(tmp_path)
    for row in rows:
        Path(row["audio_path"]).write_bytes(b"fake source audio; no model run")
    return {
        "validation_manifest": str(path),
        "validation_sha256": sha256_file(path),
        "sources": ["cv"],
        "samples_per_source": 2,
        "seed": 1,
        "output_parent": "outputs",
        "acoustic_model": "portuguese_mfa",
        "dictionary": "portuguese_brazil_mfa",
        "model_version": "v2.0.0a",
        "jobs": 1,
    }


def test_mocked_pipeline_preserves_sources_and_records_missing_alignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen_hotword.training import pt_mfa_runner as runner

    _local_assets(tmp_path, monkeypatch)
    config = _config(tmp_path)
    root = tmp_path / "pilot"
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    commands: list[list[str]] = []

    def fake_command(args: list[str], log: Path, env: dict[str, str]) -> None:
        commands.append(args)
        assert env["MFA_ROOT_DIR"] == str(root / "mfa_root")
        assert str(root) in env["TMPDIR"]
        log.write_text("mock, not H200 evidence")
        assert "download" not in args
        if args[0] == "ffmpeg":
            with wave.open(args[-1], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x10" * 16000)
        elif args[1] == "align":
            assert "--single_speaker" in args and "--no_overwrite" in args
            assert not any(a in {"train", "adapt", "train_dictionary"} for a in args)
            _alignment(root / "aligned/pt_0000.json")

    monkeypatch.setattr(runner, "run_command", fake_command)
    monkeypatch.setattr(runner.shutil, "which", lambda name: name)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "3.4.0")

    def missing_git(*args: Any, **kwargs: Any) -> None:
        raise FileNotFoundError("git unavailable in container")

    monkeypatch.setattr(runner.subprocess, "run", missing_git)
    asset_root = tmp_path / "models/mfa/pt_pilot_v2_0_0a"
    original_assets = {p: p.read_bytes() for p in asset_root.iterdir()}
    report = run_pilot(config, tmp_path, root)
    assert report["git_commit"] is None
    assert original_assets == {p: p.read_bytes() for p in asset_root.iterdir()}
    assert report["status"] == "completed"
    assert report["counts_by_status"] == {"aligned": 1, "missing_alignment": 1}
    assert report["training_started"] is False and report["labels_changed"] is False
    assert (root / "sha256.txt").is_file()
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    count = len(commands)
    with pytest.raises(FileExistsError):
        run_pilot(config, tmp_path, root)
    assert len(commands) == count


def test_failed_preflight_keeps_compact_report_and_never_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen_hotword.training import pt_mfa_runner as runner

    _local_assets(tmp_path, monkeypatch)
    config = _config(tmp_path)
    root = tmp_path / "failed"
    calls = []

    def failing(args: list[str], log: Path, env: dict[str, str]) -> None:
        calls.append(args)
        raise RuntimeError("MFA dependency failed")

    monkeypatch.setattr(runner, "run_command", failing)
    monkeypatch.setattr(runner.shutil, "which", lambda name: name)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "3.4.0")
    report = run_pilot(config, tmp_path, root)
    assert report["status"] == "failed"
    assert report["mfa_alignment_invoked"] is False
    assert calls == [["mfa", "--help"]]
    assert "MFA dependency failed" in (root / "report.json").read_text()


def _local_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from qwen_hotword.training import pt_mfa_assets as assets

    root = tmp_path / assets.DEFAULT_ASSETS
    root.mkdir(parents=True)
    receipt: dict[str, Any] = {"model_version": assets.VERSION, "assets": {}}
    specs = {}
    for kind, spec in assets.ASSETS.items():
        path = root / str(spec["name"])
        path.write_text("olá\to l a\nmundo\tm u n d o\n")
        specs[kind] = {**spec, "size_bytes": path.stat().st_size}
        receipt["assets"][kind] = {**specs[kind], "sha256": sha256_file(path)}
    (root / "assets.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(assets, "ASSETS", specs)
    return root


def test_missing_or_corrupt_assets_fail_without_external_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen_hotword.training import pt_mfa_runner as runner

    config = _config(tmp_path)
    calls = []
    monkeypatch.setattr(runner, "run_command", lambda *args: calls.append(args))
    report = run_pilot(config, tmp_path, tmp_path / "missing")
    assert report["status"] == "failed"
    assert "download_pt_mfa_assets.py" in report["error"]
    asset_root = _local_assets(tmp_path, monkeypatch)
    dictionary = asset_root / "portuguese_brazil_mfa.dict"
    dictionary.write_bytes(b"x" * dictionary.stat().st_size)
    report = run_pilot(config, tmp_path, tmp_path / "corrupt")
    assert "SHA256 mismatch" in report["error"]
    assert not report["mfa_alignment_invoked"]
    assert calls == []
