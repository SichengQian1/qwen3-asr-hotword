from __future__ import annotations

import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training import pt_mfa_assets as assets


def _downloads(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("meta.yaml", "version: mock\n")
    payloads = {"acoustic": stream.getvalue(), "dictionary": "olá\to l a\n".encode()}
    monkeypatch.setattr(
        assets,
        "ASSETS",
        {kind: {**spec, "size_bytes": len(payloads[kind])} for kind, spec in assets.ASSETS.items()},
    )
    monkeypatch.setattr(assets.shutil, "which", lambda _: "curl")
    return payloads


def test_separate_download_reuses_only_verified_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _downloads(monkeypatch)
    calls = []

    def fake_run(command: list[str], **kwargs: Any) -> None:
        calls.append(command)
        assert command[0] == "curl"
        assert "--insecure" not in command and "-k" not in command
        assert "--proto-redir" in command
        assert kwargs["check"] is True
        kind = next(k for k, spec in assets.ASSETS.items() if spec["url"] == command[-1])
        Path(command[command.index("--output") + 1]).write_bytes(payloads[kind])

    monkeypatch.setattr(assets.subprocess, "run", fake_run)
    target = tmp_path / "assets"
    assert assets.download_assets(target)["status"] == "downloaded"
    before = {p: p.read_bytes() for p in target.iterdir()}
    assert assets.download_assets(target)["status"] == "verified_existing"
    assert len(calls) == 2
    assert before == {p: p.read_bytes() for p in target.iterdir()}
    assert list(tmp_path.iterdir()) == [target]
    receipt = target / "assets.json"
    data = json.loads(receipt.read_text())
    data["assets"]["acoustic"]["url"] = "https://untrusted.invalid/model"
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata mismatch"):
        assets.download_assets(target)
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["ssl", "truncated", "invalid_zip"])
def test_download_failure_preserves_partial_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    payloads = _downloads(monkeypatch)

    def fail(command: list[str], **kwargs: Any) -> None:
        path = Path(command[command.index("--output") + 1])
        if failure == "ssl":
            path.write_bytes(b"partial")
            raise subprocess.CalledProcessError(60, command)
        path.write_bytes(b"bad" if failure == "truncated" else b"x" * len(payloads["acoustic"]))

    monkeypatch.setattr(assets.subprocess, "run", fail)
    target = tmp_path / "assets"
    for _ in range(2):
        with pytest.raises((subprocess.CalledProcessError, ValueError, zipfile.BadZipFile)):
            assets.download_assets(target)
    assert not target.exists()
    assert len(list(tmp_path.glob("assets_download_*"))) == 2
    assert all(p.stat().st_size > 0 for p in tmp_path.glob("*/portuguese_mfa.zip"))


def test_existing_incomplete_directory_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "assets"
    target.mkdir()
    marker = target / "keep"
    marker.write_text("user file")
    calls = []
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="Missing"):
        assets.download_assets(target)
    assert marker.read_text() == "user file"
    assert calls == []
