from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.io import pinned_download as module


class Response(io.BytesIO):
    def __init__(self, data: bytes, status: int = 200, content_range: str = "") -> None:
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_resumes_existing_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.pending").write_bytes(b"abc")

    def open_response(request: Any, **kwargs: Any) -> Response:
        assert request.get_header("Range") == "bytes=3-5"
        assert kwargs["context"].verify_mode == 0
        return Response(b"def", 206, "bytes 3-5/6")

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert target.read_bytes() == b"abcdef"
    assert not target.with_suffix(".bin.pending").exists()


def test_short_response_continues_instead_of_hashing_half_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def open_response(request: Any, **kwargs: Any) -> Response:
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            return Response(b"abc", 206, "bytes 0-5/6")
        return Response(b"def", 206, "bytes 3-5/6")

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    target = tmp_path / "model.bin"
    module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert calls == ["bytes=0-5", "bytes=3-5"]


def test_network_error_retries_from_written_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Interrupted(Response):
        def read(self, size: int = -1) -> bytes:
            if self.tell():
                raise OSError("connection closed")
            return super().read(size)

    calls = []

    def open_response(request: Any, **kwargs: Any) -> Response:
        calls.append(request.get_header("Range"))
        if len(calls) == 1:
            return Interrupted(b"abc", 206, "bytes 0-5/6")
        return Response(b"def", 206, "bytes 3-5/6")

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    target = tmp_path / "model.bin"
    module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert target.read_bytes() == b"abcdef"
    assert calls == ["bytes=0-5", "bytes=3-5"]


def test_ignored_range_does_not_append_full_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.pending").write_bytes(b"abc")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: Response(b"abcdef"))
    module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert target.read_bytes() == b"abcdef"


def test_wrong_range_refused_without_corrupting_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.bin"
    pending = target.with_suffix(".bin.pending")
    pending.write_bytes(b"abc")
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: Response(b"abc", 206, "bytes 0-2/6")
    )
    with pytest.raises(ValueError, match="Content-Range"):
        module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert pending.read_bytes() == b"abc"
    assert not target.exists()


def test_bad_checksum_never_promoted_and_restarts_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.bin"
    calls = []

    def open_response(*args: Any, **kwargs: Any) -> Response:
        calls.append(1)
        return Response(b"xxxxxx")

    monkeypatch.setattr(module.urllib.request, "urlopen", open_response)
    with pytest.raises(ValueError, match="pinned input"):
        module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert len(calls) == 2
    assert not target.exists()


def test_verified_pending_promoted_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "model.bin"
    target.with_suffix(".bin.pending").write_bytes(b"abcdef")

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("should not use network")

    monkeypatch.setattr(module.urllib.request, "urlopen", unexpected)
    module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    module.download("https://example.test/asset", target, digest(b"abcdef"), 6)
    assert target.read_bytes() == b"abcdef"
