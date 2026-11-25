"""Exercise TLS and password-free shared-operation API without model dependencies."""

import json
import os
import socket
import ssl
import sys
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from qwen_hotword.demo_v2 import server, session
from qwen_hotword.demo_v2.https import certificate_context


def test_tls_shared_console_without_password_and_recording_lock(monkeypatch, tmp_path):
    directory = tmp_path / "outputs/demo_tls"
    certificate_context(directory, "127.0.0.1")
    original = (directory / "127.0.0.1.crt").read_bytes()
    certificate_context(directory, "127.0.0.1")
    assert (directory / "127.0.0.1.crt").read_bytes() == original
    assert (directory / "demo-ca.key").stat().st_mode & 0o077 == 0
    assert (directory.parent / "demo_ca.crt").read_bytes() == (
        directory / "demo-ca.crt"
    ).read_bytes()
    with pytest.raises(ValueError, match="一起提供"):
        certificate_context(directory, "127.0.0.1", directory / "127.0.0.1.crt")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    static = tmp_path / "examples/zh_rag_demo_v2"
    static.mkdir(parents=True)
    (static / "index.html").write_text("demo")
    demo = MagicMock()
    demo.clips, demo.checked, demo.extras = [], set(), {"新热词": object()}
    demo.entries = [SimpleNamespace(surface=w) for w in ("北京", "新热词", "北京")]
    demo.checkpoint = "checkpoint"
    demo.snapshot.return_value = {"selected": -1}
    monkeypatch.setattr(session, "SharedStudio", lambda *_: demo)

    def fork(output):
        child = MagicMock()
        child.entries, child.extras = demo.entries.copy(), demo.extras.copy()
        child.snapshot.return_value = {"selected": -1}
        return child

    demo.fork_session.side_effect = fork
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DEMO_DECODER_PY", sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo",
            "--https",
            "--bind",
            "127.0.0.1",
            "--public-host",
            "127.0.0.1",
            "--port",
            str(port),
            "--gpu",
            "5",
            "--max-active-sessions",
            "2",
        ],
    )
    ready = threading.Event()
    servers, errors = [], []
    real_server = server.ThreadingHTTPServer

    class ObservedServer(real_server):
        def serve_forever(self):
            servers.append(self)
            ready.set()
            super().serve_forever(poll_interval=0.05)

    monkeypatch.setattr(server, "ThreadingHTTPServer", ObservedServer)

    def run():
        try:
            server.main()
        except Exception as exc:
            errors.append(exc)
            ready.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert ready.wait(10), errors
    assert not errors
    context = ssl.create_default_context(cafile=str(directory / "demo-ca.crt"))
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=context)
    )

    client_key = ""

    def request(path, payload=None, **headers):
        headers.setdefault("X-Client-Session", client_key)
        req = urllib.request.Request(
            f"https://127.0.0.1:{port}" + path,
            data=json.dumps(payload or {}).encode(),
            headers=headers,
        )
        with opener.open(req, timeout=5) as reply:
            return json.load(reply)

    try:
        with pytest.raises(urllib.error.HTTPError) as no_session:
            request("/state")
        assert no_session.value.code == 401
        client_key = request("/session")["client_session"]
        first_key = client_key
        assert request("/vocabulary") == {"words": ["北京", "新热词"], "extras": ["新热词"]}
        assert request("/state") == {
            "selected": -1,
            "occupied": False,
            "gpu": 5,
            "active_users": 0,
            "max_active_users": 2,
        }
        recording_key = request("/start", {"kind": "mic"})["session"]
        assert request("/state")["occupied"]
        with pytest.raises(urllib.error.HTTPError) as occupied:
            request("/start", {"kind": "mic"})
        assert occupied.value.code == 409
        with pytest.raises(urllib.error.HTTPError) as invalid_session:
            request("/abort", **{"X-Session": "someone-else"})
        assert invalid_session.value.code == 400
        assert request("/state")["occupied"]
        client_key = request("/session")["client_session"]
        assert client_key != first_key
        assert not request("/state")["occupied"]
        second_recording = request("/start", {"kind": "mic"})["session"]
        assert request("/state")["active_users"] == 2
        with pytest.raises(urllib.error.HTTPError) as wrong_recording:
            request("/abort", **{"X-Session": recording_key})
        assert wrong_recording.value.code == 400
        request("/abort", **{"X-Session": second_recording})
        client_key = first_key
        request("/abort", **{"X-Session": recording_key})
        assert not request("/state")["occupied"]
        demo.hunt.assert_not_called()
        demo.select.assert_called_once_with(-1)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
        report = json.loads(
            next(tmp_path.glob("outputs/zh_rag_demo_v2_*/startup.json")).read_text()
        )
        assert report["https"] and report["case_limit"] == 0
        assert report["session_isolation"] and report["max_active_sessions"] == 2
    finally:
        servers[0].shutdown()
        worker.join(5)
    assert not errors
    demo.close.assert_called_once()
