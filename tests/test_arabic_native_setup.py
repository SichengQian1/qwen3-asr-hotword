from __future__ import annotations

import hashlib
import io
import json
import ssl
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts import label_arabic_native as run


def setup_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, verify_ssl: bool
) -> tuple[dict[str, Any], bytes]:
    payload = b"pinned model bytes"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("model.db", payload)
    archive_bytes = buffer.getvalue()
    config = {
        "runtime": "runtime",
        "assets": "data",
        "installation": {
            "verify_ssl": verify_ssl,
            "pip_trusted_hosts": ["pypi.org", "files.pythonhosted.org"],
        },
        "packages": {
            "test-model": {
                "url": "https://example.org/model.zip",
                "destination": "test-model",
                "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                "files": [{"path": "model.db", "sha256": hashlib.sha256(payload).hexdigest()}],
            }
        },
    }
    python = tmp_path / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "CONFIG", config_path)
    return config, archive_bytes


@pytest.mark.parametrize("verify_ssl", [False, True])
def test_setup_scopes_ssl_policy_and_reuses_verified_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verify_ssl: bool
) -> None:
    config, archive_bytes = setup_fixture(tmp_path, monkeypatch, verify_ssl=verify_ssl)
    global_context = ssl._create_default_https_context
    with (
        patch.object(urllib.request, "urlopen", return_value=io.BytesIO(archive_bytes)) as fetch,
        patch.object(subprocess, "run") as execute,
    ):
        run.setup(config)
        context = fetch.call_args.kwargs["context"]
        assert context.verify_mode == (ssl.CERT_REQUIRED if verify_ssl else ssl.CERT_NONE)
        assert context.check_hostname is verify_ssl
        assert ssl._create_default_https_context is global_context
        for call in execute.call_args_list[:2]:
            command = call.args[0]
            if verify_ssl:
                assert "--trusted-host" not in command
            else:
                for host in config["installation"]["pip_trusted_hosts"]:
                    assert command[command.index(host) - 1] == "--trusted-host"
        assert (tmp_path / "data/test-model/model.db").read_bytes() == b"pinned model bytes"
        marker = json.loads((tmp_path / "setup_completed.json").read_text())
        assert marker["installation"] == config["installation"]
        run.setup(config)
        assert fetch.call_count == 1


def test_ssl_bypass_still_rejects_changed_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = setup_fixture(tmp_path, monkeypatch, verify_ssl=False)
    with (
        patch.object(urllib.request, "urlopen", return_value=io.BytesIO(b"changed bytes")),
        patch.object(subprocess, "run"),
        pytest.raises(ValueError, match="missing/changed pinned input"),
    ):
        run.setup(config)
    assert not (tmp_path / "data/test-model.zip").exists()
    assert not (tmp_path / "data/test-model/model.db").exists()
    assert not (tmp_path / "setup_completed.json").exists()
