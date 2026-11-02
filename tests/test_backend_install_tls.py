from __future__ import annotations

import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.evaluation import backend_install_tls as tls
from qwen_hotword.evaluation.backend_comparison import write_once


def test_explicit_ca_is_required_to_exist_and_not_replaced(tmp_path: Path) -> None:
    bundle = tmp_path / "company.pem"
    with pytest.raises(ValueError, match="existing readable PEM"):
        tls.candidates({}, bundle)
    bundle.write_text("certificate fixture")
    assert tls.candidates({"PIP_CERT": "/other.pem"}, bundle) == [bundle]
    assert bundle.read_text() == "certificate fixture"


def test_known_environment_bundles_are_ordered_and_deduplicated(tmp_path: Path) -> None:
    first, second = tmp_path / "one.pem", tmp_path / "two.pem"
    first.touch()
    second.touch()
    result = tls.candidates(
        {"PIP_CERT": str(first), "REQUESTS_CA_BUNDLE": str(first), "SSL_CERT_FILE": str(second)},
        None,
    )
    assert result[:2] == [first, second]
    assert len(result) == len(set(result))


def test_ca_must_pass_every_endpoint_and_does_not_change_environment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    one, two = tmp_path / "one.pem", tmp_path / "two.pem"
    env = {"HTTPS_PROXY": "https://private-user:private-password@example.test"}
    endpoints = {"release": "https://release.test/", "index": "https://index.test/"}
    calls = []

    def probe(bundle: Path, url: str, environ: Any) -> str:
        assert environ == env
        calls.append((bundle, url))
        if bundle == one and "index" in url:
            return "certificate_verification_failed"
        return "verified_https_http_200"

    assert tls.choose_bundle([one, two], endpoints, env, probe) == two
    assert len(calls) == 4
    assert env == {"HTTPS_PROXY": "https://private-user:private-password@example.test"}
    assert "private-password" not in capsys.readouterr().out


@pytest.mark.parametrize("bundles", [[], [Path("/fixture/company.pem")]])
def test_no_valid_ca_stops_with_actionable_error(bundles: list[Path]) -> None:
    with pytest.raises(ValueError, match="trusted CA PEM bundle"):
        tls.choose_bundle(
            bundles, {"release": "https://release.test"}, {}, lambda *a: "http_error_403"
        )


def test_https_redirect_keeps_head_and_rejects_downgrade() -> None:
    redirect = tls.HTTPSRedirects()
    request = urllib.request.Request("https://github.test/asset", method="HEAD")
    updated = redirect.redirect_request(request, None, 302, "Found", {}, "https://cdn.test/wheel")
    assert updated.get_method() == "HEAD" and updated.full_url == "https://cdn.test/wheel"
    with pytest.raises(ValueError, match="non-HTTPS"):
        redirect.redirect_request(request, None, 302, "Found", {}, "http://cdn.test/wheel")


def test_endpoint_uses_verified_ca_and_reads_no_wheel_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def context(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "strict_context"

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    class Opener:
        def open(self, request: Any, timeout: int) -> Response:
            assert request.get_method() == "HEAD" and timeout == 10
            return Response()  # No read() API: a large wheel must never be downloaded.

    monkeypatch.setattr(ssl, "create_default_context", context)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    assert tls.check_endpoint(Path("/fixture/trusted.pem"), "https://github.test/asset", {}) == (
        "verified_https_http_200"
    )
    assert seen == {"cafile": "/fixture/trusted.pem"}


def test_certificate_failure_is_distinct_and_sensitive_error_text_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def bad_context(**kwargs: Any) -> None:
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "private-proxy-token"))

    monkeypatch.setattr(ssl, "create_default_context", bad_context)
    assert tls.check_endpoint(Path("/fixture/company.pem"), "https://github.test/", {}) == (
        "certificate_verification_failed"
    )
    with pytest.raises(ValueError, match="HTTPS URLs"):
        tls.check_endpoint(Path("/fixture/company.pem"), "http://github.test/", {})


def test_launch_changes_only_child_trust_variables_and_keeps_existing_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = tmp_path / "outputs/asr_backend_comparison_v023_shared_v2/run_contract.json"
    write_once(saved, {"original": "identity"})
    before = saved.read_bytes()
    env = {"PIP_CERT": "old", "CUDA_VISIBLE_DEVICES": "4"}

    def call(args: list[str], **kwargs: Any) -> int:
        assert args == [
            sys.executable,
            "-B",
            str(tmp_path / "scripts/compare_asr_backends.py"),
            "all",
            "--gpu",
            "4",
        ]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
        assert all(kwargs["env"][key] == "/fixture/trusted.pem" for key in tls.CA_VARIABLES)
        return 7

    monkeypatch.setattr(subprocess, "call", call)
    assert tls.launch(tmp_path, Path("/fixture/trusted.pem"), 4, env) == 7
    assert saved.read_bytes() == before and env["PIP_CERT"] == "old"


def test_cli_failed_probe_does_not_launch_installation_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_once(
        tmp_path / "configs/asr_backend_comparison_v023.workzone.json",
        {
            "vllm_wheel": "https://release.test/wheel#sha256=hash",
            "torch_index": "https://torch.test",
        },
    )
    monkeypatch.setattr(sys, "argv", ["check_asr_install_tls.py", "--run", "--gpu", "4"])
    monkeypatch.setattr(tls, "candidates", lambda *a: [])
    monkeypatch.setattr(subprocess, "call", lambda *a, **kw: pytest.fail("must not run"))
    assert tls.main(tmp_path) == 1


def test_cli_passes_selected_ca_to_unchanged_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_once(
        tmp_path / "configs/asr_backend_comparison_v023.workzone.json",
        {
            "vllm_wheel": "https://release.test/wheel#sha256=hash",
            "torch_index": "https://torch.test",
        },
    )
    monkeypatch.setattr(sys, "argv", ["check_asr_install_tls.py", "--run", "--gpu", "4"])
    monkeypatch.setattr(tls, "candidates", lambda *a: [Path("/fixture/company.pem")])
    monkeypatch.setattr(tls, "choose_bundle", lambda *a: Path("/fixture/company.pem"))
    monkeypatch.setattr(tls, "launch", lambda root, ca, gpu, env: 0 if gpu == 4 else 1)
    assert tls.main(tmp_path) == 0
