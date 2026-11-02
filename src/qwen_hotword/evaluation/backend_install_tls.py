"""Check installed CA bundles before resuming the unchanged backend comparison."""

from __future__ import annotations

import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.training.stage_progress import stage_progress

CA_VARIABLES = ("PIP_CERT", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")
# Official PyPI 0.0.6 metadata URL; HEAD only, not an installation alternative.
QWEN_WHEEL_PROBE = (
    "https://files.pythonhosted.org/packages/01/12/"
    "d3027a7e4dc2eea0b12a4bf8414a7109f055004e1771666e01d8859d3ca0/"
    "qwen_asr-0.0.6-py3-none-any.whl"
)


def candidates(environ: Mapping[str, str], explicit: Path | None) -> list[Path]:
    """Only consider existing trust stores; never trust a certificate from the peer."""
    if explicit is not None:
        paths = [explicit]
    else:
        defaults = ssl.get_default_verify_paths()
        paths = [Path(environ[key]) for key in CA_VARIABLES if environ.get(key)]
        paths += [
            Path("/etc/ssl/certs/ca-certificates.crt"),
            Path("/etc/pki/tls/certs/ca-bundle.crt"),
            Path(sys.prefix) / "ssl/cert.pem",
        ]
        if defaults.cafile:
            paths.append(Path(defaults.cafile))
    result: list[Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved.is_file() and resolved not in result:
            result.append(resolved)
    if explicit is not None and not result:
        raise ValueError("--ca-bundle must name an existing readable PEM certificate bundle")
    return result


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("refusing a non-HTTPS package redirect")
        # HEAD follows the same asset redirect without downloading the large wheel.
        if req.get_method() == "HEAD":
            return urllib.request.Request(newurl, method="HEAD", headers=dict(req.headers))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_endpoint(bundle: Path, url: str, environ: Mapping[str, str]) -> str:
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("package checks require HTTPS URLs")
    try:
        context = ssl.create_default_context(cafile=str(bundle))
        handlers: list[Any] = [urllib.request.HTTPSHandler(context=context), HTTPSRedirects()]
        if environ.get("PIP_PROXY"):
            handlers.append(
                urllib.request.ProxyHandler(
                    {"http": environ["PIP_PROXY"], "https": environ["PIP_PROXY"]}
                )
            )
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "qwen-ca-check/1"}
        )
        with opener.open(request, timeout=10) as response:
            return f"verified_https_http_{response.status}"
    except urllib.error.HTTPError as error:
        # Authentication, 404, etc. remain failures: TLS alone is not download availability.
        code = error.code
        error.close()
        return f"http_error_{code}"
    except (urllib.error.URLError, ssl.SSLError, OSError, ValueError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, ssl.SSLCertVerificationError):
            return "certificate_verification_failed"
        # Never print raw errors/redirects/proxies that might contain credentials.
        return f"connection_error_{type(reason).__name__}"


def choose_bundle(
    bundles: list[Path],
    endpoints: Mapping[str, str],
    environ: Mapping[str, str],
    probe: Callable[[Path, str, Mapping[str, str]], str] = check_endpoint,
) -> Path:
    if not bundles:
        raise ValueError("no installed CA bundle found; obtain the work-zone trusted CA PEM bundle")
    with stage_progress("Check installed CA bundles", total=len(bundles), every=1) as tick:
        for number, bundle in enumerate(bundles, 1):
            print(f"Check CA bundle {number}/{len(bundles)}: {bundle}", flush=True)
            passed = True
            for label, url in endpoints.items():
                outcome = probe(bundle, url, environ)
                print(f"  {label}: {outcome}", flush=True)
                if not outcome.startswith("verified_https_http_"):
                    passed = False
                    break
            tick(number)
            if passed:
                return bundle
    raise ValueError(
        "no installed CA bundle passed all download checks; ask the work-zone administrator for "
        "a trusted CA PEM bundle and rerun with --ca-bundle /actual/path/company-ca-bundle.pem. "
        "HTTP/connection errors can also require proxy/access repair; see checks above. "
        "No model was loaded and no certificate verification was disabled."
    )


def launch(root: Path, bundle: Path, gpu: int, environ: Mapping[str, str]) -> int:
    env = dict(environ)
    for key in CA_VARIABLES:
        env[key] = str(bundle)
    # The original runner validates its existing contract; this launcher does not
    # edit it, reinstall a ready environment, or alter any inference setting.
    return subprocess.call(
        [
            sys.executable,
            "-B",
            str(root / "scripts/compare_asr_backends.py"),
            "all",
            "--gpu",
            str(gpu),
        ],
        env=env,
        cwd=root,
    )


def main(root: Path) -> int:
    import argparse

    from qwen_hotword.evaluation.backend_comparison import read

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ca-bundle", type=Path, help="trusted PEM bundle supplied by your work zone"
    )
    parser.add_argument(
        "--run", action="store_true", help="resume original test after CA checks pass"
    )
    parser.add_argument("--gpu", type=int, help="allocated physical H200 number (needed for --run)")
    args = parser.parse_args()
    if args.run and (args.gpu is None or args.gpu < 0):
        parser.error("--run requires --gpu N")
    cfg = read(root / "configs/asr_backend_comparison_v023.workzone.json")
    endpoints = {
        "vLLM release (including redirect)": cfg["vllm_wheel"].split("#", 1)[0],
        "PyPI index": "https://pypi.org/simple/qwen-asr/",
        "PyPI wheel host": QWEN_WHEEL_PROBE,
        "PyTorch CUDA index": cfg["torch_index"].rstrip("/") + "/torch/",
    }
    try:
        bundle = choose_bundle(candidates(os.environ, args.ca_bundle), endpoints, os.environ)
    except (OSError, ValueError) as error:
        print(f"CA check stopped: {error}", flush=True)
        return 1
    print(f"Trusted CA checks passed: {bundle}", flush=True)
    if args.run:
        return launch(root, bundle, args.gpu, os.environ)
    print(
        "Checks only; no packages installed or GPU used. Add --run --gpu 4 to continue.", flush=True
    )
    return 0
