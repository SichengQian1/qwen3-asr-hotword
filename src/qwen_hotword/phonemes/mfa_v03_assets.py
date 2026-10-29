"""Pinned G2P-only foundation assets and an isolated MFA runtime, never old assets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from qwen_hotword.training.stage_progress import stage_progress


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def pin(path: Path, expected: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"missing/changed pinned input: {path}")


def command(args: list[str], log: Path, label: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with stage_progress(label), log.open("a", encoding="utf-8") as stream:
        print("Log: " + str(log), flush=True)
        result = subprocess.run(
            args,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "MFA_ROOT_DIR": str(log.parent / "mfa_work")},
        )
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{label} failed ({result.returncode}); {log}\n{tail}")


def _fetch(url: str, target: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=90) as response, target.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        return
    except (urllib.error.URLError, OSError) as error:
        print(f"Python HTTPS download failed ({error}); trying host curl trust", flush=True)
        if target.exists():
            target.rename(target.with_name(target.name + ".python_failed"))
    curl = shutil.which("curl")
    if not curl:
        raise ValueError("curl required for TLS-verified, pinned asset downloads")
    subprocess.run(
        [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "2",
            "--connect-timeout",
            "30",
            "--max-time",
            "900",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--output",
            str(target),
            url,
        ],
        check=True,
        timeout=2800,
    )


def download(
    config: dict[str, Any],
    directory: Path,
    *,
    fetch: Callable[[str, Path], None] = _fetch,
    archive: Path | None = None,
) -> dict[str, Any]:
    """Reuse only exact hash matches; preserve corrupt files and failed partials."""
    if set(config["models"]) != {"en", "es", "pt"}:
        raise ValueError("exactly en/es/pt models required")
    spec_sha = hashlib.sha256(json.dumps(config["models"], sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "asset_spec.json"
    if marker.exists() and json.loads(marker.read_text())["spec_sha256"] != spec_sha:
        raise ValueError("asset specification differs; choose a new asset directory")
    if not marker.exists():
        if any(directory.iterdir()):
            raise ValueError("refusing unowned nonempty asset directory")
        write_json(marker, {"spec_sha256": spec_sha, "models": config["models"]})
    result: dict[str, Any] = {}
    total = sum(len(v["files"]) for v in config["models"].values())
    with stage_progress(
        "Download/verify fixed MFA G2P, lexicon and rules", total=total, every=1
    ) as up:
        completed = 0
        for lang, model in config["models"].items():
            if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
                raise ValueError("immutable full HF commit required")
            for relative, identity in model["files"].items():
                rel = Path(relative)
                if rel.is_absolute() or ".." in rel.parts or rel.parts[0] == "acoustic":
                    raise ValueError("only safe G2P/lexicon relative paths allowed")
                target = directory / lang / rel
                if target.exists():
                    pin(target, identity["sha256"])
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_name(target.name + ".part_" + uuid.uuid4().hex[:10])
                    url = (
                        f"https://huggingface.co/{model['repo']}/resolve/"
                        f"{model['revision']}/{relative}"
                    )
                    print(f"{lang}: {relative} ({identity['bytes']:,} bytes)", flush=True)
                    if archive is None:
                        fetch(url, partial)
                    else:
                        member = f"{lang}/{relative}"
                        with zipfile.ZipFile(archive) as offline:
                            info = offline.getinfo(member)
                            if info.file_size != identity["bytes"]:
                                raise ValueError("offline archive member size differs")
                            with offline.open(info) as source, partial.open("wb") as stream:
                                shutil.copyfileobj(source, stream, length=1024 * 1024)
                    pin(partial, identity["sha256"])
                    if partial.stat().st_size != identity["bytes"]:
                        raise ValueError(f"asset size mismatch: {partial}")
                    partial.rename(target)
                if target.stat().st_size != identity["bytes"]:
                    raise ValueError(f"asset size mismatch: {target}")
                result[f"{lang}/{relative}"] = identity
                completed += 1
                up(completed)
    receipt = {"status": "verified", "spec_sha256": spec_sha, "files": result}
    write_json(directory / "verified.json", receipt)
    return receipt


def environment(config: dict[str, Any], assets: Path, logs: Path) -> list[str]:
    """Clone the existing binary-capable aligner env, modify the new env only."""
    spec = config["environment"]
    name, base, version = spec["name"], spec["base"], spec["mfa_version"]
    if name == base or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", s) for s in (name, base)):
        raise ValueError("a distinct, safe new Conda environment name required")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise ValueError("Conda required; run inside the work-zone Conda shell")
    query = subprocess.run(
        [conda, "env", "list", "--json"], capture_output=True, text=True, check=True
    )
    names = {Path(p).name for p in json.loads(query.stdout)["envs"]}
    marker = assets / "runtime_environment.json"
    identity = {"name": name, "base": base, "mfa_version": version}
    owned = marker.exists() and json.loads(marker.read_text()) == identity
    prefix = [conda, "run", "--no-capture-output", "-n", name, "python"]
    probe = [
        "-c",
        "import importlib.metadata as m; import pynini, pywrapfst; "
        "from montreal_forced_aligner.g2p.generator import PhonetisaurusRewriter; "
        "print(m.version('montreal-forced-aligner'))",
    ]
    if name in names:
        checked = subprocess.run(prefix + probe, capture_output=True, text=True, check=False)
        if checked.returncode == 0 and checked.stdout.strip().splitlines()[-1:] == [version]:
            return prefix
        if not owned:
            raise ValueError(f"existing {name} has another MFA version; refusing to modify it")
    else:
        if base not in names:
            raise ValueError(f"base Conda environment not found: {base}")
        write_json(marker, identity)
        command(
            [conda, "create", "--yes", "--name", name, "--clone", base],
            logs / "environment.log",
            f"Clone {base} into isolated {name}",
        )
    command(
        [
            conda,
            "install",
            "--yes",
            "--name",
            name,
            "--channel",
            "conda-forge",
            f"montreal-forced-aligner={version}",
        ],
        logs / "environment.log",
        f"Install MFA {version} in {name} only",
    )
    checked = subprocess.run(prefix + probe, capture_output=True, text=True, check=True)
    if checked.stdout.strip().splitlines()[-1:] != [version]:
        raise ValueError("isolated MFA runtime version verification failed")
    return prefix


def run_worker(
    root: Path,
    config: dict[str, Any],
    assets: Path,
    prefix: list[str],
    request: Path,
    output: Path,
) -> dict[str, Any]:
    """Do not reuse partial output from a failed process."""
    marker = output.with_suffix(".completed.json")
    identity = {
        "request_sha256": sha(request),
        "asset_receipt_sha256": sha(assets / "verified.json"),
        "runtime_command": prefix,
    }
    if marker.exists():
        done = json.loads(marker.read_text())
        if done["identity"] != identity:
            raise ValueError("G2P request/assets changed; use a new output directory")
        pin(output, done["output_sha256"])
        return cast(dict[str, Any], done["report"])
    temporary = output.with_name(output.name + ".attempt_" + uuid.uuid4().hex[:8])
    command(
        prefix
        + [
            str(root / "scripts/prepare_mfa_v03.py"),
            "worker",
            "--config",
            str(root / config["_config_path"]),
            "--assets-dir",
            str(assets),
            "--request",
            str(request),
            "--worker-output",
            str(temporary),
        ],
        output.with_suffix(".log"),
        "Run real CPU MFA G2P (no Qwen, no GPU)",
    )
    report: dict[str, Any] = json.loads(
        temporary.with_suffix(temporary.suffix + ".report.json").read_text()
    )
    temporary.replace(output)
    write_json(marker, {"identity": identity, "output_sha256": sha(output), "report": report})
    return report
