"""Bounded CPU subprocesses with visible activity, live logs or written-row counts."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from qwen_hotword.training.stage_progress import stage_progress


def _stop(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def execute(
    command: list[str],
    stdout: Path,
    stderr: Path,
    *,
    label: str,
    timeout: float,
    env: dict[str, str],
    rows_total: int | None = None,
    live: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    offset, written, timed_out = 0, 0, False
    with (
        stdout.open("xb") as out,
        stderr.open("xb") as err,
        stage_progress(label, total=rows_total, every=100) as update,
    ):
        process = subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            with stdout.open("rb") as reader:
                while True:
                    remaining = timeout - (time.monotonic() - started)
                    if remaining <= 0 and process.poll() is None:
                        timed_out = True
                        _stop(process)
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=max(0.001, min(1.0, remaining)))
                    reader.seek(offset)
                    chunk = reader.read()
                    offset += len(chunk)
                    if rows_total is not None and chunk:
                        written += chunk.count(b"\n")
                        update(written)
                    if live and chunk:
                        print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                    if process.poll() is not None:
                        break
        finally:
            if process.poll() is None:
                _stop(process)
    return {
        "command": command,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
        "written_rows": written,
        "stdout": str(stdout),
        "stderr": str(stderr),
    }
