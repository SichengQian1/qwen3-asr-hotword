"""Visible activity and actual record counts without guessing completion times."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


@contextmanager
def stage_progress(
    label: str, *, total: int | None = None, every: int = 25000, interval: float = 10
) -> Iterator[Callable[[int], None]]:
    if (total is not None and total < 0) or every <= 0 or interval <= 0:
        raise ValueError("invalid progress count or interval")
    started = time.monotonic()
    stopped = threading.Event()
    completed = 0

    def show(status: str) -> None:
        count = "" if total is None and not completed else f" {completed:,}"
        if total is not None:
            fraction = min(1.0, completed / total) if total else 0.0
            filled = int(fraction * 20)
            bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
            count = f" {bar}{count}/{total:,} ({fraction:.1%})"
        print(f"{label}:{count}; {status}; elapsed={time.monotonic() - started:.1f}s", flush=True)

    def update(value: int) -> None:
        nonlocal completed
        completed = value
        if value and (value % every == 0 or value == total):
            show("active")

    def heartbeat() -> None:
        while not stopped.wait(interval):
            show("active")

    show("started")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    status = "completed"
    try:
        yield update
    except BaseException:
        status = "interrupted or failed"
        raise
    finally:
        stopped.set()
        thread.join()
        show(status)
