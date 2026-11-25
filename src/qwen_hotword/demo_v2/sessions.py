"""Isolated browser sessions sharing one serialized GPU/decoder worker."""

from __future__ import annotations

import json
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from qwen_hotword.demo_v2.session import SharedStudio


class SessionError(ValueError):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


@dataclass
class Client:
    studio: SharedStudio
    lock: Any = field(default_factory=Lock)
    recording: str = ""
    seen: float = field(default_factory=time.monotonic)


class Sessions:
    def __init__(self, model: SharedStudio, max_active: int = 4, max_sessions: int = 64):
        self.model = model
        self.max_active = max_active
        self.max_sessions = max_sessions
        self.clients: dict[str, Client] = {}
        self.lock = Lock()
        # SDK, detector hooks and private decoder socket must never run concurrently.
        self.engine = Lock()
        self.metrics_lock = Lock()
        self.metrics: deque[dict[str, Any]] = deque(maxlen=1000)
        self.steps = 0

    def _expire(self) -> None:
        # Called under registry lock. Never block on an in-flight GPU/MFA request.
        now = time.monotonic()
        for key, client in list(self.clients.items()):
            if not client.lock.acquire(blocking=False):
                continue
            try:
                age = now - client.seen
                if client.recording and age > 120:
                    client.studio.abort_pass()
                    client.studio.states = []
                    client.studio.pcm = bytearray()
                    client.recording = ""
                if age > 1800:
                    client.studio.abort_pass()
                    del self.clients[key]
            finally:
                client.lock.release()

    def create(self) -> str:
        with self.lock:
            self._expire()
            if len(self.clients) >= self.max_sessions:
                raise SessionError("网页会话已满，请关闭不用的页面，稍后重试。", 503)
            key = secrets.token_urlsafe(24)
            output = self.model.out / "sessions" / secrets.token_hex(12)
            self.clients[key] = Client(self.model.fork_session(output))
            return key

    @contextmanager
    def lease(self, key: str) -> Iterator[Client]:
        with self.lock:
            self._expire()
            client = self.clients.get(key)
        if client is None:
            raise SessionError("网页会话已过期，请刷新页面。", 401)
        with client.lock:
            with self.lock:
                if self.clients.get(key) is not client:
                    raise SessionError("网页会话已过期，请刷新页面。", 401)
                client.seen = time.monotonic()
            try:
                yield client
            finally:
                client.seen = time.monotonic()

    def capacity(self) -> dict[str, int]:
        with self.lock:
            return dict(
                active_users=sum(bool(c.recording) for c in self.clients.values()),
                max_active_users=self.max_active,
            )

    def _start(self, client: Client, kind: str) -> dict[str, str]:
        with self.lock:
            if sum(bool(c.recording) for c in self.clients.values()) >= self.max_active:
                raise SessionError(f"同时录音人数已达 {self.max_active} 人，请稍候再开始。")
            client.recording = secrets.token_urlsafe(16)
        try:
            with self.engine:
                client.studio.begin_pass(kind)
        except Exception:
            client.studio.abort_pass()
            client.recording = ""
            raise
        return {"session": client.recording}

    def _measure(self, item: dict[str, Any]) -> None:
        with self.metrics_lock:
            self.steps += 1
            self.metrics.append(item)
            groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in self.metrics:
                groups[row["active_users"]].append(row)
            summary: dict[str, Any] = {}
            for count, rows in groups.items():
                values = sorted(r["queue_seconds"] + r["compute_seconds"] for r in rows)
                summary[str(count)] = dict(
                    steps=len(rows),
                    mean_compute_seconds=sum(r["compute_seconds"] for r in rows) / len(rows),
                    mean_queue_seconds=sum(r["queue_seconds"] for r in rows) / len(rows),
                    p95_step_seconds=values[min(len(values) - 1, int(len(values) * 0.95))],
                    steps_over_2s=sum(v > 2 for v in values),
                )
            report = dict(
                total_steps=self.steps,
                window_steps=len(self.metrics),
                max_active_users=self.max_active,
                chunk_seconds=2,
                by_active_users=summary,
                note="Serial GPU worker; includes tail chunks. Not a capacity guarantee.",
            )
            path = self.model.out / "concurrency_summary.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(path)

    def dispatch(self, key: str, op: str, params: dict[str, Any], data: bytes, headers: Any) -> Any:
        with self.lease(key) as client:
            demo = client.studio
            if op == "/leave":
                demo.abort_pass()
                with self.lock:
                    del self.clients[key]
                return {}
            if op == "/state":
                return dict(demo.snapshot(), occupied=bool(client.recording), **self.capacity())
            if op == "/vocabulary":
                return dict(
                    words=sorted({e.surface for e in demo.entries}), extras=sorted(demo.extras)
                )
            if op == "/clip":
                return demo.replay_chunk(int(params.get("offset", 0)))
            if op in ("/start", "/add", "/remove", "/select") and client.recording:
                raise SessionError("本页面已有音频在识别，请先结束。")
            if op == "/start":
                return self._start(client, str(params["kind"]))
            if op == "/select":
                demo.select(int(params["i"]))
                return {}
            if op == "/add":
                return demo.register(str(params["word"]).strip())
            if op == "/remove":
                return demo.remove(str(params["word"]))
            if op not in ("/step", "/abort"):
                raise SessionError("未知接口", 404)
            if not client.recording or headers.get("X-Session") != client.recording:
                raise SessionError("录音会话已失效", 400)
            if op == "/abort":
                demo.abort_pass()
                demo.states = []
                demo.pcm = bytearray()
                client.recording = ""
                return {}
            if int(headers.get("X-Seq", -1)) != demo.seq:
                raise SessionError("音频块顺序不一致", 400)
            final = headers.get("X-Final") == "1"
            queued = time.monotonic()
            active = self.capacity()["active_users"]
            try:
                with self.engine:
                    started = time.monotonic()
                    result = demo.record_step(data, final)
                    finished = time.monotonic()
            except Exception:
                demo.abort_pass()
                demo.states = []
                demo.pcm = bytearray()
                client.recording = ""
                raise
            if final:
                client.recording = ""
                demo.states = []
                demo.pcm = bytearray()
            result.update(
                queue_seconds=started - queued,
                compute_seconds=finished - started,
                **self.capacity(),
            )
            self._measure(
                dict(
                    active_users=active,
                    **{k: result[k] for k in ("queue_seconds", "compute_seconds")},
                )
            )
            return result

    def close(self) -> None:
        for client in self.clients.values():
            client.studio.abort_pass()
