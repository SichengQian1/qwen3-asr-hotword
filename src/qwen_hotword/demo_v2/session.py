"""Live operator session; automatic case hunting does not generate operator events."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from qwen_hotword.demo.runtime import CHUNK, TABLE_SHA, Demo
from qwen_hotword.demo_v2.runtime import Studio


class SharedStudio(Studio):
    def fork_session(self, output: Path) -> SharedStudio:
        """Share read-only model weights; never share mutable stream or index state."""
        child = copy.copy(self)
        child.out = output
        output.mkdir(parents=True, exist_ok=False)
        child.det = copy.copy(self.det)
        child.init_studio()
        child.states = []
        child.pcm = bytearray()
        child.seq = 0
        child.configure("", {})
        return child

    def init_studio(self) -> None:
        self.selected = -1
        self.mic_pcm = b""
        self.history = []
        self.base_words = {e.surface for e in self.base}
        self._mic_path: Path | None = None
        self._mic_sha = ""
        self._mic_bytes = 0
        self._record_file: BinaryIO | None = None

    def abort_pass(self) -> None:
        if self._record_file is not None:
            self._record_file.close()
            self._record_file = None

    def close(self) -> None:
        if hasattr(self, "_record_file"):
            self.abort_pass()
        super().close()

    def replay_size(self) -> int:
        return (
            len(self.pcm_for(self.clips[self.selected])) if self.selected >= 0 else self._mic_bytes
        )

    def replay_chunk(self, offset: int) -> bytes:
        if offset < 0 or offset % 2:
            raise ValueError("无效的重播音频偏移")
        if self.selected >= 0:
            return self.pcm_for(self.clips[self.selected])[offset : offset + CHUNK]
        if self._mic_path is None:
            raise ValueError("请先录制并定稿一段音频")
        with self._mic_path.open("rb") as stream:
            stream.seek(offset)
            return stream.read(CHUNK)

    def begin_pass(self, kind: str) -> None:
        if kind not in ("replay", "mic") or (kind == "mic" and self.selected != -1):
            raise ValueError("音频来源与当前模式不符")
        self.abort_pass()
        if kind == "replay":
            if not self.replay_size():
                raise ValueError("请先录制并定稿一段音频")
            self.expected = (
                hashlib.sha256(self.pcm_for(self.clips[self.selected])).hexdigest()
                if self.selected >= 0
                else self._mic_sha
            )
        else:
            self.expected = ""
            folder = self.out / "recordings"
            folder.mkdir(exist_ok=True)
            self._record_path = folder / (uuid4().hex + ".s16le")
            self._record_file = self._record_path.open("xb")
        self._stream_hash = hashlib.sha256()
        self._stream_bytes = 0
        self._segments: dict[str, list[str]] = {"left": [], "right": []}
        self._segment_full = False
        self._last_result: dict[str, Any] = {}
        self.ever: set[str] = set()
        Demo.start(self)

    def select(self, i: int) -> None:
        if not -1 <= i < len(self.clips):
            raise ValueError("未知样例")
        self.configure(self.clips[i]["word"] if i >= 0 else "", self.extras)
        self.selected = i

    def log_event(self, kind: str, **fields: Any) -> None:
        c = self.clips[self.selected] if self.selected >= 0 else {}
        event = dict(
            kind=kind,
            time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=c.get("demo_source", "自由麦克风"),
            sample=c.get("sample_id", ""),
            target=c.get("word", ""),
            table=dict(
                base=TABLE_SHA,
                size=len({e.surface for e in self.entries}),
                omitted=self.omit,
                extras={w: e.pronunciation for w, e in self.extras.items()},
            ),
            **fields,
        )
        with (self.out / "session_history.jsonl").open("a", encoding="utf-8") as f:
            print(json.dumps(event, ensure_ascii=False), file=f)
        self.history.append(event)
        self.history = self.history[-100:]

    def register(self, word: str) -> dict[str, Any]:
        if word in self.extras:
            raise ValueError("此词已在本场词表中")
        result = self.add(word)
        self.log_event("add", word=word)
        return result

    def remove(self, word: str) -> dict[str, Any]:
        if word not in self.extras:
            raise ValueError("此词不在本场新增列表")
        self.configure(self.omit, {w: e for w, e in self.extras.items() if w != word})
        background = word != self.omit and word in self.base_words
        self.log_event("remove", word=word, background=background)
        return {"background": background}

    def record_step(self, data: bytes, final: bool) -> dict[str, Any]:
        if len(data) % 2 or len(data) > CHUNK or (not final and len(data) != CHUNK):
            raise ValueError("无效的音频块")
        if self._segment_full and not data and final:
            # The user may stop exactly on a segment boundary; do not finalize twice.
            result = dict(self._last_result)
            self.seq += 1
        else:
            if self._segment_full:
                seq = self.seq
                Demo.start(self)
                self.seq = seq  # HTTP chunk ordering spans ALL internal segments.
                self._segment_full = False
            segment_final = final or len(self.pcm) + len(data) == CHUNK * 15
            result = Demo.step(self, data, segment_final)
            if segment_final:
                for side in ("left", "right"):
                    self._segments[side].append(result[side])
                self._segment_full = True
            for side in ("left", "right"):
                parts = self._segments[side] + ([] if segment_final else [result[side]])
                result[side] = "\n".join(p for p in parts if p)
            self._last_result = dict(result)
        self._stream_bytes += len(data)
        self._stream_hash.update(data)
        if self._record_file is not None:
            self._record_file.write(data)
        result.update(
            seconds=self._stream_bytes / 32000,
            final=final,
            completed_segments=len(self._segments["left"]),
        )
        self.ever.update(c["word"] for c in result["candidates"])
        if final:
            key = self._stream_hash.hexdigest()
            if self.expected and key != self.expected:
                raise ValueError("重播音频与原声不一致")
            if not self.expected:
                self.abort_pass()
                self._mic_path = self._record_path
                self._mic_sha, self._mic_bytes = key, self._stream_bytes
            self.log_event(
                "test",
                audio=key,
                left=result["left"],
                right=result["right"],
                prompted=sorted(self.ever),
                seconds=self._stream_bytes / 32000,
                segments=len(self._segments["left"]),
                recording=str(self._mic_path) if self.selected < 0 else None,
            )
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "overrides": [w for w in self.extras if w != self.omit and w in self.base_words],
            "replayable": self.selected >= 0 or self._mic_path is not None,
            "replay_bytes": self.replay_size(),
        }
