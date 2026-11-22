"""Live operator session; automatic case hunting does not generate operator events."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from qwen_hotword.demo.runtime import TABLE_SHA, Demo
from qwen_hotword.demo_v2.runtime import Studio


class SharedStudio(Studio):
    def init_studio(self) -> None:
        self.selected = -1
        self.mic_pcm = b""
        self.history = []
        self.base_words = {e.surface for e in self.base}

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
        result = Demo.step(self, data, final)
        self.ever.update(c["word"] for c in result["candidates"])
        if final:
            key = hashlib.sha256(self.pcm).hexdigest()
            if self.expected and key != self.expected:
                raise ValueError("重播音频与原声不一致")
            if not self.expected:
                self.mic_pcm = bytes(self.pcm)
            self.log_event(
                "test",
                audio=key,
                left=result["left"],
                right=result["right"],
                prompted=sorted(self.ever),
            )
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "overrides": [w for w in self.extras if w != self.omit and w in self.base_words],
        }
