"""Dependency-free request/cache contract shared by the demo and real adapter."""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LANGUAGES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
CONTEXT_TEMPLATES = {
    "en": "The following terms are spelling references. Use them only when supported by the audio. "
    "Do not add a term that was not spoken.\n{words}",
    "es": "Los siguientes términos son referencias de escritura. Úsalos solo si el audio los "
    "respalda. No añadas términos que no se hayan pronunciado.\n{words}",
    "pt": "Os termos abaixo são referências de escrita. Use-os apenas quando houver apoio no "
    "áudio. Não acrescente termos que não foram falados.\n{words}",
}


def build_context(language: str, words: list[str]) -> str:
    """Same text templates as evaluation/recovered_case_run.py; no eval imports."""
    if language not in LANGUAGES or len(words) > 7:
        raise ValueError("use en/es/pt and at most seven words")
    if any(
        not word.strip() or any(marker in word for marker in ("\n", "\r", "<|", "|>", "<asr_text>"))
        for word in words
    ):
        raise ValueError("hotwords must be nonempty plain surfaces without control tokens")
    return CONTEXT_TEMPLATES[language].format(words="\n".join(words)) if words else ""


def output_length(feature_length: int) -> int:
    """Qwen3-ASR vLLM 0.23.0 acoustic frames -> audio embedding rows (NOT CTC rows)."""
    if feature_length <= 0:
        raise ValueError("feature_length must be positive")
    remainder = feature_length % 100
    after_first = (remainder - 1) // 2 + 1
    return (((after_first - 1) // 2 + 1 - 1) // 2 + 1) + feature_length // 100 * 13


def owner_tag(request_id: str) -> int:
    if not request_id:
        raise ValueError("request_id cannot be empty")
    return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:8], "big") >> 1


def build_prompt(context: str, language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError("language must be en, es or pt")
    if any(marker in context for marker in ("<|", "|>", "<asr_text>")):
        raise ValueError("context cannot contain Qwen control tokens")
    system = f"<|im_start|>system\n{context}<|im_end|>\n" if context else ""
    return (
        system
        + "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        + f"<|im_start|>assistant\nlanguage {LANGUAGES[language]}<asr_text>"
    )


@dataclass
class CacheEntry:
    owner: int
    feature_length: int
    embedding_rows: int
    value: Any
    byte_size: int
    expires_at: float
    in_use: bool = False


class AudioCache:
    """Worker-local serial cache. Values stay on their original device.

    Expiry applies before first use. A generation pins an entry until release;
    this sample does not evict an entry that may be in an active request.
    The service MUST serialize prepare/generate/release, not just each RPC.
    """

    def __init__(
        self,
        *,
        max_bytes: int = 32 * 1024 * 1024,
        ttl_seconds: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_bytes <= 0 or ttl_seconds <= 0:
            raise ValueError("cache limit and TTL must be positive")
        self.max_bytes, self.ttl_seconds, self.clock = max_bytes, ttl_seconds, clock
        self.entries: dict[int, CacheEntry] = {}

    def put(self, value: Any, *, owner: int, feature_length: int, byte_size: int) -> int:
        self.reap()
        if byte_size <= 0 or byte_size + self.bytes_used > self.max_bytes:
            raise RuntimeError("audio cache budget exceeded; do not silently re-encode")
        key = secrets.randbits(62) + 1
        while key in self.entries:
            key = secrets.randbits(62) + 1
        self.entries[key] = CacheEntry(
            owner,
            feature_length,
            output_length(feature_length),
            value,
            byte_size,
            self.clock() + self.ttl_seconds,
        )
        return key

    def get(self, key: int, *, owner: int, feature_length: int) -> Any:
        self.reap()
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError("audio cache missing/expired; no automatic Encoder fallback")
        if entry.owner != owner or entry.feature_length != feature_length:
            raise ValueError("audio cache request identity or acoustic length mismatch")
        entry.in_use = True
        return entry.value

    def release(self, key: int, *, owner: int) -> None:
        entry = self.entries.get(key)
        if entry is not None and entry.owner != owner:
            raise ValueError("cannot release another request's audio")
        self.entries.pop(key, None)

    def reap(self) -> None:
        now = self.clock()
        for key, entry in tuple(self.entries.items()):
            if not entry.in_use and entry.expires_at <= now:
                del self.entries[key]

    @property
    def bytes_used(self) -> int:
        return sum(entry.byte_size for entry in self.entries.values())
