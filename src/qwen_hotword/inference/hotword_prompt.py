from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

DEFAULT_PT_BR_PROMPT_TEMPLATE = (
    "As palavras a seguir podem aparecer no áudio e servem apenas como referência "
    "de grafia. Use-as somente se forem realmente faladas; não as inclua à força "
    "na transcrição: {hotwords}"
)


def normalize_match_words(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return tuple(re.sub(r"\s+", " ", without_punctuation).strip().split())


def strict_phrase_match(text: str, phrase: str) -> bool:
    text_words = normalize_match_words(text)
    phrase_words = normalize_match_words(phrase)
    if not phrase_words or len(phrase_words) > len(text_words):
        return False
    width = len(phrase_words)
    return any(
        text_words[start : start + width] == phrase_words
        for start in range(0, len(text_words) - width + 1)
    )


def strict_matched_surfaces(
    text: str,
    surfaces: Iterable[str],
) -> tuple[str, ...]:
    return tuple(surface for surface in surfaces if strict_phrase_match(text, surface))


def build_hotword_prompt(
    hotwords: Iterable[str],
    *,
    template: str = DEFAULT_PT_BR_PROMPT_TEMPLATE,
) -> str:
    if template.count("{hotwords}") != 1:
        raise ValueError("prompt template must contain exactly one {hotwords} placeholder")
    ordered: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for raw_hotword in hotwords:
        surface = unicodedata.normalize("NFKC", raw_hotword).strip()
        key = normalize_match_words(surface)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(surface)
    if not ordered:
        return ""
    return template.format(hotwords=", ".join(ordered))
