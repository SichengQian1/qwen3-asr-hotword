from __future__ import annotations

import re
import unicodedata

EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
}

PT_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "esta",
    "este",
    "eu",
    "foi",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "um",
    "uma",
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = normalized.replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens: list[str] = []
    current: list[str] = []
    for char in normalized:
        if char.isalpha() or char in {"'", "-"}:
            current.append(char)
            continue
        if current:
            tokens.append("".join(current).strip("'-"))
            current = []
    if current:
        tokens.append("".join(current).strip("'-"))
    return [token for token in tokens if token]


def stopwords_for_language(language: str) -> set[str]:
    if language.lower().startswith("pt"):
        return PT_STOPWORDS
    return EN_STOPWORDS


def contains_token_sequence(tokens: list[str], query: tuple[str, ...]) -> bool:
    if not query or len(query) > len(tokens):
        return False
    width = len(query)
    for index in range(len(tokens) - width + 1):
        if tuple(tokens[index : index + width]) == query:
            return True
    return False


def language_code_for_id(language: str) -> str:
    lowered = language.lower().replace("_", "-")
    if lowered.startswith("pt"):
        return "ptbr"
    if lowered.startswith("en"):
        return "en"
    return re.sub(r"[^a-z0-9]+", "", lowered) or "xx"

