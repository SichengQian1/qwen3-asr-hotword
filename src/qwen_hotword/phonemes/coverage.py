from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPECIAL_TOKENS = {"<blank>", "<unk>"}
STRESS_MARKS = {"ˈ", "ˌ"}
SKIP_CHARS = {
    " ",
    "\t",
    "\n",
    "\r",
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "¿",
    "¡",
    "'",
    "\"",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "/",
    "|",
    "‖",
}

PHONE_REPLACEMENTS = (
    ("t͡ʃ", "tʃ"),
    ("d͡ʒ", "dʒ"),
    ("t͜ʃ", "tʃ"),
    ("d͜ʒ", "dʒ"),
    ("ʷ", "w"),
    ("aɪ", "aj"),
    ("aʊ", "aw"),
    ("eɪ", "ej"),
    ("oʊ", "ow"),
    ("ɔɪ", "ɔj"),
    ("g", "ɡ"),
)


@dataclass(frozen=True)
class PhonemeVocab:
    """CTC phoneme vocabulary loaded from a JSON config."""

    tokens: tuple[str, ...]
    phone_tokens: tuple[str, ...]
    token_to_id: dict[str, int]


@dataclass(frozen=True)
class TokenizationResult:
    """Result of matching a phonemized string against a phoneme vocabulary."""

    normalized_ipa: str
    tokens: list[str]
    token_ids: list[int]
    oov_units: list[str]


def load_phoneme_vocab(path: str | Path) -> PhonemeVocab:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens = tuple(str(token) for token in value["tokens"])
    token_to_id = {normalization_key(token): index for index, token in enumerate(tokens)}
    phone_tokens = tuple(
        sorted(
            (normalization_key(token) for token in tokens if token not in SPECIAL_TOKENS),
            key=len,
            reverse=True,
        )
    )
    return PhonemeVocab(tokens=tokens, phone_tokens=phone_tokens, token_to_id=token_to_id)


def normalization_key(text: str) -> str:
    """Normalize phones for robust matching of combining marks.

    The v0.2 vocabulary keeps nasal vowels as base-letter plus combining tilde.
    NFD matching avoids false OOVs when a G2P backend emits a precomposed form
    such as "ẽ" instead of "e" + combining tilde.
    """

    return unicodedata.normalize("NFD", text)


def normalize_ipa_for_vocab(ipa: str) -> str:
    normalized = normalization_key(ipa)
    for source, target in PHONE_REPLACEMENTS:
        normalized = normalized.replace(normalization_key(source), normalization_key(target))
    normalized = "".join(char for char in normalized if char not in STRESS_MARKS)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def tokenize_ipa_to_vocab(ipa: str, vocab: PhonemeVocab) -> TokenizationResult:
    normalized = normalize_ipa_for_vocab(ipa)
    tokens: list[str] = []
    token_ids: list[int] = []
    oov_units: list[str] = []

    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char in SKIP_CHARS:
            index += 1
            continue

        matched = None
        for phone in vocab.phone_tokens:
            if normalized.startswith(phone, index):
                matched = phone
                break

        if matched is None:
            oov_units.append(char)
            index += 1
            continue

        token_id = vocab.token_to_id[matched]
        tokens.append(vocab.tokens[token_id])
        token_ids.append(token_id)
        index += len(matched)

    return TokenizationResult(
        normalized_ipa=normalized,
        tokens=tokens,
        token_ids=token_ids,
        oov_units=oov_units,
    )


def espeak_language_code(language: str) -> str:
    lowered = language.lower().replace("_", "-")
    if lowered.startswith("pt"):
        return "pt-br"
    if lowered.startswith("es"):
        return "es"
    if lowered.startswith("en"):
        return "en-us"
    return lowered


def coerce_record_text(row: dict[str, Any], text_column: str | None) -> str | None:
    columns = (text_column,) if text_column else (
        "text",
        "sentence",
        "transcript",
        "transcription",
        "normalized_text",
        "raw_transcription",
    )
    for column in columns:
        value = row.get(column) if column else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def coerce_record_language(
    row: dict[str, Any],
    language_column: str | None,
    default_language: str | None,
) -> str | None:
    columns = (language_column,) if language_column else ("language", "lang", "locale")
    for column in columns:
        value = row.get(column) if column else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_language
