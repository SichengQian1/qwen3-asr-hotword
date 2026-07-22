from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import PhonemeVocab


@dataclass(frozen=True)
class HotwordEntry:
    hotword_id: str
    language: str
    surface: str
    normalized: str
    words: tuple[str, ...]
    pronunciation: str
    phoneme_tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    source: str
    validation_occurrences: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["words"] = list(self.words)
        value["phoneme_tokens"] = list(self.phoneme_tokens)
        value["token_ids"] = list(self.token_ids)
        return value


def load_hotword_table(
    path: str | Path,
    *,
    vocab: PhonemeVocab,
    blank_id: int = 0,
) -> list[HotwordEntry]:
    table_path = Path(path).expanduser()
    if not table_path.is_file():
        raise FileNotFoundError(f"hotword table does not exist: {table_path}")
    entries: list[HotwordEntry] = []
    seen_ids: set[str] = set()
    seen_pronunciations: set[tuple[str, tuple[int, ...]]] = set()
    with table_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid hotword JSON at {table_path}:{line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(f"hotword row {line_number} must be an object")
            entry = _entry_from_dict(raw, line_number=line_number, vocab=vocab)
            if entry.hotword_id in seen_ids:
                raise ValueError(f"duplicate hotword ID: {entry.hotword_id}")
            seen_ids.add(entry.hotword_id)
            pronunciation_key = (entry.language, entry.token_ids)
            if pronunciation_key in seen_pronunciations:
                raise ValueError(
                    f"duplicate pronunciation in hotword table: {entry.hotword_id}"
                )
            seen_pronunciations.add(pronunciation_key)
            if any(
                token_id == blank_id or token_id < 0 or token_id >= len(vocab.tokens)
                for token_id in entry.token_ids
            ):
                raise ValueError(f"hotword {entry.hotword_id} has invalid CTC token IDs")
            entries.append(entry)
    if not entries:
        raise ValueError(f"hotword table is empty: {table_path}")
    return entries


def write_hotword_table(path: str | Path, entries: list[HotwordEntry]) -> None:
    if not entries:
        raise ValueError("cannot write an empty hotword table")
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(destination)


def _entry_from_dict(
    raw: dict[str, Any],
    *,
    line_number: int,
    vocab: PhonemeVocab,
) -> HotwordEntry:
    words = _string_tuple(raw.get("words"), "words", line_number)
    phoneme_tokens = _string_tuple(
        raw.get("phoneme_tokens"),
        "phoneme_tokens",
        line_number,
    )
    raw_ids = raw.get("token_ids")
    if not isinstance(raw_ids, list) or not raw_ids or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in raw_ids
    ):
        raise ValueError(f"hotword row {line_number} has invalid token_ids")
    token_ids = tuple(raw_ids)
    if any(token_id < 0 or token_id >= len(vocab.tokens) for token_id in token_ids):
        raise ValueError(f"hotword row {line_number} has out-of-range token_ids")
    expected_tokens = tuple(vocab.tokens[token_id] for token_id in token_ids)
    if phoneme_tokens != expected_tokens:
        raise ValueError(
            f"hotword row {line_number} phoneme tokens do not match token IDs"
        )
    occurrences = raw.get("validation_occurrences")
    if not isinstance(occurrences, int) or isinstance(occurrences, bool) or occurrences <= 0:
        raise ValueError(
            f"hotword row {line_number} has invalid validation_occurrences"
        )
    return HotwordEntry(
        hotword_id=_required_string(raw, "hotword_id", line_number),
        language=_required_string(raw, "language", line_number),
        surface=_required_string(raw, "surface", line_number),
        normalized=_required_string(raw, "normalized", line_number),
        words=words,
        pronunciation=_required_string(raw, "pronunciation", line_number),
        phoneme_tokens=phoneme_tokens,
        token_ids=token_ids,
        source=_required_string(raw, "source", line_number),
        validation_occurrences=occurrences,
    )


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"hotword row {line_number} has invalid {key}")
    return value.strip()


def _string_tuple(value: object, key: str, line_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"hotword row {line_number} has invalid {key}")
    return tuple(value)
