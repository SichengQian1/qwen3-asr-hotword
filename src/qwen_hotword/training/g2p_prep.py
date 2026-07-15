from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

INTERNAL_CONNECTORS = {"'", "-"}
CONNECTOR_REPLACEMENTS = {
    "‘": "'",
    "’": "'",
    "ʼ": "'",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
}


@dataclass(frozen=True)
class G2pWordlistSummary:
    tsv_path: str
    output_dir: str
    records_seen: int
    records_with_text: int
    empty_text_records: int
    total_word_tokens: int
    unique_words: int
    fragments_with_digits: int
    unique_fragments_with_digits: int
    minimum_word_count: int
    words_path: str
    word_counts_path: str
    character_counts_path: str
    digit_fragments_path: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_training_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    for source, target in CONNECTOR_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_letter_or_mark(character: str) -> bool:
    return unicodedata.category(character)[0] in {"L", "M"}


def extract_word_tokens(text: str) -> list[str]:
    normalized = normalize_training_text(text)
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            token = "".join(current).strip("'-")
            if token:
                tokens.append(token)
            current.clear()

    for index, character in enumerate(normalized):
        if _is_letter_or_mark(character):
            current.append(character)
            continue
        next_character = normalized[index + 1] if index + 1 < len(normalized) else ""
        if (
            character in INTERNAL_CONNECTORS
            and current
            and next_character
            and _is_letter_or_mark(next_character)
        ):
            current.append(character)
            continue
        flush()
    flush()
    return tokens


def digit_fragments(text: str) -> list[str]:
    normalized = normalize_training_text(text)
    return [fragment for fragment in normalized.split() if any(char.isdigit() for char in fragment)]


def prepare_mfa_wordlist(
    tsv_path: str | Path,
    output_dir: str | Path,
    *,
    text_column: str = "text",
    max_records: int = 0,
    minimum_word_count: int = 1,
) -> G2pWordlistSummary:
    path = Path(tsv_path).expanduser()
    destination = Path(output_dir).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TSV does not exist: {path}")
    if max_records < 0:
        raise ValueError("max_records must be non-negative")
    if minimum_word_count <= 0:
        raise ValueError("minimum_word_count must be positive")

    word_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    digit_counts: Counter[str] = Counter()
    records_seen = 0
    records_with_text = 0
    empty_text_records = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        if text_column not in fieldnames:
            raise ValueError(
                f"TSV is missing text column {text_column!r}; found {sorted(fieldnames)}"
            )
        for row in reader:
            if max_records > 0 and records_seen >= max_records:
                break
            records_seen += 1
            text = str(row.get(text_column) or "").strip()
            if not text:
                empty_text_records += 1
                continue
            records_with_text += 1
            normalized = normalize_training_text(text)
            word_counts.update(extract_word_tokens(normalized))
            character_counts.update(char for char in normalized if not char.isspace())
            digit_counts.update(digit_fragments(normalized))

    retained_words = sorted(
        word for word, count in word_counts.items() if count >= minimum_word_count
    )
    destination.mkdir(parents=True, exist_ok=True)
    words_path = destination / "words.txt"
    word_counts_path = destination / "word_counts.tsv"
    character_counts_path = destination / "character_counts.tsv"
    digit_fragments_path = destination / "fragments_with_digits.tsv"

    words_path.write_text(
        "".join(f"{word}\n" for word in retained_words),
        encoding="utf-8",
    )
    _write_counter(word_counts_path, ("word", "count"), word_counts)
    _write_counter(character_counts_path, ("character", "count"), character_counts)
    _write_counter(digit_fragments_path, ("fragment", "count"), digit_counts)

    return G2pWordlistSummary(
        tsv_path=str(path),
        output_dir=str(destination),
        records_seen=records_seen,
        records_with_text=records_with_text,
        empty_text_records=empty_text_records,
        total_word_tokens=sum(word_counts.values()),
        unique_words=len(word_counts),
        fragments_with_digits=sum(digit_counts.values()),
        unique_fragments_with_digits=len(digit_counts),
        minimum_word_count=minimum_word_count,
        words_path=str(words_path),
        word_counts_path=str(word_counts_path),
        character_counts_path=str(character_counts_path),
        digit_fragments_path=str(digit_fragments_path),
        status="pass" if records_seen > 0 and retained_words else "fail",
    )


def _write_counter(
    path: Path,
    header: tuple[str, str],
    counts: Counter[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([value, count])
