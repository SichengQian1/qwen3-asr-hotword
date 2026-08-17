from __future__ import annotations

import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ACUTE_TRANSLATION = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
    }
)


def diagnose_spanish_mfa_audit(
    audit_dir: str | Path,
    *,
    max_items: int = 30,
) -> dict[str, Any]:
    root = Path(audit_dir).expanduser()
    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"MFA audit summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dictionary_path = Path(str(summary["dictionary_path"])).expanduser()
    dictionary_words = _read_dictionary_words(dictionary_path)
    missing = _read_word_counts(root / "missing_words.tsv")

    categories: Counter[str] = Counter()
    acute_recoverable: list[tuple[str, int, str]] = []
    all_marks_recoverable: list[tuple[str, int, str]] = []
    unrecovered: list[tuple[str, int]] = []
    for word, count in missing:
        if any(character in word for character in "áéíóú"):
            categories["acute_accent"] += 1
        if "ñ" in word:
            categories["enye"] += 1
        if "ü" in word:
            categories["diaeresis"] += 1

        acute_candidate = word.translate(ACUTE_TRANSLATION)
        all_marks_candidate = strip_all_combining_marks(word)
        if acute_candidate in dictionary_words:
            acute_recoverable.append((word, count, acute_candidate))
        elif all_marks_candidate in dictionary_words:
            all_marks_recoverable.append((word, count, all_marks_candidate))
        else:
            unrecovered.append((word, count))

    oov_units = []
    for row in _read_tsv(root / "oov_phone_counts.tsv"):
        unit = row["oov_unit"]
        oov_units.append(
            {
                "unit": unit,
                "unicode": [
                    {
                        "codepoint": f"U+{ord(character):04X}",
                        "name": unicodedata.name(character, "UNKNOWN"),
                    }
                    for character in unit
                ],
                "dictionary_count": int(row["dictionary_count"]),
                "corpus_weighted_count": int(row["corpus_weighted_count"]),
            }
        )

    oov_pronunciations = _read_tsv(root / "words_with_oov_phones.tsv")[:max_items]
    extra_words = [row["word"] for row in _read_tsv(root / "extra_dictionary_words.tsv")]
    report: dict[str, Any] = {
        "audit_dir": str(root),
        "dictionary_path": str(dictionary_path),
        "missing_unique_words": len(missing),
        "missing_corpus_tokens": sum(count for _, count in missing),
        "missing_categories": dict(sorted(categories.items())),
        "acute_recoverable_unique_words": len(acute_recoverable),
        "acute_recoverable_corpus_tokens": sum(count for _, count, _ in acute_recoverable),
        "all_marks_recoverable_unique_words": len(all_marks_recoverable),
        "all_marks_recoverable_corpus_tokens": sum(
            count for _, count, _ in all_marks_recoverable
        ),
        "unrecovered_unique_words": len(unrecovered),
        "unrecovered_corpus_tokens": sum(count for _, count in unrecovered),
        "top_acute_recoverable": _top_recoverable(acute_recoverable, max_items),
        "top_all_marks_recoverable": _top_recoverable(
            all_marks_recoverable, max_items
        ),
        "top_unrecovered": [
            {"word": word, "corpus_count": count}
            for word, count in _sorted_by_count(unrecovered)[:max_items]
        ],
        "oov_units": oov_units,
        "top_pronunciations_with_oov": oov_pronunciations,
        "extra_dictionary_words": len(extra_words),
        "top_extra_dictionary_words": extra_words[:max_items],
        "status": "pass",
    }
    (root / "spanish_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def strip_all_combining_marks(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", text)
        if unicodedata.category(character) != "Mn"
    )


def _read_dictionary_words(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"MFA dictionary does not exist: {path}")
    words: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"invalid MFA dictionary line {line_number}: expected WORD PHONES"
                )
            words.add(parts[0])
    return words


def _read_word_counts(path: Path) -> list[tuple[str, int]]:
    return [
        (row["word"], int(row["corpus_count"]))
        for row in _read_tsv(path)
        if row["word"]
    ]


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required diagnostic TSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def _top_recoverable(
    rows: list[tuple[str, int, str]], max_items: int
) -> list[dict[str, str | int]]:
    return [
        {"word": word, "corpus_count": count, "dictionary_candidate": candidate}
        for word, count, candidate in sorted(rows, key=lambda row: (-row[1], row[0]))[
            :max_items
        ]
    ]


def _sorted_by_count(rows: list[tuple[str, int]]) -> list[tuple[str, int]]:
    return sorted(rows, key=lambda row: (-row[1], row[0]))
