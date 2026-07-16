from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab


@dataclass(frozen=True)
class MfaDictionaryAuditSummary:
    words_path: str
    dictionary_path: str
    vocab_path: str
    word_counts_path: str | None
    input_unique_words: int
    dictionary_unique_words: int
    dictionary_pronunciations: int
    missing_words: int
    extra_dictionary_words: int
    duplicate_pronunciation_entries: int
    dictionary_word_coverage: float
    corpus_token_coverage: float | None
    words_with_oov_phones: int
    pronunciations_with_oov_phones: int
    oov_phone_units: int
    corpus_weighted_oov_phone_units: int | None
    ctc_output_classes: int
    training_labels_ready: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_words(path: str | Path) -> set[str]:
    word_path = Path(path).expanduser()
    return {
        line.strip()
        for line in word_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }


def load_word_counts(path: str | Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"word", "count"}.issubset(reader.fieldnames):
            raise ValueError("word-count TSV must contain word and count columns")
        for row in reader:
            word = str(row.get("word") or "").strip()
            if word:
                counts[word] = int(row["count"])
    return counts


def load_mfa_dictionary(path: str | Path) -> dict[str, list[str]]:
    pronunciations: dict[str, list[str]] = defaultdict(list)
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"invalid MFA dictionary line {line_number}: expected WORD PHONES"
                )
            word, pronunciation = parts
            pronunciations[word].append(pronunciation)
    return dict(pronunciations)


def audit_mfa_dictionary(
    words_path: str | Path,
    dictionary_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    word_counts_path: str | Path | None = None,
) -> MfaDictionaryAuditSummary:
    words_file = Path(words_path).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (words_file, dictionary_file, vocab_file):
        if not path.is_file():
            raise FileNotFoundError(f"required file does not exist: {path}")

    input_words = load_words(words_file)
    dictionary = load_mfa_dictionary(dictionary_file)
    dictionary_words = set(dictionary)
    word_counts = load_word_counts(word_counts_path) if word_counts_path else None
    vocab = load_phoneme_vocab(vocab_file)

    missing_words = input_words - dictionary_words
    extra_words = dictionary_words - input_words
    duplicate_entries = sum(max(0, len(values) - 1) for values in dictionary.values())
    raw_phone_counts: Counter[str] = Counter()
    normalized_phone_counts: Counter[str] = Counter()
    oov_counts: Counter[str] = Counter()
    weighted_oov_counts: Counter[str] = Counter()
    words_with_oov: list[tuple[str, int, str, str]] = []
    pronunciations_with_oov = 0

    for word, pronunciations in dictionary.items():
        corpus_count = word_counts.get(word, 0) if word_counts is not None else 1
        for pronunciation in pronunciations:
            raw_phone_counts.update(pronunciation.split())
            tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
            normalized_phone_counts.update(tokenized.tokens)
            if tokenized.oov_units:
                pronunciations_with_oov += 1
                oov_counts.update(tokenized.oov_units)
                pronunciation_oov_counts = Counter(tokenized.oov_units)
                weighted_oov_counts.update(
                    {
                        unit: count * corpus_count
                        for unit, count in pronunciation_oov_counts.items()
                    }
                )
                words_with_oov.append(
                    (word, corpus_count, pronunciation, " ".join(tokenized.oov_units))
                )

    dictionary_coverage = (
        1.0 if not input_words else (len(input_words) - len(missing_words)) / len(input_words)
    )
    corpus_coverage: float | None = None
    if word_counts is not None:
        total_tokens = sum(word_counts.get(word, 0) for word in input_words)
        missing_tokens = sum(word_counts.get(word, 0) for word in missing_words)
        corpus_coverage = (
            1.0 if total_tokens == 0 else (total_tokens - missing_tokens) / total_tokens
        )

    destination.mkdir(parents=True, exist_ok=True)
    _write_word_list(destination / "missing_words.tsv", missing_words, word_counts)
    _write_word_list(destination / "extra_dictionary_words.tsv", extra_words, word_counts)
    _write_counter(destination / "raw_mfa_phone_counts.tsv", "phone", raw_phone_counts)
    _write_counter(
        destination / "normalized_vocab_phone_counts.tsv",
        "phone",
        normalized_phone_counts,
    )
    _write_oov_counts(destination / "oov_phone_counts.tsv", oov_counts, weighted_oov_counts)
    _write_words_with_oov(destination / "words_with_oov_phones.tsv", words_with_oov)

    summary = MfaDictionaryAuditSummary(
        words_path=str(words_file),
        dictionary_path=str(dictionary_file),
        vocab_path=str(vocab_file),
        word_counts_path=str(Path(word_counts_path).expanduser()) if word_counts_path else None,
        input_unique_words=len(input_words),
        dictionary_unique_words=len(dictionary_words),
        dictionary_pronunciations=sum(len(values) for values in dictionary.values()),
        missing_words=len(missing_words),
        extra_dictionary_words=len(extra_words),
        duplicate_pronunciation_entries=duplicate_entries,
        dictionary_word_coverage=dictionary_coverage,
        corpus_token_coverage=corpus_coverage,
        words_with_oov_phones=len({row[0] for row in words_with_oov}),
        pronunciations_with_oov_phones=pronunciations_with_oov,
        oov_phone_units=sum(oov_counts.values()),
        corpus_weighted_oov_phone_units=(
            sum(weighted_oov_counts.values()) if word_counts is not None else None
        ),
        ctc_output_classes=len(vocab.tokens),
        training_labels_ready=not missing_words and not oov_counts,
        status="pass",
    )
    (destination / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _write_word_list(
    path: Path,
    words: set[str],
    word_counts: Counter[str] | None,
) -> None:
    counts = word_counts if word_counts is not None else Counter()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["word", "corpus_count"])
        for word in sorted(words, key=lambda value: (-counts[value], value)):
            writer.writerow([word, counts[word] if word_counts is not None else ""])


def _write_counter(path: Path, label: str, counts: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([label, "count"])
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([value, count])


def _write_oov_counts(
    path: Path,
    counts: Counter[str],
    weighted_counts: Counter[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["oov_unit", "dictionary_count", "corpus_weighted_count"])
        for unit, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([unit, count, weighted_counts[unit]])


def _write_words_with_oov(path: Path, rows: list[tuple[str, int, str, str]]) -> None:
    sorted_rows = sorted(rows, key=_word_with_oov_sort_key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["word", "corpus_count", "pronunciation", "oov_units"])
        writer.writerows(sorted_rows)


def _word_with_oov_sort_key(row: tuple[str, int, str, str]) -> tuple[int, str, str]:
    return (-row[1], row[0], row[2])
