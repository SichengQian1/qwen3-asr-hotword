from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.text import contains_token_sequence, stopwords_for_language
from qwen_hotword.hotwords.registry import HotwordEntry, write_hotword_table
from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.edit_distance import sequence_edit_distance
from qwen_hotword.training.g2p_prep import extract_word_tokens
from qwen_hotword.training.mfa_audit import load_mfa_dictionary


@dataclass(frozen=True)
class SimulatedHotwordCase:
    case_id: str
    sample_id: str
    case_type: str
    language: str
    active_hotword_ids: tuple[str, ...]
    expected_hotword_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["active_hotword_ids"] = list(self.active_hotword_ids)
        value["expected_hotword_ids"] = list(self.expected_hotword_ids)
        return value


@dataclass(frozen=True)
class SimulatedHotwordSummary:
    validation_manifest_path: str
    validation_manifest_sha256: str
    dictionary_path: str
    vocab_path: str
    hotword_table_path: str
    cases_path: str
    validation_records: int
    candidate_phrases: int
    selected_hotwords: int
    single_word_hotwords: int
    phrase_hotwords: int
    positive_cases: int
    negative_cases: int
    active_hotwords_per_case: int
    test_set_used: bool
    seed: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _PhraseCandidate:
    words: tuple[str, ...]
    language: str
    sample_ids: set[str] = field(default_factory=set)

    @property
    def occurrences(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True)
class _ManifestRecord:
    sample_id: str
    language: str
    words: tuple[str, ...]


def build_simulated_hotword_assets(
    validation_manifest_path: str | Path,
    dictionary_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    hotword_count: int = 50,
    case_count: int = 200,
    active_hotwords_per_case: int = 10,
    positive_ratio: float = 0.5,
    min_words: int = 1,
    max_words: int = 2,
    min_phonemes: int = 4,
    max_phonemes: int = 24,
    max_occurrences: int = 20,
    seed: int = 20_260_722,
) -> SimulatedHotwordSummary:
    if hotword_count <= 1 or case_count <= 1 or active_hotwords_per_case <= 1:
        raise ValueError("hotword, case, and active-candidate counts must exceed one")
    if active_hotwords_per_case > hotword_count:
        raise ValueError("active candidates cannot exceed the simulated hotword count")
    if not 0.0 < positive_ratio < 1.0:
        raise ValueError("positive_ratio must be between zero and one")
    if min_words <= 0 or max_words < min_words:
        raise ValueError("hotword word-width bounds are invalid")
    if min_phonemes <= 0 or max_phonemes < min_phonemes:
        raise ValueError("hotword phoneme-length bounds are invalid")
    if max_occurrences <= 0:
        raise ValueError("max_occurrences must be positive")

    manifest = Path(validation_manifest_path).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (manifest, dictionary_file, vocab_file):
        if not path.is_file():
            raise FileNotFoundError(f"required hotword input does not exist: {path}")
    records = _load_validation_records(manifest)
    vocab = load_phoneme_vocab(vocab_file)
    dictionary = load_mfa_dictionary(dictionary_file)
    phrase_candidates = _collect_phrase_candidates(
        records,
        min_words=min_words,
        max_words=max_words,
        max_occurrences=max_occurrences,
    )
    eligible = _build_entries(
        phrase_candidates,
        dictionary,
        vocab,
        min_phonemes=min_phonemes,
        max_phonemes=max_phonemes,
    )
    selected = _balanced_select(eligible, count=hotword_count, seed=seed)
    if len(selected) < hotword_count:
        raise ValueError(
            f"only {len(selected)} unique simulated hotwords satisfy the requested contract"
        )
    cases = _build_cases(
        records,
        selected,
        case_count=case_count,
        active_hotwords_per_case=active_hotwords_per_case,
        positive_ratio=positive_ratio,
        seed=seed,
    )
    if not cases or not any(case.expected_hotword_ids for case in cases):
        raise RuntimeError("simulated hotword cases contain no positive examples")
    if not any(not case.expected_hotword_ids for case in cases):
        raise RuntimeError("simulated hotword cases contain no negative examples")

    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / "simulated_hotwords.jsonl"
    cases_path = destination / "simulated_hotword_cases.jsonl"
    write_hotword_table(table_path, selected)
    _write_jsonl(cases_path, [case.to_dict() for case in cases])
    positive_cases = sum(bool(case.expected_hotword_ids) for case in cases)
    summary = SimulatedHotwordSummary(
        validation_manifest_path=str(manifest),
        validation_manifest_sha256=_sha256_file(manifest),
        dictionary_path=str(dictionary_file),
        vocab_path=str(vocab_file),
        hotword_table_path=str(table_path),
        cases_path=str(cases_path),
        validation_records=len(records),
        candidate_phrases=len(phrase_candidates),
        selected_hotwords=len(selected),
        single_word_hotwords=sum(len(entry.words) == 1 for entry in selected),
        phrase_hotwords=sum(len(entry.words) > 1 for entry in selected),
        positive_cases=positive_cases,
        negative_cases=len(cases) - positive_cases,
        active_hotwords_per_case=active_hotwords_per_case,
        test_set_used=False,
        seed=seed,
        status="pass",
    )
    _write_json(destination / "simulated_hotword_summary.json", summary.to_dict())
    return summary


def load_simulated_cases(path: str | Path) -> list[SimulatedHotwordCase]:
    case_path = Path(path).expanduser()
    cases: list[SimulatedHotwordCase] = []
    seen_ids: set[str] = set()
    seen_samples: set[str] = set()
    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"simulated case row {line_number} must be an object")
            case = SimulatedHotwordCase(
                case_id=_required_string(raw, "case_id", line_number),
                sample_id=_required_string(raw, "sample_id", line_number),
                case_type=_required_string(raw, "case_type", line_number),
                language=_required_string(raw, "language", line_number),
                active_hotword_ids=_string_tuple(raw, "active_hotword_ids", line_number),
                expected_hotword_ids=_string_tuple(
                    raw,
                    "expected_hotword_ids",
                    line_number,
                    allow_empty=True,
                ),
            )
            if case.case_id in seen_ids or case.sample_id in seen_samples:
                raise ValueError("simulated cases must have unique IDs and sample IDs")
            if not set(case.expected_hotword_ids).issubset(case.active_hotword_ids):
                raise ValueError(f"case {case.case_id} expects an inactive hotword")
            seen_ids.add(case.case_id)
            seen_samples.add(case.sample_id)
            cases.append(case)
    if not cases:
        raise ValueError(f"simulated case table is empty: {case_path}")
    return cases


def _load_validation_records(path: Path) -> list[_ManifestRecord]:
    records: list[_ManifestRecord] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("split") != "validation":
                raise ValueError(
                    f"manifest row {line_number} is not formal validation data"
                )
            sample_id = _required_string(raw, "id", line_number)
            if sample_id in seen_ids:
                raise ValueError(f"duplicate validation sample ID: {sample_id}")
            seen_ids.add(sample_id)
            words = tuple(extract_word_tokens(_required_string(raw, "text", line_number)))
            if not words:
                raise ValueError(f"validation row {line_number} has no normalized words")
            records.append(
                _ManifestRecord(
                    sample_id=sample_id,
                    language=_required_string(raw, "language", line_number),
                    words=words,
                )
            )
    if not records:
        raise ValueError(f"validation manifest is empty: {path}")
    return records


def _collect_phrase_candidates(
    records: list[_ManifestRecord],
    *,
    min_words: int,
    max_words: int,
    max_occurrences: int,
) -> list[_PhraseCandidate]:
    candidates: dict[tuple[str, tuple[str, ...]], _PhraseCandidate] = {}
    for record in records:
        stopwords = stopwords_for_language(record.language)
        seen: set[tuple[str, ...]] = set()
        for width in range(min_words, max_words + 1):
            for start in range(0, len(record.words) - width + 1):
                words = record.words[start : start + width]
                if words in seen or sum(len(word) for word in words) < 5:
                    continue
                seen.add(words)
                if words[0] in stopwords or words[-1] in stopwords:
                    continue
                key = (record.language, words)
                candidate = candidates.setdefault(
                    key,
                    _PhraseCandidate(words=words, language=record.language),
                )
                candidate.sample_ids.add(record.sample_id)
    return [
        candidate
        for candidate in candidates.values()
        if candidate.occurrences <= max_occurrences
    ]


def _build_entries(
    candidates: list[_PhraseCandidate],
    dictionary: dict[str, list[str]],
    vocab: PhonemeVocab,
    *,
    min_phonemes: int,
    max_phonemes: int,
) -> list[HotwordEntry]:
    entries: list[HotwordEntry] = []
    for candidate in candidates:
        if any(word not in dictionary or not dictionary[word] for word in candidate.words):
            continue
        pronunciation = " ".join(dictionary[word][0] for word in candidate.words)
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        if tokenized.oov_units or not min_phonemes <= len(tokenized.token_ids) <= max_phonemes:
            continue
        surface = " ".join(candidate.words)
        entries.append(
            HotwordEntry(
                hotword_id="pending",
                language=candidate.language,
                surface=surface,
                normalized=surface,
                words=candidate.words,
                pronunciation=pronunciation,
                phoneme_tokens=tuple(tokenized.tokens),
                token_ids=tuple(tokenized.token_ids),
                source="simulated_from_validation_manifest_and_mfa",
                validation_occurrences=candidate.occurrences,
            )
        )
    return entries


def _balanced_select(
    entries: list[HotwordEntry],
    *,
    count: int,
    seed: int,
) -> list[HotwordEntry]:
    rng = random.Random(seed)
    groups = {
        "word": [entry for entry in entries if len(entry.words) == 1],
        "phrase": [entry for entry in entries if len(entry.words) > 1],
    }
    for values in groups.values():
        rng.shuffle(values)
        values.sort(
            key=lambda item: (
                item.validation_occurrences,
                -len(item.token_ids),
                item.normalized,
            )
        )
    target_phrases = count // 2
    provisional = groups["phrase"][:target_phrases] + groups["word"][: count - target_phrases]
    selected_keys = {(entry.language, entry.normalized) for entry in provisional}
    remaining = [
        entry
        for entry in groups["phrase"][target_phrases:] + groups["word"][count - target_phrases :]
        if (entry.language, entry.normalized) not in selected_keys
    ]
    selected: list[HotwordEntry] = []
    seen_pronunciations: set[tuple[str, tuple[int, ...]]] = set()
    for entry in provisional + remaining:
        key = (entry.language, entry.token_ids)
        if key in seen_pronunciations:
            continue
        seen_pronunciations.add(key)
        selected.append(
            HotwordEntry(
                hotword_id=f"sim_hw_ptbr_{len(selected) + 1:04d}",
                language=entry.language,
                surface=entry.surface,
                normalized=entry.normalized,
                words=entry.words,
                pronunciation=entry.pronunciation,
                phoneme_tokens=entry.phoneme_tokens,
                token_ids=entry.token_ids,
                source=entry.source,
                validation_occurrences=entry.validation_occurrences,
            )
        )
        if len(selected) == count:
            break
    return selected


def _build_cases(
    records: list[_ManifestRecord],
    entries: list[HotwordEntry],
    *,
    case_count: int,
    active_hotwords_per_case: int,
    positive_ratio: float,
    seed: int,
) -> list[SimulatedHotwordCase]:
    rng = random.Random(seed)
    expected_by_sample: dict[str, tuple[str, ...]] = {}
    for record in records:
        matches = tuple(
            entry.hotword_id
            for entry in entries
            if entry.language == record.language
            and contains_token_sequence(list(record.words), entry.words)
        )
        expected_by_sample[record.sample_id] = matches[:active_hotwords_per_case]
    positives = [record for record in records if expected_by_sample[record.sample_id]]
    negatives = [record for record in records if not expected_by_sample[record.sample_id]]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_target = min(round(case_count * positive_ratio), len(positives))
    negative_target = min(case_count - positive_target, len(negatives))
    cases: list[SimulatedHotwordCase] = []
    entry_by_id = {entry.hotword_id: entry for entry in entries}
    for record in positives[:positive_target]:
        expected = expected_by_sample[record.sample_id]
        active = _active_hotwords(
            expected,
            entries,
            entry_by_id,
            count=active_hotwords_per_case,
            rng=rng,
        )
        cases.append(
            SimulatedHotwordCase(
                case_id=f"sim_positive_{len(cases) + 1:05d}",
                sample_id=record.sample_id,
                case_type="positive_confusable",
                language=record.language,
                active_hotword_ids=active,
                expected_hotword_ids=expected,
            )
        )
    for record in negatives[:negative_target]:
        available = [entry.hotword_id for entry in entries if entry.language == record.language]
        rng.shuffle(available)
        cases.append(
            SimulatedHotwordCase(
                case_id=f"sim_negative_{len(cases) + 1:05d}",
                sample_id=record.sample_id,
                case_type="negative",
                language=record.language,
                active_hotword_ids=tuple(available[:active_hotwords_per_case]),
                expected_hotword_ids=(),
            )
        )
    rng.shuffle(cases)
    return cases


def _active_hotwords(
    expected_ids: tuple[str, ...],
    entries: list[HotwordEntry],
    entry_by_id: dict[str, HotwordEntry],
    *,
    count: int,
    rng: random.Random,
) -> tuple[str, ...]:
    active = list(expected_ids)
    expected = entry_by_id[expected_ids[0]]
    confusable = sorted(
        (
            (
                sequence_edit_distance(expected.token_ids, entry.token_ids)
                / max(len(expected.token_ids), len(entry.token_ids)),
                entry.hotword_id,
            )
            for entry in entries
            if entry.language == expected.language and entry.hotword_id not in expected_ids
        ),
        key=lambda item: (item[0], item[1]),
    )
    active.extend(hotword_id for _, hotword_id in confusable[: min(3, count - len(active))])
    available = [
        entry.hotword_id
        for entry in entries
        if entry.language == expected.language and entry.hotword_id not in active
    ]
    rng.shuffle(available)
    active.extend(available[: max(0, count - len(active))])
    return tuple(active[:count])


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {key}")
    return value.strip()


def _string_tuple(
    raw: dict[str, Any],
    key: str,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"row {line_number} has invalid {key}")
    if not value and not allow_empty:
        raise ValueError(f"row {line_number} has empty {key}")
    return tuple(value)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
