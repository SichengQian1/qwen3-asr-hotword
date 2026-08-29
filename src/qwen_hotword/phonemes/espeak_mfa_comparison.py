from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.g2p_prep import extract_word_tokens
from qwen_hotword.training.mfa_audit import load_mfa_dictionary

LANGUAGE_ORDER = ("en", "es", "pt")
ESPEAK_LANGUAGE_CODES = {"en": "en-us", "es": "es-419", "pt": "pt-br"}
PRIMARY_STRATA = (
    "orthographic_risk",
    "long_word",
    "low_frequency",
    "mid_frequency",
    "high_frequency",
)
LANGUAGE_SWITCH_RE = re.compile(r"\([a-z]{2,3}(?:-[a-z0-9]+)*\)", re.IGNORECASE)

PhonemizeBatch = Callable[[list[str], str], list[str]]


@dataclass(frozen=True)
class NamedPath:
    language: str
    path: Path


@dataclass(frozen=True)
class CandidateWord:
    word: str
    corpus_count: int
    mfa_pronunciation: str
    mfa_tokens: tuple[str, ...]
    mfa_token_ids: tuple[int, ...]
    frequency_tier: str
    length_bucket: str
    risk_tags: tuple[str, ...]


@dataclass(frozen=True)
class SelectedWord:
    candidate: CandidateWord
    sampling_stratum: str
    sampling_rank: str


def parse_named_path(value: str) -> NamedPath:
    language, separator, raw_path = value.partition("=")
    language = language.strip().lower()
    raw_path = raw_path.strip()
    if not separator or language not in LANGUAGE_ORDER or not raw_path:
        raise ValueError("path arguments must use en=PATH, es=PATH, or pt=PATH")
    return NamedPath(language=language, path=Path(raw_path).expanduser())


def compare_espeak_mfa(
    manifests: Mapping[str, Path],
    dictionaries: Mapping[str, Path],
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    phonemize_batch: PhonemizeBatch,
    sample_size: int = 500,
    seed: int = 20_260_829,
    tool_metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compare eSpeak-ng with existing MFA pronunciations on train-manifest words."""

    _validate_language_paths(manifests, "manifest")
    _validate_language_paths(dictionaries, "MFA dictionary")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    destination = Path(output_dir).expanduser()
    _require_new_output_dir(destination)
    vocab_file = Path(vocab_path).expanduser()
    if not vocab_file.is_file():
        raise FileNotFoundError(f"phoneme vocabulary does not exist: {vocab_file}")
    vocab = load_phoneme_vocab(vocab_file)

    selections: dict[str, list[SelectedWord]] = {}
    source_stats: dict[str, dict[str, Any]] = {}
    for language in LANGUAGE_ORDER:
        counts, manifest_stats = _manifest_word_counts(
            manifests[language],
            expected_language=language,
        )
        candidates, eligibility = _eligible_candidates(
            counts,
            dictionaries[language],
            vocab,
        )
        selections[language] = _select_stratified(
            candidates,
            language=language,
            sample_size=sample_size,
            seed=seed,
        )
        source_stats[language] = {**manifest_stats, **eligibility}

    sampled_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    confusion_counts: Counter[tuple[str, str, str]] = Counter()
    for language in LANGUAGE_ORDER:
        selected = selections[language]
        words = [item.candidate.word for item in selected]
        ipa_values = phonemize_batch(words, ESPEAK_LANGUAGE_CODES[language])
        if len(ipa_values) != len(words):
            raise RuntimeError(
                f"phonemizer returned {len(ipa_values)} rows for {len(words)} {language} words"
            )
        for item, espeak_ipa in zip(selected, ipa_values, strict=True):
            sampled = _sampled_word_row(language, item)
            sampled_rows.append(sampled)
            comparison = _comparison_row(
                sampled,
                espeak_ipa,
                vocab,
                confusion_counts,
            )
            comparison_rows.append(comparison)

    summary = _build_summary(
        comparison_rows,
        source_stats=source_stats,
        sample_size=sample_size,
        seed=seed,
        ctc_output_classes=len(vocab.tokens),
    )
    config: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "espeak_ng_vs_mfa_g2p_feasibility_selection",
        "git_commit": _git_commit(),
        "languages": list(LANGUAGE_ORDER),
        "espeak_language_codes": ESPEAK_LANGUAGE_CODES,
        "sample_size_per_language": sample_size,
        "sampling_seed": seed,
        "sampling_policy": (
            "100-per-primary-stratum-at-500; deterministic SHA256 ranking; "
            "shortfalls filled from remaining eligible words"
        ),
        "primary_strata": list(PRIMARY_STRATA),
        "comparison_space": "current_90_class_ctc_vocab_after_existing_normalization",
        "acceptance_threshold": None,
        "input_identity": {
            language: {
                "manifest": _file_identity(manifests[language]),
                "mfa_dictionary": _file_identity(dictionaries[language]),
            }
            for language in LANGUAGE_ORDER
        },
        "vocab": _file_identity(vocab_file),
        "tool_metadata": dict(sorted((tool_metadata or {}).items())),
        "outputs_are_selection_evidence_not_training_data": True,
        "test_set_used": False,
    }
    destination.mkdir(parents=True)
    _write_json(destination / "run_config.json", config)
    _write_jsonl(destination / "sampled_words.jsonl", sampled_rows)
    _write_jsonl(destination / "word_comparisons.jsonl", comparison_rows)
    _write_json(destination / "summary.json", summary)
    _write_language_summary(destination / "language_summary.tsv", summary)
    _write_confusions(destination / "phone_confusions.tsv", confusion_counts)
    _write_oov_units(destination / "oov_units.tsv", comparison_rows)
    _write_review_cases(destination / "manual_review.tsv", comparison_rows)
    _write_sha256_manifest(destination)
    return summary


def _validate_language_paths(paths: Mapping[str, Path], label: str) -> None:
    if set(paths) != set(LANGUAGE_ORDER):
        raise ValueError(f"{label} paths must contain exactly en, es, and pt")
    for language in LANGUAGE_ORDER:
        if not paths[language].is_file():
            raise FileNotFoundError(f"{label} does not exist for {language}: {paths[language]}")


def _require_new_output_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output directory already exists; refusing to overwrite: {path}")


def _manifest_word_counts(
    path: Path,
    *,
    expected_language: str,
) -> tuple[Counter[str], dict[str, Any]]:
    counts: Counter[str] = Counter()
    records = 0
    records_with_words = 0
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row: {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"manifest row is not an object: {path}:{line_number}")
            if raw.get("split") != "train":
                raise ValueError(f"manifest row is not train-only: {path}:{line_number}")
            raw_language = raw.get("language")
            if not isinstance(raw_language, str) or not _matches_language(
                raw_language, expected_language
            ):
                raise ValueError(
                    f"manifest language does not match {expected_language}: "
                    f"{path}:{line_number}"
                )
            records += 1
            raw_words = raw.get("words")
            if isinstance(raw_words, list) and all(isinstance(word, str) for word in raw_words):
                words = [word for word in raw_words if word]
            else:
                text = raw.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"manifest row has no text or words: {path}:{line_number}")
                words = extract_word_tokens(text)
            if words:
                records_with_words += 1
                counts.update(words)
    return counts, {
        "manifest_records": records,
        "manifest_records_with_words": records_with_words,
        "manifest_word_tokens": sum(counts.values()),
        "manifest_unique_words": len(counts),
    }


def _matches_language(raw_language: str, expected_language: str) -> bool:
    normalized = raw_language.casefold().replace("_", "-")
    return normalized == expected_language or normalized.startswith(expected_language + "-")


def _eligible_candidates(
    counts: Counter[str],
    dictionary_path: Path,
    vocab: PhonemeVocab,
) -> tuple[list[CandidateWord], dict[str, Any]]:
    raw_dictionary = load_mfa_dictionary(dictionary_path)
    unique_dictionary = {
        word: tuple(dict.fromkeys(pronunciations))
        for word, pronunciations in raw_dictionary.items()
    }
    shared = sorted(set(counts) & set(unique_dictionary))
    frequency_tiers = _frequency_tiers(shared, counts)
    candidates: list[CandidateWord] = []
    ambiguous = 0
    mfa_oov = 0
    empty = 0
    for word in shared:
        pronunciations = unique_dictionary[word]
        if len(pronunciations) != 1:
            ambiguous += 1
            continue
        pronunciation = pronunciations[0].strip()
        if not pronunciation:
            empty += 1
            continue
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        if tokenized.oov_units or not tokenized.tokens:
            mfa_oov += 1
            continue
        candidates.append(
            CandidateWord(
                word=word,
                corpus_count=counts[word],
                mfa_pronunciation=pronunciation,
                mfa_tokens=tuple(_nfc(token) for token in tokenized.tokens),
                mfa_token_ids=tuple(tokenized.token_ids),
                frequency_tier=frequency_tiers[word],
                length_bucket=_length_bucket(word),
                risk_tags=_risk_tags(word),
            )
        )
    return candidates, {
        "mfa_dictionary_unique_words": len(raw_dictionary),
        "manifest_dictionary_intersection": len(shared),
        "excluded_ambiguous_mfa_words": ambiguous,
        "excluded_empty_mfa_words": empty,
        "excluded_mfa_vocab_oov_words": mfa_oov,
        "eligible_words": len(candidates),
    }


def _frequency_tiers(words: Sequence[str], counts: Counter[str]) -> dict[str, str]:
    ranked = sorted(words, key=lambda word: (-counts[word], word))
    result: dict[str, str] = {}
    total = max(1, len(ranked))
    for index, word in enumerate(ranked):
        fraction = index / total
        if fraction < 1 / 3:
            tier = "high"
        elif fraction < 2 / 3:
            tier = "mid"
        else:
            tier = "low"
        result[word] = tier
    return result


def _length_bucket(word: str) -> str:
    length = sum(1 for char in word if unicodedata.category(char)[0] in {"L", "M"})
    if length <= 4:
        return "short"
    if length >= 10:
        return "long"
    return "medium"


def _risk_tags(word: str) -> tuple[str, ...]:
    tags: list[str] = []
    if "'" in word or "-" in word:
        tags.append("connector")
    if any(ord(char) > 127 for char in word):
        tags.append("non_ascii")
    if any(unicodedata.combining(char) for char in unicodedata.normalize("NFD", word)):
        tags.append("diacritic")
    return tuple(tags)


def _select_stratified(
    candidates: Sequence[CandidateWord],
    *,
    language: str,
    sample_size: int,
    seed: int,
) -> list[SelectedWord]:
    if len(candidates) < sample_size:
        raise ValueError(
            f"only {len(candidates)} eligible {language} words; need {sample_size}"
        )
    quota = sample_size // len(PRIMARY_STRATA)
    remainder = sample_size % len(PRIMARY_STRATA)
    selected_words: set[str] = set()
    selected: list[SelectedWord] = []
    for index, stratum in enumerate(PRIMARY_STRATA):
        target = quota + (1 if index < remainder else 0)
        pool = [
            candidate
            for candidate in candidates
            if candidate.word not in selected_words and _in_primary_stratum(candidate, stratum)
        ]
        for candidate in sorted(
            pool,
            key=lambda item: _stable_rank(seed, language, stratum, item.word),
        )[:target]:
            selected_words.add(candidate.word)
            selected.append(
                SelectedWord(
                    candidate=candidate,
                    sampling_stratum=stratum,
                    sampling_rank=_stable_rank(seed, language, stratum, candidate.word),
                )
            )
    shortfall = sample_size - len(selected)
    if shortfall:
        remaining = [candidate for candidate in candidates if candidate.word not in selected_words]
        for candidate in sorted(
            remaining,
            key=lambda item: _stable_rank(seed, language, "fallback", item.word),
        )[:shortfall]:
            selected.append(
                SelectedWord(
                    candidate=candidate,
                    sampling_stratum="fallback",
                    sampling_rank=_stable_rank(seed, language, "fallback", candidate.word),
                )
            )
    if len(selected) != sample_size:
        raise RuntimeError(f"failed to select {sample_size} words for {language}")
    return sorted(selected, key=lambda item: (item.sampling_stratum, item.sampling_rank))


def _in_primary_stratum(candidate: CandidateWord, stratum: str) -> bool:
    if stratum == "orthographic_risk":
        return bool(candidate.risk_tags)
    if stratum == "long_word":
        return candidate.length_bucket == "long"
    return candidate.frequency_tier == stratum.removesuffix("_frequency")


def _stable_rank(seed: int, language: str, stratum: str, word: str) -> str:
    value = f"{seed}\0{language}\0{stratum}\0{word}".encode()
    return hashlib.sha256(value).hexdigest()


def _sampled_word_row(language: str, selected: SelectedWord) -> dict[str, Any]:
    candidate = selected.candidate
    return {
        "language": language,
        "espeak_language": ESPEAK_LANGUAGE_CODES[language],
        "word": candidate.word,
        "corpus_count": candidate.corpus_count,
        "sampling_stratum": selected.sampling_stratum,
        "sampling_rank": selected.sampling_rank,
        "frequency_tier": candidate.frequency_tier,
        "length_bucket": candidate.length_bucket,
        "risk_tags": list(candidate.risk_tags),
        "mfa_pronunciation": candidate.mfa_pronunciation,
        "mfa_tokens": list(candidate.mfa_tokens),
        "mfa_token_ids": list(candidate.mfa_token_ids),
    }


def _comparison_row(
    sampled: dict[str, Any],
    raw_espeak_ipa: str,
    vocab: PhonemeVocab,
    confusion_counts: Counter[tuple[str, str, str]],
) -> dict[str, Any]:
    ipa = str(raw_espeak_ipa).strip()
    switches = LANGUAGE_SWITCH_RE.findall(ipa)
    comparison_ipa = LANGUAGE_SWITCH_RE.sub("", ipa)
    tokenized = tokenize_ipa_to_vocab(comparison_ipa, vocab)
    espeak_tokens = [_nfc(token) for token in tokenized.tokens]
    oov_units = [_nfc(unit) for unit in tokenized.oov_units]
    mfa_tokens = [str(token) for token in sampled["mfa_tokens"]]
    alignment = align_phone_sequences(mfa_tokens, espeak_tokens)
    substitutions = 0
    insertions = 0
    deletions = 0
    language = str(sampled["language"])
    for reference, hypothesis in alignment:
        if reference == hypothesis:
            continue
        confusion_counts[(language, reference or "<eps>", hypothesis or "<eps>")] += 1
        if reference is None:
            insertions += 1
        elif hypothesis is None:
            deletions += 1
        else:
            substitutions += 1
    edit_distance = substitutions + insertions + deletions
    per = edit_distance / len(mfa_tokens)
    return {
        **sampled,
        "mfa_conversion_success": True,
        "mfa_vocab_complete": True,
        "espeak_raw_ipa": ipa,
        "espeak_comparison_ipa": comparison_ipa.strip(),
        "espeak_conversion_success": bool(ipa),
        "espeak_language_switch_flags": switches,
        "espeak_tokens": espeak_tokens,
        "espeak_token_ids": tokenized.token_ids,
        "espeak_oov_units": oov_units,
        "espeak_vocab_complete": bool(espeak_tokens) and not oov_units,
        "exact_token_match": mfa_tokens == espeak_tokens,
        "phone_edit_distance": edit_distance,
        "phone_error_rate": per,
        "phone_similarity": max(0.0, 1.0 - per),
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "mfa_phone_count": len(mfa_tokens),
        "espeak_phone_count": len(espeak_tokens),
        "phone_length_ratio": len(espeak_tokens) / len(mfa_tokens),
        "alignment": [
            {"mfa": reference, "espeak": hypothesis}
            for reference, hypothesis in alignment
        ],
    }


def align_phone_sequences(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> list[tuple[str | None, str | None]]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        moves[row][0] = "delete"
    for column in range(1, columns):
        costs[0][column] = column
        moves[0][column] = "insert"
    for row in range(1, rows):
        for column in range(1, columns):
            equal = reference[row - 1] == hypothesis[column - 1]
            choices = (
                (costs[row - 1][column - 1] + (0 if equal else 1), "equal" if equal else "sub"),
                (costs[row - 1][column] + 1, "delete"),
                (costs[row][column - 1] + 1, "insert"),
            )
            costs[row][column], moves[row][column] = min(choices, key=lambda item: item[0])
    alignment: list[tuple[str | None, str | None]] = []
    row = len(reference)
    column = len(hypothesis)
    while row or column:
        move = moves[row][column]
        if move in {"equal", "sub"}:
            alignment.append((reference[row - 1], hypothesis[column - 1]))
            row -= 1
            column -= 1
        elif move == "delete":
            alignment.append((reference[row - 1], None))
            row -= 1
        else:
            alignment.append((None, hypothesis[column - 1]))
            column -= 1
    alignment.reverse()
    return alignment


def _build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    source_stats: Mapping[str, dict[str, Any]],
    sample_size: int,
    seed: int,
    ctc_output_classes: int,
) -> dict[str, Any]:
    by_language = {
        language: _metrics([row for row in rows if row["language"] == language])
        for language in LANGUAGE_ORDER
    }
    category_metrics: dict[str, Any] = {}
    for language in LANGUAGE_ORDER:
        language_rows = [row for row in rows if row["language"] == language]
        category_metrics[language] = {
            "sampling_stratum": _group_metrics(language_rows, "sampling_stratum"),
            "frequency_tier": _group_metrics(language_rows, "frequency_tier"),
            "length_bucket": _group_metrics(language_rows, "length_bucket"),
            "risk_tag": _risk_group_metrics(language_rows),
        }
    return {
        "schema_version": 1,
        "status": "completed",
        "purpose": "feasibility_information_collection_without_acceptance_threshold",
        "sample_size_per_language": sample_size,
        "total_sampled_words": len(rows),
        "sampling_seed": seed,
        "ctc_output_classes": ctc_output_classes,
        "source_stats": source_stats,
        "by_language": by_language,
        "category_metrics": category_metrics,
        "interpretation_policy": (
            "MFA is the current label-system reference, not an absolute pronunciation truth; "
            "manual review is required for high-distance and OOV cases"
        ),
        "test_set_used": False,
    }


def _metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    pers = [float(row["phone_error_rate"]) for row in rows]
    distances = [int(row["phone_edit_distance"]) for row in rows]
    return {
        "words": count,
        "mfa_conversion_success": sum(bool(row["mfa_conversion_success"]) for row in rows),
        "espeak_conversion_success": sum(bool(row["espeak_conversion_success"]) for row in rows),
        "espeak_vocab_complete_words": sum(bool(row["espeak_vocab_complete"]) for row in rows),
        "words_with_espeak_oov": sum(bool(row["espeak_oov_units"]) for row in rows),
        "espeak_oov_units": sum(len(row["espeak_oov_units"]) for row in rows),
        "words_with_language_switch": sum(
            bool(row["espeak_language_switch_flags"]) for row in rows
        ),
        "exact_token_matches": sum(bool(row["exact_token_match"]) for row in rows),
        "exact_token_match_rate": _rate(
            sum(bool(row["exact_token_match"]) for row in rows), count
        ),
        "mean_phone_edit_distance": statistics.fmean(distances) if distances else None,
        "mean_phone_error_rate": statistics.fmean(pers) if pers else None,
        "median_phone_error_rate": statistics.median(pers) if pers else None,
        "p90_phone_error_rate": _percentile(pers, 0.90),
        "per_le_0_1_rate": _rate(sum(value <= 0.1 for value in pers), count),
        "per_le_0_2_rate": _rate(sum(value <= 0.2 for value in pers), count),
        "per_gt_0_3_rate": _rate(sum(value > 0.3 for value in pers), count),
        "substitutions": sum(int(row["substitutions"]) for row in rows),
        "insertions": sum(int(row["insertions"]) for row in rows),
        "deletions": sum(int(row["deletions"]) for row in rows),
    }


def _group_metrics(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _metrics(value) for key, value in sorted(groups.items())}


def _risk_group_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_tags = row["risk_tags"]
        tags = raw_tags if isinstance(raw_tags, list) and raw_tags else ["none"]
        for tag in tags:
            groups[str(tag)].append(row)
    return {key: _metrics(value) for key, value in sorted(groups.items())}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _write_language_summary(path: Path, summary: Mapping[str, Any]) -> None:
    fields = (
        "words",
        "espeak_conversion_success",
        "espeak_vocab_complete_words",
        "words_with_espeak_oov",
        "espeak_oov_units",
        "words_with_language_switch",
        "exact_token_matches",
        "exact_token_match_rate",
        "mean_phone_error_rate",
        "median_phone_error_rate",
        "p90_phone_error_rate",
        "per_le_0_1_rate",
        "per_le_0_2_rate",
        "per_gt_0_3_rate",
    )
    lines = ["language\t" + "\t".join(fields)]
    by_language = summary["by_language"]
    assert isinstance(by_language, dict)
    for language in LANGUAGE_ORDER:
        metrics = by_language[language]
        lines.append(language + "\t" + "\t".join(str(metrics[field]) for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_confusions(
    path: Path, counts: Counter[tuple[str, str, str]]
) -> None:
    lines = ["language\tmfa_phone\tespeak_phone\tcount"]
    for (language, reference, hypothesis), count in sorted(
        counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"{language}\t{reference}\t{hypothesis}\t{count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_oov_units(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for unit in row["espeak_oov_units"]:
            counts[(str(row["language"]), str(unit))] += 1
    lines = ["language\toov_unit\tunicode_codepoints\tcount"]
    for (language, unit), count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        codepoints = " ".join(f"U+{ord(char):04X}" for char in unit)
        lines.append(f"{language}\t{unit}\t{codepoints}\t{count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_review_cases(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for language in LANGUAGE_ORDER:
        language_rows = [row for row in rows if row["language"] == language]
        for row in sorted(
            language_rows,
            key=lambda item: (-float(item["phone_error_rate"]), str(item["word"])),
        )[:20]:
            selected[(language, str(row["word"]))] = row
        for row in language_rows:
            if row["espeak_oov_units"] or row["espeak_language_switch_flags"]:
                selected[(language, str(row["word"]))] = row
    lines = [
        "language\tword\tcorpus_count\tmfa_pronunciation\tespeak_raw_ipa\t"
        "mfa_tokens\tespeak_tokens\tespeak_oov_units\tphone_error_rate\t"
        "review_label\treviewer_notes"
    ]
    ordered = sorted(
        selected.values(),
        key=lambda item: (str(item["language"]), str(item["word"])),
    )
    for row in ordered:
        values = (
            row["language"],
            row["word"],
            row["corpus_count"],
            row["mfa_pronunciation"],
            row["espeak_raw_ipa"],
            " ".join(row["mfa_tokens"]),
            " ".join(row["espeak_tokens"]),
            " ".join(row["espeak_oov_units"]),
            row["phone_error_rate"],
            "",
            "",
        )
        lines.append("\t".join(_tsv(value) for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tsv(value: object) -> str:
    return str(value).replace("\t", " ").replace("\n", " ")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _file_identity(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_manifest(directory: Path) -> None:
    paths = sorted(path for path in directory.iterdir() if path.name != "sha256.txt")
    content = "".join(f"{_sha256(path)}  {path.name}\n" for path in paths if path.is_file())
    (directory / "sha256.txt").write_text(content, encoding="utf-8")


def _git_commit() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def environment_tool_metadata() -> dict[str, str]:
    metadata: dict[str, str] = {}
    try:
        from importlib.metadata import version

        metadata["phonemizer_version"] = version("phonemizer")
    except Exception:  # noqa: BLE001 - diagnostic metadata must not hide the real run
        metadata["phonemizer_version"] = "unknown"
    try:
        result = subprocess.run(
            ["espeak-ng", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        metadata["espeak_ng_version"] = (result.stdout or result.stderr).strip()
    except (OSError, subprocess.CalledProcessError):
        metadata["espeak_ng_version"] = "unknown"
    metadata["python_executable"] = os.path.realpath(sys.executable)
    metadata["python_version"] = sys.version.split()[0]
    return metadata
