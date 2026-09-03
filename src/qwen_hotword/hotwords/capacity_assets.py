from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.text import (
    contains_token_sequence,
    stopwords_for_language,
    tokenize_words,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.edit_distance import sequence_edit_distance
from qwen_hotword.training.mfa_audit import load_mfa_dictionary

DEFAULT_SIZES = (100, 500, 1_000, 2_000, 4_000, 5_000, 10_000)
DEFAULT_WORD_COUNT_WEIGHTS = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}
PROFILE_NAMES = ("representative", "hard_negative")


@dataclass(frozen=True)
class CapacityCase:
    case_id: str
    sample_id: str
    reference_text: str
    normalized_reference_text: str
    language: str
    primary_group: str
    expected_hotword_ids: tuple[str, ...]
    active_hotword_ids: tuple[str, ...]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class TrainCandidate:
    entry: HotwordEntry
    train_occurrences: int
    word_count: int
    phone_count: int
    frequency_band: str
    phone_length_band: str


def parse_capacity_sizes(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        except ValueError as error:
            raise ValueError("capacity sizes must be comma-separated integers") from error
    else:
        sizes = tuple(value)
    if not sizes or any(size < 100 for size in sizes):
        raise ValueError("capacity sizes must be non-empty and at least 100")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("capacity sizes must be unique and strictly increasing")
    if sizes[0] != 100:
        raise ValueError("capacity sizes must retain the sealed 100-hotword baseline")
    return sizes


def build_hotword_capacity_assets(
    *,
    training_manifest_path: str | Path,
    dictionary_path: str | Path,
    vocab_path: str | Path,
    base_hotwords_path: str | Path,
    base_cases_path: str | Path,
    output_dir: str | Path,
    selection_path: str | Path | None = None,
    sizes: Sequence[int] = DEFAULT_SIZES,
    seed: int = 20_260_818,
    maximum_ngram_words: int = 4,
    candidate_pool_multiplier: int = 3,
    word_count_weights: Mapping[int, float] = DEFAULT_WORD_COUNT_WEIGHTS,
    print_progress: bool = True,
    language: str = "pt-BR",
) -> dict[str, object]:
    resolved_sizes = parse_capacity_sizes(sizes)
    if maximum_ngram_words != 4:
        raise ValueError("capacity v1 is sealed to real 1-4 word train n-grams")
    if candidate_pool_multiplier < 2:
        raise ValueError("candidate_pool_multiplier must be at least two")
    weights = _validate_weights(word_count_weights)
    normalized_language = _normalize_language(language)
    paths = {
        "training_manifest": Path(training_manifest_path).expanduser(),
        "dictionary": Path(dictionary_path).expanduser(),
        "vocab": Path(vocab_path).expanduser(),
        "base_hotwords": Path(base_hotwords_path).expanduser(),
        "base_cases": Path(base_cases_path).expanduser(),
        "output": Path(output_dir).expanduser(),
    }
    if selection_path is not None:
        paths["selection"] = Path(selection_path).expanduser()
    for key in ("training_manifest", "dictionary", "vocab", "base_hotwords", "base_cases"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"capacity input does not exist: {paths[key]}")
    destination = paths["output"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"capacity output must be a new empty directory: {destination}")

    vocab = load_phoneme_vocab(paths["vocab"])
    base_entries = load_hotword_table(paths["base_hotwords"], vocab=vocab, blank_id=0)
    cases = load_capacity_base_cases(paths["base_cases"])
    if "selection" in paths:
        cases = _select_capacity_cases(cases, paths["selection"])
    base_by_id = {entry.hotword_id: entry for entry in base_entries}
    _validate_base_contract(cases, base_by_id, language=normalized_language)
    language_tag = cases[0].language

    maximum_size = resolved_sizes[-1]
    reservoir_limits = _word_count_quotas(
        maximum_size * candidate_pool_multiplier,
        weights,
    )
    selected_surfaces = _sample_train_ngrams(
        paths["training_manifest"],
        reservoir_limits=reservoir_limits,
        maximum_ngram_words=maximum_ngram_words,
        seed=seed,
        print_progress=print_progress,
        language=language_tag,
    )
    occurrence_counts = _count_selected_train_ngrams(
        paths["training_manifest"],
        selected_surfaces=set(selected_surfaces),
        maximum_ngram_words=maximum_ngram_words,
        print_progress=print_progress,
    )
    dictionary = load_mfa_dictionary(paths["dictionary"])
    candidates, rejection_counts = _materialize_train_candidates(
        selected_surfaces,
        occurrence_counts=occurrence_counts,
        dictionary=dictionary,
        vocab=vocab,
        base_entries=base_entries,
        seed=seed,
        language=language_tag,
    )
    representative_order = _representative_order(candidates, weights=weights, seed=seed)
    if len(representative_order) < maximum_size:
        raise ValueError(
            f"only {len(representative_order)} valid train-only candidates; "
            f"at least {maximum_size} are required"
        )
    candidate_by_id = {candidate.entry.hotword_id: candidate for candidate in candidates}
    all_entries = {**base_by_id, **{key: value.entry for key, value in candidate_by_id.items()}}
    hard_orders = {
        case.case_id: _hard_negative_order(
            case,
            candidate_by_id=candidate_by_id,
            base_by_id=base_by_id,
            seed=seed,
        )
        for case in cases
        if case.expected_hotword_ids
    }

    destination.mkdir(parents=True, exist_ok=True)
    pool_path = destination / "candidate_pool.jsonl"
    _write_candidate_pool(pool_path, candidates)
    level_summaries: dict[str, dict[str, Any]] = {}
    for profile in PROFILE_NAMES:
        level_summaries[profile] = {}
        for size in resolved_sizes:
            level = _build_level(
                profile=profile,
                size=size,
                cases=cases,
                all_entries=all_entries,
                candidate_by_id=candidate_by_id,
                representative_order=representative_order,
                hard_orders=hard_orders,
                seed=seed,
            )
            level_dir = destination / profile / f"size_{size}"
            level_dir.mkdir(parents=True, exist_ok=False)
            hotword_path = level_dir / "hotwords.jsonl"
            case_path = level_dir / "cases.jsonl"
            summary_path = level_dir / "summary.json"
            _write_hotword_rows(hotword_path, level["entries"], candidate_by_id)
            _write_jsonl(case_path, level["case_rows"])
            summary = {
                "profile": profile,
                "active_hotwords_per_case": size,
                "cases": len(cases),
                "positive_cases": sum(bool(case.expected_hotword_ids) for case in cases),
                "negative_cases": sum(not case.expected_hotword_ids for case in cases),
                "expected_hotwords": sum(len(case.expected_hotword_ids) for case in cases),
                "hotword_table_entries": len(level["entries"]),
                "word_count_distribution": level["word_count_distribution"],
                "frequency_band_distribution": level["frequency_band_distribution"],
                "phone_length_band_distribution": level["phone_length_band_distribution"],
                "hotwords_path": str(hotword_path),
                "cases_path": str(case_path),
                "hotwords_sha256": _sha256_file(hotword_path),
                "cases_sha256": _sha256_file(case_path),
                "test_set_used": False,
                "status": "pass",
            }
            _write_json(summary_path, summary)
            level_summaries[profile][str(size)] = summary

    summary_path = destination / "asset_summary.json"
    config_path = destination / "run_config.json"
    config = {
        "schema_version": 1,
        "purpose": "language_specific_hotword_capacity_assets",
        "language": normalized_language,
        "language_tag": language_tag,
        "sizes": list(resolved_sizes),
        "profiles": list(PROFILE_NAMES),
        "seed": seed,
        "maximum_ngram_words": maximum_ngram_words,
        "candidate_pool_multiplier": candidate_pool_multiplier,
        "word_count_weights": {str(key): value for key, value in weights.items()},
        "inputs": {key: _file_identity(path) for key, path in paths.items() if key != "output"},
        "test_set_used": False,
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "base_hotwords": len(base_entries),
        "base_cases": len(cases),
        "base_expected_hotwords": sum(len(case.expected_hotword_ids) for case in cases),
        "sampled_train_ngrams": len(selected_surfaces),
        "valid_train_candidates": len(candidates),
        "candidate_rejection_counts": dict(sorted(rejection_counts.items())),
        "candidate_pool_path": str(pool_path),
        "candidate_pool_sha256": _sha256_file(pool_path),
        "levels": level_summaries,
        "test_set_used": False,
    }
    _write_json(config_path, config)
    _write_json(summary_path, summary)
    _write_sha256_manifest(destination)
    return summary


def load_capacity_base_cases(path: str | Path) -> tuple[CapacityCase, ...]:
    rows: list[CapacityCase] = []
    seen: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_object(line, Path(path), line_number)
            case_id = _required_string(raw, "case_id", line_number)
            if case_id in seen:
                raise ValueError(f"duplicate capacity base case ID: {case_id}")
            active = _string_tuple(raw, "active_hotword_ids", line_number, allow_empty=False)
            expected = _string_tuple(raw, "expected_hotword_ids", line_number, allow_empty=True)
            rows.append(
                CapacityCase(
                    case_id=case_id,
                    sample_id=_required_string(raw, "sample_id", line_number),
                    reference_text=_required_string(raw, "reference_text", line_number),
                    normalized_reference_text=_required_string(
                        raw, "normalized_reference_text", line_number
                    ),
                    language=_required_string(raw, "language", line_number),
                    primary_group=_required_string(raw, "primary_group", line_number),
                    expected_hotword_ids=expected,
                    active_hotword_ids=active,
                    raw=raw,
                )
            )
            seen.add(case_id)
    if not rows:
        raise ValueError("capacity base case table is empty")
    return tuple(rows)


def _select_capacity_cases(
    cases: Sequence[CapacityCase], selection_path: Path
) -> tuple[CapacityCase, ...]:
    raw = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("capacity selection must be a JSON object")
    if raw.get("test_set_used") is not False:
        raise ValueError("capacity selection must explicitly record test_set_used=false")
    if raw.get("selection_profile") != "formal100":
        raise ValueError("capacity selection must use the sealed formal100 profile")
    retrieval_mode = raw.get("retrieval_mode")
    if retrieval_mode not in {None, "operating"}:
        raise ValueError("capacity selection must be legacy formal100 or operating retrieval")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("capacity selection has no samples")
    selected_ids: list[str] = []
    selected_rows: dict[str, dict[str, Any]] = {}
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"capacity selection sample {index} is not an object")
        case_id = sample.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"capacity selection sample {index} has invalid case_id")
        selected_ids.append(case_id)
        selected_rows[case_id] = sample
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("capacity selection contains duplicate case IDs")
    by_id = {case.case_id: case for case in cases}
    missing = set(selected_ids) - set(by_id)
    if missing:
        raise ValueError(f"capacity selection refers to unknown cases: {sorted(missing)}")
    selected_cases: list[CapacityCase] = []
    for case_id in selected_ids:
        case = by_id[case_id]
        sample = selected_rows[case_id]
        sample_id = sample.get("sample_id")
        if sample_id != case.sample_id:
            raise ValueError(f"capacity selection sample mismatch for case {case_id}")
        expected_raw = sample.get("expected_hotword_ids")
        if not isinstance(expected_raw, list) or any(
            not isinstance(item, str) or not item for item in expected_raw
        ):
            raise ValueError(f"capacity selection case {case_id} has invalid expected IDs")
        expected = tuple(expected_raw)
        if not set(expected).issubset(case.active_hotword_ids):
            raise ValueError(f"capacity selection case {case_id} expects inactive hotwords")
        case_raw = dict(case.raw)
        case_raw["expected_hotword_ids"] = list(expected)
        if "expected_surfaces" in sample:
            case_raw["expected_surfaces"] = sample["expected_surfaces"]
        selected_cases.append(
            CapacityCase(
                case_id=case.case_id,
                sample_id=case.sample_id,
                reference_text=case.reference_text,
                normalized_reference_text=case.normalized_reference_text,
                language=case.language,
                primary_group=case.primary_group,
                expected_hotword_ids=expected,
                active_hotword_ids=case.active_hotword_ids,
                raw=case_raw,
            )
        )
    return tuple(selected_cases)


def _validate_base_contract(
    cases: Sequence[CapacityCase],
    entries: Mapping[str, HotwordEntry],
    *,
    language: str,
) -> None:
    for case in cases:
        if len(case.active_hotword_ids) != 100 or len(set(case.active_hotword_ids)) != 100:
            raise ValueError(f"base case {case.case_id} must have 100 unique active hotwords")
        missing = set(case.active_hotword_ids) - set(entries)
        if missing:
            raise ValueError(f"base case {case.case_id} has unknown hotwords: {sorted(missing)}")
        if not set(case.expected_hotword_ids).issubset(case.active_hotword_ids):
            raise ValueError(f"base case {case.case_id} expects inactive hotwords")
        if _normalize_language(case.language) != language:
            raise ValueError(
                f"capacity case {case.case_id} language {case.language!r} "
                f"does not match requested {language!r}"
            )


def _sample_train_ngrams(
    path: Path,
    *,
    reservoir_limits: Mapping[int, int],
    maximum_ngram_words: int,
    seed: int,
    print_progress: bool,
    language: str,
) -> dict[str, tuple[str, ...]]:
    heaps: dict[int, list[tuple[int, str]]] = defaultdict(list)
    selected: dict[int, dict[str, tuple[str, ...]]] = defaultdict(dict)
    stopwords = stopwords_for_language(language)
    records = 0
    for records, raw in enumerate(_iter_train_rows(path), start=1):
        words = tuple(tokenize_words(_required_string(raw, "text", records)))
        for ngram in _ngrams(words, maximum_ngram_words):
            if all(word in stopwords for word in ngram):
                continue
            width = min(len(ngram), 4)
            surface = " ".join(ngram)
            current = selected[width]
            if surface in current:
                continue
            rank = _stable_int(seed, f"{width}\0{surface}")
            heap = heaps[width]
            limit = reservoir_limits[width]
            item = (-rank, surface)
            if len(heap) < limit:
                heapq.heappush(heap, item)
                current[surface] = ngram
            elif item > heap[0]:
                _, removed = heapq.heapreplace(heap, item)
                current.pop(removed)
                current[surface] = ngram
        if print_progress and records % 50_000 == 0:
            print(
                f"capacity candidate sampling records={records} "
                f"selected={sum(len(value) for value in selected.values())}",
                flush=True,
            )
    if records == 0:
        raise ValueError("training manifest contains no train records")
    return {
        surface: words for width in sorted(selected) for surface, words in selected[width].items()
    }


def _count_selected_train_ngrams(
    path: Path,
    *,
    selected_surfaces: set[str],
    maximum_ngram_words: int,
    print_progress: bool,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    records = 0
    for records, raw in enumerate(_iter_train_rows(path), start=1):
        words = tuple(tokenize_words(_required_string(raw, "text", records)))
        for ngram in _ngrams(words, maximum_ngram_words):
            surface = " ".join(ngram)
            if surface in selected_surfaces:
                counts[surface] += 1
        if print_progress and records % 50_000 == 0:
            print(f"capacity occurrence pass records={records}", flush=True)
    return counts


def _materialize_train_candidates(
    selected: Mapping[str, tuple[str, ...]],
    *,
    occurrence_counts: Mapping[str, int],
    dictionary: Mapping[str, Sequence[str]],
    vocab: PhonemeVocab,
    base_entries: Sequence[HotwordEntry],
    seed: int,
    language: str,
) -> tuple[tuple[TrainCandidate, ...], Counter[str]]:
    rejected: Counter[str] = Counter()
    seen_pronunciations = {(entry.language, entry.token_ids) for entry in base_entries}
    base_surfaces = {entry.normalized for entry in base_entries}
    candidates: list[TrainCandidate] = []
    for surface, words in sorted(
        selected.items(), key=lambda item: (_stable_int(seed, item[0]), item[0])
    ):
        if surface in base_surfaces:
            rejected["duplicate_base_surface"] += 1
            continue
        phones: list[str] = []
        token_ids: list[int] = []
        pronunciations: list[str] = []
        invalid_reason: str | None = None
        for word in words:
            values = dictionary.get(word)
            if not values:
                invalid_reason = "dictionary_missing"
                break
            if len(values) != 1:
                invalid_reason = "ambiguous_pronunciation"
                break
            pronunciation = values[0]
            tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
            if tokenized.oov_units or not tokenized.token_ids:
                invalid_reason = "phone_oov_or_empty"
                break
            pronunciations.append(pronunciation)
            phones.extend(tokenized.tokens)
            token_ids.extend(tokenized.token_ids)
        if invalid_reason is not None:
            rejected[invalid_reason] += 1
            continue
        if len(token_ids) < 4:
            rejected["fewer_than_four_phonemes"] += 1
            continue
        key = (language, tuple(token_ids))
        if key in seen_pronunciations:
            rejected["duplicate_pronunciation"] += 1
            continue
        seen_pronunciations.add(key)
        occurrences = int(occurrence_counts.get(surface, 0))
        if occurrences <= 0:
            rejected["missing_train_occurrence"] += 1
            continue
        digest = hashlib.sha256(f"{language}\0{surface}".encode()).hexdigest()[:16]
        language_id = "".join(character for character in language.lower() if character.isalnum())
        entry = HotwordEntry(
            hotword_id=f"capacity_{language_id}_{digest}",
            language=language,
            surface=surface,
            normalized=surface,
            words=words,
            pronunciation=" ".join(pronunciations),
            phoneme_tokens=tuple(phones),
            token_ids=tuple(token_ids),
            source="capacity_v1_train_only_real_ngram",
            validation_occurrences=1,
        )
        candidates.append(
            TrainCandidate(
                entry=entry,
                train_occurrences=occurrences,
                word_count=min(len(words), 4),
                phone_count=len(token_ids),
                frequency_band=_frequency_band(occurrences),
                phone_length_band=_phone_length_band(len(token_ids)),
            )
        )
    return tuple(candidates), rejected


def _normalize_language(value: str) -> str:
    normalized = value.strip().replace("_", "-").lower()
    aliases = {
        "en": "English",
        "en-us": "English",
        "english": "English",
        "es": "Spanish",
        "es-419": "Spanish",
        "spanish": "Spanish",
        "pt": "Portuguese",
        "pt-br": "Portuguese",
        "portuguese": "Portuguese",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(f"unsupported capacity language: {value!r}")
    return resolved


def _representative_order(
    candidates: Sequence[TrainCandidate],
    *,
    weights: Mapping[int, float],
    seed: int,
) -> tuple[str, ...]:
    by_width: dict[int, deque[TrainCandidate]] = {}
    for width in weights:
        values = [item for item in candidates if item.word_count == width]
        values.sort(
            key=lambda item: (
                _band_rank(item.frequency_band),
                _band_rank(item.phone_length_band),
                _stable_int(seed, item.entry.hotword_id),
                item.entry.hotword_id,
            )
        )
        by_width[width] = deque(_interleave_bands(values, seed=seed))
    selected_counts = dict.fromkeys(weights, 0)
    ordered: list[str] = []
    while any(by_width[width] for width in by_width):
        next_width = max(
            (width for width in weights if by_width[width]),
            key=lambda width: (
                weights[width] * (len(ordered) + 1) - selected_counts[width],
                -width,
            ),
        )
        item = by_width[next_width].popleft()
        ordered.append(item.entry.hotword_id)
        selected_counts[next_width] += 1
    return tuple(ordered)


def _interleave_bands(values: Sequence[TrainCandidate], *, seed: int) -> tuple[TrainCandidate, ...]:
    buckets: dict[tuple[str, str], deque[TrainCandidate]] = defaultdict(deque)
    for item in values:
        buckets[(item.frequency_band, item.phone_length_band)].append(item)
    for key in buckets:
        ordered = sorted(
            buckets[key],
            key=lambda item: (_stable_int(seed, item.entry.hotword_id), item.entry.hotword_id),
        )
        buckets[key] = deque(ordered)
    keys = sorted(buckets)
    output: list[TrainCandidate] = []
    while any(buckets[key] for key in keys):
        for key in keys:
            if buckets[key]:
                output.append(buckets[key].popleft())
    return tuple(output)


def _build_level(
    *,
    profile: str,
    size: int,
    cases: Sequence[CapacityCase],
    all_entries: Mapping[str, HotwordEntry],
    candidate_by_id: Mapping[str, TrainCandidate],
    representative_order: Sequence[str],
    hard_orders: Mapping[str, Sequence[str]],
    seed: int,
) -> dict[str, Any]:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown capacity profile: {profile}")
    case_rows: list[dict[str, object]] = []
    used_ids: set[str] = set()
    word_counts: Counter[int] = Counter()
    frequency_bands: Counter[str] = Counter()
    phone_bands: Counter[str] = Counter()
    for case in cases:
        active = list(case.active_hotword_ids)
        if size > 100:
            order = (
                representative_order
                if profile == "representative" or not case.expected_hotword_ids
                else hard_orders[case.case_id]
            )
            reference_words = tokenize_words(case.normalized_reference_text)
            for hotword_id in order:
                if len(active) >= size:
                    break
                if hotword_id in active:
                    continue
                entry = candidate_by_id[hotword_id].entry
                if contains_token_sequence(reference_words, entry.words):
                    continue
                active.append(hotword_id)
        if len(active) != size or len(set(active)) != size:
            raise ValueError(
                f"profile {profile} case {case.case_id} could only activate "
                f"{len(active)}/{size} unique hotwords"
            )
        used_ids.update(active)
        for hotword_id in active:
            entry = all_entries[hotword_id]
            candidate = candidate_by_id.get(hotword_id)
            word_counts[min(len(entry.words), 4)] += 1
            frequency_bands[candidate.frequency_band if candidate else "base_v3"] += 1
            phone_bands[candidate.phone_length_band if candidate else "base_v3"] += 1
        raw = dict(case.raw)
        raw["active_hotword_ids"] = active
        raw["capacity_profile"] = profile
        raw["capacity_size"] = size
        raw["capacity_seed"] = seed
        case_rows.append(raw)
    entries = tuple(all_entries[hotword_id] for hotword_id in sorted(used_ids))
    return {
        "entries": entries,
        "case_rows": case_rows,
        "word_count_distribution": {str(key): value for key, value in sorted(word_counts.items())},
        "frequency_band_distribution": dict(sorted(frequency_bands.items())),
        "phone_length_band_distribution": dict(sorted(phone_bands.items())),
    }


def _hard_negative_order(
    case: CapacityCase,
    *,
    candidate_by_id: Mapping[str, TrainCandidate],
    base_by_id: Mapping[str, HotwordEntry],
    seed: int,
) -> tuple[str, ...]:
    targets = [base_by_id[hotword_id].token_ids for hotword_id in case.expected_hotword_ids]
    scored: list[tuple[float, int, int, str]] = []
    for hotword_id, candidate in candidate_by_id.items():
        best_ratio = 1.0
        best_distance = max(len(candidate.entry.token_ids), 1)
        for target in targets:
            distance = sequence_edit_distance(target, candidate.entry.token_ids)
            ratio = distance / max(len(target), len(candidate.entry.token_ids))
            if (ratio, distance) < (best_ratio, best_distance):
                best_ratio, best_distance = ratio, distance
        scored.append(
            (
                best_ratio,
                abs(candidate.phone_count - min(map(len, targets))),
                _stable_int(seed, f"{case.case_id}\0{hotword_id}"),
                hotword_id,
            )
        )
    scored.sort()
    return tuple(item[-1] for item in scored)


def _iter_train_rows(path: Path) -> Iterable[dict[str, Any]]:
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_object(line, path, line_number)
            split = _required_string(raw, "split", line_number)
            if split == "test":
                raise ValueError("sealed test data is forbidden in capacity asset construction")
            if split != "train":
                raise ValueError(
                    f"capacity training manifest row {line_number} is not train: {split!r}"
                )
            seen += 1
            yield raw
    if seen == 0:
        raise ValueError("capacity training manifest is empty")


def _ngrams(words: tuple[str, ...], maximum_words: int) -> Iterable[tuple[str, ...]]:
    for width in range(1, min(maximum_words, len(words)) + 1):
        for start in range(0, len(words) - width + 1):
            yield words[start : start + width]


def _word_count_quotas(total: int, weights: Mapping[int, float]) -> dict[int, int]:
    quotas = {width: int(total * weight) for width, weight in weights.items()}
    remainder = total - sum(quotas.values())
    for width in sorted(weights, key=lambda item: (-weights[item], item))[:remainder]:
        quotas[width] += 1
    return quotas


def _validate_weights(values: Mapping[int, float]) -> dict[int, float]:
    weights = {int(key): float(value) for key, value in values.items()}
    if set(weights) != {1, 2, 3, 4} or any(value <= 0.0 for value in weights.values()):
        raise ValueError("word_count_weights must define positive weights for 1,2,3,4+")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("word_count_weights must sum to one")
    return weights


def _frequency_band(count: int) -> str:
    if count >= 6:
        return "high_6_plus"
    if count >= 2:
        return "medium_2_5"
    return "low_1"


def _phone_length_band(count: int) -> str:
    if count <= 6:
        return "short_4_6"
    if count <= 10:
        return "medium_7_10"
    return "long_11_plus"


def _band_rank(value: str) -> int:
    if value.startswith("high") or value.startswith("short"):
        return 0
    if value.startswith("medium"):
        return 1
    return 2


def _stable_int(seed: int, value: str) -> int:
    digest = hashlib.blake2b(f"{seed}\0{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _write_candidate_pool(path: Path, candidates: Sequence[TrainCandidate]) -> None:
    rows = []
    for candidate in sorted(candidates, key=lambda item: item.entry.hotword_id):
        row = candidate.entry.to_dict()
        row.update(
            {
                "capacity_train_occurrences": candidate.train_occurrences,
                "capacity_word_count": candidate.word_count,
                "capacity_phone_count": candidate.phone_count,
                "capacity_frequency_band": candidate.frequency_band,
                "capacity_phone_length_band": candidate.phone_length_band,
                "validation_occurrences_semantics": (
                    "schema compatibility only; capacity_train_occurrences is authoritative"
                ),
            }
        )
        rows.append(row)
    _write_jsonl(path, rows)


def _write_hotword_rows(
    path: Path,
    entries: Sequence[HotwordEntry],
    candidate_by_id: Mapping[str, TrainCandidate],
) -> None:
    rows = []
    for entry in entries:
        row = entry.to_dict()
        candidate = candidate_by_id.get(entry.hotword_id)
        if candidate is not None:
            row["capacity_train_occurrences"] = candidate.train_occurrences
            row["capacity_word_count"] = candidate.word_count
            row["capacity_phone_count"] = candidate.phone_count
            row["capacity_frequency_band"] = candidate.frequency_band
            row["capacity_phone_length_band"] = candidate.phone_length_band
        rows.append(row)
    _write_jsonl(path, rows)


def _load_object(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"JSON row {line_number} must be an object")
    return raw


def _required_string(raw: Mapping[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {key}")
    return value.strip()


def _string_tuple(
    raw: Mapping[str, Any], key: str, line_number: int, *, allow_empty: bool
) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"row {line_number} has invalid {key}")
    if not value and not allow_empty:
        raise ValueError(f"row {line_number} has empty {key}")
    return tuple(value)


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_sha256_manifest(destination: Path) -> None:
    paths = sorted(
        path for path in destination.rglob("*") if path.is_file() and path.name != "sha256.txt"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(destination)}" for path in paths]
    (destination / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
