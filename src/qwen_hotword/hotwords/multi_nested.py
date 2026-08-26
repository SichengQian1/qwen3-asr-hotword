from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.text import (
    contains_token_sequence,
    stopwords_for_language,
    tokenize_words,
)
from qwen_hotword.hotwords.registry import HotwordEntry, write_hotword_table
from qwen_hotword.hotwords.scoring import HotwordMatch, HotwordScoringConfig, score_hotwords
from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.edit_distance import sequence_edit_distance
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, load_feature_shard

DEFAULT_SEED = 20_260_804
GROUP_TARGETS: dict[str, int] = {
    "single_hotword": 30,
    "two_independent": 30,
    "three_independent": 40,
    "nested_short_only": 20,
    "nested_long_present": 30,
    "nested_family_plus_two": 30,
    "negative": 30,
}
ASSET_FILENAMES = (
    "multi_nested_hotwords_v3.jsonl",
    "hotword_families_v3.jsonl",
    "multi_nested_cases_v3.jsonl",
    "sample_selection_v3.json",
    "asset_summary_v3.json",
)
SCORE_FILENAMES = (
    "hotword_case_scores_v3.jsonl",
    "multi_nested_evaluation_report_v3.json",
)


@dataclass(frozen=True)
class ValidationSample:
    sample_id: str
    audio_path: str
    reference_text: str
    normalized_text: str
    language: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    language: str
    words: tuple[str, ...]
    pronunciation: str
    phoneme_tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    sample_spans: Mapping[str, tuple[tuple[int, int], ...]]

    @property
    def key(self) -> str:
        return f"{self.language}\0{' '.join(self.words)}"

    @property
    def occurrences(self) -> int:
        return len(self.sample_spans)


@dataclass(frozen=True)
class HotwordFamily:
    family_id: str
    short_hotword_id: str
    long_hotword_id: str
    short_surface: str
    long_surface: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiNestedCase:
    case_id: str
    sample_id: str
    audio_path: str
    reference_text: str
    normalized_reference_text: str
    language: str
    primary_group: str
    expected_hotword_ids: tuple[str, ...]
    expected_surfaces: tuple[str, ...]
    expected_word_spans: Mapping[str, tuple[int, int]]
    containment_expected_ids: tuple[str, ...]
    longest_match_expected_ids: tuple[str, ...]
    active_hotword_ids: tuple[str, ...]
    nested_family_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]
    independent_expected_ids: tuple[str, ...]
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "expected_hotword_ids",
            "expected_surfaces",
            "containment_expected_ids",
            "longest_match_expected_ids",
            "active_hotword_ids",
            "nested_family_ids",
            "hard_negative_ids",
            "independent_expected_ids",
        ):
            value[key] = list(value[key])
        value["expected_word_spans"] = {
            key: list(span) for key, span in self.expected_word_spans.items()
        }
        return value


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    sample_id: str
    primary_group: str
    ranked_matches: tuple[HotwordMatch, ...]
    operating_matches: tuple[HotwordMatch, ...]
    effective_time_steps: int = 0
    decoded_token_count: int = 0


@dataclass
class _CaseDraft:
    sample: ValidationSample
    group: str
    expected_keys: tuple[str, ...]
    containment_keys: tuple[str, ...]
    longest_keys: tuple[str, ...]
    independent_keys: tuple[str, ...]
    family_keys: tuple[tuple[str, str], ...]
    spans: dict[str, tuple[int, int]]
    reason: str


def build_multi_nested_assets(
    validation_manifest_path: str | Path,
    dictionary_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    active_hotwords_per_case: int = 100,
    group_targets: Mapping[str, int] | None = None,
) -> dict[str, object]:
    if active_hotwords_per_case != 100:
        raise ValueError("v3 contract requires exactly 100 active hotwords per case")
    targets = dict(group_targets or GROUP_TARGETS)
    if set(targets) != set(GROUP_TARGETS) or any(value < 0 for value in targets.values()):
        raise ValueError("group targets must define every v3 group with non-negative counts")
    manifest = Path(validation_manifest_path).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (manifest, dictionary_file, vocab_file):
        if not path.is_file():
            raise FileNotFoundError(f"required v3 asset input does not exist: {path}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"v3 output must be a new empty directory: {destination}")

    samples = load_validation_samples(manifest)
    vocab = load_phoneme_vocab(vocab_file)
    dictionary = load_mfa_dictionary(dictionary_file)
    candidates = _deduplicate_pronunciations(
        _collect_candidates(samples, dictionary, vocab), seed=seed
    )
    if len(candidates) < active_hotwords_per_case:
        raise ValueError(
            f"only {len(candidates)} valid natural candidates; 100 active hotwords are required"
        )
    by_key = {candidate.key: candidate for candidate in candidates}
    family_keys = _collect_family_keys(candidates)
    drafts = _select_case_drafts(samples, by_key, family_keys, targets, seed)
    if not drafts:
        raise ValueError("no natural validation cases satisfy the v3 contract")

    required_keys = {key for draft in drafts for key in draft.containment_keys}
    required_keys.update(key for draft in drafts for pair in draft.family_keys for key in pair)
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (item.key not in required_keys, _stable_rank(seed, item.key), item.key),
    )
    pool = ordered_candidates[: max(500, active_hotwords_per_case)]
    pool_keys = {candidate.key for candidate in pool}
    missing_required = required_keys - pool_keys
    if missing_required:
        pool.extend(by_key[key] for key in sorted(missing_required))
    pool.sort(key=lambda item: (item.language, item.words))
    entries: list[HotwordEntry] = []
    id_by_key: dict[str, str] = {}
    for index, candidate in enumerate(pool, start=1):
        hotword_id = f"sim_v3_hw_ptbr_{index:04d}"
        id_by_key[candidate.key] = hotword_id
        surface = " ".join(candidate.words)
        entries.append(
            HotwordEntry(
                hotword_id=hotword_id,
                language=candidate.language,
                surface=surface,
                normalized=surface,
                words=candidate.words,
                pronunciation=candidate.pronunciation,
                phoneme_tokens=candidate.phoneme_tokens,
                token_ids=candidate.token_ids,
                source="simulated_v3_multi_nested_from_validation_manifest_and_mfa",
                validation_occurrences=candidate.occurrences,
            )
        )
    entry_by_id = {entry.hotword_id: entry for entry in entries}

    used_pairs = {pair for draft in drafts for pair in draft.family_keys}
    families: list[HotwordFamily] = []
    family_id_by_pair: dict[tuple[str, str], str] = {}
    for index, pair in enumerate(sorted(used_pairs), start=1):
        family_id = f"sim_v3_family_{index:04d}"
        family_id_by_pair[pair] = family_id
        short, long = pair
        families.append(
            HotwordFamily(
                family_id=family_id,
                short_hotword_id=id_by_key[short],
                long_hotword_id=id_by_key[long],
                short_surface=" ".join(by_key[short].words),
                long_surface=" ".join(by_key[long].words),
            )
        )

    cases = [
        _materialize_case(
            draft,
            index=index,
            entries=entries,
            entry_by_id=entry_by_id,
            id_by_key=id_by_key,
            family_id_by_pair=family_id_by_pair,
            active_count=active_hotwords_per_case,
            seed=seed,
        )
        for index, draft in enumerate(drafts, start=1)
    ]
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("v3 case IDs are not unique")
    if len({case.audio_path for case in cases}) != len(cases):
        raise RuntimeError("v3 primary case audio paths are not disjoint")
    if any(len(case.active_hotword_ids) != active_hotwords_per_case for case in cases):
        raise RuntimeError("v3 case does not contain exactly 100 active hotwords")

    destination.mkdir(parents=True, exist_ok=True)
    hotword_path = destination / ASSET_FILENAMES[0]
    family_path = destination / ASSET_FILENAMES[1]
    case_path = destination / ASSET_FILENAMES[2]
    selection_path = destination / ASSET_FILENAMES[3]
    summary_path = destination / ASSET_FILENAMES[4]
    write_hotword_table(hotword_path, entries)
    _write_jsonl(family_path, [family.to_dict() for family in families])
    _write_jsonl(case_path, [case.to_dict() for case in cases])
    _write_json(
        selection_path,
        {
            "seed": seed,
            "audio_disjoint_verified": True,
            "speaker_disjoint_verified": False,
            "cases": [case.to_dict() for case in cases],
        },
    )
    actual = {name: sum(case.primary_group == name for case in cases) for name in targets}
    shortages = {
        name: {
            "target": targets[name],
            "actual": actual[name],
            "missing": targets[name] - actual[name],
        }
        for name in targets
        if actual[name] < targets[name]
    }
    critical_minimum = min(actual["nested_short_only"], actual["nested_long_present"])
    conclusion_scope = "formal" if critical_minimum >= 10 else "smoke_insufficient_data"
    summary: dict[str, object] = {
        "asset_version": "simulated-hotwords-v3-multi-nested",
        "evaluation_scope": "validation_multi_nested_hotword_eval",
        "validation_manifest_path": str(manifest),
        "validation_manifest_sha256": _sha256_file(manifest),
        "dictionary_path": str(dictionary_file),
        "dictionary_sha256": _sha256_file(dictionary_file),
        "vocab_path": str(vocab_file),
        "vocab_sha256": _sha256_file(vocab_file),
        "output_dir": str(destination),
        "hotword_table_path": str(hotword_path),
        "families_path": str(family_path),
        "cases_path": str(case_path),
        "sample_selection_path": str(selection_path),
        "validation_records": len(samples),
        "candidate_hotwords": len(candidates),
        "selected_hotwords": len(entries),
        "nested_families": len(families),
        "target_case_counts": targets,
        "actual_case_counts": actual,
        "case_shortages": shortages,
        "total_cases": len(cases),
        "active_hotwords_per_case": active_hotwords_per_case,
        "fixed_seed": seed,
        "audio_disjoint_verified": True,
        "speaker_disjoint_verified": False,
        "test_set_used": False,
        "conclusion_scope": conclusion_scope,
        "status": "pass" if conclusion_scope == "formal" else "insufficient_data",
        "limitations": [
            "natural validation occurrences only; no audio concatenation or fabricated text",
            "speaker IDs are unavailable, so only audio-disjoint selection is verified",
            "strict normalized complete-word/contiguous-phrase matching; no aliases",
        ],
    }
    _write_json(summary_path, summary)
    return summary


def load_validation_samples(path: str | Path) -> tuple[ValidationSample, ...]:
    manifest = Path(path).expanduser()
    samples: list[ValidationSample] = []
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"manifest row {line_number} must be an object")
            split = _required_string(raw, "split", line_number)
            if split == "test":
                raise ValueError("sealed test data is forbidden in v3 evaluation")
            if split != "validation":
                raise ValueError(f"manifest row {line_number} is not validation: {split!r}")
            sample_id = _required_string(raw, "id", line_number)
            audio_path = _required_string(raw, "audio_path", line_number)
            if sample_id in seen_ids or audio_path in seen_audio:
                raise ValueError("validation manifest has duplicate IDs or audio paths")
            text = _required_string(raw, "text", line_number)
            words = tuple(tokenize_words(text))
            if not words:
                raise ValueError(f"manifest row {line_number} has no normalized words")
            seen_ids.add(sample_id)
            seen_audio.add(audio_path)
            samples.append(
                ValidationSample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    reference_text=text,
                    normalized_text=" ".join(words),
                    language=_required_string(raw, "language", line_number),
                    words=words,
                )
            )
    if not samples:
        raise ValueError(f"validation manifest is empty: {manifest}")
    return tuple(samples)


def load_multi_nested_cases(
    path: str | Path,
    *,
    expected_active_hotwords: int | None = 100,
) -> tuple[MultiNestedCase, ...]:
    if expected_active_hotwords is not None and expected_active_hotwords <= 0:
        raise ValueError("expected active hotword count must be positive")
    rows: list[MultiNestedCase] = []
    seen_cases: set[str] = set()
    seen_samples: set[str] = set()
    seen_audio: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"v3 case row {line_number} must be an object")
            spans_raw = raw.get("expected_word_spans")
            if not isinstance(spans_raw, dict):
                raise ValueError(f"v3 case row {line_number} has invalid spans")
            spans: dict[str, tuple[int, int]] = {}
            for key, value in spans_raw.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, list)
                    or len(value) != 2
                    or any(not isinstance(item, int) for item in value)
                ):
                    raise ValueError(f"v3 case row {line_number} has invalid span")
                spans[key] = (value[0], value[1])
            row = MultiNestedCase(
                case_id=_required_string(raw, "case_id", line_number),
                sample_id=_required_string(raw, "sample_id", line_number),
                audio_path=_required_string(raw, "audio_path", line_number),
                reference_text=_required_string(raw, "reference_text", line_number),
                normalized_reference_text=_required_string(
                    raw, "normalized_reference_text", line_number
                ),
                language=_required_string(raw, "language", line_number),
                primary_group=_required_string(raw, "primary_group", line_number),
                expected_hotword_ids=_string_tuple(raw, "expected_hotword_ids", line_number, True),
                expected_surfaces=_string_tuple(raw, "expected_surfaces", line_number, True),
                expected_word_spans=spans,
                containment_expected_ids=_string_tuple(
                    raw, "containment_expected_ids", line_number, True
                ),
                longest_match_expected_ids=_string_tuple(
                    raw, "longest_match_expected_ids", line_number, True
                ),
                active_hotword_ids=_string_tuple(raw, "active_hotword_ids", line_number),
                nested_family_ids=_string_tuple(raw, "nested_family_ids", line_number, True),
                hard_negative_ids=_string_tuple(raw, "hard_negative_ids", line_number, True),
                independent_expected_ids=_string_tuple(
                    raw, "independent_expected_ids", line_number, True
                ),
                selection_reason=_required_string(raw, "selection_reason", line_number),
            )
            if len(set(row.active_hotword_ids)) != len(row.active_hotword_ids):
                raise ValueError(f"v3 case {row.case_id} must have unique active hotwords")
            if (
                expected_active_hotwords is not None
                and len(row.active_hotword_ids) != expected_active_hotwords
            ):
                raise ValueError(
                    f"v3 case {row.case_id} must have "
                    f"{expected_active_hotwords} unique active hotwords"
                )
            if not set(row.containment_expected_ids).issubset(row.active_hotword_ids):
                raise ValueError(f"v3 case {row.case_id} expects inactive hotwords")
            if (
                row.case_id in seen_cases
                or row.sample_id in seen_samples
                or row.audio_path in seen_audio
            ):
                raise ValueError("v3 cases must have unique case, sample, and audio IDs")
            seen_cases.add(row.case_id)
            seen_samples.add(row.sample_id)
            seen_audio.add(row.audio_path)
            rows.append(row)
    if not rows:
        raise ValueError("v3 case table is empty")
    return tuple(rows)


def load_hotword_families(path: str | Path) -> tuple[HotwordFamily, ...]:
    families: list[HotwordFamily] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"family row {line_number} must be an object")
            families.append(
                HotwordFamily(
                    family_id=_required_string(raw, "family_id", line_number),
                    short_hotword_id=_required_string(raw, "short_hotword_id", line_number),
                    long_hotword_id=_required_string(raw, "long_hotword_id", line_number),
                    short_surface=_required_string(raw, "short_surface", line_number),
                    long_surface=_required_string(raw, "long_surface", line_number),
                )
            )
    return tuple(families)


def load_multi_nested_case_scores(path: str | Path) -> tuple[CaseScore, ...]:
    rows: list[CaseScore] = []
    seen_cases: set[str] = set()
    seen_samples: set[str] = set()
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"v3 score row {line_number} must be an object")
            case_id = _required_string(raw, "case_id", line_number)
            sample_id = _required_string(raw, "sample_id", line_number)
            if case_id in seen_cases or sample_id in seen_samples:
                raise ValueError("v3 scores must have unique case and sample IDs")
            ranking_raw = raw.get("ranking_top5")
            operating_raw = raw.get("operating_matches")
            if not isinstance(ranking_raw, list) or not isinstance(operating_raw, list):
                raise ValueError(f"v3 score row {line_number} has invalid match lists")
            rows.append(
                CaseScore(
                    case_id=case_id,
                    sample_id=sample_id,
                    primary_group=_required_string(raw, "primary_group", line_number),
                    ranked_matches=tuple(
                        _hotword_match_from_dict(item, line_number) for item in ranking_raw
                    ),
                    operating_matches=tuple(
                        _hotword_match_from_dict(item, line_number) for item in operating_raw
                    ),
                    effective_time_steps=_required_nonnegative_int(
                        raw, "effective_time_steps", line_number
                    ),
                    decoded_token_count=_required_nonnegative_int(
                        raw, "decoded_token_count", line_number
                    ),
                )
            )
            seen_cases.add(case_id)
            seen_samples.add(sample_id)
    if not rows:
        raise ValueError("v3 score table is empty")
    return tuple(rows)


def evaluate_multi_nested_case_scores(
    cases: Sequence[MultiNestedCase],
    hotwords: Sequence[HotwordEntry],
    families: Sequence[HotwordFamily],
    scores: Sequence[CaseScore],
) -> dict[str, object]:
    if {case.case_id for case in cases} != {score.case_id for score in scores}:
        raise ValueError("v3 cases and scores do not contain the same case IDs")
    score_by_id = {score.case_id: score for score in scores}
    entry_by_id = {entry.hotword_id: entry for entry in hotwords}
    family_by_id = {family.family_id: family for family in families}
    overall = _metric_block(cases, score_by_id, ground_truth="containment")
    by_group = {
        group: _metric_block(
            [case for case in cases if case.primary_group == group],
            score_by_id,
            ground_truth="containment",
        )
        for group in GROUP_TARGETS
    }
    form_ids = _form_hotword_ids(hotwords, families)
    by_form = {
        name: {
            "ranking": _ranking_for_ids(cases, score_by_id, ids),
            "operating": _operating_for_ids(cases, score_by_id, ids),
        }
        for name, ids in form_ids.items()
    }
    by_length: dict[str, object] = {}
    length_ids: dict[str, set[str]] = {}
    for name, minimum, maximum in (
        ("phonemes_4_7", 4, 7),
        ("phonemes_8_12", 8, 12),
        ("phonemes_13_18", 13, 18),
        ("phonemes_19_plus", 19, 10_000),
    ):
        ids = {entry.hotword_id for entry in hotwords if minimum <= len(entry.token_ids) <= maximum}
        length_ids[name] = ids
        by_length[name] = {
            "ranking": _ranking_for_ids(cases, score_by_id, ids),
            "operating": _operating_for_ids(cases, score_by_id, ids),
        }
    by_form_and_length = {
        form_name: {
            length_name: {
                "ranking": _ranking_for_ids(cases, score_by_id, form_set & bucket_set),
                "operating": _operating_for_ids(cases, score_by_id, form_set & bucket_set),
            }
            for length_name, bucket_set in length_ids.items()
        }
        for form_name, form_set in form_ids.items()
    }

    short_only = [case for case in cases if case.primary_group == "nested_short_only"]
    long_present = [
        case
        for case in cases
        if case.primary_group in {"nested_long_present", "nested_family_plus_two"}
    ]

    def case_short_ids(case: MultiNestedCase) -> set[str]:
        return {family_by_id[family_id].short_hotword_id for family_id in case.nested_family_ids}

    def case_long_ids(case: MultiNestedCase) -> set[str]:
        return {family_by_id[family_id].long_hotword_id for family_id in case.nested_family_ids}

    containment = _metric_block(long_present, score_by_id, ground_truth="containment")
    longest = _metric_block(
        long_present,
        score_by_id,
        ground_truth="longest",
        redundant_by_case={
            case.case_id: set(case.containment_expected_ids) - set(case.longest_match_expected_ids)
            for case in long_present
        },
    )
    short_only_long_ranking_trigger_cases = 0
    short_only_long_operating_trigger_cases = 0
    family_slots: list[int] = []
    redundant_hits = 0
    crowding_cases: list[dict[str, object]] = []
    for case in cases:
        score = score_by_id[case.case_id]
        ranked_ids = [match.hotword_id for match in score.ranked_matches[:5]]
        case_family_ids = {
            hotword_id
            for family_id in case.nested_family_ids
            for hotword_id in (
                family_by_id[family_id].short_hotword_id,
                family_by_id[family_id].long_hotword_id,
            )
        }
        if case.primary_group == "nested_short_only":
            corresponding_long_ids = {
                family_by_id[family_id].long_hotword_id for family_id in case.nested_family_ids
            }
            operating_ids = {match.hotword_id for match in score.operating_matches}
            short_only_long_ranking_trigger_cases += bool(set(ranked_ids) & corresponding_long_ids)
            short_only_long_operating_trigger_cases += bool(operating_ids & corresponding_long_ids)
        if case_family_ids:
            occupied = [hotword_id for hotword_id in ranked_ids if hotword_id in case_family_ids]
            family_slots.append(len(occupied))
            redundant_hits += max(0, len(occupied) - 1)
            missed = [
                hotword_id
                for hotword_id in case.independent_expected_ids
                if hotword_id not in ranked_ids
            ]
            if missed and len(occupied) > 1:
                crowding_cases.append(
                    {
                        "case_id": case.case_id,
                        "sample_id": case.sample_id,
                        "missed_independent_hotword_ids": missed,
                        "missed_independent_surfaces": [
                            entry_by_id[item].surface for item in missed
                        ],
                        "occupying_family_hotword_ids": occupied,
                        "occupying_family_surfaces": [
                            entry_by_id[item].surface for item in occupied
                        ],
                        "attribution": (
                            "family occupied multiple Top-5 slots while an independent "
                            "target was absent"
                        ),
                    }
                )
    ordinary_three = [case for case in cases if case.primary_group == "three_independent"]
    nested_plus_two = [case for case in cases if case.primary_group == "nested_family_plus_two"]
    ordinary_other_recall = _recall_for_case_ids(
        ordinary_three, score_by_id, lambda case: set(case.independent_expected_ids), 5
    )
    nested_other_recall = _recall_for_case_ids(
        nested_plus_two, score_by_id, lambda case: set(case.independent_expected_ids), 5
    )
    nested_metrics: dict[str, object] = {
        "short_only_short_recall_at_5": _recall_for_case_ids(
            short_only, score_by_id, case_short_ids, 5
        ),
        "short_only_short_operating_recall": _operating_recall_for_case_ids(
            short_only, score_by_id, case_short_ids
        ),
        "short_only_long_ranking_false_trigger_rate_at_5": _safe_ratio(
            short_only_long_ranking_trigger_cases, len(short_only)
        ),
        "short_only_long_operating_false_trigger_rate": _safe_ratio(
            short_only_long_operating_trigger_cases, len(short_only)
        ),
        "long_present_long_recall_at_5": _recall_for_case_ids(
            long_present, score_by_id, case_long_ids, 5
        ),
        "long_present_long_operating_recall": _operating_recall_for_case_ids(
            long_present, score_by_id, case_long_ids
        ),
        "long_present_short_simultaneous_recall_at_5": _recall_for_case_ids(
            long_present,
            score_by_id,
            case_short_ids,
            5,
        ),
        "long_present_short_operating_recall": _operating_recall_for_case_ids(
            long_present,
            score_by_id,
            case_short_ids,
        ),
        "containment": containment,
        "longest_match": longest,
        "mean_family_slots_in_top5": _safe_ratio(sum(family_slots), len(family_slots)),
        "family_duplicate_or_redundant_hits": redundant_hits,
        "nested_plus_two_independent_recall_at_5": nested_other_recall,
        "ordinary_three_independent_recall_at_5": ordinary_other_recall,
        "slot_crowding_loss": ordinary_other_recall - nested_other_recall,
        "crowding_attribution_cases": crowding_cases,
    }
    single_recall = _recall_value(by_form["single_word"], 5)
    compound_recall = _recall_value(by_form["multiword_non_nested"], 5)
    return {
        "overall": overall,
        "by_primary_group": by_group,
        "by_hotword_form": by_form,
        "by_phoneme_length": by_length,
        "by_hotword_form_and_phoneme_length": by_form_and_length,
        "multiword_minus_single_recall_at_5": compound_recall - single_recall,
        "nested": nested_metrics,
        "per_case": [_per_case_metrics(case, score_by_id[case.case_id]) for case in cases],
    }


def score_multi_nested_cases(
    checkpoint_path: str | Path,
    cache: DiskFeatureCache,
    vocab: PhonemeVocab,
    hotwords: Sequence[HotwordEntry],
    families: Sequence[HotwordFamily],
    cases: Sequence[MultiNestedCase],
    output_dir: str | Path,
    *,
    device: Any,
    manifest_path: str | Path,
    dictionary_path: str | Path,
    vocab_path: str | Path,
    hotword_table_path: str | Path,
    families_path: str | Path,
    cases_path: str | Path,
    asset_summary_path: str | Path,
    batch_size: int = 128,
    threshold: float = 0.86,
    top_k: int = 5,
    minimum_phonemes: int = 4,
    maximum_edit_ratio: float = 0.35,
    posterior_weight: float = 0.25,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.0,
) -> dict[str, object]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from qwen_hotword.hotwords.evaluation import _load_checkpoint
    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
        ctc_head_config,
    )

    if cache.split != "validation":
        raise ValueError("v3 evaluation accepts only validation feature caches")
    if (
        (threshold, top_k, minimum_phonemes, maximum_edit_ratio, posterior_weight)
        != (
            0.86,
            5,
            4,
            0.35,
            0.25,
        )
        or minimum_posterior_confidence != 0.0
        or minimum_top1_margin != 0.0
    ):
        raise ValueError("v3 scoring parameters are fixed and cannot be tuned in this run")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    incompatible = {
        path.name
        for path in destination.iterdir()
        if path.name not in {*ASSET_FILENAMES, *SCORE_FILENAMES}
    }
    if incompatible:
        raise FileExistsError(
            f"v3 score output contains incompatible files: {sorted(incompatible)}"
        )
    for filename in SCORE_FILENAMES:
        if (destination / filename).exists():
            raise FileExistsError(f"refusing to overwrite existing v3 score output: {filename}")
    checkpoint = Path(checkpoint_path).expanduser()
    payload = _load_checkpoint(checkpoint, vocab)
    head = build_ctc_head_from_checkpoint(payload)
    if not isinstance(head, TemporalUpsampleCtcHead) or head.time_upsampling_factor != 2:
        raise ValueError("v3 evaluation requires the Temporal 2x CTC Head")
    head = head.to(device=device, dtype=torch.float32)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    by_id = {entry.hotword_id: entry for entry in hotwords}
    case_by_sample = {case.sample_id: case for case in cases}
    if len(case_by_sample) != len(cases):
        raise ValueError("v3 cases have duplicate sample IDs")
    cached_ids = {sample_id for shard in cache.shards for sample_id in shard.sample_ids}
    missing = set(case_by_sample) - cached_ids
    if missing:
        raise ValueError(f"{len(missing)} v3 cases are absent from the validation cache")
    ranking_config = HotwordScoringConfig(
        score_threshold=0.0,
        top_k=100,
        minimum_phonemes=minimum_phonemes,
        maximum_edit_ratio=1.0,
        posterior_weight=posterior_weight,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.0,
    )
    scores: list[CaseScore] = []
    started = time.monotonic()
    processed = 0
    with torch.no_grad():
        for shard_index, descriptor in enumerate(cache.shards, start=1):
            wanted = set(descriptor.sample_ids) & set(case_by_sample)
            if not wanted:
                continue
            samples = [
                sample
                for sample in load_feature_shard(descriptor, num_classes=len(vocab.tokens))
                if sample.sample_id in wanted
            ]
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                hidden = pad_sequence(
                    [sample.hidden_states for sample in batch], batch_first=True, padding_value=0.0
                ).to(device=device, dtype=torch.float32)
                lengths = torch.tensor(
                    [sample.hidden_states.shape[0] for sample in batch],
                    dtype=torch.long,
                    device=device,
                )
                logits = head(hidden, input_lengths=lengths)
                effective_lengths = head.output_lengths(lengths)
                for row_index, sample in enumerate(batch):
                    case = case_by_sample[sample.sample_id]
                    active = [by_id[item] for item in case.active_hotword_ids]
                    result = score_hotwords(
                        logits[row_index],
                        input_length=int(effective_lengths[row_index].item()),
                        hotwords=active,
                        config=ranking_config,
                        blank_id=0,
                    )
                    all_ranked = result.ranked_matches
                    ranked = all_ranked[:top_k]
                    operating = tuple(
                        match
                        for match in all_ranked
                        if match.score >= threshold
                        and match.edit_ratio <= maximum_edit_ratio
                        and match.posterior_confidence >= minimum_posterior_confidence
                    )[:top_k]
                    scores.append(
                        CaseScore(
                            case_id=case.case_id,
                            sample_id=case.sample_id,
                            primary_group=case.primary_group,
                            ranked_matches=ranked,
                            operating_matches=operating,
                            effective_time_steps=result.effective_time_steps,
                            decoded_token_count=len(result.decoded_token_ids),
                        )
                    )
                processed += len(batch)
                elapsed = time.monotonic() - started
                rate = processed / elapsed if elapsed else 0.0
                eta = (len(cases) - processed) / rate if rate else 0.0
                print(
                    f"v3 scoring cases={processed}/{len(cases)} "
                    f"shard={shard_index}/{len(cache.shards)} "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s eta={eta:.1f}s",
                    flush=True,
                )
    if len(scores) != len(cases):
        raise RuntimeError(f"v3 scored {len(scores)} cases, expected {len(cases)}")
    scores.sort(key=lambda item: item.case_id)
    metrics = evaluate_multi_nested_case_scores(cases, hotwords, families, scores)
    family_by_id = {family.family_id: family for family in families}
    case_by_id = {case.case_id: case for case in cases}
    score_rows = [_score_row(case_by_id[score.case_id], score, family_by_id) for score in scores]
    score_path = destination / SCORE_FILENAMES[0]
    report_path = destination / SCORE_FILENAMES[1]
    _write_jsonl(score_path, score_rows)
    cache_index = cache.root / "cache_index.json"
    asset_summary_file = Path(asset_summary_path)
    asset_summary = json.loads(asset_summary_file.read_text(encoding="utf-8"))
    if not isinstance(asset_summary, dict):
        raise ValueError("v3 asset summary must be a JSON object")
    report: dict[str, object] = {
        "purpose": "validation_multi_nested_ctc_hotword_specialized_evaluation",
        "evaluation_scope": "validation_multi_nested_hotword_eval",
        "test_set_used": False,
        "feature_cache_reused": True,
        "encoder_feature_extraction_performed": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "manifest_path": str(Path(manifest_path)),
        "manifest_sha256": _sha256_file(Path(manifest_path)),
        "validation_cache_dir": str(cache.root),
        "cache_index_path": str(cache_index),
        "cache_index_sha256": _sha256_file(cache_index),
        "vocab_path": str(Path(vocab_path)),
        "vocab_sha256": _sha256_file(Path(vocab_path)),
        "dictionary_path": str(Path(dictionary_path)),
        "dictionary_sha256": _sha256_file(Path(dictionary_path)),
        "hotword_table_path": str(Path(hotword_table_path)),
        "hotword_table_sha256": _sha256_file(Path(hotword_table_path)),
        "families_path": str(Path(families_path)),
        "families_sha256": _sha256_file(Path(families_path)),
        "cases_path": str(Path(cases_path)),
        "cases_sha256": _sha256_file(Path(cases_path)),
        "asset_summary_path": str(asset_summary_file),
        "asset_summary_sha256": _sha256_file(asset_summary_file),
        "fixed_seed": asset_summary.get("fixed_seed"),
        "target_case_counts": asset_summary.get("target_case_counts"),
        "actual_case_counts": asset_summary.get("actual_case_counts"),
        "asset_conclusion_scope": asset_summary.get("conclusion_scope"),
        "head_config": ctc_head_config(head),
        "scoring_config": {
            "top_k": top_k,
            "threshold": threshold,
            "maximum_edit_ratio": maximum_edit_ratio,
            "posterior_weight": posterior_weight,
            "minimum_posterior_confidence": minimum_posterior_confidence,
            "minimum_phonemes": minimum_phonemes,
            "minimum_top1_margin": minimum_top1_margin,
            "time_axis": "temporal_upsample_2x_only",
        },
        "ranking_definition": "forced Top-5 without score threshold; not a deployment trigger",
        "operating_definition": "threshold=0.86 plus edit/posterior guards, capped at Top-5",
        "ground_truth_definitions": {
            "containment": "all naturally spoken complete nested members count as expected",
            "longest_match": (
                "only the longest family member is a target; shorter hits are redundant"
            ),
        },
        "metric_definitions": {
            "micro_recall_at_k": "retrieved expected hotwords divided by expected hotwords",
            "any_hit_at_k": (
                "positive cases with at least one expected Top-K hit divided by positive cases"
            ),
            "all_hit_at_k": (
                "positive cases with every expected hotword in Top-K divided by positive cases"
            ),
            "mean_hits_at_k": "retrieved expected hotwords divided by positive cases",
            "raw_precision_at_5": (
                "expected hits divided by forced ranked candidates; a three-target case "
                "returning five candidates has a 60% ceiling by construction, not a model "
                "accuracy ceiling"
            ),
            "slot_crowding_loss": (
                "ordinary three-independent Recall@5 minus the two-independent-target "
                "Recall@5 in nested-family-plus-two cases"
            ),
        },
        "engineering_reference_targets": {
            "three_hotword_micro_recall_at_5": 0.95,
            "three_hotword_all_3_hit_at_5": 0.85,
            "operating_precision": 0.90,
            "negative_case_false_positive_rate": 0.03,
            "short_only_long_operating_false_trigger_rate": 0.03,
            "slot_crowding_loss": 0.05,
            "classification": "project engineering references, not universal standards",
        },
        "metrics": metrics,
        "case_scores_path": str(score_path),
        "scoring_seconds": time.monotonic() - started,
        "limitations": [
            "natural validation only; no speaker-disjoint claim",
            "ranking Top-5 forces candidates even for negative audio",
            "no production longest-match suppression is applied",
        ],
        "execution_status": "pass",
        "status": (
            "pass" if asset_summary.get("conclusion_scope") == "formal" else "insufficient_data"
        ),
    }
    _write_json(report_path, report)
    return report


def _collect_candidates(
    samples: Sequence[ValidationSample],
    dictionary: Mapping[str, list[str]],
    vocab: PhonemeVocab,
) -> list[Candidate]:
    spans: dict[tuple[str, tuple[str, ...]], dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        stopwords = stopwords_for_language(sample.language)
        for width in (1, 2, 3):
            for start in range(len(sample.words) - width + 1):
                words = sample.words[start : start + width]
                if sum(len(word) for word in words) < 5:
                    continue
                if words[0] in stopwords or words[-1] in stopwords:
                    continue
                spans[(sample.language, words)][sample.sample_id].append((start, start + width))
    candidates: list[Candidate] = []
    for (language, words), sample_spans in spans.items():
        if any(word not in dictionary or not dictionary[word] for word in words):
            continue
        pronunciation = " ".join(dictionary[word][0] for word in words)
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        if tokenized.oov_units or not 4 <= len(tokenized.token_ids) <= 48:
            continue
        candidates.append(
            Candidate(
                language=language,
                words=words,
                pronunciation=pronunciation,
                phoneme_tokens=tuple(tokenized.tokens),
                token_ids=tuple(tokenized.token_ids),
                sample_spans={key: tuple(value) for key, value in sample_spans.items()},
            )
        )
    candidates.sort(key=lambda item: item.key)
    return candidates


def _collect_family_keys(candidates: Sequence[Candidate]) -> list[tuple[str, str]]:
    by_words = {(candidate.language, candidate.words): candidate for candidate in candidates}
    pairs: list[tuple[str, str]] = []
    for long in candidates:
        if len(long.words) < 2:
            continue
        for start in range(len(long.words)):
            short = by_words.get((long.language, (long.words[start],)))
            if short is None:
                continue
            short_only = set(short.sample_spans) - set(long.sample_spans)
            if short_only and long.sample_spans:
                pairs.append((short.key, long.key))
    return sorted(set(pairs))


def _deduplicate_pronunciations(candidates: Sequence[Candidate], *, seed: int) -> list[Candidate]:
    selected: list[Candidate] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -item.occurrences,
            _stable_rank(seed, f"pronunciation:{item.key}"),
            item.key,
        ),
    ):
        key = (candidate.language, candidate.token_ids)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    selected.sort(key=lambda item: item.key)
    return selected


def _select_case_drafts(
    samples: Sequence[ValidationSample],
    by_key: Mapping[str, Candidate],
    family_keys: Sequence[tuple[str, str]],
    targets: Mapping[str, int],
    seed: int,
) -> list[_CaseDraft]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    candidate_keys_by_sample: dict[str, list[str]] = defaultdict(list)
    for key, candidate in by_key.items():
        for sample_id in candidate.sample_spans:
            candidate_keys_by_sample[sample_id].append(key)
    used_audio: set[str] = set()
    drafts: list[_CaseDraft] = []

    def add(draft: _CaseDraft) -> bool:
        if draft.sample.audio_path in used_audio:
            return False
        used_audio.add(draft.sample.audio_path)
        drafts.append(draft)
        return True

    family_occurrences: list[tuple[int, tuple[str, str], str]] = []
    for pair in family_keys:
        short, long = pair
        for sample_id in by_key[long].sample_spans:
            family_occurrences.append(
                (_stable_rank(seed, f"long:{short}:{long}:{sample_id}"), pair, sample_id)
            )
    family_occurrences.sort()
    count = 0
    for _, pair, sample_id in family_occurrences:
        if count >= targets["nested_family_plus_two"]:
            break
        sample = sample_by_id[sample_id]
        independent = _independent_keys(
            sample,
            candidate_keys_by_sample[sample_id],
            by_key,
            excluded=set(pair),
            count=2,
            seed=seed,
            preferred_width=1,
        )
        if len(independent) != 2:
            continue
        expected = (pair[0], pair[1], *independent)
        spans = _spans_for_keys(sample_id, independent, by_key)
        spans.update(_nested_family_spans(sample_id, pair, by_key))
        if add(
            _CaseDraft(
                sample,
                "nested_family_plus_two",
                expected,
                expected,
                (pair[1], *independent),
                independent,
                (pair,),
                spans,
                f"natural long family plus two non-overlapping independent hotwords; seed={seed}",
            )
        ):
            count += 1

    short_occurrences: list[tuple[int, tuple[str, str], str]] = []
    for pair in family_keys:
        short, long = pair
        for sample_id in set(by_key[short].sample_spans) - set(by_key[long].sample_spans):
            short_occurrences.append(
                (_stable_rank(seed, f"short:{short}:{long}:{sample_id}"), pair, sample_id)
            )
    short_occurrences.sort()
    count = 0
    for _, pair, sample_id in short_occurrences:
        if count >= targets["nested_short_only"]:
            break
        sample = sample_by_id[sample_id]
        spans = _spans_for_keys(sample_id, (pair[0],), by_key)
        if add(
            _CaseDraft(
                sample,
                "nested_short_only",
                (pair[0],),
                (pair[0],),
                (pair[0],),
                (),
                (pair,),
                spans,
                f"natural short member without complete long phrase; seed={seed}",
            )
        ):
            count += 1

    count = 0
    for _, pair, sample_id in family_occurrences:
        if count >= targets["nested_long_present"]:
            break
        sample = sample_by_id[sample_id]
        spans = _nested_family_spans(sample_id, pair, by_key)
        if add(
            _CaseDraft(
                sample,
                "nested_long_present",
                pair,
                pair,
                (pair[1],),
                (),
                (pair,),
                spans,
                f"natural long phrase with contained short member; seed={seed}",
            )
        ):
            count += 1

    for group, desired, multiplicity in (
        ("three_independent", targets["three_independent"], 3),
        ("two_independent", targets["two_independent"], 2),
        ("single_hotword", targets["single_hotword"], 1),
    ):
        count = 0
        ordered_samples = sorted(
            samples, key=lambda item: _stable_rank(seed, f"{group}:{item.sample_id}")
        )
        for sample in ordered_samples:
            if count >= desired:
                break
            keys = _independent_keys(
                sample,
                candidate_keys_by_sample[sample.sample_id],
                by_key,
                excluded=set(),
                count=multiplicity,
                seed=seed,
                preferred_width=(1 if count % 2 == 0 else 2) if multiplicity == 1 else None,
            )
            if len(keys) != multiplicity:
                continue
            spans = _spans_for_keys(sample.sample_id, keys, by_key)
            if add(
                _CaseDraft(
                    sample,
                    group,
                    keys,
                    keys,
                    keys,
                    keys,
                    (),
                    spans,
                    f"{multiplicity} natural independent non-overlapping hotword(s); seed={seed}",
                )
            ):
                count += 1

    negative_count = 0
    for sample in sorted(
        samples, key=lambda item: _stable_rank(seed, f"negative:{item.sample_id}")
    ):
        if negative_count >= targets["negative"]:
            break
        if add(
            _CaseDraft(
                sample,
                "negative",
                (),
                (),
                (),
                (),
                (),
                {},
                f"deterministic natural negative with strictly absent active hotwords; seed={seed}",
            )
        ):
            negative_count += 1
    return drafts


def _independent_keys(
    sample: ValidationSample,
    keys: Sequence[str],
    by_key: Mapping[str, Candidate],
    *,
    excluded: set[str],
    count: int,
    seed: int,
    preferred_width: int | None = None,
) -> tuple[str, ...]:
    eligible = [key for key in keys if key not in excluded and len(by_key[key].words) <= 2]
    eligible.sort(
        key=lambda key: (
            preferred_width is not None and len(by_key[key].words) != preferred_width,
            _stable_rank(seed, f"independent:{sample.sample_id}:{key}"),
            len(by_key[key].words),
            key,
        )
    )
    chosen: list[str] = []
    chosen_spans = [
        span
        for key in excluded
        if sample.sample_id in by_key[key].sample_spans
        for span in by_key[key].sample_spans[sample.sample_id]
    ]
    for key in eligible:
        candidate = by_key[key]
        if any(
            _contains_words(candidate.words, by_key[other].words) for other in (*chosen, *excluded)
        ):
            continue
        span = candidate.sample_spans[sample.sample_id][0]
        if any(_overlaps(span, other) for other in chosen_spans):
            continue
        chosen.append(key)
        chosen_spans.append(span)
        if len(chosen) == count:
            return tuple(chosen)
    return ()


def _materialize_case(
    draft: _CaseDraft,
    *,
    index: int,
    entries: Sequence[HotwordEntry],
    entry_by_id: Mapping[str, HotwordEntry],
    id_by_key: Mapping[str, str],
    family_id_by_pair: Mapping[tuple[str, str], str],
    active_count: int,
    seed: int,
) -> MultiNestedCase:
    expected_ids = tuple(id_by_key[key] for key in draft.expected_keys)
    containment_ids = tuple(id_by_key[key] for key in draft.containment_keys)
    longest_ids = tuple(id_by_key[key] for key in draft.longest_keys)
    family_member_ids = {id_by_key[key] for pair in draft.family_keys for key in pair}
    required = list(dict.fromkeys((*containment_ids, *family_member_ids)))
    absent = [
        entry
        for entry in entries
        if entry.hotword_id not in required
        and not contains_token_sequence(list(draft.sample.words), entry.words)
    ]
    reference_targets = [entry_by_id[item] for item in containment_ids]
    absent.sort(
        key=lambda entry: (
            _nearest_edit_ratio(entry, reference_targets),
            _stable_rank(seed, f"active:{draft.sample.sample_id}:{entry.hotword_id}"),
        )
    )
    family_hard = sorted(family_member_ids - set(containment_ids))
    hard = tuple(
        (family_hard + [entry.hotword_id for entry in absent])[
            : min(10, len(absent) + len(family_hard))
        ]
    )
    selected = required + [entry.hotword_id for entry in absent]
    selected = list(dict.fromkeys(selected))[:active_count]
    if len(selected) != active_count:
        raise ValueError(f"case {draft.sample.sample_id} cannot fill 100 absent active hotwords")
    return MultiNestedCase(
        case_id=f"sim_v3_{draft.group}_{index:04d}",
        sample_id=draft.sample.sample_id,
        audio_path=draft.sample.audio_path,
        reference_text=draft.sample.reference_text,
        normalized_reference_text=draft.sample.normalized_text,
        language=draft.sample.language,
        primary_group=draft.group,
        expected_hotword_ids=expected_ids,
        expected_surfaces=tuple(entry_by_id[item].surface for item in expected_ids),
        expected_word_spans={id_by_key[key]: span for key, span in draft.spans.items()},
        containment_expected_ids=containment_ids,
        longest_match_expected_ids=longest_ids,
        active_hotword_ids=tuple(selected),
        nested_family_ids=tuple(family_id_by_pair[pair] for pair in draft.family_keys),
        hard_negative_ids=hard,
        independent_expected_ids=tuple(id_by_key[key] for key in draft.independent_keys),
        selection_reason=draft.reason,
    )


def _metric_block(
    cases: Sequence[MultiNestedCase],
    score_by_id: Mapping[str, CaseScore],
    *,
    ground_truth: str,
    redundant_by_case: Mapping[str, set[str]] | None = None,
) -> dict[str, object]:
    def expected(case: MultiNestedCase) -> set[str]:
        values = (
            case.containment_expected_ids
            if ground_truth == "containment"
            else case.longest_match_expected_ids
        )
        return set(values)

    ranking: list[dict[str, object]] = []
    for k in (1, 3, 5):
        expected_total = hits = positive_cases = any_hits = all_hits = 0
        for case in cases:
            truth = expected(case)
            if not truth:
                continue
            positive_cases += 1
            expected_total += len(truth)
            ranked = {match.hotword_id for match in score_by_id[case.case_id].ranked_matches[:k]}
            found = len(truth & ranked)
            hits += found
            any_hits += bool(found)
            all_hits += found == len(truth)
        ranking.append(
            {
                "k": k,
                "expected_hotwords": expected_total,
                "retrieved_expected_hotwords": hits,
                "micro_recall_at_k": _safe_ratio(hits, expected_total),
                "any_hit_at_k": _safe_ratio(any_hits, positive_cases),
                "all_hit_at_k": _safe_ratio(all_hits, positive_cases),
                "mean_hits_at_k": _safe_ratio(hits, positive_cases),
            }
        )
    expected_total = selected_total = true_total = positive = positive_hit = negative = (
        negative_fp
    ) = 0
    raw_hits = raw_selected = all_three = three_cases = 0
    for case in cases:
        truth = expected(case)
        score = score_by_id[case.case_id]
        ranked_ids = {match.hotword_id for match in score.ranked_matches[:5]}
        selected_ids = {match.hotword_id for match in score.operating_matches}
        precision_selected_ids = selected_ids - (
            redundant_by_case.get(case.case_id, set()) if redundant_by_case else set()
        )
        raw_hits += len(truth & ranked_ids)
        raw_selected += len(score.ranked_matches[:5])
        if truth:
            positive += 1
            expected_total += len(truth)
            positive_hit += bool(truth & selected_ids)
            if len(truth) == 3:
                three_cases += 1
                all_three += truth.issubset(ranked_ids)
        else:
            negative += 1
            negative_fp += bool(selected_ids)
        selected_total += len(precision_selected_ids)
        true_total += len(truth & selected_ids)
    precision = _safe_ratio(true_total, selected_total)
    recall = _safe_ratio(true_total, expected_total)
    return {
        "case_count": len(cases),
        "ranking": ranking,
        "raw_precision_at_5": _safe_ratio(raw_hits, raw_selected),
        "all_3_hit_at_5": _safe_ratio(all_three, three_cases),
        "operating": {
            "expected_hotwords": expected_total,
            "selected_hotwords": selected_total,
            "true_positive_hotwords": true_total,
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2 * precision * recall, precision + recall),
            "positive_case_hit_rate": _safe_ratio(positive_hit, positive),
            "negative_case_false_positive_rate": _safe_ratio(negative_fp, negative),
        },
    }


def _ranking_for_ids(
    cases: Sequence[MultiNestedCase], score_by_id: Mapping[str, CaseScore], ids: set[str]
) -> list[dict[str, object]]:
    filtered: list[MultiNestedCase] = []
    for case in cases:
        expected = tuple(item for item in case.containment_expected_ids if item in ids)
        if expected:
            filtered.append(
                MultiNestedCase(**{**asdict(case), "containment_expected_ids": expected})
            )
    return _metric_block(filtered, score_by_id, ground_truth="containment")["ranking"]  # type: ignore[return-value]


def _operating_for_ids(
    cases: Sequence[MultiNestedCase], score_by_id: Mapping[str, CaseScore], ids: set[str]
) -> dict[str, object]:
    expected_total = selected = true = 0
    for case in cases:
        truth = set(case.containment_expected_ids) & ids
        if not truth:
            continue
        selected_ids = {
            match.hotword_id
            for match in score_by_id[case.case_id].operating_matches
            if match.hotword_id in ids
        }
        expected_total += len(truth)
        selected += len(selected_ids)
        true += len(truth & selected_ids)
    precision = _safe_ratio(true, selected)
    recall = _safe_ratio(true, expected_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2 * precision * recall, precision + recall),
    }


def _form_hotword_ids(
    hotwords: Sequence[HotwordEntry], families: Sequence[HotwordFamily]
) -> dict[str, set[str]]:
    short = {family.short_hotword_id for family in families}
    long = {family.long_hotword_id for family in families}
    return {
        "single_word": {
            entry.hotword_id
            for entry in hotwords
            if len(entry.words) == 1 and entry.hotword_id not in short
        },
        "multiword_non_nested": {
            entry.hotword_id
            for entry in hotwords
            if len(entry.words) > 1 and entry.hotword_id not in long
        },
        "nested_short": short,
        "nested_long": long,
    }


def _per_case_metrics(case: MultiNestedCase, score: CaseScore) -> dict[str, object]:
    truth = set(case.containment_expected_ids)
    ranked = {match.hotword_id for match in score.ranked_matches[:5]}
    selected = {match.hotword_id for match in score.operating_matches}
    return {
        "case_id": case.case_id,
        "sample_id": case.sample_id,
        "primary_group": case.primary_group,
        "expected_count": len(truth),
        "ranking_correct_hits_at_5": len(truth & ranked),
        "ranking_missed_at_5": len(truth - ranked),
        "ranking_wrong_candidates_at_5": len(ranked - truth),
        "operating_correct_hits": len(truth & selected),
        "operating_missed": len(truth - selected),
        "operating_false_triggers": len(selected - truth),
    }


def _score_row(
    case: MultiNestedCase,
    score: CaseScore,
    family_by_id: Mapping[str, HotwordFamily],
) -> dict[str, object]:
    ranking_ids = [match.hotword_id for match in score.ranked_matches]
    operating_ids = [match.hotword_id for match in score.operating_matches]
    family_ids = {
        item
        for family_id in case.nested_family_ids
        for item in (
            family_by_id[family_id].short_hotword_id,
            family_by_id[family_id].long_hotword_id,
        )
    }
    missed_independent = [item for item in case.independent_expected_ids if item not in ranking_ids]
    occupying = [item for item in ranking_ids if item in family_ids]
    return {
        "case_id": case.case_id,
        "sample_id": case.sample_id,
        "primary_group": case.primary_group,
        "effective_time_steps": score.effective_time_steps,
        "decoded_token_count": score.decoded_token_count,
        "ranking_top5": [match.to_dict() for match in score.ranked_matches],
        "operating_matches": [match.to_dict() for match in score.operating_matches],
        "ranking_containment_hits": [
            item for item in ranking_ids if item in case.containment_expected_ids
        ],
        "ranking_longest_match_hits": [
            item for item in ranking_ids if item in case.longest_match_expected_ids
        ],
        "operating_containment_hits": [
            item for item in operating_ids if item in case.containment_expected_ids
        ],
        "operating_longest_match_hits": [
            item for item in operating_ids if item in case.longest_match_expected_ids
        ],
        "family_slot_occupancy": len(occupying),
        "family_member_hits": occupying,
        "crowded_out_independent_hotword_ids": missed_independent if len(occupying) > 1 else [],
        "ranking_false_triggers": [
            item for item in ranking_ids if item not in case.containment_expected_ids
        ],
        "operating_false_triggers": [
            item for item in operating_ids if item not in case.containment_expected_ids
        ],
    }


def _spans_for_keys(
    sample_id: str, keys: Iterable[str], by_key: Mapping[str, Candidate]
) -> dict[str, tuple[int, int]]:
    return {key: by_key[key].sample_spans[sample_id][0] for key in keys}


def _nested_family_spans(
    sample_id: str,
    pair: tuple[str, str],
    by_key: Mapping[str, Candidate],
) -> dict[str, tuple[int, int]]:
    short_key, long_key = pair
    long_span = by_key[long_key].sample_spans[sample_id][0]
    nested_short = next(
        (
            span
            for span in by_key[short_key].sample_spans[sample_id]
            if long_span[0] <= span[0] and span[1] <= long_span[1]
        ),
        None,
    )
    if nested_short is None:
        raise RuntimeError("nested family short span is not contained in the long span")
    return {short_key: nested_short, long_key: long_span}


def _nearest_edit_ratio(entry: HotwordEntry, targets: Sequence[HotwordEntry]) -> float:
    if not targets:
        return 1.0
    return min(
        sequence_edit_distance(entry.token_ids, target.token_ids)
        / max(len(entry.token_ids), len(target.token_ids))
        for target in targets
    )


def _recall_for_case_ids(
    cases: Sequence[MultiNestedCase],
    score_by_id: Mapping[str, CaseScore],
    expected: Any,
    k: int,
) -> float:
    total = hits = 0
    for case in cases:
        truth = expected(case)
        ranked = {match.hotword_id for match in score_by_id[case.case_id].ranked_matches[:k]}
        total += len(truth)
        hits += len(truth & ranked)
    return _safe_ratio(hits, total)


def _operating_recall_for_case_ids(
    cases: Sequence[MultiNestedCase],
    score_by_id: Mapping[str, CaseScore],
    expected: Any,
) -> float:
    total = hits = 0
    for case in cases:
        truth = expected(case)
        selected = {match.hotword_id for match in score_by_id[case.case_id].operating_matches}
        total += len(truth)
        hits += len(truth & selected)
    return _safe_ratio(hits, total)


def _recall_value(block: object, k: int) -> float:
    if not isinstance(block, dict) or not isinstance(block.get("ranking"), list):
        return 0.0
    for item in block["ranking"]:
        if isinstance(item, dict) and item.get("k") == k:
            value = item.get("micro_recall_at_k")
            return float(value) if isinstance(value, (int, float)) else 0.0
    return 0.0


def _contains_words(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return contains_token_sequence(list(left), right) or contains_token_sequence(list(right), left)


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _stable_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _required_string(raw: Mapping[str, object], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {key}")
    return value.strip()


def _required_nonnegative_int(raw: Mapping[str, object], key: str, line_number: int) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"row {line_number} has invalid {key}")
    return value


def _hotword_match_from_dict(value: object, line_number: int) -> HotwordMatch:
    if not isinstance(value, dict):
        raise ValueError(f"v3 score row {line_number} has a non-object match")

    def number(key: str) -> float:
        raw = value.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"v3 score row {line_number} match has invalid {key}")
        return float(raw)

    def integer(key: str) -> int:
        raw = value.get(key)
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ValueError(f"v3 score row {line_number} match has invalid {key}")
        return raw

    def optional_integer(key: str) -> int | None:
        raw = value.get(key)
        if raw is None:
            return None
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise ValueError(f"v3 score row {line_number} match has invalid {key}")
        return raw

    return HotwordMatch(
        hotword_id=_required_string(value, "hotword_id", line_number),
        surface=_required_string(value, "surface", line_number),
        language=_required_string(value, "language", line_number),
        score=number("score"),
        edit_similarity=number("edit_similarity"),
        edit_distance=integer("edit_distance"),
        edit_ratio=number("edit_ratio"),
        posterior_confidence=number("posterior_confidence"),
        decoded_start=integer("decoded_start"),
        decoded_end=integer("decoded_end"),
        start_step=optional_integer("start_step"),
        end_step=optional_integer("end_step"),
    )


def _string_tuple(
    raw: Mapping[str, object], key: str, line_number: int, allow_empty: bool = False
) -> tuple[str, ...]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"row {line_number} has invalid {key}")
    return tuple(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
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
