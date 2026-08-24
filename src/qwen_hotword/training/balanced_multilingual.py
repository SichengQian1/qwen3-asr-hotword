from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

LANGUAGE_ORDER = ("en", "es", "pt")
EXPECTED_LANGUAGE_TAGS = {
    "en": frozenset({"en-US"}),
    "es": frozenset({"es"}),
    "pt": frozenset({"pt", "pt-BR"}),
}
DATASET_VERSION = "en-es-pt-temporal2x-balanced-v2"
VALIDATION_DATASET_VERSION = "en-es-pt-temporal2x-balanced-validation-v1"
EXPERIMENT_NAME = "full-ctc-v1"


@dataclass(frozen=True)
class LanguagePool:
    name: str
    root: Path


@dataclass(frozen=True)
class Candidate:
    priority: str
    sample_id: str
    audio_path: str
    duration_seconds: float
    source_corpus: str
    release_source: str
    speaker_id: str | None


@dataclass
class CountHours:
    records: int = 0
    duration_seconds: float = 0.0

    def add(self, duration_seconds: float) -> None:
        self.records += 1
        self.duration_seconds += duration_seconds

    def to_dict(self) -> dict[str, int | float]:
        return {
            "records": self.records,
            "hours": self.duration_seconds / 3600.0,
        }


def parse_language_pool(value: str) -> LanguagePool:
    name, separator, raw_path = value.partition("=")
    name = name.strip()
    raw_path = raw_path.strip()
    if not separator or not re.fullmatch(r"[a-z][a-z0-9_-]*", name) or not raw_path:
        raise ValueError("language must use NAME=POOL_DIR with a safe lowercase name")
    return LanguagePool(name, Path(raw_path).expanduser())


def parse_named_source(value: str) -> tuple[str, str]:
    language, separator, source = value.partition("=")
    language = language.strip()
    source = source.strip()
    if (
        not separator
        or not re.fullmatch(r"[a-z][a-z0-9_-]*", language)
        or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", source)
    ):
        raise ValueError("include-all-source must use LANGUAGE=SOURCE_CORPUS")
    return language, source


def build_balanced_multilingual_training(
    pools: list[LanguagePool],
    output_dir: str | Path,
    *,
    target_hours: float = 150.0,
    seed: int = 20_260_824,
    include_all_sources: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Derive a duration-balanced train set without reading sealed test manifests."""

    by_name = {pool.name: pool for pool in pools}
    if len(by_name) != len(pools):
        raise ValueError("language pool names must be unique")
    if tuple(sorted(by_name)) != tuple(sorted(LANGUAGE_ORDER)):
        raise ValueError("language pools must contain exactly en, es, and pt")
    if not math.isfinite(target_hours) or target_hours <= 0.0:
        raise ValueError("target_hours must be finite and positive")
    target_seconds = target_hours * 3600.0
    mandatory = _normalize_mandatory_sources(include_all_sources)
    destination = Path(output_dir).expanduser()
    _require_empty_directory(destination)

    selected: dict[str, set[str]] = {}
    selected_metrics: dict[str, dict[str, Any]] = {}
    input_identities: dict[str, dict[str, Any]] = {}
    for language in LANGUAGE_ORDER:
        pool = by_name[language]
        summary_path = pool.root / "split_summary.json"
        train_path = pool.root / "full_ctc_train.jsonl"
        for path in (summary_path, train_path):
            if not path.is_file():
                raise FileNotFoundError(f"required input does not exist: {path}")
        pool_summary = _load_object(summary_path)
        verified_train_sha256 = _validate_pool_summary(
            pool_summary,
            pool,
            train_path,
        )
        scan = _scan_train_manifest(
            train_path,
            language=language,
            seed=seed,
            expected_tags=EXPECTED_LANGUAGE_TAGS[language],
        )
        _validate_scanned_train(scan, pool_summary, language)
        chosen, metrics = _select_candidates(
            scan["candidates"],
            target_seconds=target_seconds,
            mandatory_sources=mandatory[language],
            language=language,
        )
        selected[language] = chosen
        selected_metrics[language] = metrics
        input_identities[language] = {
            "pool_dir": str(pool.root),
            "split_summary": _file_identity(summary_path),
            "train_manifest": _known_file_identity(
                train_path,
                verified_train_sha256,
            ),
            "sealed_validation_reference": _split_reference(
                pool_summary, "validation"
            ),
            "sealed_test_reference": _split_reference(pool_summary, "test"),
        }

    destination.mkdir(parents=True)
    language_paths = {
        language: destination / f"full_ctc_train_{language}.jsonl"
        for language in LANGUAGE_ORDER
    }
    combined_path = destination / "full_ctc_train.jsonl"
    config_path = destination / "selection_config.json"
    summary_path = destination / "selection_summary.json"
    output_paths = (*language_paths.values(), combined_path, config_path, summary_path)
    temporary_paths = {
        path: path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in output_paths
    }

    try:
        written = _write_language_manifests(
            by_name,
            selected,
            language_paths,
            temporary_paths,
        )
        combined_metrics = _write_interleaved_manifest(
            temporary_paths[combined_path],
            {language: temporary_paths[path] for language, path in language_paths.items()},
        )
        _validate_written_selection(selected_metrics, written, combined_metrics)
        output_sha256 = {
            path.name: _sha256_file(temporary_paths[path])
            for path in (*language_paths.values(), combined_path)
        }
        hour_values = [
            float(selected_metrics[language]["selected"]["hours"])
            for language in LANGUAGE_ORDER
        ]
        summary: dict[str, Any] = {
            "schema_version": 2,
            "status": "pass",
            "dataset_version": DATASET_VERSION,
            "output_dir": str(destination),
            "language_order": list(LANGUAGE_ORDER),
            "balance_definition": "train_audio_hours_with_shared_temporal_2x_exposure",
            "target_hours_per_language": target_hours,
            "selection_seed": seed,
            "selection_policy": (
                "include_mandatory_sources_then_stable_hash_until_target_minimum"
            ),
            "interleave_policy": "emit_next_language_with_least_cumulative_audio",
            "language_metrics": selected_metrics,
            "combined_records": int(combined_metrics["records"]),
            "combined_audio_hours": float(combined_metrics["hours"]),
            "selected_hour_spread": max(hour_values) - min(hour_values),
            "duplicate_selected_ids": int(combined_metrics["duplicate_ids"]),
            "duplicate_selected_audio_paths": int(
                combined_metrics["duplicate_audio_paths"]
            ),
            "output_paths": {
                "combined_train": str(combined_path),
                "per_language_train": {
                    language: str(path) for language, path in language_paths.items()
                },
            },
            "output_sha256": output_sha256,
            "validation_test_policy": (
                "independent sealed source splits referenced from input summaries; "
                "not copied, opened, resampled, or combined"
            ),
            "test_set_used": False,
            "test_set_content_read": False,
            "source_manifests_modified": False,
            "manifest_contract": {
                "experiment": EXPERIMENT_NAME,
                "dataset_version": DATASET_VERSION,
                "source_dataset_version": "preserved from each input row",
                "balanced_language_bucket": "en, es, or pt",
            },
        }
        config = {
            "schema_version": 2,
            "dataset_version": DATASET_VERSION,
            "inputs": input_identities,
            "language_order": list(LANGUAGE_ORDER),
            "target_hours_per_language": target_hours,
            "selection_seed": seed,
            "mandatory_source_corpora": {
                language: sorted(mandatory[language]) for language in LANGUAGE_ORDER
            },
            "selection_policy": summary["selection_policy"],
            "interleave_policy": summary["interleave_policy"],
            "manifest_normalization": {
                "experiment": EXPERIMENT_NAME,
                "dataset_version": DATASET_VERSION,
                "source_dataset_version": "preserved from each input row",
                "balanced_language_bucket": "en, es, or pt",
            },
            "test_set_policy": summary["validation_test_policy"],
        }
        temporary_paths[config_path].write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_paths[summary_path].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in output_paths:
            temporary_paths[path].replace(path)
    except Exception:
        _cleanup(temporary_paths.values())
        raise

    _write_sha256_manifest(destination)
    return summary


def build_balanced_multilingual_validation(
    pools: list[LanguagePool],
    training_root: str | Path,
    output_dir: str | Path,
    *,
    target_hours: float = 4.0,
    seed: int = 20_260_824,
) -> dict[str, Any]:
    """Derive an equal-duration validation set without opening sealed test data."""

    by_name = {pool.name: pool for pool in pools}
    if len(by_name) != len(pools) or set(by_name) != set(LANGUAGE_ORDER):
        raise ValueError("language pools must contain exactly en, es, and pt")
    if not math.isfinite(target_hours) or target_hours <= 0.0:
        raise ValueError("target_hours must be finite and positive")
    target_seconds = target_hours * 3600.0
    destination = Path(output_dir).expanduser()
    _require_empty_directory(destination)

    train_root = Path(training_root).expanduser()
    train_path = train_root / "full_ctc_train.jsonl"
    train_summary_path = train_root / "selection_summary.json"
    train_ids, train_audio, training_identity = _validate_training_selection(
        train_path,
        train_summary_path,
    )

    selected: dict[str, set[str]] = {}
    selected_metrics: dict[str, dict[str, Any]] = {}
    input_identities: dict[str, dict[str, Any]] = {}
    validation_paths: dict[str, Path] = {}
    for language in LANGUAGE_ORDER:
        pool = by_name[language]
        summary_path = pool.root / "split_summary.json"
        validation_path = pool.root / "full_ctc_validation.jsonl"
        for path in (summary_path, validation_path):
            if not path.is_file():
                raise FileNotFoundError(f"required input does not exist: {path}")
        pool_summary = _load_object(summary_path)
        verified_sha256 = _validate_pool_split_summary(
            pool_summary,
            pool,
            validation_path,
            split="validation",
        )
        scan = _scan_train_manifest(
            validation_path,
            language=language,
            seed=seed,
            expected_tags=EXPECTED_LANGUAGE_TAGS[language],
            split="validation",
        )
        _validate_scanned_train(scan, pool_summary, language, split="validation")
        candidates = scan["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, Candidate)
            if candidate.sample_id in train_ids or candidate.audio_path in train_audio:
                raise ValueError(
                    f"{language} validation overlaps selected balanced training data"
                )
        chosen, metrics = _select_candidates(
            candidates,
            target_seconds=target_seconds,
            mandatory_sources=frozenset(),
            language=language,
        )
        selected[language] = chosen
        selected_metrics[language] = metrics
        validation_paths[language] = validation_path
        input_identities[language] = {
            "pool_dir": str(pool.root),
            "split_summary": _file_identity(summary_path),
            "validation_manifest": _known_file_identity(
                validation_path,
                verified_sha256,
            ),
            "sealed_test_reference": _split_reference(pool_summary, "test"),
        }

    destination.mkdir(parents=True)
    language_paths = {
        language: destination / f"full_ctc_validation_{language}.jsonl"
        for language in LANGUAGE_ORDER
    }
    combined_path = destination / "full_ctc_validation.jsonl"
    config_path = destination / "selection_config.json"
    summary_path = destination / "selection_summary.json"
    output_paths = (*language_paths.values(), combined_path, config_path, summary_path)
    temporary_paths = {
        path: path.with_name(f".{path.name}.{os.getpid()}.tmp") for path in output_paths
    }

    try:
        written = _write_validation_manifests(
            validation_paths,
            selected,
            language_paths,
            temporary_paths,
        )
        combined_metrics = _write_interleaved_manifest(
            temporary_paths[combined_path],
            {language: temporary_paths[path] for language, path in language_paths.items()},
            dataset_version=VALIDATION_DATASET_VERSION,
            expected_split="validation",
            forbidden_ids=train_ids,
            forbidden_audio_paths=train_audio,
        )
        _validate_written_selection(selected_metrics, written, combined_metrics)
        output_sha256 = {
            path.name: _sha256_file(temporary_paths[path])
            for path in (*language_paths.values(), combined_path)
        }
        hour_values = [
            float(selected_metrics[language]["selected"]["hours"])
            for language in LANGUAGE_ORDER
        ]
        summary: dict[str, Any] = {
            "schema_version": 1,
            "status": "pass",
            "dataset_version": VALIDATION_DATASET_VERSION,
            "output_dir": str(destination),
            "language_order": list(LANGUAGE_ORDER),
            "balance_definition": "validation_audio_hours",
            "target_hours_per_language": target_hours,
            "selection_seed": seed,
            "selection_policy": "stable_hash_until_target_minimum",
            "interleave_policy": "emit_next_language_with_least_cumulative_audio",
            "language_metrics": selected_metrics,
            "combined_records": int(combined_metrics["records"]),
            "combined_audio_hours": float(combined_metrics["hours"]),
            "selected_hour_spread": max(hour_values) - min(hour_values),
            "cross_train_id_overlaps": int(combined_metrics["forbidden_id_overlaps"]),
            "cross_train_audio_overlaps": int(
                combined_metrics["forbidden_audio_overlaps"]
            ),
            "duplicate_selected_ids": int(combined_metrics["duplicate_ids"]),
            "duplicate_selected_audio_paths": int(
                combined_metrics["duplicate_audio_paths"]
            ),
            "manifest_contract": {
                "experiment": EXPERIMENT_NAME,
                "dataset_version": VALIDATION_DATASET_VERSION,
                "source_dataset_version": "preserved from each input row",
                "balanced_language_bucket": "en, es, or pt",
                "split": "validation",
            },
            "output_paths": {
                "combined_validation": str(combined_path),
                "per_language_validation": {
                    language: str(path) for language, path in language_paths.items()
                },
            },
            "output_sha256": output_sha256,
            "source_validation_content_read": True,
            "test_set_used": False,
            "test_set_content_read": False,
            "source_manifests_modified": False,
        }
        config = {
            "schema_version": 1,
            "dataset_version": VALIDATION_DATASET_VERSION,
            "balanced_training": training_identity,
            "inputs": input_identities,
            "language_order": list(LANGUAGE_ORDER),
            "target_hours_per_language": target_hours,
            "selection_seed": seed,
            "selection_policy": summary["selection_policy"],
            "interleave_policy": summary["interleave_policy"],
            "manifest_normalization": summary["manifest_contract"],
            "test_set_policy": "sealed references recorded; content not opened",
        }
        temporary_paths[config_path].write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_paths[summary_path].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in output_paths:
            temporary_paths[path].replace(path)
    except Exception:
        _cleanup(temporary_paths.values())
        raise

    _write_sha256_manifest(destination)
    return summary


def _normalize_mandatory_sources(
    values: Mapping[str, Iterable[str]] | None,
) -> dict[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {
        language: frozenset() for language in LANGUAGE_ORDER
    }
    if values is None:
        return result
    unknown = set(values) - set(LANGUAGE_ORDER)
    if unknown:
        raise ValueError(f"mandatory sources have unknown languages: {sorted(unknown)}")
    for language, sources in values.items():
        normalized = frozenset(source.strip() for source in sources if source.strip())
        result[language] = normalized
    return result


def _validate_pool_summary(
    summary: Mapping[str, Any],
    pool: LanguagePool,
    train_path: Path,
) -> str:
    if summary.get("status") != "pass":
        raise ValueError(f"pool summary status is not pass: {pool.name}")
    if summary.get("test_set_sealed") is not True or summary.get("test_set_used") is not False:
        raise ValueError(f"pool test boundary is not sealed and unused: {pool.name}")
    paths = summary.get("manifest_paths")
    hashes = summary.get("manifest_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError(f"pool summary has no manifest identities: {pool.name}")
    recorded_train = paths.get("train")
    expected_hash = hashes.get("train")
    if (
        not isinstance(recorded_train, str)
        or Path(recorded_train).expanduser().resolve() != train_path.resolve()
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise ValueError(f"pool train identity is invalid: {pool.name}")
    actual_hash = _sha256_file(train_path)
    if actual_hash != expected_hash:
        raise ValueError(f"pool train SHA256 mismatch: {pool.name}")
    return actual_hash


def _validate_pool_split_summary(
    summary: Mapping[str, Any],
    pool: LanguagePool,
    manifest_path: Path,
    *,
    split: str,
) -> str:
    if split not in {"train", "validation"}:
        raise ValueError("balanced selection accepts only train or validation")
    if summary.get("status") != "pass":
        raise ValueError(f"pool summary status is not pass: {pool.name}")
    if summary.get("test_set_sealed") is not True or summary.get("test_set_used") is not False:
        raise ValueError(f"pool test boundary is not sealed and unused: {pool.name}")
    paths = summary.get("manifest_paths")
    hashes = summary.get("manifest_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError(f"pool summary has no manifest identities: {pool.name}")
    recorded_path = paths.get(split)
    expected_hash = hashes.get(split)
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).expanduser().resolve() != manifest_path.resolve()
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise ValueError(f"pool {split} identity is invalid: {pool.name}")
    actual_hash = _sha256_file(manifest_path)
    if actual_hash != expected_hash:
        raise ValueError(f"pool {split} SHA256 mismatch: {pool.name}")
    return actual_hash


def _validate_training_selection(
    train_path: Path,
    summary_path: Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    for path in (train_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"required balanced training input does not exist: {path}")
    summary = _load_object(summary_path)
    if (
        summary.get("status") != "pass"
        or summary.get("dataset_version") != DATASET_VERSION
        or summary.get("test_set_used") is not False
        or summary.get("test_set_content_read") is not False
    ):
        raise ValueError("balanced training summary is not an unused-test v2 selection")
    hashes = summary.get("output_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("balanced training summary has no output SHA256")
    expected_hash = hashes.get(train_path.name)
    if not isinstance(expected_hash, str) or _sha256_file(train_path) != expected_hash:
        raise ValueError("balanced training manifest SHA256 mismatch")
    ids: set[str] = set()
    audio_paths: set[str] = set()
    with train_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_row(line, train_path, line_number)
            if (
                raw.get("experiment") != EXPERIMENT_NAME
                or raw.get("dataset_version") != DATASET_VERSION
                or raw.get("split") != "train"
            ):
                raise ValueError(
                    f"balanced training row has incompatible contract: {line_number}"
                )
            sample_id = _required_string(raw, "id", train_path, line_number)
            audio = _required_string(raw, "audio_path", train_path, line_number)
            if sample_id in ids or audio in audio_paths:
                raise ValueError("balanced training manifest contains duplicate identity")
            ids.add(sample_id)
            audio_paths.add(audio)
    if len(ids) != summary.get("combined_records"):
        raise ValueError("balanced training record count differs from summary")
    return ids, audio_paths, {
        "manifest": _known_file_identity(train_path, expected_hash),
        "summary": _file_identity(summary_path),
        "records": len(ids),
        "content_read_for_identity_only": True,
    }


def _scan_train_manifest(
    path: Path,
    *,
    language: str,
    seed: int,
    expected_tags: frozenset[str],
    split: str = "train",
) -> dict[str, Any]:
    candidates: list[Candidate] = []
    ids: set[str] = set()
    audio_paths: set[str] = set()
    source_metrics: defaultdict[str, CountHours] = defaultdict(CountHours)
    release_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    speaker_ids: set[str] = set()
    missing_speaker_records = 0
    total = CountHours()
    factor_counts: Counter[int] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_row(line, path, line_number)
            if raw.get("split") != split:
                raise ValueError(
                    f"non-{split} row in {split} manifest: {path}:{line_number}"
                )
            sample_id = _required_string(raw, "id", path, line_number)
            audio = _required_string(raw, "audio_path", path, line_number)
            if sample_id in ids or audio in audio_paths:
                raise ValueError(
                    f"duplicate ID or audio in {split} manifest: {path}:{line_number}"
                )
            ids.add(sample_id)
            audio_paths.add(audio)
            tag = _required_string(raw, "language", path, line_number)
            if tag not in expected_tags:
                raise ValueError(f"unexpected {language} language tag: {tag!r}")
            duration = _required_duration(raw, path, line_number)
            source = _required_string(raw, "source_corpus", path, line_number)
            release = _required_string(raw, "release_source", path, line_number)
            factor = raw.get("ctc_time_upsampling_factor")
            if factor != 2:
                raise ValueError(f"balanced pool requires Temporal 2x: {path}:{line_number}")
            speaker_value = raw.get("speaker_id")
            speaker = speaker_value.strip() if isinstance(speaker_value, str) else None
            if speaker:
                speaker_ids.add(speaker)
            else:
                missing_speaker_records += 1
            priority = _stable_digest(f"{seed}\0{language}\0{sample_id}\0{audio}")
            candidates.append(
                Candidate(priority, sample_id, audio, duration, source, release, speaker)
            )
            total.add(duration)
            source_metrics[source].add(duration)
            release_counts[release] += 1
            language_counts[tag] += 1
            factor_counts[int(factor)] += 1
    if not candidates:
        raise ValueError(f"{split} manifest is empty: {path}")
    return {
        "candidates": candidates,
        "total": total,
        "source_metrics": source_metrics,
        "release_counts": release_counts,
        "language_counts": language_counts,
        "speaker_count": len(speaker_ids),
        "missing_speaker_records": missing_speaker_records,
        "factor_counts": factor_counts,
    }


def _validate_scanned_train(
    scan: Mapping[str, Any],
    summary: Mapping[str, Any],
    language: str,
    *,
    split: str = "train",
) -> None:
    split_records = summary.get("split_records")
    split_hours = summary.get("split_audio_hours")
    if not isinstance(split_records, dict) or not isinstance(split_hours, dict):
        raise ValueError(f"pool summary has invalid split metrics: {language}")
    total = scan["total"]
    assert isinstance(total, CountHours)
    if total.records != split_records.get(split) or not math.isclose(
        total.duration_seconds / 3600.0,
        float(split_hours.get(split, -1.0)),
        abs_tol=1e-6,
    ):
        raise ValueError(f"scanned {split} metrics differ from summary: {language}")


def _select_candidates(
    candidates: list[Candidate],
    *,
    target_seconds: float,
    mandatory_sources: frozenset[str],
    language: str,
) -> tuple[set[str], dict[str, Any]]:
    available_sources = {candidate.source_corpus for candidate in candidates}
    missing_sources = mandatory_sources - available_sources
    if missing_sources:
        raise ValueError(
            f"mandatory {language} sources are absent: {sorted(missing_sources)}"
        )
    mandatory = [
        candidate for candidate in candidates if candidate.source_corpus in mandatory_sources
    ]
    optional = [
        candidate for candidate in candidates if candidate.source_corpus not in mandatory_sources
    ]
    selected = {candidate.sample_id for candidate in mandatory}
    selected_seconds = sum(candidate.duration_seconds for candidate in mandatory)
    if selected_seconds > target_seconds:
        raise ValueError(f"mandatory {language} sources exceed target hours")
    for candidate in sorted(optional, key=lambda item: (item.priority, item.sample_id)):
        if selected_seconds >= target_seconds:
            break
        selected.add(candidate.sample_id)
        selected_seconds += candidate.duration_seconds
    if selected_seconds < target_seconds:
        raise ValueError(f"{language} pool cannot reach target hours")
    chosen = [candidate for candidate in candidates if candidate.sample_id in selected]
    sources: defaultdict[str, CountHours] = defaultdict(CountHours)
    releases: defaultdict[str, CountHours] = defaultdict(CountHours)
    speakers: set[str] = set()
    missing_speaker_records = 0
    maximum_duration = 0.0
    for candidate in chosen:
        sources[candidate.source_corpus].add(candidate.duration_seconds)
        releases[candidate.release_source].add(candidate.duration_seconds)
        maximum_duration = max(maximum_duration, candidate.duration_seconds)
        if candidate.speaker_id:
            speakers.add(candidate.speaker_id)
        else:
            missing_speaker_records += 1
    return selected, {
        "available": CountHours(
            records=len(candidates),
            duration_seconds=sum(item.duration_seconds for item in candidates),
        ).to_dict(),
        "selected": CountHours(
            records=len(chosen),
            duration_seconds=selected_seconds,
        ).to_dict(),
        "overshoot_seconds": selected_seconds - target_seconds,
        "maximum_selected_record_seconds": maximum_duration,
        "mandatory_source_corpora": sorted(mandatory_sources),
        "mandatory": CountHours(
            records=len(mandatory),
            duration_seconds=sum(item.duration_seconds for item in mandatory),
        ).to_dict(),
        "selected_source_metrics": {
            key: value.to_dict() for key, value in sorted(sources.items())
        },
        "selected_release_metrics": {
            key: value.to_dict() for key, value in sorted(releases.items())
        },
        "selected_speakers_with_ids": len(speakers),
        "selected_records_without_speaker_id": missing_speaker_records,
    }


def _write_language_manifests(
    pools: Mapping[str, LanguagePool],
    selected: Mapping[str, set[str]],
    paths: Mapping[str, Path],
    temporary_paths: Mapping[Path, Path],
) -> dict[str, CountHours]:
    result: dict[str, CountHours] = {}
    for language in LANGUAGE_ORDER:
        metric = CountHours()
        seen: set[str] = set()
        input_path = pools[language].root / "full_ctc_train.jsonl"
        output_path = temporary_paths[paths[language]]
        with (
            input_path.open(encoding="utf-8") as input_handle,
            output_path.open("w", encoding="utf-8") as output_handle,
        ):
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                raw = _load_row(line, input_path, line_number)
                sample_id = _required_string(raw, "id", input_path, line_number)
                if sample_id not in selected[language]:
                    continue
                source_dataset_version = _required_string(
                    raw,
                    "dataset_version",
                    input_path,
                    line_number,
                )
                raw["source_dataset_version"] = source_dataset_version
                raw["dataset_version"] = DATASET_VERSION
                raw["experiment"] = EXPERIMENT_NAME
                raw["balanced_language_bucket"] = language
                seen.add(sample_id)
                metric.add(_required_duration(raw, input_path, line_number))
                output_handle.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")
        if seen != selected[language]:
            raise RuntimeError(f"selected {language} IDs were not written exactly once")
        result[language] = metric
    return result


def _write_validation_manifests(
    input_paths: Mapping[str, Path],
    selected: Mapping[str, set[str]],
    paths: Mapping[str, Path],
    temporary_paths: Mapping[Path, Path],
) -> dict[str, CountHours]:
    result: dict[str, CountHours] = {}
    for language in LANGUAGE_ORDER:
        metric = CountHours()
        seen: set[str] = set()
        input_path = input_paths[language]
        output_path = temporary_paths[paths[language]]
        with (
            input_path.open(encoding="utf-8") as input_handle,
            output_path.open("w", encoding="utf-8") as output_handle,
        ):
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                raw = _load_row(line, input_path, line_number)
                sample_id = _required_string(raw, "id", input_path, line_number)
                if sample_id not in selected[language]:
                    continue
                if raw.get("split") != "validation":
                    raise ValueError(
                        f"selected validation row has wrong split: {input_path}:{line_number}"
                    )
                source_dataset_version = _required_string(
                    raw,
                    "dataset_version",
                    input_path,
                    line_number,
                )
                raw["source_dataset_version"] = source_dataset_version
                raw["dataset_version"] = VALIDATION_DATASET_VERSION
                raw["experiment"] = EXPERIMENT_NAME
                raw["balanced_language_bucket"] = language
                seen.add(sample_id)
                metric.add(_required_duration(raw, input_path, line_number))
                output_handle.write(json.dumps(raw, ensure_ascii=False, sort_keys=True) + "\n")
        if seen != selected[language]:
            raise RuntimeError(f"selected {language} validation IDs were not written exactly once")
        result[language] = metric
    return result


def _write_interleaved_manifest(
    path: Path,
    language_paths: Mapping[str, Path],
    *,
    dataset_version: str = DATASET_VERSION,
    expected_split: str = "train",
    forbidden_ids: set[str] | frozenset[str] = frozenset(),
    forbidden_audio_paths: set[str] | frozenset[str] = frozenset(),
) -> dict[str, int | float]:
    handles: dict[str, TextIO] = {
        language: language_paths[language].open(encoding="utf-8")
        for language in LANGUAGE_ORDER
    }
    pending: dict[str, str | None] = {
        language: handles[language].readline() or None for language in LANGUAGE_ORDER
    }
    emitted_seconds = {language: 0.0 for language in LANGUAGE_ORDER}
    ids: set[str] = set()
    audio_paths: set[str] = set()
    duplicates_ids = 0
    duplicate_audio = 0
    forbidden_id_overlaps = 0
    forbidden_audio_overlaps = 0
    records = 0
    try:
        with path.open("w", encoding="utf-8") as output_handle:
            while any(value is not None for value in pending.values()):
                active = [language for language in LANGUAGE_ORDER if pending[language] is not None]
                language = min(active, key=lambda item: (emitted_seconds[item], item))
                line = pending[language]
                assert line is not None
                raw = _load_row(line, language_paths[language], records + 1)
                if raw.get("experiment") != EXPERIMENT_NAME:
                    raise ValueError(
                        f"{language} output row has incompatible experiment"
                    )
                if raw.get("dataset_version") != dataset_version:
                    raise ValueError(
                        f"{language} output row has incompatible dataset_version"
                    )
                if raw.get("balanced_language_bucket") != language:
                    raise ValueError(
                        f"{language} output row has incompatible language bucket"
                    )
                if raw.get("split") != expected_split:
                    raise ValueError(f"{language} output row has incompatible split")
                sample_id = _required_string(raw, "id", language_paths[language], records + 1)
                audio = _required_string(raw, "audio_path", language_paths[language], records + 1)
                if sample_id in ids:
                    duplicates_ids += 1
                if audio in audio_paths:
                    duplicate_audio += 1
                if duplicates_ids or duplicate_audio:
                    raise ValueError("selected languages contain duplicate IDs or audio paths")
                if sample_id in forbidden_ids:
                    forbidden_id_overlaps += 1
                if audio in forbidden_audio_paths:
                    forbidden_audio_overlaps += 1
                if forbidden_id_overlaps or forbidden_audio_overlaps:
                    raise ValueError("selected manifest overlaps forbidden training identities")
                ids.add(sample_id)
                audio_paths.add(audio)
                duration = _required_duration(raw, language_paths[language], records + 1)
                emitted_seconds[language] += duration
                records += 1
                output_handle.write(line)
                pending[language] = handles[language].readline() or None
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "records": records,
        "hours": sum(emitted_seconds.values()) / 3600.0,
        "duplicate_ids": duplicates_ids,
        "duplicate_audio_paths": duplicate_audio,
        "forbidden_id_overlaps": forbidden_id_overlaps,
        "forbidden_audio_overlaps": forbidden_audio_overlaps,
    }


def _validate_written_selection(
    expected: Mapping[str, Mapping[str, Any]],
    written: Mapping[str, CountHours],
    combined: Mapping[str, int | float],
) -> None:
    expected_records = 0
    expected_seconds = 0.0
    for language in LANGUAGE_ORDER:
        selected = expected[language]["selected"]
        assert isinstance(selected, dict)
        records = int(selected["records"])
        seconds = float(selected["hours"]) * 3600.0
        if written[language].records != records or not math.isclose(
            written[language].duration_seconds, seconds, abs_tol=1e-5
        ):
            raise RuntimeError(f"written {language} selection differs from plan")
        expected_records += records
        expected_seconds += seconds
    if int(combined["records"]) != expected_records or not math.isclose(
        float(combined["hours"]) * 3600.0,
        expected_seconds,
        abs_tol=1e-5,
    ):
        raise RuntimeError("combined manifest differs from language selections")


def _split_reference(summary: Mapping[str, Any], split: str) -> dict[str, Any]:
    paths = summary["manifest_paths"]
    hashes = summary["manifest_sha256"]
    records = summary["split_records"]
    hours = summary["split_audio_hours"]
    assert all(isinstance(value, dict) for value in (paths, hashes, records, hours))
    return {
        "path": paths[split],
        "sha256": hashes[split],
        "records": records[split],
        "hours": hours[split],
        "content_read": False,
    }


def _load_row(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}:{line_number}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest row is not an object: {path}:{line_number}")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return value


def _required_string(
    raw: Mapping[str, Any], key: str, path: Path, line_number: int
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record has invalid {key}: {path}:{line_number}")
    return value.strip()


def _required_duration(raw: Mapping[str, Any], path: Path, line_number: int) -> float:
    value = raw.get("duration_seconds")
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"record has invalid duration: {path}:{line_number}")
    return float(value)


def _file_identity(path: Path) -> dict[str, int | str]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _known_file_identity(path: Path, sha256: str) -> dict[str, int | str]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_empty_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"output path is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def _cleanup(paths: Iterable[Path]) -> None:
    for path in paths:
        with suppress(FileNotFoundError):
            path.unlink()


def _write_sha256_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "sha256.txt"
    )
    lines = [f"{_sha256_file(path)}  {path.name}" for path in paths]
    (directory / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
