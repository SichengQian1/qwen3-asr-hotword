from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

SPLITS = ("train", "validation", "test")
ALLOWED_SOURCE_SPLITS = {*SPLITS, "unsplit"}
CTC_LENGTH_ISSUE = "ctc_length_infeasible"
DATASET_VERSION = "spanish-temporal2x-combined-v1"


@dataclass(frozen=True)
class SpanishCorpusInput:
    name: str
    manifest_dir: Path
    source_tsv: Path


@dataclass(frozen=True)
class SourceMetadata:
    speaker_id: str
    source_split: str


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
            "hours": round(self.duration_seconds / 3600.0, 6),
        }


def parse_named_path(value: str, *, label: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    name = name.strip()
    raw_path = raw_path.strip()
    if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or not raw_path:
        raise ValueError(f"{label} must use NAME=PATH with a safe lowercase name")
    return name, Path(raw_path).expanduser()


def combine_spanish_inputs(
    manifest_values: Iterable[str],
    source_values: Iterable[str],
) -> list[SpanishCorpusInput]:
    manifests = _unique_named_paths(manifest_values, label="corpus")
    sources = _unique_named_paths(source_values, label="source-tsv")
    if set(manifests) != set(sources):
        raise ValueError("corpus and source-tsv names must match exactly")
    return [
        SpanishCorpusInput(name, manifest_dir, sources[name])
        for name, manifest_dir in manifests.items()
    ]


def build_spanish_temporal2x_training(
    corpora: list[SpanishCorpusInput],
    output_dir: str | Path,
    *,
    train_fraction: float = 0.96,
    validation_fraction: float = 0.02,
    test_fraction: float = 0.02,
    time_upsampling_factor: int = 2,
    release_max_effective_ratio: float = 0.90,
    speaker_split_seed: int = 20_260_824,
) -> dict[str, Any]:
    """Build a Spanish Temporal 2x pool without changing established splits."""

    if not corpora:
        raise ValueError("at least one corpus is required")
    names = [corpus.name for corpus in corpora]
    if len(set(names)) != len(names):
        raise ValueError("corpus names must be unique")
    if time_upsampling_factor <= 1:
        raise ValueError("time_upsampling_factor must exceed one")
    if not 0.0 < release_max_effective_ratio <= 1.0:
        raise ValueError("release_max_effective_ratio must be within (0, 1]")
    fractions = _validate_fractions(
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    destination = Path(output_dir).expanduser()
    _require_empty_directory(destination)

    source_metadata: dict[str, dict[str, SourceMetadata]] = {}
    source_summaries: dict[str, dict[str, Any]] = {}
    input_identities: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        summary_path = corpus.manifest_dir / "summary.json"
        ready_path = corpus.manifest_dir / "train_ready.jsonl"
        review_path = corpus.manifest_dir / "needs_review.jsonl"
        for path in (summary_path, ready_path, review_path, corpus.source_tsv):
            if not path.is_file():
                raise FileNotFoundError(f"required input does not exist: {path}")
        source_summary = _load_object(summary_path)
        if source_summary.get("status") != "pass":
            raise ValueError(f"manifest summary status is not pass: {summary_path}")
        source_summaries[corpus.name] = source_summary
        source_metadata[corpus.name] = _read_source_metadata(corpus.source_tsv)
        input_identities[corpus.name] = {
            "manifest_dir": str(corpus.manifest_dir),
            "summary": _file_identity(summary_path),
            "ready_manifest": _file_identity(ready_path),
            "review_manifest": _file_identity(review_path),
            "source_tsv": _file_identity(corpus.source_tsv),
        }

    speaker_split, assignment_report = _resolve_speaker_splits(
        corpora,
        source_metadata,
        fractions=fractions,
        time_upsampling_factor=time_upsampling_factor,
        release_max_effective_ratio=release_max_effective_ratio,
        seed=speaker_split_seed,
    )

    destination.mkdir(parents=True)
    manifest_paths = {
        split: destination / f"full_ctc_{split}.jsonl" for split in SPLITS
    }
    config_path = destination / "split_config.json"
    summary_path = destination / "split_summary.json"
    speaker_path = destination / "speaker_split_assignments.tsv"
    temporary_paths = {
        path: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for path in (*manifest_paths.values(), config_path, summary_path, speaker_path)
    }
    handles: dict[str, TextIO] = {}
    split_metrics = {split: CountHours() for split in SPLITS}
    corpus_metrics: dict[str, dict[str, CountHours]] = {
        corpus.name: {
            "original_ready": CountHours(),
            "temporal_2x_recovered": CountHours(),
            "review_not_released": CountHours(),
            **{f"split_{split}": CountHours() for split in SPLITS},
        }
        for corpus in corpora
    }
    speakers_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    audio_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    all_ids: set[str] = set()
    all_audio: set[str] = set()
    language_counts: Counter[str] = Counter()

    try:
        handles = {
            split: temporary_paths[path].open("w", encoding="utf-8")
            for split, path in manifest_paths.items()
        }
        for corpus in corpora:
            metadata_by_audio = source_metadata[corpus.name]
            for raw, release_source, path, line_number in _iter_manifest_rows(
                corpus,
                time_upsampling_factor=time_upsampling_factor,
                release_max_effective_ratio=release_max_effective_ratio,
                include_not_released=True,
            ):
                duration = _required_duration(raw, path, line_number)
                audio_path = _required_string(raw, "audio_path", path, line_number)
                metadata = metadata_by_audio.get(audio_path)
                if metadata is None:
                    raise ValueError(
                        f"manifest audio is absent from source TSV: {path}:{line_number}"
                    )
                if release_source == "review_not_released":
                    corpus_metrics[corpus.name][release_source].add(duration)
                    continue
                split = _record_split(
                    raw,
                    metadata,
                    speaker_split,
                    path=path,
                    line_number=line_number,
                )
                record = _release_record(
                    raw,
                    corpus_name=corpus.name,
                    release_source=release_source,
                    split=split,
                    speaker_id=metadata.speaker_id,
                    source_split=metadata.source_split,
                    time_upsampling_factor=time_upsampling_factor,
                    path=path,
                    line_number=line_number,
                )
                sample_id = str(record["id"])
                if sample_id in all_ids:
                    raise ValueError(f"duplicate sample ID: {sample_id}")
                if audio_path in all_audio:
                    raise ValueError(f"duplicate audio path: {audio_path}")
                all_ids.add(sample_id)
                all_audio.add(audio_path)
                ids_by_split[split].add(sample_id)
                audio_by_split[split].add(audio_path)
                speakers_by_split[split].add(metadata.speaker_id)
                handles[split].write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                split_metrics[split].add(duration)
                corpus_metrics[corpus.name][release_source].add(duration)
                corpus_metrics[corpus.name][f"split_{split}"].add(duration)
                language_counts[str(record["language"])] += 1
    except Exception:
        for handle in handles.values():
            handle.close()
        _cleanup(temporary_paths.values())
        raise
    else:
        for handle in handles.values():
            handle.close()

    for corpus in corpora:
        metrics = corpus_metrics[corpus.name]
        ready_records = metrics["original_ready"].records
        review_records = (
            metrics["temporal_2x_recovered"].records
            + metrics["review_not_released"].records
        )
        source_summary = source_summaries[corpus.name]
        _validate_source_partition(
            source_summary,
            source_metadata_records=len(source_metadata[corpus.name]),
            ready_records=ready_records,
            review_records=review_records,
            corpus_name=corpus.name,
        )

    if any(metric.records == 0 for metric in split_metrics.values()):
        _cleanup(temporary_paths.values())
        raise ValueError("combined output contains an empty split")
    id_overlaps = _cross_split_overlap_count(ids_by_split)
    audio_overlaps = _cross_split_overlap_count(audio_by_split)
    speaker_overlaps = _cross_split_overlap_count(speakers_by_split)
    if id_overlaps or audio_overlaps or speaker_overlaps:
        _cleanup(temporary_paths.values())
        raise RuntimeError("combined output contains cross-split leakage")

    _write_speaker_assignments(
        temporary_paths[speaker_path],
        speaker_split,
        assignment_report["speaker_release_seconds"],
    )
    manifest_sha256 = {
        split: _sha256_file(temporary_paths[path])
        for split, path in manifest_paths.items()
    }
    total_seconds = sum(metric.duration_seconds for metric in split_metrics.values())
    serialized_corpus_metrics = {
        name: {metric: value.to_dict() for metric, value in metrics.items()}
        for name, metrics in corpus_metrics.items()
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "dataset_version": DATASET_VERSION,
        "output_dir": str(destination),
        "corpus_order": names,
        "split_strategy": "preserve_explicit_split_and_assign_unsplit_by_speaker",
        "split_fractions": fractions,
        "speaker_split_seed": speaker_split_seed,
        "time_upsampling_factor": time_upsampling_factor,
        "release_max_effective_ratio": release_max_effective_ratio,
        "source_records": len(all_ids),
        "released_records": len(all_ids),
        "original_ready_records": sum(
            metrics["original_ready"].records for metrics in corpus_metrics.values()
        ),
        "recovered_records": sum(
            metrics["temporal_2x_recovered"].records
            for metrics in corpus_metrics.values()
        ),
        "review_not_released_records": sum(
            metrics["review_not_released"].records
            for metrics in corpus_metrics.values()
        ),
        "input_manifest_records": len(all_ids)
        + sum(
            metrics["review_not_released"].records
            for metrics in corpus_metrics.values()
        ),
        "total_audio_hours": total_seconds / 3600.0,
        "split_records": {
            split: split_metrics[split].records for split in SPLITS
        },
        "split_audio_hours": {
            split: split_metrics[split].duration_seconds / 3600.0
            for split in SPLITS
        },
        "corpus_metrics": serialized_corpus_metrics,
        "language_counts": dict(sorted(language_counts.items())),
        "speakers_by_split": {
            split: len(speakers_by_split[split]) for split in SPLITS
        },
        "cross_split_id_overlaps": id_overlaps,
        "cross_split_audio_overlaps": audio_overlaps,
        "cross_split_speaker_overlaps": speaker_overlaps,
        "explicit_split_records_preserved": assignment_report[
            "explicit_split_records"
        ],
        "unsplit_speakers_assigned": assignment_report["unsplit_speakers"],
        "unsplit_assignment": assignment_report["unsplit_assignment"],
        "manifest_paths": {
            split: str(path) for split, path in manifest_paths.items()
        },
        "manifest_sha256": manifest_sha256,
        "speaker_assignment_path": str(speaker_path),
        "source_manifests_modified": False,
        "test_set_used": False,
        "test_set_content_processed_for_mechanical_copy": True,
        "test_set_sealed": True,
    }
    config = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "inputs": input_identities,
        "corpus_order": names,
        "split_strategy": summary["split_strategy"],
        "split_fractions_for_unsplit_speakers": fractions,
        "speaker_split_seed": speaker_split_seed,
        "release_policy": {
            "include_original_ready": True,
            "include_exact_issue_set": [CTC_LENGTH_ISSUE],
            "time_upsampling_factor": time_upsampling_factor,
            "maximum_effective_ratio": release_max_effective_ratio,
            "include_other_issues": False,
        },
        "test_set_policy": (
            "existing test assignments preserved; output sealed; not used for tuning"
        ),
    }
    try:
        temporary_paths[config_path].write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_paths[summary_path].write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in (*manifest_paths.values(), config_path, summary_path, speaker_path):
            temporary_paths[path].replace(path)
    finally:
        _cleanup(temporary_paths.values())
    _write_sha256_manifest(destination)
    return summary


def _unique_named_paths(values: Iterable[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, path = parse_named_path(value, label=label)
        if name in result:
            raise ValueError(f"duplicate {label} name: {name}")
        result[name] = path
    return result


def _read_source_metadata(path: Path) -> dict[str, SourceMetadata]:
    result: dict[str, SourceMetadata] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"audio", "speaker_id", "source_split"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"source TSV is missing columns {missing}: {path}")
        for row_number, row in enumerate(reader, start=2):
            audio = str(row.get("audio") or "").strip()
            speaker_id = str(row.get("speaker_id") or "").strip()
            source_split = str(row.get("source_split") or "").strip()
            if not audio or not speaker_id or source_split not in ALLOWED_SOURCE_SPLITS:
                raise ValueError(f"invalid source metadata at {path}:{row_number}")
            if audio in result:
                raise ValueError(f"duplicate source TSV audio: {audio}")
            result[audio] = SourceMetadata(speaker_id, source_split)
    if not result:
        raise ValueError(f"source TSV contains no records: {path}")
    return result


def _resolve_speaker_splits(
    corpora: list[SpanishCorpusInput],
    metadata: Mapping[str, Mapping[str, SourceMetadata]],
    *,
    fractions: Mapping[str, float],
    time_upsampling_factor: int,
    release_max_effective_ratio: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    fixed: dict[str, str] = {}
    unsplit_seconds: defaultdict[str, float] = defaultdict(float)
    explicit_records = 0
    for corpus in corpora:
        metadata_by_audio = metadata[corpus.name]
        for raw, release_source, path, line_number in _iter_manifest_rows(
            corpus,
            time_upsampling_factor=time_upsampling_factor,
            release_max_effective_ratio=release_max_effective_ratio,
            include_not_released=False,
        ):
            if release_source == "review_not_released":
                continue
            audio = _required_string(raw, "audio_path", path, line_number)
            source = metadata_by_audio.get(audio)
            if source is None:
                raise ValueError(f"manifest audio is absent from source TSV: {audio}")
            raw_split = _required_string(raw, "split", path, line_number)
            if raw_split != source.source_split:
                raise ValueError(
                    f"manifest/source split mismatch for {audio}: "
                    f"{raw_split!r} != {source.source_split!r}"
                )
            if source.source_split == "unsplit":
                unsplit_seconds[source.speaker_id] += _required_duration(
                    raw, path, line_number
                )
            else:
                explicit_records += 1
                previous = fixed.get(source.speaker_id)
                if previous is not None and previous != source.source_split:
                    raise ValueError(
                        f"speaker occurs in multiple explicit splits: {source.speaker_id}"
                    )
                fixed[source.speaker_id] = source.source_split

    assignment = dict(fixed)
    totals = {split: 0.0 for split in SPLITS}
    unsplit_total = sum(unsplit_seconds.values())
    targets = {split: unsplit_total * fractions[split] for split in SPLITS}
    unassigned = [speaker for speaker in unsplit_seconds if speaker not in fixed]
    for speaker in unsplit_seconds:
        if speaker in fixed:
            totals[fixed[speaker]] += unsplit_seconds[speaker]
    ordered = sorted(
        unassigned,
        key=lambda speaker: (
            -unsplit_seconds[speaker],
            _stable_digest(f"{seed}\0speaker\0{speaker}"),
            speaker,
        ),
    )
    for speaker in ordered:
        split = max(
            SPLITS,
            key=lambda candidate: (
                (targets[candidate] - totals[candidate])
                / max(targets[candidate], 1e-12),
                _stable_digest(f"{seed}\0split\0{speaker}\0{candidate}"),
            ),
        )
        assignment[speaker] = split
        totals[split] += unsplit_seconds[speaker]
    if unsplit_seconds and any(
        not any(assignment[speaker] == split for speaker in unsplit_seconds)
        for split in SPLITS
    ):
        raise ValueError("unsplit speaker assignment produced an empty split")
    report = {
        "explicit_split_records": explicit_records,
        "unsplit_speakers": len(unsplit_seconds),
        "unsplit_assignment": {
            split: {
                "speakers": sum(
                    assignment[speaker] == split for speaker in unsplit_seconds
                ),
                "hours": round(totals[split] / 3600.0, 6),
                "target_hours": round(targets[split] / 3600.0, 6),
            }
            for split in SPLITS
        },
        "speaker_release_seconds": dict(unsplit_seconds),
    }
    return assignment, report


def _iter_manifest_rows(
    corpus: SpanishCorpusInput,
    *,
    time_upsampling_factor: int,
    release_max_effective_ratio: float,
    include_not_released: bool,
) -> Iterable[tuple[dict[str, Any], str, Path, int]]:
    ready_path = corpus.manifest_dir / "train_ready.jsonl"
    review_path = corpus.manifest_dir / "needs_review.jsonl"
    with ready_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_row(line, ready_path, line_number)
            if raw.get("training_ready") is not True or raw.get("label_status") != "ready":
                raise ValueError(f"invalid ready row: {ready_path}:{line_number}")
            yield raw, "original_ready", ready_path, line_number
    with review_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_row(line, review_path, line_number)
            released = _recommended_temporal_recovery(
                raw,
                time_upsampling_factor=time_upsampling_factor,
                release_max_effective_ratio=release_max_effective_ratio,
                path=review_path,
                line_number=line_number,
            )
            if released:
                yield raw, "temporal_2x_recovered", review_path, line_number
            elif include_not_released:
                yield raw, "review_not_released", review_path, line_number


def _recommended_temporal_recovery(
    raw: Mapping[str, Any],
    *,
    time_upsampling_factor: int,
    release_max_effective_ratio: float,
    path: Path,
    line_number: int,
) -> bool:
    if raw.get("training_ready") is not False or raw.get("label_status") != "needs_review":
        raise ValueError(f"invalid review row: {path}:{line_number}")
    issues = raw.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"review row has no issues: {path}:{line_number}")
    reasons: set[str] = set()
    for issue in issues:
        if not isinstance(issue, dict) or not isinstance(issue.get("reason"), str):
            raise ValueError(f"invalid review issue: {path}:{line_number}")
        reasons.add(str(issue["reason"]))
    if reasons != {CTC_LENGTH_ISSUE}:
        return False
    estimated = raw.get("estimated_ctc_input_length")
    minimum = raw.get("ctc_minimum_input_length")
    if not isinstance(estimated, int) or not isinstance(minimum, int):
        raise ValueError(f"temporal review row lacks lengths: {path}:{line_number}")
    if estimated >= minimum:
        raise ValueError(f"temporal issue is inconsistent: {path}:{line_number}")
    effective = estimated * time_upsampling_factor
    return effective >= minimum and minimum / effective <= release_max_effective_ratio


def _record_split(
    raw: Mapping[str, Any],
    metadata: SourceMetadata,
    speaker_split: Mapping[str, str],
    *,
    path: Path,
    line_number: int,
) -> str:
    raw_split = _required_string(raw, "split", path, line_number)
    if raw_split != metadata.source_split:
        raise ValueError(f"manifest/source split mismatch: {path}:{line_number}")
    if metadata.source_split in SPLITS:
        return metadata.source_split
    split = speaker_split.get(metadata.speaker_id)
    if split not in SPLITS:
        raise ValueError(f"unassigned speaker: {metadata.speaker_id}")
    return split


def _release_record(
    raw: Mapping[str, Any],
    *,
    corpus_name: str,
    release_source: str,
    split: str,
    speaker_id: str,
    source_split: str,
    time_upsampling_factor: int,
    path: Path,
    line_number: int,
) -> dict[str, Any]:
    token_ids = raw.get("phoneme_token_ids")
    if not isinstance(token_ids, list) or not token_ids or any(
        not isinstance(token, int) or isinstance(token, bool) for token in token_ids
    ):
        raise ValueError(f"record has invalid phoneme_token_ids: {path}:{line_number}")
    minimum = raw.get("ctc_minimum_input_length")
    estimated = raw.get("estimated_ctc_input_length")
    if not isinstance(minimum, int) or not isinstance(estimated, int):
        raise ValueError(f"record has invalid CTC lengths: {path}:{line_number}")
    effective = estimated * time_upsampling_factor
    if effective < minimum:
        raise ValueError(f"released record is not Temporal 2x feasible: {path}:{line_number}")
    duration = _required_duration(raw, path, line_number)
    return {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "split": split,
        "id": _required_string(raw, "id", path, line_number),
        "audio_path": _required_string(raw, "audio_path", path, line_number),
        "audio_relative": _required_string(raw, "audio_relative", path, line_number),
        "text": _required_string(raw, "text", path, line_number),
        "language": _required_string(raw, "language", path, line_number),
        "speaker_id": speaker_id,
        "source_split": source_split,
        "split_assignment_source": (
            "preserved_source_split" if source_split in SPLITS else "speaker_assignment"
        ),
        "phoneme_token_ids": token_ids,
        "label_length": len(token_ids),
        "ctc_minimum_input_length": minimum,
        "estimated_ctc_input_length": estimated,
        "ctc_time_upsampling_factor": time_upsampling_factor,
        "effective_ctc_input_length": effective,
        "effective_ctc_target_ratio": minimum / effective,
        "duration_seconds": duration,
        "source_corpus": corpus_name,
        "source_dataset": raw.get("dataset"),
        "release_source": release_source,
        "source_tsv": _required_string(raw, "source_tsv", path, line_number),
        "source_row_number": raw.get("row_number"),
        "split_hash": raw.get("split_hash"),
    }


def _load_row(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"manifest row is not an object: {path}:{line_number}")
    return raw


def _load_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return raw


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


def _validate_fractions(
    train: float, validation: float, test: float
) -> dict[str, float]:
    values = {"train": train, "validation": validation, "test": test}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("all split fractions must be finite and positive")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    return values


def _validate_source_partition(
    summary: Mapping[str, Any],
    *,
    source_metadata_records: int,
    ready_records: int,
    review_records: int,
    corpus_name: str,
) -> None:
    expected_ready = summary.get("ready_records")
    expected_review = summary.get("review_records")
    expected_source = summary.get("source_records")
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (expected_ready, expected_review, expected_source)
    ):
        raise ValueError(f"source manifest summary has invalid counts: {corpus_name}")
    if ready_records != expected_ready or review_records != expected_review:
        raise ValueError(f"source manifest row counts differ from summary: {corpus_name}")
    if ready_records + review_records != expected_source:
        raise ValueError(f"source manifest partition is inconsistent: {corpus_name}")
    if source_metadata_records != expected_source:
        raise ValueError(f"source TSV count differs from manifest summary: {corpus_name}")


def _write_speaker_assignments(
    path: Path,
    assignments: Mapping[str, str],
    release_seconds: Mapping[str, float],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("speaker_id", "split", "unsplit_release_hours"),
            delimiter="\t",
        )
        writer.writeheader()
        for speaker in sorted(assignments):
            writer.writerow(
                {
                    "speaker_id": speaker,
                    "split": assignments[speaker],
                    "unsplit_release_hours": (
                        f"{release_seconds.get(speaker, 0.0) / 3600.0:.9f}"
                    ),
                }
            )


def _cross_split_overlap_count(values: Mapping[str, set[str]]) -> int:
    return sum(
        len(values[left] & values[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    )


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_identity(path: Path) -> dict[str, int | str]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_manifest(destination: Path) -> None:
    lines = []
    for path in sorted(value for value in destination.rglob("*") if value.is_file()):
        if path.name == "sha256.txt":
            continue
        lines.append(f"{_sha256_file(path)}  {path.relative_to(destination)}\n")
    (destination / "sha256.txt").write_text("".join(lines), encoding="utf-8")


def _require_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {path}")


def _cleanup(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
