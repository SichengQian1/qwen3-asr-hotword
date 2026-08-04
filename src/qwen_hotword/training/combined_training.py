from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO, cast

from qwen_hotword.training.full_training import assign_full_training_split

SPLIT_NAMES = ("train", "validation", "test")
EXPERIMENT_NAME = "full-ctc-v1"
DATASET_VERSION = "temporal2x-combined-v1"
CTC_LENGTH_ISSUE = "ctc_length_infeasible"


@dataclass(frozen=True)
class CombinedCorpusInput:
    name: str
    manifest_dir: Path


@dataclass
class CountHours:
    records: int = 0
    duration_seconds: float = 0.0

    def add(self, duration_seconds: float) -> None:
        self.records += 1
        self.duration_seconds += duration_seconds

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "hours": round(self.duration_seconds / 3600.0, 6),
        }


@dataclass(frozen=True)
class CombinedTrainingSummary:
    output_dir: str
    dataset_version: str
    experiment: str
    corpus_order: list[str]
    split_strategy: str
    split_fractions: dict[str, float]
    time_upsampling_factor: int
    release_max_effective_ratio: float
    source_records: int
    original_ready_records: int
    recovered_records: int
    total_audio_hours: float
    split_records: dict[str, int]
    split_audio_hours: dict[str, float]
    corpus_metrics: dict[str, dict[str, dict[str, object]]]
    language_counts: dict[str, int]
    manifest_paths: dict[str, str]
    manifest_sha256: dict[str, str]
    duplicate_ids: int
    duplicate_audio_paths: int
    cross_split_id_overlaps: int
    cross_split_audio_overlaps: int
    source_manifests_modified: bool
    test_set_used: bool
    test_set_sealed: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_combined_corpus_spec(value: str) -> CombinedCorpusInput:
    if "=" not in value:
        raise ValueError("corpus must use NAME=MANIFEST_DIR")
    name, raw_path = value.split("=", maxsplit=1)
    name = name.strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(
            "corpus name must start with an alphanumeric character and contain "
            "only lowercase letters, digits, '_' or '-'"
        )
    if not raw_path.strip():
        raise ValueError("corpus manifest directory must not be empty")
    return CombinedCorpusInput(name=name, manifest_dir=Path(raw_path).expanduser())


def build_temporal2x_combined_training(
    corpora: list[CombinedCorpusInput],
    output_dir: str | Path,
    *,
    train_fraction: float = 0.96,
    validation_fraction: float = 0.02,
    test_fraction: float = 0.02,
    time_upsampling_factor: int = 2,
    release_max_effective_ratio: float = 0.90,
    progress_every: int = 50_000,
    print_progress: bool = True,
) -> CombinedTrainingSummary:
    if not corpora:
        raise ValueError("at least one corpus is required")
    names = [corpus.name for corpus in corpora]
    if len(set(names)) != len(names):
        raise ValueError("corpus names must be unique")
    if time_upsampling_factor <= 1:
        raise ValueError("time_upsampling_factor must exceed one")
    if not 0.0 < release_max_effective_ratio <= 1.0:
        raise ValueError("release_max_effective_ratio must be within (0, 1]")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    _validate_fractions(fractions)

    destination = Path(output_dir).expanduser()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"output path is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    manifest_paths = {split: destination / f"full_ctc_{split}.jsonl" for split in SPLIT_NAMES}
    config_path = destination / "split_config.json"
    summary_path = destination / "split_summary.json"
    temporary_paths = {
        path: path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for path in (*manifest_paths.values(), config_path, summary_path)
    }
    for temporary in temporary_paths.values():
        if temporary.exists():
            raise FileExistsError(f"temporary output already exists: {temporary}")

    handles: dict[str, TextIO] = {}
    split_metrics = {split: CountHours() for split in SPLIT_NAMES}
    corpus_metrics: dict[str, dict[str, CountHours]] = {
        corpus.name: {
            "original_ready": CountHours(),
            "temporal_2x_recovered": CountHours(),
            **{f"split_{split}": CountHours() for split in SPLIT_NAMES},
        }
        for corpus in corpora
    }
    input_identities: dict[str, dict[str, object]] = {}
    language_counts: dict[str, int] = {}
    ids_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    audio_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    all_ids: set[str] = set()
    all_audio: set[str] = set()
    duplicate_ids = 0
    duplicate_audio_paths = 0
    started = time.monotonic()

    def emit(raw: dict[str, Any], corpus_name: str, release_source: str, line_number: int) -> None:
        nonlocal duplicate_ids, duplicate_audio_paths
        record = _combined_training_record(
            raw,
            corpus_name=corpus_name,
            release_source=release_source,
            fractions=fractions,
            time_upsampling_factor=time_upsampling_factor,
            line_number=line_number,
        )
        sample_id = str(record["id"])
        audio_path = str(record["audio_path"])
        if sample_id in all_ids:
            duplicate_ids += 1
            raise ValueError(f"duplicate sample ID across combined corpora: {sample_id}")
        if audio_path in all_audio:
            duplicate_audio_paths += 1
            raise ValueError(f"duplicate audio path across combined corpora: {audio_path}")
        all_ids.add(sample_id)
        all_audio.add(audio_path)
        split = str(record["split"])
        duration = cast(float, record["duration_seconds"])
        handles[split].write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        ids_by_split[split].add(sample_id)
        audio_by_split[split].add(audio_path)
        split_metrics[split].add(duration)
        corpus_metrics[corpus_name][f"split_{split}"].add(duration)
        language = str(record["language"])
        language_counts[language] = language_counts.get(language, 0) + 1

    try:
        handles = {
            split: temporary_paths[path].open("w", encoding="utf-8")
            for split, path in manifest_paths.items()
        }
        for corpus in corpora:
            ready_path = corpus.manifest_dir / "train_ready.jsonl"
            review_path = corpus.manifest_dir / "needs_review.jsonl"
            for path in (ready_path, review_path):
                if not path.is_file():
                    raise FileNotFoundError(f"required corpus manifest does not exist: {path}")

            ready_digest = hashlib.sha256()
            ready_seen = 0
            with ready_path.open("rb") as input_handle:
                for line_number, raw_line in enumerate(input_handle, start=1):
                    ready_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    raw = _load_row(raw_line, ready_path, line_number)
                    duration = _ready_duration(raw, ready_path, line_number)
                    emit(raw, corpus.name, "original_ready", line_number)
                    corpus_metrics[corpus.name]["original_ready"].add(duration)
                    ready_seen += 1
                    _print_progress(
                        corpus.name,
                        "ready",
                        ready_seen,
                        progress_every,
                        started,
                        print_progress,
                    )

            review_digest = hashlib.sha256()
            review_seen = 0
            recovered_seen = 0
            with review_path.open("rb") as input_handle:
                for line_number, raw_line in enumerate(input_handle, start=1):
                    review_digest.update(raw_line)
                    if not raw_line.strip():
                        continue
                    raw = _load_row(raw_line, review_path, line_number)
                    review_seen += 1
                    if _is_recommended_temporal_recovery(
                        raw,
                        time_upsampling_factor=time_upsampling_factor,
                        release_max_effective_ratio=release_max_effective_ratio,
                        path=review_path,
                        line_number=line_number,
                    ):
                        duration = _required_duration(raw, review_path, line_number)
                        emit(raw, corpus.name, "temporal_2x_recovery", line_number)
                        corpus_metrics[corpus.name]["temporal_2x_recovered"].add(duration)
                        recovered_seen += 1
                    _print_progress(
                        corpus.name,
                        "review",
                        review_seen,
                        progress_every,
                        started,
                        print_progress,
                    )

            input_identities[corpus.name] = {
                "manifest_dir": str(corpus.manifest_dir),
                "ready_manifest": _streamed_identity(
                    ready_path, ready_digest.hexdigest(), ready_seen
                ),
                "review_manifest": _streamed_identity(
                    review_path, review_digest.hexdigest(), review_seen
                ),
            }
            if print_progress:
                print(
                    f"combined_manifest corpus={corpus.name} complete "
                    f"ready={ready_seen} recovered={recovered_seen}",
                    flush=True,
                )
    except Exception:
        for handle in handles.values():
            handle.close()
        _cleanup_temporary_paths(temporary_paths.values())
        raise
    else:
        for handle in handles.values():
            handle.close()

    source_records = len(all_ids)
    if source_records == 0 or any(metric.records == 0 for metric in split_metrics.values()):
        _cleanup_temporary_paths(temporary_paths.values())
        raise ValueError("combined stable split produced an empty dataset or split")
    cross_split_id_overlaps = _cross_split_overlap_count(ids_by_split)
    cross_split_audio_overlaps = _cross_split_overlap_count(audio_by_split)
    if cross_split_id_overlaps or cross_split_audio_overlaps:
        _cleanup_temporary_paths(temporary_paths.values())
        raise RuntimeError("combined manifests contain cross-split overlap")

    manifest_sha256 = {
        split: _sha256_file(temporary_paths[path]) for split, path in manifest_paths.items()
    }
    original_ready_records = sum(
        metrics["original_ready"].records for metrics in corpus_metrics.values()
    )
    recovered_records = sum(
        metrics["temporal_2x_recovered"].records for metrics in corpus_metrics.values()
    )
    total_seconds = sum(metric.duration_seconds for metric in split_metrics.values())
    serialized_corpus_metrics = {
        name: {key: value.to_dict() for key, value in metrics.items()}
        for name, metrics in corpus_metrics.items()
    }
    summary = CombinedTrainingSummary(
        output_dir=str(destination),
        dataset_version=DATASET_VERSION,
        experiment=EXPERIMENT_NAME,
        corpus_order=names,
        split_strategy="existing_stable_split_hash",
        split_fractions=fractions,
        time_upsampling_factor=time_upsampling_factor,
        release_max_effective_ratio=release_max_effective_ratio,
        source_records=source_records,
        original_ready_records=original_ready_records,
        recovered_records=recovered_records,
        total_audio_hours=total_seconds / 3600.0,
        split_records={split: metric.records for split, metric in split_metrics.items()},
        split_audio_hours={
            split: metric.duration_seconds / 3600.0 for split, metric in split_metrics.items()
        },
        corpus_metrics=serialized_corpus_metrics,
        language_counts=dict(sorted(language_counts.items())),
        manifest_paths={split: str(path) for split, path in manifest_paths.items()},
        manifest_sha256=manifest_sha256,
        duplicate_ids=duplicate_ids,
        duplicate_audio_paths=duplicate_audio_paths,
        cross_split_id_overlaps=cross_split_id_overlaps,
        cross_split_audio_overlaps=cross_split_audio_overlaps,
        source_manifests_modified=False,
        test_set_used=False,
        test_set_sealed=True,
        status="pass",
    )
    config = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "experiment": EXPERIMENT_NAME,
        "purpose": "combined_temporal_2x_full_training_manifests",
        "inputs": input_identities,
        "corpus_order": names,
        "split_strategy": "existing_stable_split_hash",
        "split_fractions": fractions,
        "release_policy": {
            "include_original_ready": True,
            "include_exact_issue_set": [CTC_LENGTH_ISSUE],
            "time_upsampling_factor": time_upsampling_factor,
            "maximum_effective_ratio": release_max_effective_ratio,
            "include_other_issues": False,
        },
        "manifest_contract": {
            "ctc_time_upsampling_factor": time_upsampling_factor,
            "estimated_ctc_input_length": "original Qwen encoder CTC frame length",
            "effective_ctc_input_length": (
                "estimated_ctc_input_length * ctc_time_upsampling_factor"
            ),
        },
        "source_manifests_modified": False,
        "test_set_policy": "sealed; do not use for checkpoint selection or tuning",
    }
    try:
        temporary_paths[config_path].write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_paths[summary_path].write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in (*manifest_paths.values(), config_path, summary_path):
            temporary_paths[path].replace(path)
    finally:
        _cleanup_temporary_paths(temporary_paths.values())
    return summary


def _combined_training_record(
    raw: dict[str, Any],
    *,
    corpus_name: str,
    release_source: str,
    fractions: dict[str, float],
    time_upsampling_factor: int,
    line_number: int,
) -> dict[str, object]:
    sample_id = _required_string(raw, "id", line_number)
    audio_path = _required_string(raw, "audio_path", line_number)
    token_ids = raw.get("phoneme_token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"source row {line_number} has no phoneme_token_ids")
    if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids):
        raise ValueError(f"source row {line_number} has non-integer token IDs")
    if raw.get("label_length") != len(token_ids):
        raise ValueError(f"source row {line_number} label_length does not match token IDs")
    minimum_length = raw.get("ctc_minimum_input_length")
    estimated_length = raw.get("estimated_ctc_input_length")
    if not isinstance(minimum_length, int) or minimum_length < len(token_ids):
        raise ValueError(f"source row {line_number} has invalid CTC minimum length")
    if not isinstance(estimated_length, int) or estimated_length <= 0:
        raise ValueError(f"source row {line_number} has invalid estimated CTC length")
    effective_length = estimated_length * time_upsampling_factor
    if effective_length < minimum_length:
        raise ValueError(f"source row {line_number} is not Temporal 2x CTC-feasible")
    duration = raw.get("duration_seconds")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
        raise ValueError(f"source row {line_number} has invalid duration_seconds")
    split_hash = raw.get("split_hash")
    if not isinstance(split_hash, int | float) or isinstance(split_hash, bool):
        raise ValueError(f"source row {line_number} has invalid split_hash")
    split_hash = float(split_hash)
    if not 0.0 <= split_hash < 1.0:
        raise ValueError(f"source row {line_number} split_hash is outside [0, 1)")
    split = assign_full_training_split(split_hash, fractions)
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "dataset_version": DATASET_VERSION,
        "split": split,
        "id": sample_id,
        "audio_path": audio_path,
        "audio_relative": _required_string(raw, "audio_relative", line_number),
        "text": _required_string(raw, "text", line_number),
        "language": _required_string(raw, "language", line_number),
        "phoneme_token_ids": token_ids,
        "label_length": len(token_ids),
        "ctc_minimum_input_length": minimum_length,
        "estimated_ctc_input_length": estimated_length,
        "ctc_time_upsampling_factor": time_upsampling_factor,
        "effective_ctc_input_length": effective_length,
        "effective_ctc_target_ratio": minimum_length / effective_length,
        "duration_seconds": float(duration),
        "source_corpus": corpus_name,
        "source_dataset": raw.get("dataset"),
        "release_source": release_source,
        "source_tsv": _required_string(raw, "source_tsv", line_number),
        "source_row_number": raw.get("row_number"),
        "split_hash": split_hash,
    }


def _is_recommended_temporal_recovery(
    raw: dict[str, Any],
    *,
    time_upsampling_factor: int,
    release_max_effective_ratio: float,
    path: Path,
    line_number: int,
) -> bool:
    if raw.get("training_ready") is not False or raw.get("label_status") != "needs_review":
        raise ValueError(f"review row has invalid status: {path}:{line_number}")
    issues = raw.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"review row has invalid issues: {path}:{line_number}")
    reasons: list[str] = []
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError(f"review row has invalid issue entry: {path}:{line_number}")
        reason = issue.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"review row has invalid issue reason: {path}:{line_number}")
        reasons.append(reason)
    if set(reasons) != {CTC_LENGTH_ISSUE}:
        return False
    estimated = raw.get("estimated_ctc_input_length")
    minimum = raw.get("ctc_minimum_input_length")
    if not isinstance(estimated, int) or estimated <= 0:
        raise ValueError(f"pure temporal row has invalid estimated length: {path}:{line_number}")
    if not isinstance(minimum, int) or minimum <= 0:
        raise ValueError(f"pure temporal row has invalid minimum length: {path}:{line_number}")
    if estimated >= minimum:
        raise ValueError(f"pure temporal row is already 1x feasible: {path}:{line_number}")
    effective_length = estimated * time_upsampling_factor
    effective_ratio = minimum / effective_length
    return effective_length >= minimum and effective_ratio <= release_max_effective_ratio


def _ready_duration(raw: dict[str, Any], path: Path, line_number: int) -> float:
    if raw.get("training_ready") is not True or raw.get("label_status") != "ready":
        raise ValueError(f"ready row has invalid status: {path}:{line_number}")
    estimated = raw.get("estimated_ctc_input_length")
    minimum = raw.get("ctc_minimum_input_length")
    if not isinstance(estimated, int) or not isinstance(minimum, int) or estimated < minimum:
        raise ValueError(f"ready row is not 1x CTC-feasible: {path}:{line_number}")
    return _required_duration(raw, path, line_number)


def _required_duration(raw: dict[str, Any], path: Path, line_number: int) -> float:
    value = raw.get("duration_seconds")
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"row has invalid duration: {path}:{line_number}")
    if value <= 0:
        raise ValueError(f"row has non-positive duration: {path}:{line_number}")
    return float(value)


def _load_row(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"manifest row must be an object: {path}:{line_number}")
    return raw


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source row {line_number} has invalid {key}")
    return value


def _validate_fractions(fractions: dict[str, float]) -> None:
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("all split fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("train, validation, and test fractions must sum to one")


def _cross_split_overlap_count(values: dict[str, set[str]]) -> int:
    return sum(
        len(values[left] & values[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )


def _streamed_identity(path: Path, sha256: str, records: int) -> dict[str, object]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256,
        "records": records,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cleanup_temporary_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _print_progress(
    corpus: str,
    stage: str,
    records: int,
    progress_every: int,
    started: float,
    enabled: bool,
) -> None:
    if enabled and records % progress_every == 0:
        elapsed = max(time.monotonic() - started, 1e-9)
        print(
            f"combined_manifest corpus={corpus} stage={stage} records={records} "
            f"elapsed={elapsed:.1f}s records_per_second={records / elapsed:.1f}",
            flush=True,
        )
