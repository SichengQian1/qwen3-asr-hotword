from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO, cast

SPLIT_NAMES = ("train", "validation", "test")
EXPERIMENT_NAME = "full-ctc-v1"


@dataclass(frozen=True)
class FullTrainingSplitSummary:
    source_manifest_path: str
    output_dir: str
    experiment: str
    split_strategy: str
    split_fractions: dict[str, float]
    source_records: int
    split_records: dict[str, int]
    split_audio_hours: dict[str, float]
    manifest_paths: dict[str, str]
    manifest_sha256: dict[str, str]
    duplicate_ids: int
    duplicate_audio_paths: int
    cross_split_id_overlaps: int
    cross_split_audio_overlaps: int
    test_set_sealed: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_full_training_splits(
    source_manifest: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.96,
    validation_fraction: float = 0.02,
    test_fraction: float = 0.02,
    overwrite: bool = False,
) -> FullTrainingSplitSummary:
    source = Path(source_manifest).expanduser()
    destination = Path(output_dir).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"source training manifest does not exist: {source}")
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    _validate_fractions(fractions)

    destination.mkdir(parents=True, exist_ok=True)
    manifest_paths = {
        split: destination / f"full_ctc_{split}.jsonl" for split in SPLIT_NAMES
    }
    summary_path = destination / "split_summary.json"
    config_path = destination / "split_config.json"
    existing_outputs = [
        path
        for path in (*manifest_paths.values(), summary_path, config_path)
        if path.exists()
    ]
    if existing_outputs and not overwrite:
        raise ValueError(
            "output directory already contains a full-training split; "
            "use --overwrite only when intentionally rebuilding it"
        )

    temporary_paths = {
        split: path.with_suffix(path.suffix + ".tmp")
        for split, path in manifest_paths.items()
    }
    handles: dict[str, TextIO] = {}
    record_counts = dict.fromkeys(SPLIT_NAMES, 0)
    duration_seconds = dict.fromkeys(SPLIT_NAMES, 0.0)
    ids_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    audio_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    all_ids: set[str] = set()
    all_audio: set[str] = set()
    source_records = 0
    duplicate_ids = 0
    duplicate_audio_paths = 0

    try:
        handles = {
            split: path.open("w", encoding="utf-8")
            for split, path in temporary_paths.items()
        }
        with source.open("r", encoding="utf-8") as input_handle:
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                raw = _load_source_row(line, source, line_number)
                sample_id = _required_string(raw, "id", line_number)
                audio_path = _required_string(raw, "audio_path", line_number)
                if sample_id in all_ids:
                    duplicate_ids += 1
                    raise ValueError(
                        f"duplicate sample ID at source row {line_number}: {sample_id}"
                    )
                if audio_path in all_audio:
                    duplicate_audio_paths += 1
                    raise ValueError(
                        f"duplicate audio path at source row {line_number}: {audio_path}"
                    )
                all_ids.add(sample_id)
                all_audio.add(audio_path)

                split_hash = raw.get("split_hash")
                if not isinstance(split_hash, float | int) or isinstance(split_hash, bool):
                    raise ValueError(f"source row {line_number} has invalid split_hash")
                split_hash = float(split_hash)
                if not 0.0 <= split_hash < 1.0:
                    raise ValueError(f"source row {line_number} split_hash is outside [0, 1)")
                split = assign_full_training_split(split_hash, fractions)
                record = _training_record(raw, split=split, line_number=line_number)
                handles[split].write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                ids_by_split[split].add(sample_id)
                audio_by_split[split].add(audio_path)
                record_counts[split] += 1
                duration_seconds[split] += cast(float, record["duration_seconds"])
                source_records += 1
    except Exception:
        for handle in handles.values():
            handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise
    else:
        for handle in handles.values():
            handle.close()

    if source_records == 0:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise ValueError("source training manifest contains no records")
    if any(record_counts[split] == 0 for split in SPLIT_NAMES):
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise ValueError("stable split produced an empty train, validation, or test set")

    for split in SPLIT_NAMES:
        temporary_paths[split].replace(manifest_paths[split])

    cross_split_id_overlaps = _cross_split_overlap_count(ids_by_split)
    cross_split_audio_overlaps = _cross_split_overlap_count(audio_by_split)
    manifest_sha256 = {
        split: _sha256_file(path) for split, path in manifest_paths.items()
    }
    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "source_manifest": _file_identity(source),
        "split_strategy": "existing_stable_split_hash",
        "split_fractions": fractions,
        "test_set_policy": "sealed; do not use for checkpoint selection or tuning",
    }
    _write_json(config_path, config)
    status = (
        "pass"
        if source_records == sum(record_counts.values())
        and cross_split_id_overlaps == 0
        and cross_split_audio_overlaps == 0
        else "fail"
    )
    summary = FullTrainingSplitSummary(
        source_manifest_path=str(source),
        output_dir=str(destination),
        experiment=EXPERIMENT_NAME,
        split_strategy="existing_stable_split_hash",
        split_fractions=fractions,
        source_records=source_records,
        split_records=record_counts,
        split_audio_hours={
            split: duration_seconds[split] / 3600 for split in SPLIT_NAMES
        },
        manifest_paths={split: str(path) for split, path in manifest_paths.items()},
        manifest_sha256=manifest_sha256,
        duplicate_ids=duplicate_ids,
        duplicate_audio_paths=duplicate_audio_paths,
        cross_split_id_overlaps=cross_split_id_overlaps,
        cross_split_audio_overlaps=cross_split_audio_overlaps,
        test_set_sealed=True,
        status=status,
    )
    _write_json(summary_path, summary.to_dict())
    return summary


def assign_full_training_split(
    split_hash: float,
    fractions: dict[str, float],
) -> str:
    train_boundary = fractions["train"]
    validation_boundary = train_boundary + fractions["validation"]
    if split_hash < train_boundary:
        return "train"
    if split_hash < validation_boundary:
        return "validation"
    return "test"


def _training_record(
    raw: dict[str, Any],
    *,
    split: str,
    line_number: int,
) -> dict[str, object]:
    if raw.get("training_ready") is not True or raw.get("label_status") != "ready":
        raise ValueError(f"source row {line_number} is not marked training-ready")
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
    if not isinstance(estimated_length, int) or estimated_length < minimum_length:
        raise ValueError(f"source row {line_number} is not physically CTC-feasible")
    duration = raw.get("duration_seconds")
    if not isinstance(duration, float | int) or isinstance(duration, bool) or duration <= 0:
        raise ValueError(f"source row {line_number} has invalid duration_seconds")

    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "split": split,
        "id": _required_string(raw, "id", line_number),
        "audio_path": _required_string(raw, "audio_path", line_number),
        "audio_relative": _required_string(raw, "audio_relative", line_number),
        "text": _required_string(raw, "text", line_number),
        "language": _required_string(raw, "language", line_number),
        "phoneme_token_ids": token_ids,
        "label_length": len(token_ids),
        "ctc_minimum_input_length": minimum_length,
        "estimated_ctc_input_length": estimated_length,
        "duration_seconds": float(duration),
        "source_tsv": _required_string(raw, "source_tsv", line_number),
        "source_row_number": raw.get("row_number"),
        "split_hash": float(raw["split_hash"]),
    }


def _load_source_row(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"source row {line_number} must be a JSON object")
    return raw


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source row {line_number} has invalid {key}")
    return value


def _validate_fractions(fractions: dict[str, float]) -> None:
    if any(fractions[split] <= 0 for split in SPLIT_NAMES):
        raise ValueError("all split fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("train, validation, and test fractions must sum to one")


def _cross_split_overlap_count(values: dict[str, set[str]]) -> int:
    return sum(
        len(values[left] & values[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
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


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
