from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestMetadata:
    status: str
    duration_seconds: float


@dataclass
class SpeakerAggregate:
    records: int = 0
    duration_seconds: float = 0.0
    shards: set[str] = field(default_factory=set)


def audit_swift_english_speakers(
    source_tsv: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Audit a stable speaker prefix inferred from Swift English WAV basenames."""

    source_path = Path(source_tsv).expanduser()
    manifest_root = Path(manifest_dir).expanduser()
    destination = Path(output_dir).expanduser()
    ready_path = manifest_root / "train_ready.jsonl"
    review_path = manifest_root / "needs_review.jsonl"
    summary_path = manifest_root / "summary.json"
    for path in (source_path, ready_path, review_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")
    _require_empty_directory(destination)

    manifest, manifest_counts, manifest_seconds = _read_manifests(
        ready_path,
        review_path,
    )
    manifest_summary = _load_object(summary_path)
    _validate_manifest_summary(
        manifest_summary,
        ready_records=manifest_counts["ready"],
        review_records=manifest_counts["needs_review"],
    )

    destination.mkdir(parents=True)
    inventory_path = destination / "speaker_inventory.tsv"
    speakers_path = destination / "speaker_summary.tsv"
    failures_path = destination / "parse_failures.tsv"
    source_audio: set[str] = set()
    manifest_missing = 0
    parse_failures: list[dict[str, object]] = []
    speaker_rows: defaultdict[str, SpeakerAggregate] = defaultdict(SpeakerAggregate)
    prefix_component_counts: Counter[int] = Counter()
    first_component_counts: Counter[str] = Counter()
    third_component_counts: Counter[str] = Counter()
    utterance_suffix_digit_records = 0
    duplicate_utterance_keys = 0
    utterance_keys: set[tuple[str, str]] = set()

    with (
        source_path.open(encoding="utf-8-sig", newline="") as source_handle,
        inventory_path.open("w", encoding="utf-8", newline="") as output_handle,
    ):
        reader = csv.DictReader(source_handle, delimiter="\t")
        required = {"audio", "text"}
        missing_columns = sorted(required - set(reader.fieldnames or ()))
        if missing_columns:
            raise ValueError(f"source TSV is missing columns: {missing_columns}")
        writer = csv.DictWriter(
            output_handle,
            fieldnames=(
                "audio",
                "speaker_id",
                "source_split",
                "utterance_id",
                "shard",
                "manifest_status",
                "duration_seconds",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for row_number, row in enumerate(reader, start=2):
            audio = str(row.get("audio") or "").strip()
            text = str(row.get("text") or "").strip()
            if not audio or not text:
                raise ValueError(f"source TSV row {row_number} has empty audio or text")
            if audio in source_audio:
                raise ValueError(f"source TSV contains duplicate audio: {audio}")
            source_audio.add(audio)
            metadata = manifest.get(audio)
            if metadata is None:
                manifest_missing += 1
                parse_failures.append(
                    {
                        "row_number": row_number,
                        "audio": audio,
                        "detail": "missing_manifest_record",
                    }
                )
                continue
            parsed = _parse_speaker_basename(Path(audio).name)
            if parsed is None:
                parse_failures.append(
                    {
                        "row_number": row_number,
                        "audio": audio,
                        "detail": "basename_has_no_speaker_utterance_boundary",
                    }
                )
                continue
            speaker_id, utterance_id, components = parsed
            prefix_component_counts[len(components)] += 1
            first_component_counts[components[0]] += 1
            if len(components) >= 3:
                third_component_counts[components[2]] += 1
            if utterance_id.isdigit():
                utterance_suffix_digit_records += 1
            utterance_key = (speaker_id, utterance_id)
            if utterance_key in utterance_keys:
                duplicate_utterance_keys += 1
            utterance_keys.add(utterance_key)
            shard = Path(audio).parent.name
            aggregate = speaker_rows[speaker_id]
            aggregate.records += 1
            aggregate.duration_seconds += metadata.duration_seconds
            aggregate.shards.add(shard)
            writer.writerow(
                {
                    "audio": audio,
                    "speaker_id": speaker_id,
                    "source_split": "unsplit",
                    "utterance_id": utterance_id,
                    "shard": shard,
                    "manifest_status": metadata.status,
                    "duration_seconds": f"{metadata.duration_seconds:.6f}",
                }
            )

    manifest_extra = len(set(manifest) - source_audio)
    _write_speaker_summary(speakers_path, speaker_rows)
    _write_failures(failures_path, parse_failures)
    clip_counts = [aggregate.records for aggregate in speaker_rows.values()]
    speaker_hours = [
        aggregate.duration_seconds / 3600.0 for aggregate in speaker_rows.values()
    ]
    speakers_across_shards = sum(
        len(aggregate.shards) > 1 for aggregate in speaker_rows.values()
    )
    status = "pass"
    if parse_failures or manifest_missing or manifest_extra or duplicate_utterance_keys:
        status = "warn"
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "speaker_key_policy": "wav_stem_without_final_underscore_component",
        "source_tsv": str(source_path),
        "manifest_dir": str(manifest_root),
        "output_dir": str(destination),
        "source_records": len(source_audio),
        "manifest_records": len(manifest),
        "manifest_status_counts": dict(sorted(manifest_counts.items())),
        "manifest_status_hours": {
            key: round(value / 3600.0, 6)
            for key, value in sorted(manifest_seconds.items())
        },
        "manifest_missing_records": manifest_missing,
        "manifest_extra_records": manifest_extra,
        "parse_failure_records": len(parse_failures),
        "parsed_records": sum(aggregate.records for aggregate in speaker_rows.values()),
        "speaker_count": len(speaker_rows),
        "prefix_component_counts": {
            str(key): value for key, value in sorted(prefix_component_counts.items())
        },
        "first_component_counts": dict(sorted(first_component_counts.items())),
        "third_component_counts": dict(sorted(third_component_counts.items())),
        "utterance_suffix_digit_records": utterance_suffix_digit_records,
        "duplicate_speaker_utterance_keys": duplicate_utterance_keys,
        "speakers_across_multiple_shards": speakers_across_shards,
        "clips_per_speaker": _distribution(clip_counts),
        "hours_per_speaker": _distribution(speaker_hours),
        "speaker_inventory_path": str(inventory_path),
        "speaker_summary_path": str(speakers_path),
        "parse_failures_path": str(failures_path),
        "input_sha256": {
            "source_tsv": _sha256_file(source_path),
            "manifest_summary": _sha256_file(summary_path),
            "ready_manifest": _sha256_file(ready_path),
            "review_manifest": _sha256_file(review_path),
        },
    }
    _write_json(destination / "summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def _read_manifests(
    ready_path: Path,
    review_path: Path,
) -> tuple[dict[str, ManifestMetadata], Counter[str], defaultdict[str, float]]:
    result: dict[str, ManifestMetadata] = {}
    counts: Counter[str] = Counter()
    seconds: defaultdict[str, float] = defaultdict(float)
    for path, status in ((ready_path, "ready"), (review_path, "needs_review")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid manifest JSON: {path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"manifest row is not an object: {path}:{line_number}")
                audio = row.get("audio_path")
                duration = row.get("duration_seconds")
                if not isinstance(audio, str) or not audio.strip():
                    raise ValueError(f"manifest row has invalid audio: {path}:{line_number}")
                if (
                    not isinstance(duration, int | float)
                    or isinstance(duration, bool)
                    or not math.isfinite(float(duration))
                    or float(duration) <= 0.0
                ):
                    raise ValueError(f"manifest row has invalid duration: {path}:{line_number}")
                if audio in result:
                    raise ValueError(f"duplicate manifest audio: {audio}")
                result[audio] = ManifestMetadata(status, float(duration))
                counts[status] += 1
                seconds[status] += float(duration)
    return result, counts, seconds


def _parse_speaker_basename(
    basename: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    stem = Path(basename).stem
    speaker_id, separator, utterance_id = stem.rpartition("_")
    if not separator or not speaker_id or not utterance_id:
        return None
    components = tuple(component for component in speaker_id.split("_") if component)
    if not components or "_".join(components) != speaker_id:
        return None
    return speaker_id, utterance_id, components


def _write_speaker_summary(
    path: Path,
    speakers: dict[str, SpeakerAggregate],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("speaker_id", "records", "hours", "shards"),
            delimiter="\t",
        )
        writer.writeheader()
        for speaker_id, aggregate in sorted(speakers.items()):
            writer.writerow(
                {
                    "speaker_id": speaker_id,
                    "records": aggregate.records,
                    "hours": f"{aggregate.duration_seconds / 3600.0:.9f}",
                    "shards": len(aggregate.shards),
                }
            )


def _write_failures(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row_number", "audio", "detail"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def _distribution(values: list[int] | list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p95": None,
            "maximum": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "median": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "maximum": ordered[-1],
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _validate_manifest_summary(
    summary: dict[str, Any],
    *,
    ready_records: int,
    review_records: int,
) -> None:
    if summary.get("status") != "pass":
        raise ValueError("manifest summary status is not pass")
    if summary.get("ready_records") != ready_records:
        raise ValueError("ready manifest count differs from summary")
    if summary.get("review_records") != review_records:
        raise ValueError("review manifest count differs from summary")
    if summary.get("source_records") != ready_records + review_records:
        raise ValueError("manifest source partition differs from summary")


def _load_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return raw


def _require_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {path}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
