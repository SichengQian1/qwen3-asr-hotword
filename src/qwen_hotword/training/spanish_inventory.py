from __future__ import annotations

import csv
import hashlib
import json
import math
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.training.experiment_a import AudioMetadata, read_audio_metadata
from qwen_hotword.training.g2p_prep import normalize_training_text

_CV_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
_TARGET_MARKERS = (
    "argentina",
    "argentinean",
    "argentino",
    "cordoba",
    "cordobes",
    "rioplatense",
)
_LATIN_AMERICAN_MARKERS = (
    "america central",
    "andino",
    "bolivia",
    "caribe",
    "centroamerica",
    "chile",
    "colombia",
    "costa rica",
    "cuba",
    "ecuador",
    "el salvador",
    "guatemala",
    "honduras",
    "latinoamerica",
    "latin america",
    "mexico",
    "nicaragua",
    "panama",
    "paraguay",
    "peru",
    "puerto rico",
    "republica dominicana",
    "uruguay",
    "venezuela",
)
_PENINSULAR_MARKERS = (
    "andalucia",
    "andaluz",
    "asturias",
    "castilla",
    "espana",
    "galicia",
    "madrid",
    "peninsular",
)


@dataclass(frozen=True)
class InventoryRow:
    row_number: int
    audio: str
    text: str


@dataclass(frozen=True)
class AudioProbe:
    status: str
    duration_seconds: float | None
    sample_rate: int | None
    error: str


@dataclass(frozen=True)
class CommonVoiceMetadata:
    path: str
    client_id: str
    sentence_id: str
    sentence: str
    locale: str
    accent: str
    gender: str
    up_votes: str
    down_votes: str
    official_split: str


def classify_spanish_accent(accent: str) -> str:
    """Return a conservative metadata tier without claiming acoustic dialect identity."""

    normalized = _fold_text(accent)
    if not normalized:
        return "unknown"
    has_target = any(marker in normalized for marker in _TARGET_MARKERS)
    has_latin_american = has_target or any(
        marker in normalized for marker in _LATIN_AMERICAN_MARKERS
    )
    has_peninsular = any(marker in normalized for marker in _PENINSULAR_MARKERS)
    if has_latin_american and has_peninsular:
        return "mixed_latin_american_peninsular"
    if has_target:
        return "argentinian_rioplatense_metadata"
    if has_latin_american:
        return "latin_american_metadata"
    if has_peninsular:
        return "peninsular_metadata"
    return "other_unclassified_metadata"


def audit_spanish_candidate_inventory(
    mls_tsv: str | Path,
    common_voice_tsv: str | Path,
    common_voice_root: str | Path,
    rioplatense_tsv: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 16,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    """Audit duration, CV metadata tiers, split leakage, and core clip overlap."""

    mls_path = Path(mls_tsv).expanduser()
    cv_path = Path(common_voice_tsv).expanduser()
    cv_root = Path(common_voice_root).expanduser()
    core_path = Path(rioplatense_tsv).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (mls_path, cv_path, core_path):
        if not path.is_file():
            raise FileNotFoundError(f"required TSV does not exist: {path}")
    if not cv_root.is_dir():
        raise FileNotFoundError(f"Common Voice root does not exist: {cv_root}")
    if workers <= 0 or progress_every < 0:
        raise ValueError("workers must be positive and progress_every must be non-negative")
    _require_empty_directory(destination)

    started = time.monotonic()
    mls_rows = _read_inventory_tsv(mls_path)
    cv_rows = _read_inventory_tsv(cv_path)
    core_clips = _read_core_clips(core_path)
    cv_metadata, metadata_report, metadata_paths = _load_common_voice_metadata(cv_root)

    destination.mkdir(parents=True)
    mls_probes = _probe_audio_rows(
        mls_rows,
        workers=workers,
        progress_every=progress_every,
        label="mls",
    )
    cv_probes = _probe_audio_rows(
        cv_rows,
        workers=workers,
        progress_every=progress_every,
        label="common_voice",
    )

    mls_inventory_path = destination / "mls_inventory.tsv"
    cv_inventory_path = destination / "common_voice_inventory.tsv"
    mls_summary = _write_mls_inventory(mls_rows, mls_probes, mls_inventory_path)
    cv_summary = _write_common_voice_inventory(
        cv_rows,
        cv_probes,
        cv_metadata,
        core_clips,
        cv_inventory_path,
    )
    cv_summary["metadata_source"] = metadata_report

    input_sha256 = {
        "mls_tsv": _sha256_file(mls_path),
        "common_voice_tsv": _sha256_file(cv_path),
        "rioplatense_tsv": _sha256_file(core_path),
        **{
            f"common_voice_metadata_{path.stem}": _sha256_file(path)
            for path in metadata_paths
        },
    }
    run_config = {
        "schema_version": 1,
        "purpose": "spanish_candidate_inventory_and_common_voice_metadata_join",
        "mls_tsv": str(mls_path),
        "common_voice_tsv": str(cv_path),
        "common_voice_root": str(cv_root),
        "rioplatense_tsv": str(core_path),
        "workers": workers,
        "progress_every": progress_every,
        "join_key": "audio basename / Common Voice path",
        "input_sha256": input_sha256,
        "dialect_policy": {
            "labels_are_metadata_tiers_not_acoustic_dialect_predictions": True,
            "target_markers": list(_TARGET_MARKERS),
            "latin_american_markers": list(_LATIN_AMERICAN_MARKERS),
            "peninsular_markers": list(_PENINSULAR_MARKERS),
            "mls_policy": "source_unknown_likely_peninsular_fallback_only",
        },
    }
    _write_json(destination / "run_config.json", run_config)
    _write_json(destination / "mls_summary.json", mls_summary)
    _write_json(destination / "common_voice_summary.json", cv_summary)

    status = "pass"
    errors: list[str] = []
    for source_name, source_summary in (("mls", mls_summary), ("common_voice", cv_summary)):
        if source_summary["invalid_audio_records"]:
            status = "warn"
            errors.append(f"{source_name}: invalid audio records")
    if cv_summary["metadata_missing_records"]:
        status = "warn"
        errors.append("common_voice: source rows missing original metadata")
    if cv_summary["metadata_text_mismatch_records"]:
        status = "warn"
        errors.append("common_voice: Swift text differs from original metadata")
    if cv_summary["unexpected_locale_records"]:
        status = "warn"
        errors.append("common_voice: joined metadata contains a non-es locale")
    if metadata_report["cross_split_clip_overlaps"]:
        status = "warn"
        errors.append("common_voice: original metadata has cross-split clip overlap")
    if metadata_report["duplicate_validated_paths"] or metadata_report["metadata_conflicts"]:
        status = "warn"
        errors.append("common_voice: original metadata contains duplicate/conflicting rows")

    total_candidate_hours = float(mls_summary["valid_audio_hours"]) + float(
        cv_summary["valid_audio_hours"]
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "errors": errors,
        "output_dir": str(destination),
        "mls": mls_summary,
        "common_voice": cv_summary,
        "total_candidate_records": len(mls_rows) + len(cv_rows),
        "total_valid_audio_hours": round(total_candidate_hours, 6),
        "elapsed_seconds": time.monotonic() - started,
        "sealed_core_policy": {
            "rioplatense_overlap_is_never_eligible_for_auxiliary_training": True,
            "overlap_counts_by_core_split": cv_summary["core_overlap_counts"],
            "overlap_hours_by_core_split": cv_summary["core_overlap_hours"],
        },
        "next_step": (
            "set source-hour quotas only after inspecting metadata tier, official split, "
            "and core-overlap hours"
        ),
    }
    _write_json(destination / "summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def _read_inventory_tsv(path: Path) -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    audio_values: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = sorted({"audio", "text"} - fields)
        if missing:
            raise ValueError(f"TSV {path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            audio = str(row.get("audio") or "").strip()
            text = str(row.get("text") or "").strip()
            if not audio or not text:
                raise ValueError(f"TSV {path} row {row_number} has empty audio or text")
            if audio in audio_values:
                raise ValueError(f"TSV {path} contains duplicate audio: {audio}")
            audio_values.add(audio)
            rows.append(InventoryRow(row_number=row_number, audio=audio, text=text))
    if not rows:
        raise ValueError(f"TSV {path} contains no data rows")
    return rows


def _read_core_clips(path: Path) -> dict[str, str]:
    clips: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = sorted({"audio", "source_split"} - fields)
        if missing:
            raise ValueError(f"core TSV {path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            clip = Path(str(row.get("audio") or "").strip()).name
            split = str(row.get("source_split") or "").strip()
            if not clip or split not in {"train", "validation", "test"}:
                raise ValueError(f"core TSV {path} row {row_number} has invalid clip or split")
            previous = clips.get(clip)
            if previous is not None and previous != split:
                raise ValueError(f"core clip occurs in multiple splits: {clip}")
            clips[clip] = split
    return clips


def _load_common_voice_metadata(
    root: Path,
) -> tuple[dict[str, CommonVoiceMetadata], dict[str, Any], tuple[Path, ...]]:
    validated_path = root / "validated.tsv"
    if not validated_path.is_file():
        raise FileNotFoundError(f"Common Voice validated.tsv does not exist: {validated_path}")
    metadata: dict[str, dict[str, str]] = {}
    duplicate_validated_paths = 0
    for row_number, row in _iter_common_voice_rows(validated_path):
        audio_name = row["path"].strip()
        if not audio_name:
            raise ValueError(f"Common Voice {validated_path} row {row_number} has empty path")
        if audio_name in metadata:
            duplicate_validated_paths += 1
            continue
        metadata[audio_name] = row
    validated_records = len(metadata)

    official_splits: dict[str, str] = {}
    cross_split_clip_overlaps = 0
    split_counts: Counter[str] = Counter()
    metadata_conflicts: Counter[str] = Counter()
    metadata_paths = [validated_path]
    for filename, output_split in _CV_SPLITS.items():
        split_path = root / f"{filename}.tsv"
        if not split_path.is_file():
            raise FileNotFoundError(f"Common Voice split TSV does not exist: {split_path}")
        metadata_paths.append(split_path)
        for row_number, row in _iter_common_voice_rows(split_path):
            audio_name = row["path"].strip()
            if not audio_name:
                raise ValueError(f"Common Voice {split_path} row {row_number} has empty path")
            previous_split = official_splits.get(audio_name)
            if previous_split is not None and previous_split != output_split:
                cross_split_clip_overlaps += 1
                continue
            official_splits[audio_name] = output_split
            split_counts[output_split] += 1
            existing = metadata.get(audio_name)
            if existing is None:
                metadata[audio_name] = row
            else:
                for field in ("client_id", "sentence", "locale"):
                    if existing.get(field, "").strip() != row.get(field, "").strip():
                        metadata_conflicts[field] += 1

    result: dict[str, CommonVoiceMetadata] = {}
    for audio_name, row in metadata.items():
        result[audio_name] = CommonVoiceMetadata(
            path=audio_name,
            client_id=row.get("client_id", "").strip(),
            sentence_id=row.get("sentence_id", "").strip(),
            sentence=row.get("sentence", "").strip(),
            locale=row.get("locale", "").strip(),
            accent=(row.get("accents") or row.get("accent") or "").strip(),
            gender=row.get("gender", "").strip(),
            up_votes=row.get("up_votes", "").strip(),
            down_votes=row.get("down_votes", "").strip(),
            official_split=official_splits.get(audio_name, "unassigned_validated"),
        )
    report = {
        "validated_records": validated_records,
        "metadata_union_records": len(result),
        "official_split_counts": dict(sorted(split_counts.items())),
        "unassigned_validated_records": sum(
            value.official_split == "unassigned_validated" for value in result.values()
        ),
        "duplicate_validated_paths": duplicate_validated_paths,
        "cross_split_clip_overlaps": cross_split_clip_overlaps,
        "metadata_conflicts": dict(sorted(metadata_conflicts.items())),
    }
    return result, report, tuple(metadata_paths)


def _iter_common_voice_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        required = {"client_id", "path", "sentence"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Common Voice TSV {path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Common Voice TSV {path} row {row_number} has extra columns")
            yield row_number, {key: value or "" for key, value in row.items()}


def _probe_audio_rows(
    rows: list[InventoryRow],
    *,
    workers: int,
    progress_every: int,
    label: str,
) -> list[AudioProbe]:
    def probe(row: InventoryRow) -> AudioProbe:
        path = Path(row.audio).expanduser()
        if not path.is_file():
            return AudioProbe("missing", None, None, str(path))
        try:
            metadata: AudioMetadata = read_audio_metadata(path)
        except (OSError, RuntimeError, ValueError, EOFError) as error:
            return AudioProbe("invalid", None, None, f"{type(error).__name__}: {error}")
        if metadata.duration_seconds <= 0 or metadata.sample_rate <= 0:
            return AudioProbe("invalid", None, None, "non-positive duration or sample rate")
        return AudioProbe(
            "ok",
            metadata.duration_seconds,
            metadata.sample_rate,
            "",
        )

    probes: list[AudioProbe] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(probe, rows), start=1):
            probes.append(result)
            if progress_every and index % progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"spanish_inventory source={label} audio={index}/{len(rows)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return probes


def _write_mls_inventory(
    rows: list[InventoryRow],
    probes: list[AudioProbe],
    output_path: Path,
) -> dict[str, Any]:
    fields = (
        "row_number",
        "audio",
        "source_id",
        "speaker_id",
        "duration_seconds",
        "sample_rate",
        "audio_status",
        "audio_error",
        "dialect_tier",
        "training_pool",
    )
    output_rows: list[dict[str, object]] = []
    speakers: set[str] = set()
    for row, probe in zip(rows, probes, strict=True):
        source_id = Path(row.audio).stem
        speaker_id = _mls_speaker_id(source_id)
        if speaker_id:
            speakers.add(speaker_id)
        output_rows.append(
            {
                "row_number": row.row_number,
                "audio": row.audio,
                "source_id": source_id,
                "speaker_id": speaker_id,
                "duration_seconds": _duration_value(probe),
                "sample_rate": probe.sample_rate or "",
                "audio_status": probe.status,
                "audio_error": probe.error,
                "dialect_tier": "mls_source_unknown_likely_peninsular",
                "training_pool": "fallback_mls_only" if probe.status == "ok" else "exclude",
            }
        )
    _write_tsv(output_path, fields, output_rows)
    summary = _audio_summary(probes)
    summary.update(
        {
            "records": len(rows),
            "speaker_count_from_source_id": len(speakers),
            "dialect_tier": "mls_source_unknown_likely_peninsular",
            "training_policy": "fallback_only_with_final_train_hour_cap",
            "inventory_path": str(output_path),
        }
    )
    return summary


def _write_common_voice_inventory(
    rows: list[InventoryRow],
    probes: list[AudioProbe],
    metadata: Mapping[str, CommonVoiceMetadata],
    core_clips: Mapping[str, str],
    output_path: Path,
) -> dict[str, Any]:
    fields = (
        "row_number",
        "audio",
        "source_id",
        "speaker_id",
        "official_split",
        "locale",
        "accent",
        "accent_tier",
        "gender",
        "up_votes",
        "down_votes",
        "core_overlap_split",
        "duration_seconds",
        "sample_rate",
        "audio_status",
        "audio_error",
        "metadata_text_match",
        "training_pool",
    )
    output_rows: list[dict[str, object]] = []
    tier_counts: Counter[str] = Counter()
    tier_seconds: defaultdict[str, float] = defaultdict(float)
    split_counts: Counter[str] = Counter()
    split_seconds: defaultdict[str, float] = defaultdict(float)
    pool_counts: Counter[str] = Counter()
    pool_seconds: defaultdict[str, float] = defaultdict(float)
    overlap_counts: Counter[str] = Counter()
    overlap_seconds: defaultdict[str, float] = defaultdict(float)
    speakers: set[str] = set()
    metadata_missing = 0
    text_mismatches = 0
    locale_counts: Counter[str] = Counter()
    accent_counts: Counter[str] = Counter()

    for row, probe in zip(rows, probes, strict=True):
        audio_name = Path(row.audio).name
        source_id = Path(audio_name).stem
        meta = metadata.get(audio_name)
        if meta is None:
            metadata_missing += 1
            speaker_id = ""
            official_split = "metadata_missing"
            locale = ""
            accent = ""
            gender = ""
            up_votes = ""
            down_votes = ""
            text_match = ""
            accent_tier = "unknown"
        else:
            speaker_id = meta.client_id
            official_split = meta.official_split
            locale = meta.locale
            accent = meta.accent
            gender = meta.gender
            up_votes = meta.up_votes
            down_votes = meta.down_votes
            text_match_bool = normalize_training_text(row.text) == normalize_training_text(
                meta.sentence
            )
            text_match = "true" if text_match_bool else "false"
            text_mismatches += int(not text_match_bool)
            accent_tier = classify_spanish_accent(accent)
            if speaker_id:
                speakers.add(speaker_id)
            locale_counts[locale or "<missing>"] += 1
            accent_counts[accent or "<missing>"] += 1

        core_overlap_split = core_clips.get(audio_name, "")
        training_pool = _training_pool(
            probe.status,
            official_split,
            accent_tier,
            bool(core_overlap_split),
            locale,
            text_match,
        )
        seconds = probe.duration_seconds or 0.0
        tier_counts[accent_tier] += 1
        tier_seconds[accent_tier] += seconds
        split_counts[official_split] += 1
        split_seconds[official_split] += seconds
        pool_counts[training_pool] += 1
        pool_seconds[training_pool] += seconds
        if core_overlap_split:
            overlap_counts[core_overlap_split] += 1
            overlap_seconds[core_overlap_split] += seconds
        output_rows.append(
            {
                "row_number": row.row_number,
                "audio": row.audio,
                "source_id": source_id,
                "speaker_id": speaker_id,
                "official_split": official_split,
                "locale": locale,
                "accent": accent,
                "accent_tier": accent_tier,
                "gender": gender,
                "up_votes": up_votes,
                "down_votes": down_votes,
                "core_overlap_split": core_overlap_split,
                "duration_seconds": _duration_value(probe),
                "sample_rate": probe.sample_rate or "",
                "audio_status": probe.status,
                "audio_error": probe.error,
                "metadata_text_match": text_match,
                "training_pool": training_pool,
            }
        )
    _write_tsv(output_path, fields, output_rows)
    summary = _audio_summary(probes)
    summary.update(
        {
            "records": len(rows),
            "metadata_joined_records": len(rows) - metadata_missing,
            "metadata_missing_records": metadata_missing,
            "metadata_text_mismatch_records": text_mismatches,
            "unexpected_locale_records": sum(
                count for locale, count in locale_counts.items() if locale != "es"
            ),
            "speaker_count": len(speakers),
            "locale_counts": _sorted_counter(locale_counts),
            "accent_counts": _sorted_counter(accent_counts),
            "accent_tier_counts": _sorted_counter(tier_counts),
            "accent_tier_hours": _hours_counter(tier_seconds),
            "official_split_counts": _sorted_counter(split_counts),
            "official_split_hours": _hours_counter(split_seconds),
            "training_pool_counts": _sorted_counter(pool_counts),
            "training_pool_hours": _hours_counter(pool_seconds),
            "core_overlap_counts": _sorted_counter(overlap_counts),
            "core_overlap_hours": _hours_counter(overlap_seconds),
            "inventory_path": str(output_path),
        }
    )
    return summary


def _training_pool(
    audio_status: str,
    official_split: str,
    accent_tier: str,
    core_overlap: bool,
    locale: str,
    metadata_text_match: str,
) -> str:
    if audio_status != "ok":
        return "exclude_invalid_audio"
    if core_overlap:
        return "exclude_core_overlap"
    if official_split == "metadata_missing":
        return "exclude_metadata_missing"
    if locale != "es":
        return "exclude_unexpected_locale"
    if metadata_text_match == "false":
        return "exclude_metadata_text_mismatch"
    if official_split in {"validation", "test"}:
        return "exclude_official_holdout"
    if accent_tier == "argentinian_rioplatense_metadata":
        return "priority_argentinian_rioplatense"
    if accent_tier == "latin_american_metadata":
        return "priority_latin_american"
    if accent_tier == "mixed_latin_american_peninsular":
        return "candidate_mixed"
    if accent_tier == "peninsular_metadata":
        return "fallback_peninsular"
    if accent_tier == "unknown":
        return "candidate_unknown"
    return "candidate_other_unclassified"


def _audio_summary(probes: list[AudioProbe]) -> dict[str, Any]:
    durations = sorted(
        probe.duration_seconds
        for probe in probes
        if probe.status == "ok" and probe.duration_seconds is not None
    )
    status_counts = Counter(probe.status for probe in probes)
    sample_rates = Counter(
        str(probe.sample_rate) for probe in probes if probe.sample_rate is not None
    )
    return {
        "valid_audio_records": status_counts["ok"],
        "invalid_audio_records": len(probes) - status_counts["ok"],
        "audio_status_counts": _sorted_counter(status_counts),
        "sample_rate_counts": _sorted_counter(sample_rates),
        "valid_audio_hours": round(sum(durations) / 3600, 6),
        "duration_seconds": {
            "count": len(durations),
            "minimum": durations[0] if durations else None,
            "mean": sum(durations) / len(durations) if durations else None,
            "median": _percentile(durations, 0.50),
            "p90": _percentile(durations, 0.90),
            "p95": _percentile(durations, 0.95),
            "maximum": durations[-1] if durations else None,
        },
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _mls_speaker_id(source_id: str) -> str:
    pieces = source_id.split("_")
    return pieces[0] if len(pieces) >= 3 and pieces[0].isdigit() else ""


def _fold_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value.casefold())
        if unicodedata.category(character) != "Mn"
    )


def _duration_value(probe: AudioProbe) -> str:
    return f"{probe.duration_seconds:.6f}" if probe.duration_seconds is not None else ""


def _write_tsv(
    path: Path,
    fields: tuple[str, ...],
    rows: Iterable[Mapping[str, object]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_manifest(destination: Path) -> None:
    rows = []
    for path in sorted(value for value in destination.rglob("*") if value.is_file()):
        if path.name == "sha256.txt":
            continue
        rows.append(f"{_sha256_file(path)}  {path.relative_to(destination)}\n")
    (destination / "sha256.txt").write_text("".join(rows), encoding="utf-8")


def _sorted_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted((key, value) for key, value in counter.items() if value))


def _hours_counter(counter: Mapping[str, float]) -> dict[str, float]:
    return dict(
        sorted((key, round(value / 3600, 6)) for key, value in counter.items() if value)
    )
