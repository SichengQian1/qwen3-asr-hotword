from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_inventory import classify_spanish_accent

SPLITS = ("train", "validation", "test")
EXPLICIT_LATIN_AMERICAN_TIERS = {
    "argentinian_rioplatense_metadata",
    "latin_american_metadata",
}


@dataclass(frozen=True)
class CandidateRow:
    row_number: int
    audio: str
    text: str
    source_id: str
    speaker_id: str
    accent: str
    accent_tier: str
    duration_seconds: float
    source_split: str


def assign_spanish_speaker_split(
    speaker_id: str,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> str:
    """Assign one speaker to one deterministic split."""

    fractions = _validate_fractions(
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    fraction = _stable_fraction(f"{seed}\0{speaker_id}")
    train_boundary = fractions["train"]
    validation_boundary = train_boundary + fractions["validation"]
    if fraction < train_boundary:
        return "train"
    if fraction < validation_boundary:
        return "validation"
    return "test"


def select_spanish_auxiliary_pool(
    source_tsv: str | Path,
    inventory_tsv: str | Path,
    rioplatense_tsv: str | Path,
    output_dir: str | Path,
    *,
    target_hours: float = 170.0,
    train_fraction: float = 0.96,
    validation_fraction: float = 0.02,
    test_fraction: float = 0.02,
    maximum_latin_american_speaker_hours: float = 1.0,
    seed: int = 20_260_824,
) -> dict[str, Any]:
    """Select a reproducible, speaker-disjoint explicit Latin-American CV pool."""

    source_path = Path(source_tsv).expanduser()
    inventory_path = Path(inventory_tsv).expanduser()
    core_path = Path(rioplatense_tsv).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (source_path, inventory_path, core_path):
        if not path.is_file():
            raise FileNotFoundError(f"required TSV does not exist: {path}")
    if not math.isfinite(target_hours) or target_hours <= 0:
        raise ValueError("target_hours must be finite and positive")
    if (
        not math.isfinite(maximum_latin_american_speaker_hours)
        or maximum_latin_american_speaker_hours <= 0
    ):
        raise ValueError(
            "maximum_latin_american_speaker_hours must be finite and positive"
        )
    fractions = _validate_fractions(
        train_fraction,
        validation_fraction,
        test_fraction,
    )
    _require_empty_directory(destination)

    source_rows = _read_source_rows(source_path)
    core_speakers = _read_core_speakers(core_path)
    candidates, exclusion_counts, exclusion_seconds = _read_candidates(
        inventory_path,
        source_rows,
        core_speakers,
        seed=seed,
        fractions=fractions,
    )
    eligible_tier_counts: Counter[str] = Counter(row.accent_tier for row in candidates)
    eligible_tier_seconds: defaultdict[str, float] = defaultdict(float)
    eligible_split_seconds: defaultdict[str, float] = defaultdict(float)
    eligible_speakers_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for row in candidates:
        eligible_tier_seconds[row.accent_tier] += row.duration_seconds
        eligible_split_seconds[row.source_split] += row.duration_seconds
        eligible_speakers_by_split[row.source_split].add(row.speaker_id)

    by_split_tier: dict[str, dict[str, list[CandidateRow]]] = {
        split: {tier: [] for tier in EXPLICIT_LATIN_AMERICAN_TIERS}
        for split in SPLITS
    }
    for candidate in candidates:
        by_split_tier[candidate.source_split][candidate.accent_tier].append(candidate)

    target_seconds = {
        split: target_hours * 3600.0 * fractions[split] for split in SPLITS
    }
    selected: list[CandidateRow] = []
    selected_audio: set[str] = set()
    speaker_cap_seconds = maximum_latin_american_speaker_hours * 3600.0
    decision_counts: Counter[str] = Counter(exclusion_counts)
    decision_seconds: defaultdict[str, float] = defaultdict(float, exclusion_seconds)

    for split in SPLITS:
        priority = sorted(
            by_split_tier[split]["argentinian_rioplatense_metadata"],
            key=lambda row: (row.speaker_id, row.row_number),
        )
        for row in priority:
            selected.append(row)
            selected_audio.add(row.audio)
            decision_counts["selected_argentinian_rioplatense"] += 1
            decision_seconds["selected_argentinian_rioplatense"] += row.duration_seconds

        current_seconds = sum(row.duration_seconds for row in priority)
        latin_rows = by_split_tier[split]["latin_american_metadata"]
        capped_groups, capped_out = _cap_speaker_rows(
            latin_rows,
            maximum_seconds=speaker_cap_seconds,
            seed=seed,
        )
        for row in capped_out:
            decision_counts["not_selected_speaker_cap"] += 1
            decision_seconds["not_selected_speaker_cap"] += row.duration_seconds

        groups = sorted(
            capped_groups.items(),
            key=lambda item: (_stable_digest(f"{seed}\0select\0{item[0]}"), item[0]),
        )
        for group_index, (_speaker_id, rows) in enumerate(groups):
            if current_seconds >= target_seconds[split]:
                for _remaining_speaker, remaining_rows in groups[group_index:]:
                    for row in remaining_rows:
                        decision_counts["not_selected_quota"] += 1
                        decision_seconds["not_selected_quota"] += row.duration_seconds
                break
            for row in rows:
                if row.audio in selected_audio:
                    raise ValueError(f"duplicate selected audio: {row.audio}")
                selected.append(row)
                selected_audio.add(row.audio)
                current_seconds += row.duration_seconds
                decision_counts["selected_latin_american"] += 1
                decision_seconds["selected_latin_american"] += row.duration_seconds

    selected.sort(key=lambda row: row.row_number)
    split_seconds: defaultdict[str, float] = defaultdict(float)
    split_counts: Counter[str] = Counter()
    tier_seconds: defaultdict[str, float] = defaultdict(float)
    tier_counts: Counter[str] = Counter()
    speakers_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for row in selected:
        split_counts[row.source_split] += 1
        split_seconds[row.source_split] += row.duration_seconds
        tier_counts[row.accent_tier] += 1
        tier_seconds[row.accent_tier] += row.duration_seconds
        speakers_by_split[row.source_split].add(row.speaker_id)
    selected_core_speaker_counts: Counter[str] = Counter(
        core_speakers[row.speaker_id]
        for row in selected
        if row.speaker_id in core_speakers
    )
    if any(split != "train" for split in selected_core_speaker_counts):
        raise RuntimeError("a core validation/test speaker entered the auxiliary pool")

    short_splits = [
        split
        for split in SPLITS
        if split_seconds[split] + 1e-9 < target_seconds[split]
    ]
    if short_splits:
        raise ValueError(
            "insufficient eligible speaker-disjoint hours for splits: "
            + ", ".join(short_splits)
        )
    overlaps = _speaker_overlaps(speakers_by_split)
    if any(overlaps.values()):
        raise RuntimeError(f"selected speakers overlap across splits: {overlaps}")

    destination.mkdir(parents=True)
    source_output = destination / "source.tsv"
    inventory_output = destination / "selected_inventory.tsv"
    _write_selected_source(source_output, selected)
    _write_selected_inventory(inventory_output, selected)

    selected_seconds = sum(split_seconds.values())
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "selection_policy": "explicit_latin_american_metadata_speaker_disjoint",
        "source_tsv": str(source_path),
        "inventory_tsv": str(inventory_path),
        "rioplatense_tsv": str(core_path),
        "output_dir": str(destination),
        "source_records": len(source_rows),
        "eligible_records": len(candidates),
        "eligible_hours": round(
            sum(row.duration_seconds for row in candidates) / 3600.0,
            6,
        ),
        "eligible_speakers": len({row.speaker_id for row in candidates}),
        "eligible_tier_counts": dict(sorted(eligible_tier_counts.items())),
        "eligible_tier_hours": _hours_mapping(eligible_tier_seconds),
        "eligible_split_hours": _hours_mapping(eligible_split_seconds),
        "eligible_speakers_by_split": {
            split: len(eligible_speakers_by_split[split]) for split in SPLITS
        },
        "target_hours": target_hours,
        "selected_records": len(selected),
        "selected_hours": round(selected_seconds / 3600.0, 6),
        "selected_speakers": len({row.speaker_id for row in selected}),
        "split_fractions": fractions,
        "target_split_hours": _hours_mapping(target_seconds),
        "selected_split_counts": _ordered_counts(split_counts),
        "selected_split_hours": _hours_mapping(split_seconds),
        "selected_speakers_by_split": {
            split: len(speakers_by_split[split]) for split in SPLITS
        },
        "cross_split_speaker_overlaps": overlaps,
        "selected_core_speaker_counts": dict(sorted(selected_core_speaker_counts.items())),
        "selected_tier_counts": dict(sorted(tier_counts.items())),
        "selected_tier_hours": _hours_mapping(tier_seconds),
        "decision_counts": dict(sorted(decision_counts.items())),
        "decision_hours": _hours_mapping(decision_seconds),
        "maximum_latin_american_speaker_hours": (
            maximum_latin_american_speaker_hours
        ),
        "core_speaker_policy": {
            "core_train_speakers_forced_to_train": True,
            "core_validation_or_test_speakers_excluded": True,
        },
        "excluded_dialect_policy": {
            "unknown_common_voice": True,
            "peninsular_common_voice": True,
            "other_unclassified_common_voice": True,
            "mls_not_an_input": True,
        },
        "seed": seed,
        "source_output_path": str(source_output),
        "selected_inventory_path": str(inventory_output),
        "input_sha256": {
            "source_tsv": _sha256_file(source_path),
            "inventory_tsv": _sha256_file(inventory_path),
            "rioplatense_tsv": _sha256_file(core_path),
        },
    }
    _write_json(destination / "summary.json", summary)
    _write_json(
        destination / "selection_config.json",
        {
            "schema_version": 1,
            "target_hours": target_hours,
            "split_fractions": fractions,
            "maximum_latin_american_speaker_hours": (
                maximum_latin_american_speaker_hours
            ),
            "seed": seed,
            "eligible_accent_tiers": sorted(EXPLICIT_LATIN_AMERICAN_TIERS),
            "central_america_reclassified_from_raw_accent": True,
        },
    )
    _write_sha256_manifest(destination)
    return summary


def _read_source_rows(path: Path) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = sorted({"audio", "text"} - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"source TSV is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            audio = str(row.get("audio") or "").strip()
            text = str(row.get("text") or "").strip()
            if not audio or not text:
                raise ValueError(f"source TSV row {row_number} has empty audio or text")
            if audio in result:
                raise ValueError(f"source TSV contains duplicate audio: {audio}")
            result[audio] = (row_number, text)
    if not result:
        raise ValueError("source TSV contains no data rows")
    return result


def _read_core_speakers(path: Path) -> dict[str, str]:
    speaker_splits: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"speaker_id", "source_split"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Rioplatense TSV is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            speaker_id = str(row.get("speaker_id") or "").strip()
            split = str(row.get("source_split") or "").strip()
            if split not in SPLITS:
                raise ValueError(f"Rioplatense TSV row {row_number} has invalid split")
            if not speaker_id:
                continue
            previous = speaker_splits.get(speaker_id)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Rioplatense speaker occurs in multiple splits: {speaker_id}"
                )
            speaker_splits[speaker_id] = split
    return speaker_splits


def _read_candidates(
    path: Path,
    source_rows: Mapping[str, tuple[int, str]],
    core_speakers: Mapping[str, str],
    *,
    seed: int,
    fractions: Mapping[str, float],
) -> tuple[list[CandidateRow], Counter[str], defaultdict[str, float]]:
    candidates: list[CandidateRow] = []
    decisions: Counter[str] = Counter()
    decision_seconds: defaultdict[str, float] = defaultdict(float)
    seen_audio: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "audio",
            "source_id",
            "speaker_id",
            "official_split",
            "locale",
            "accent",
            "core_overlap_split",
            "duration_seconds",
            "audio_status",
            "metadata_text_match",
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"inventory TSV is missing columns: {missing}")
        for inventory_row_number, row in enumerate(reader, start=2):
            audio = str(row.get("audio") or "").strip()
            if not audio or audio in seen_audio:
                raise ValueError(
                    f"inventory TSV row {inventory_row_number} has empty/duplicate audio"
                )
            seen_audio.add(audio)
            source = source_rows.get(audio)
            if source is None:
                raise ValueError(f"inventory audio is absent from source TSV: {audio}")
            row_number, text = source
            duration = _parse_duration(row.get("duration_seconds"), inventory_row_number)
            speaker_id = str(row.get("speaker_id") or "").strip()
            accent = str(row.get("accent") or "").strip()
            accent_tier = classify_spanish_accent(accent)
            reason = _candidate_exclusion_reason(row, speaker_id, accent_tier, core_speakers)
            if reason is not None:
                decisions[reason] += 1
                decision_seconds[reason] += duration
                continue
            core_split = core_speakers.get(speaker_id)
            source_split = (
                "train"
                if core_split == "train"
                else assign_spanish_speaker_split(
                    speaker_id,
                    seed=seed,
                    train_fraction=fractions["train"],
                    validation_fraction=fractions["validation"],
                    test_fraction=fractions["test"],
                )
            )
            candidates.append(
                CandidateRow(
                    row_number=row_number,
                    audio=audio,
                    text=text,
                    source_id=str(row.get("source_id") or "").strip(),
                    speaker_id=speaker_id,
                    accent=accent,
                    accent_tier=accent_tier,
                    duration_seconds=duration,
                    source_split=source_split,
                )
            )
    if len(seen_audio) != len(source_rows):
        raise ValueError(
            "source/inventory record count differs: "
            f"source={len(source_rows)} inventory={len(seen_audio)}"
        )
    return candidates, decisions, decision_seconds


def _candidate_exclusion_reason(
    row: Mapping[str | None, str | None],
    speaker_id: str,
    accent_tier: str,
    core_speakers: Mapping[str, str],
) -> str | None:
    if str(row.get("audio_status") or "").strip() != "ok":
        return "exclude_invalid_audio"
    if str(row.get("core_overlap_split") or "").strip():
        return "exclude_core_audio_overlap"
    if str(row.get("official_split") or "").strip() != "train":
        return "exclude_not_official_train"
    if str(row.get("locale") or "").strip() != "es":
        return "exclude_unexpected_locale"
    if str(row.get("metadata_text_match") or "").strip() != "true":
        return "exclude_metadata_text_mismatch"
    if not speaker_id:
        return "exclude_missing_speaker"
    if core_speakers.get(speaker_id) in {"validation", "test"}:
        return "exclude_core_holdout_speaker_overlap"
    if accent_tier not in EXPLICIT_LATIN_AMERICAN_TIERS:
        return f"exclude_accent_tier_{accent_tier}"
    return None


def _cap_speaker_rows(
    rows: Iterable[CandidateRow],
    *,
    maximum_seconds: float,
    seed: int,
) -> tuple[dict[str, list[CandidateRow]], list[CandidateRow]]:
    by_speaker: defaultdict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        by_speaker[row.speaker_id].append(row)
    selected: dict[str, list[CandidateRow]] = {}
    capped_out: list[CandidateRow] = []
    for speaker_id, speaker_rows in by_speaker.items():
        ordered = sorted(
            speaker_rows,
            key=lambda row: (_stable_digest(f"{seed}\0row\0{row.audio}"), row.row_number),
        )
        kept: list[CandidateRow] = []
        seconds = 0.0
        for row in ordered:
            if kept and seconds + row.duration_seconds > maximum_seconds:
                capped_out.append(row)
                continue
            kept.append(row)
            seconds += row.duration_seconds
        selected[speaker_id] = kept
    return selected, capped_out


def _write_selected_source(path: Path, rows: Iterable[CandidateRow]) -> None:
    fields = (
        "audio",
        "text",
        "source_split",
        "speaker_id",
        "source_id",
        "source_subset",
        "accent_tier",
        "accent",
        "original_row_number",
    )
    output = (
        {
            "audio": row.audio,
            "text": row.text,
            "source_split": row.source_split,
            "speaker_id": row.speaker_id,
            "source_id": row.source_id,
            "source_subset": f"common_voice_v25_{row.accent_tier}",
            "accent_tier": row.accent_tier,
            "accent": row.accent,
            "original_row_number": row.row_number,
        }
        for row in rows
    )
    _write_tsv(path, fields, output)


def _write_selected_inventory(path: Path, rows: Iterable[CandidateRow]) -> None:
    fields = (
        "original_row_number",
        "audio",
        "source_id",
        "speaker_id",
        "source_split",
        "duration_seconds",
        "accent_tier",
        "accent",
    )
    output = (
        {
            "original_row_number": row.row_number,
            "audio": row.audio,
            "source_id": row.source_id,
            "speaker_id": row.speaker_id,
            "source_split": row.source_split,
            "duration_seconds": f"{row.duration_seconds:.6f}",
            "accent_tier": row.accent_tier,
            "accent": row.accent,
        }
        for row in rows
    )
    _write_tsv(path, fields, output)


def _validate_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, float]:
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    if any(not math.isfinite(value) or value < 0 for value in fractions.values()):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    if fractions["train"] <= 0:
        raise ValueError("train fraction must be positive")
    return fractions


def _parse_duration(value: str | None, row_number: int) -> float:
    try:
        duration = float(value or "")
    except ValueError as error:
        raise ValueError(f"inventory row {row_number} has invalid duration") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"inventory row {row_number} has non-positive duration")
    return duration


def _speaker_overlaps(speakers: Mapping[str, set[str]]) -> dict[str, int]:
    return {
        "train__validation": len(speakers["train"] & speakers["validation"]),
        "train__test": len(speakers["train"] & speakers["test"]),
        "validation__test": len(speakers["validation"] & speakers["test"]),
    }


def _ordered_counts(values: Mapping[str, int]) -> dict[str, int]:
    return {split: values.get(split, 0) for split in SPLITS}


def _hours_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(value / 3600.0, 6) for key, value in sorted(values.items())}


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_fraction(value: str) -> float:
    raw = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") / 2**64


def _require_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {path}")


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
