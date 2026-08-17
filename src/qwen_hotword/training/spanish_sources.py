from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

CANONICAL_FIELDS = (
    "audio",
    "text",
    "source_id",
    "speaker_id",
    "source_split",
    "language",
    "dialect",
    "source_subset",
    "gender",
    "accent",
    "sentence_id",
    "up_votes",
    "down_votes",
)

_SLR61_INDEX_FILES = {
    "female": "line_index_female.tsv",
    "male": "line_index_male.tsv",
    "weather_es_ar": "es_ar_line_index_weather.tsv",
}
_CV_SPLITS = {"train": "train", "dev": "validation", "test": "test"}


@dataclass(frozen=True)
class Slr61ConversionSummary:
    dataset: str
    source_root: str
    output_tsv_path: str
    source_records: int
    written_records: int
    skipped_records: int
    input_record_counts: dict[str, int]
    speaker_count: int
    gender_counts: dict[str, int]
    dialect_counts: dict[str, int]
    audio_extension_counts: dict[str, int]
    duplicate_source_ids: int
    duplicate_audio_values: int
    missing_audio_files: int
    audio_files_under_root: int
    indexed_audio_files: int
    excluded_peninsular_weather_audio_files: int
    unexpected_unindexed_audio_files: int
    issue_counts: dict[str, int]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CommonVoiceConversionSummary:
    dataset: str
    corpus_root: str
    output_tsv_path: str
    source_records: int
    written_records: int
    skipped_records: int
    split_record_counts: dict[str, int]
    speaker_count: int
    speakers_by_split: dict[str, int]
    cross_split_speaker_overlaps: dict[str, int]
    locale_counts: dict[str, int]
    accent_counts: dict[str, int]
    audio_extension_counts: dict[str, int]
    duplicate_source_ids: int
    duplicate_audio_values: int
    missing_audio_files: int
    audio_files_under_root: int
    unreferenced_audio_files: int
    issue_counts: dict[str, int]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def convert_slr61_argentinian_to_tsv(
    source_root: str | Path,
    output_tsv_path: str | Path,
    *,
    check_audio: bool = False,
    scan_audio_inventory: bool = False,
) -> Slr61ConversionSummary:
    root = Path(source_root).expanduser()
    downloads = root / "downloads"
    audio_root = root / "extracted"
    destination = Path(output_tsv_path).expanduser()
    if not downloads.is_dir():
        raise FileNotFoundError(f"SLR61 downloads directory does not exist: {downloads}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"SLR61 extracted directory does not exist: {audio_root}")

    input_rows: list[tuple[str, str, str]] = []
    input_record_counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    for subset, filename in _SLR61_INDEX_FILES.items():
        index_path = downloads / filename
        rows, malformed = _read_slr61_index(index_path)
        if malformed:
            issues["malformed_index_row"] += malformed
        input_record_counts[subset] = len(rows) + malformed
        input_rows.extend((subset, source_id, text) for source_id, text in rows)

    rows_to_write: list[dict[str, str]] = []
    source_ids: set[str] = set()
    audio_values: set[str] = set()
    speakers: set[str] = set()
    gender_counts: Counter[str] = Counter()
    dialect_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    duplicate_source_ids = 0
    duplicate_audio_values = 0
    missing_audio_files = 0

    for subset, source_id, text in input_rows:
        gender, speaker_id = _parse_slr61_source_id(source_id)
        if not speaker_id:
            issues["invalid_source_id"] += 1
            continue
        relative_audio = (
            Path("es-ar") / f"{source_id}.wav"
            if subset == "weather_es_ar"
            else Path(f"{source_id}.wav")
        )
        audio_path = audio_root / relative_audio
        audio_value = str(audio_path)
        if source_id in source_ids:
            duplicate_source_ids += 1
            issues["duplicate_source_id"] += 1
            continue
        if audio_value in audio_values:
            duplicate_audio_values += 1
            issues["duplicate_audio"] += 1
            continue
        if check_audio and not audio_path.is_file():
            missing_audio_files += 1
            issues["missing_audio_file"] += 1

        source_ids.add(source_id)
        audio_values.add(audio_value)
        speakers.add(speaker_id)
        gender_counts[gender] += 1
        dialect_counts["argentinian"] += 1
        extension_counts[audio_path.suffix.lower() or "<none>"] += 1
        rows_to_write.append(
            {
                "audio": audio_value,
                "text": text,
                "source_id": source_id,
                "speaker_id": speaker_id,
                "source_split": "unsplit",
                "language": "es",
                "dialect": "argentinian",
                "source_subset": subset,
                "gender": gender,
                "accent": "",
                "sentence_id": "",
                "up_votes": "",
                "down_votes": "",
            }
        )

    inventory = set(audio_root.rglob("*.wav")) if scan_audio_inventory else set()
    referenced = {Path(value) for value in audio_values}
    unindexed = inventory - referenced
    excluded_peninsular = {
        path for path in unindexed if _is_relative_to(path, audio_root / "es-es")
    }
    unexpected_unindexed = unindexed - excluded_peninsular
    if unexpected_unindexed:
        issues["unexpected_unindexed_audio_file"] += len(unexpected_unindexed)

    _write_canonical_tsv(destination, rows_to_write)
    source_records = sum(input_record_counts.values())
    summary = Slr61ConversionSummary(
        dataset="slr61_argentinian_spanish",
        source_root=str(root),
        output_tsv_path=str(destination),
        source_records=source_records,
        written_records=len(rows_to_write),
        skipped_records=source_records - len(rows_to_write),
        input_record_counts=_sorted_counter(input_record_counts),
        speaker_count=len(speakers),
        gender_counts=_sorted_counter(gender_counts),
        dialect_counts=_sorted_counter(dialect_counts),
        audio_extension_counts=_sorted_counter(extension_counts),
        duplicate_source_ids=duplicate_source_ids,
        duplicate_audio_values=duplicate_audio_values,
        missing_audio_files=missing_audio_files,
        audio_files_under_root=len(inventory),
        indexed_audio_files=len(referenced),
        excluded_peninsular_weather_audio_files=len(excluded_peninsular),
        unexpected_unindexed_audio_files=len(unexpected_unindexed),
        issue_counts=_sorted_counter(issues),
        status="pass"
        if rows_to_write
        and source_records == len(rows_to_write)
        and not issues
        and (not check_audio or missing_audio_files == 0)
        else "warn",
    )
    _write_summary(destination.parent / "slr61_conversion_summary.json", summary.to_dict())
    return summary


def convert_common_voice_rioplatense_to_tsv(
    corpus_root: str | Path,
    output_tsv_path: str | Path,
    *,
    check_audio: bool = False,
    scan_audio_inventory: bool = False,
) -> CommonVoiceConversionSummary:
    root = Path(corpus_root).expanduser()
    clips_root = root / "clips"
    destination = Path(output_tsv_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Common Voice corpus directory does not exist: {root}")
    if not clips_root.is_dir():
        raise FileNotFoundError(f"Common Voice clips directory does not exist: {clips_root}")

    rows_to_write: list[dict[str, str]] = []
    split_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    accent_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    source_ids: set[str] = set()
    audio_values: set[str] = set()
    speakers: set[str] = set()
    speakers_by_split: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_source_ids = 0
    duplicate_audio_values = 0
    missing_audio_files = 0
    source_records = 0

    for source_split, output_split in _CV_SPLITS.items():
        source_path = root / f"{source_split}.tsv"
        for _row_number, row in _read_common_voice_tsv(source_path):
            source_records += 1
            split_counts[output_split] += 1
            audio_name = row["path"].strip()
            text = row["sentence"].strip()
            speaker_id = row["client_id"].strip()
            locale = row["locale"].strip()
            accent = row["accents"].strip()
            source_id = Path(audio_name).stem
            audio_path = clips_root / audio_name
            audio_value = str(audio_path)

            if not audio_name:
                issues["missing_audio"] += 1
            if not text:
                issues["missing_text"] += 1
            if not speaker_id:
                issues["missing_speaker_id"] += 1
            if locale != "es":
                issues["unexpected_locale"] += 1
            if _integer_or_none(row["up_votes"]) is None:
                issues["invalid_up_votes"] += 1
            if _integer_or_none(row["down_votes"]) is None:
                issues["invalid_down_votes"] += 1
            if source_id in source_ids:
                duplicate_source_ids += 1
                issues["duplicate_source_id"] += 1
                continue
            if audio_value in audio_values:
                duplicate_audio_values += 1
                issues["duplicate_audio"] += 1
                continue
            if not audio_name or not text:
                continue
            if check_audio and not audio_path.is_file():
                missing_audio_files += 1
                issues["missing_audio_file"] += 1

            source_ids.add(source_id)
            audio_values.add(audio_value)
            if speaker_id:
                speakers.add(speaker_id)
                speakers_by_split[output_split].add(speaker_id)
            locale_counts[locale or "<missing>"] += 1
            accent_counts[accent or "<missing>"] += 1
            extension_counts[audio_path.suffix.lower() or "<none>"] += 1
            rows_to_write.append(
                {
                    "audio": audio_value,
                    "text": text,
                    "source_id": source_id,
                    "speaker_id": speaker_id,
                    "source_split": output_split,
                    "language": "es",
                    "dialect": "rioplatense",
                    "source_subset": f"common_voice_v26_{source_split}",
                    "gender": row["gender"].strip(),
                    "accent": accent,
                    "sentence_id": row["sentence_id"].strip(),
                    "up_votes": row["up_votes"].strip(),
                    "down_votes": row["down_votes"].strip(),
                }
            )

    overlaps = _cross_split_overlaps(speakers_by_split)
    if any(overlaps.values()):
        issues["cross_split_speaker_overlap"] += sum(overlaps.values())
    inventory = set(clips_root.rglob("*.mp3")) if scan_audio_inventory else set()
    referenced = {Path(value) for value in audio_values}
    unreferenced = inventory - referenced
    if unreferenced:
        issues["unreferenced_audio_file"] += len(unreferenced)

    _write_canonical_tsv(destination, rows_to_write)
    summary = CommonVoiceConversionSummary(
        dataset="common_voice_rioplatense_v26",
        corpus_root=str(root),
        output_tsv_path=str(destination),
        source_records=source_records,
        written_records=len(rows_to_write),
        skipped_records=source_records - len(rows_to_write),
        split_record_counts=_sorted_counter(split_counts),
        speaker_count=len(speakers),
        speakers_by_split={key: len(speakers_by_split[key]) for key in sorted(_CV_SPLITS.values())},
        cross_split_speaker_overlaps=overlaps,
        locale_counts=_sorted_counter(locale_counts),
        accent_counts=_sorted_counter(accent_counts),
        audio_extension_counts=_sorted_counter(extension_counts),
        duplicate_source_ids=duplicate_source_ids,
        duplicate_audio_values=duplicate_audio_values,
        missing_audio_files=missing_audio_files,
        audio_files_under_root=len(inventory),
        unreferenced_audio_files=len(unreferenced),
        issue_counts=_sorted_counter(issues),
        status="pass"
        if rows_to_write
        and source_records == len(rows_to_write)
        and not issues
        and (not check_audio or missing_audio_files == 0)
        else "warn",
    )
    _write_summary(
        destination.parent / "common_voice_conversion_summary.json", summary.to_dict()
    )
    return summary


def _read_slr61_index(path: Path) -> tuple[list[tuple[str, str]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"SLR61 index does not exist: {path}")
    rows: list[tuple[str, str]] = []
    malformed = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 2 or not row[0].strip() or not row[1].strip():
                malformed += 1
                continue
            rows.append((row[0].strip(), row[1].strip()))
    return rows, malformed


def _read_common_voice_tsv(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    required = {
        "client_id",
        "path",
        "sentence_id",
        "sentence",
        "up_votes",
        "down_votes",
        "gender",
        "accents",
        "locale",
    }
    if not path.is_file():
        raise FileNotFoundError(f"Common Voice split TSV does not exist: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Common Voice TSV {path} is missing columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Common Voice TSV {path} row {row_number} has extra columns")
            yield row_number, {key: value or "" for key, value in row.items()}


def _parse_slr61_source_id(source_id: str) -> tuple[str, str]:
    pieces = source_id.split("_")
    if len(pieces) < 3 or pieces[0] not in {"arf", "arm"} or not pieces[1].isdigit():
        return "", ""
    return ("female" if pieces[0] == "arf" else "male", "_".join(pieces[:2]))


def _cross_split_overlaps(
    speakers_by_split: Mapping[str, set[str]],
) -> dict[str, int]:
    splits = tuple(sorted(_CV_SPLITS.values()))
    return {
        f"{left}__{right}": len(speakers_by_split[left] & speakers_by_split[right])
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    }


def _write_canonical_tsv(destination: Path, rows: Iterable[dict[str, str]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)


def _write_summary(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sorted_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted((key, value) for key, value in counter.items() if value))


def _integer_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
