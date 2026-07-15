from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AuditSample:
    row_number: int
    audio_relative: str
    audio_path: str
    audio_exists: bool
    text: str


@dataclass(frozen=True)
class TsvAudit:
    tsv_path: str
    audio_root: str
    rows_scanned: int
    rows_with_audio: int
    rows_with_text: int
    resolved_audio_files: int
    missing_audio_files: int
    absolute_audio_values: int
    duplicate_audio_values: int
    minimum_text_characters: int
    maximum_text_characters: int
    mean_text_characters: float
    samples: tuple[AuditSample, ...]
    errors: tuple[str, ...]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_training_tsv(
    tsv_path: str | Path,
    audio_root: str | Path,
    *,
    audio_column: str = "audio",
    text_column: str = "text",
    max_records: int = 1000,
    sample_count: int = 3,
) -> TsvAudit:
    path = Path(tsv_path).expanduser()
    root = Path(audio_root).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TSV does not exist: {path}")
    if not root.is_dir():
        raise FileNotFoundError(f"audio root does not exist: {root}")
    if max_records < 0:
        raise ValueError("max_records must be non-negative")
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")

    rows_scanned = 0
    rows_with_audio = 0
    rows_with_text = 0
    resolved_audio_files = 0
    missing_audio_files = 0
    absolute_audio_values = 0
    duplicate_audio_values = 0
    text_lengths: list[int] = []
    samples: list[AuditSample] = []
    seen_audio: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing_columns = [
            column for column in (audio_column, text_column) if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"TSV is missing required columns {missing_columns}; "
                f"found {sorted(fieldnames)}"
            )

        for row_number, row in enumerate(reader, start=2):
            if max_records > 0 and rows_scanned >= max_records:
                break
            rows_scanned += 1
            audio_value = str(row.get(audio_column) or "").strip()
            text = str(row.get(text_column) or "").strip()

            if audio_value:
                rows_with_audio += 1
            if text:
                rows_with_text += 1
                text_lengths.append(len(text))

            relative_path = Path(audio_value) if audio_value else Path()
            if audio_value and relative_path.is_absolute():
                absolute_audio_values += 1
                resolved_path = relative_path
            else:
                resolved_path = root / relative_path

            audio_key = str(resolved_path)
            if audio_value and audio_key in seen_audio:
                duplicate_audio_values += 1
            elif audio_value:
                seen_audio.add(audio_key)

            audio_exists = bool(audio_value and resolved_path.is_file())
            if audio_exists:
                resolved_audio_files += 1
            elif audio_value:
                missing_audio_files += 1

            if len(samples) < sample_count:
                samples.append(
                    AuditSample(
                        row_number=row_number,
                        audio_relative=audio_value,
                        audio_path=str(resolved_path),
                        audio_exists=audio_exists,
                        text=text,
                    )
                )

    errors: list[str] = []
    if rows_scanned == 0:
        errors.append("TSV contains no data rows")
    if rows_with_audio != rows_scanned:
        errors.append(f"{rows_scanned - rows_with_audio} rows have empty audio values")
    if rows_with_text != rows_scanned:
        errors.append(f"{rows_scanned - rows_with_text} rows have empty text values")
    if missing_audio_files:
        errors.append(f"{missing_audio_files} relative audio paths did not resolve")
    if absolute_audio_values:
        errors.append(
            f"{absolute_audio_values} audio values are absolute; expected relative paths"
        )

    total_text_characters = sum(text_lengths)
    return TsvAudit(
        tsv_path=str(path),
        audio_root=str(root),
        rows_scanned=rows_scanned,
        rows_with_audio=rows_with_audio,
        rows_with_text=rows_with_text,
        resolved_audio_files=resolved_audio_files,
        missing_audio_files=missing_audio_files,
        absolute_audio_values=absolute_audio_values,
        duplicate_audio_values=duplicate_audio_values,
        minimum_text_characters=min(text_lengths, default=0),
        maximum_text_characters=max(text_lengths, default=0),
        mean_text_characters=(
            total_text_characters / len(text_lengths) if text_lengths else 0.0
        ),
        samples=tuple(samples),
        errors=tuple(errors),
        status="pass" if not errors else "fail",
    )
