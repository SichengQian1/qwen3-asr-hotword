from __future__ import annotations

import csv
import importlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.config import EvalAssetConfig, SourceConfig
from qwen_hotword.evaluation.records import Utterance

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".opus", ".ogg", ".m4a")
TEXT_COLUMNS = ("sentence", "text", "transcript", "transcription", "raw_transcription")
PATH_COLUMNS = ("audio_path", "path", "file", "filename", "audio")


def scan_sources(config: EvalAssetConfig) -> list[Utterance]:
    all_utterances: list[Utterance] = []
    for index, source in enumerate(config.sources):
        utterances = list(_scan_source(source))
        utterances = _filter_utterances(utterances, source)
        rng = random.Random(config.sampling.seed + index)
        rng.shuffle(utterances)
        if config.sampling.max_utterances_per_source > 0:
            utterances = utterances[: config.sampling.max_utterances_per_source]
        all_utterances.extend(sorted(utterances, key=lambda item: item.utt_id))
    return sorted(all_utterances, key=lambda item: item.utt_id)


def _scan_source(source: SourceConfig) -> Iterable[Utterance]:
    dataset = source.dataset.lower().replace("_", "-")
    if "librispeech" in dataset and "mls" not in dataset:
        return _scan_librispeech(source)
    if "common" in dataset and "voice" in dataset:
        return _scan_common_voice(source)
    if dataset.startswith("mls") or "multilingual-librispeech" in dataset:
        return _scan_mls(source)
    return _scan_generic_metadata(source)


def _source_metadata(source: SourceConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    metadata = {
        "use_for_hotwords": str(source.use_for_hotwords).lower(),
        "use_for_cases": str(source.use_for_cases).lower(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _filter_utterances(
    utterances: list[Utterance],
    source: SourceConfig,
) -> list[Utterance]:
    split_filter = {split.lower() for split in source.include_splits}
    if not split_filter:
        return utterances
    return [
        utterance
        for utterance in utterances
        if utterance.split.lower() in split_filter
    ]


def _scan_librispeech(source: SourceConfig) -> list[Utterance]:
    root = source.root
    utterances: list[Utterance] = []
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        split = _infer_split(root, transcript_path)
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                raw_utt_id, text = parts
                audio_path = transcript_path.parent / f"{raw_utt_id}.flac"
                if not audio_path.is_file():
                    continue
                utterances.append(
                    Utterance(
                        utt_id=f"{source.name}_{split}_{raw_utt_id}",
                        dataset=source.name,
                        split=split,
                        language=source.language,
                        audio_path=str(audio_path),
                        text=text,
                        metadata=_source_metadata(source),
                    )
                )
    utterances.extend(_scan_librispeech_tsv(source))
    return utterances


def _scan_librispeech_tsv(source: SourceConfig) -> list[Utterance]:
    root = source.root
    utterances: list[Utterance] = []
    for transcript_path in sorted(root.glob("trans_*.tsv")):
        split = transcript_path.stem.removeprefix("trans_")
        audio_root = root / split
        audio_index = _build_audio_index(audio_root if audio_root.is_dir() else root)
        rows = _read_tsv_with_fallback_header(transcript_path)
        for row in rows:
            text = _first_string(row, TEXT_COLUMNS)
            audio_value = _first_value(row, PATH_COLUMNS)
            raw_utt_id = _first_string(row, ("id", "utt_id", "audio_id", "key"))
            if raw_utt_id is None and isinstance(audio_value, str):
                raw_utt_id = Path(audio_value).stem
            if not text or not raw_utt_id:
                continue
            audio_path = None
            if isinstance(audio_value, str):
                audio_path = _resolve_audio_path(transcript_path.parent, audio_value, root=root)
            if audio_path is None:
                audio_path = audio_index.get(raw_utt_id) or audio_index.get(Path(raw_utt_id).stem)
            if audio_path is None:
                continue
            utterances.append(
                Utterance(
                    utt_id=f"{source.name}_{split}_{raw_utt_id}",
                    dataset=source.name,
                    split=split,
                    language=source.language,
                    audio_path=str(audio_path),
                    text=text,
                    metadata=_source_metadata(source),
                )
            )
    return utterances


def _scan_mls(source: SourceConfig) -> list[Utterance]:
    root = source.root
    split_by_id = _load_mls_splits(root, source.include_splits)
    audio_index = _build_mls_audio_index(root, source.include_splits)
    utterances: list[Utterance] = []
    for transcript_path in sorted(root.rglob("transcripts.txt")):
        inferred_split = _infer_split(root, transcript_path)
        if source.include_splits and inferred_split.lower() not in {
            item.lower() for item in source.include_splits
        }:
            continue
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parsed = _parse_transcript_line(line)
                if parsed is None:
                    continue
                raw_utt_id, text = parsed
                split = split_by_id.get(raw_utt_id, inferred_split)
                if source.include_splits and split.lower() not in {
                    item.lower() for item in source.include_splits
                }:
                    continue
                audio_path = audio_index.get(raw_utt_id)
                if audio_path is None:
                    continue
                utterances.append(
                    Utterance(
                        utt_id=f"{source.name}_{split}_{raw_utt_id}",
                        dataset=source.name,
                        split=split,
                        language=source.language,
                        audio_path=str(audio_path),
                        text=text,
                        metadata=_source_metadata(source),
                    )
                )
    return utterances


def _scan_common_voice(source: SourceConfig) -> list[Utterance]:
    root = source.root
    utterances: list[Utterance] = []
    locale_filter = {locale.lower().replace("_", "-") for locale in source.include_locales}
    for tsv_path in sorted(root.rglob("*.tsv")):
        split = tsv_path.stem
        if source.include_splits and split.lower() not in {
            item.lower() for item in source.include_splits
        }:
            continue
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_index, row in enumerate(reader):
                text = row.get("sentence") or row.get("text")
                relative_audio = row.get("path")
                if not text or not relative_audio:
                    continue
                locale = _infer_locale(root, tsv_path, row)
                if locale_filter and locale.lower().replace("_", "-") not in locale_filter:
                    continue
                audio_path = _resolve_audio_path(tsv_path.parent, relative_audio, root=root)
                if audio_path is None:
                    continue
                raw_utt_id = Path(relative_audio).stem or str(row_index)
                utterances.append(
                    Utterance(
                        utt_id=f"{source.name}_{split}_{raw_utt_id}",
                        dataset=source.name,
                        split=split,
                        language=source.language,
                        audio_path=str(audio_path),
                        text=text,
                        metadata=_source_metadata(source, {"locale": locale}),
                    )
                )
    return utterances


def _scan_generic_metadata(source: SourceConfig) -> list[Utterance]:
    root = source.root
    utterances: list[Utterance] = []
    for metadata_path in sorted(_iter_metadata_files(root)):
        split = _infer_split(root, metadata_path)
        rows = _read_metadata_rows(metadata_path)
        for row_index, row in enumerate(rows):
            utterance = _row_to_utterance(source, metadata_path, row, row_index, split)
            if utterance is not None:
                utterances.append(utterance)
    return utterances


def _iter_metadata_files(root: Path) -> Iterable[Path]:
    for suffix in ("*.jsonl", "*.csv", "*.tsv", "*.parquet"):
        yield from root.rglob(suffix)


def _read_metadata_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    if path.suffix in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if path.suffix == ".parquet":
        try:
            pq: Any = importlib.import_module("pyarrow.parquet")
        except ImportError:
            return []
        table = pq.read_table(path)
        return [dict(row) for row in table.to_pylist() if isinstance(row, dict)]
    return []


def _read_tsv_with_fallback_header(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        sample = handle.readline()
        handle.seek(0)
        columns = sample.rstrip("\n").split("\t")
        known_columns = set(TEXT_COLUMNS + PATH_COLUMNS + ("id", "utt_id", "audio_id", "key"))
        has_header = any(column in known_columns for column in columns)
        if has_header:
            return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
        reader = csv.reader(handle, delimiter="\t")
        rows = []
        for row in reader:
            if len(row) >= 3:
                rows.append({"id": row[0], "path": row[1], "text": row[2]})
            elif len(row) == 2:
                rows.append({"id": row[0], "text": row[1]})
        return rows


def _row_to_utterance(
    source: SourceConfig,
    metadata_path: Path,
    row: dict[str, Any],
    row_index: int,
    split: str,
) -> Utterance | None:
    text = _first_string(row, TEXT_COLUMNS)
    audio_value = _first_value(row, PATH_COLUMNS)
    if text is None or audio_value is None:
        return None
    audio_path = _audio_value_to_path(audio_value, metadata_path.parent, source.root)
    if audio_path is None:
        return None
    raw_utt_id = (
        _first_string(row, ("id", "utt_id", "audio_id"))
        or f"{metadata_path.stem}_{row_index}"
    )
    return Utterance(
        utt_id=f"{source.name}_{split}_{raw_utt_id}",
        dataset=source.name,
        split=split,
        language=source.language,
        audio_path=str(audio_path),
        text=text,
        metadata=_source_metadata(source),
    )


def _first_string(row: dict[str, Any], columns: tuple[str, ...]) -> str | None:
    value = _first_value(row, columns)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_value(row: dict[str, Any], columns: tuple[str, ...]) -> Any | None:
    for column in columns:
        if column in row and row[column] not in {None, ""}:
            return row[column]
    return None


def _audio_value_to_path(value: Any, base_dir: Path, root: Path) -> Path | None:
    if isinstance(value, dict):
        nested = value.get("path") or value.get("array")
        if isinstance(nested, str):
            return _resolve_audio_path(base_dir, nested, root=root)
        return None
    if isinstance(value, str):
        return _resolve_audio_path(base_dir, value, root=root)
    return None


def _resolve_audio_path(base_dir: Path, audio_value: str, *, root: Path) -> Path | None:
    candidate = Path(audio_value)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                base_dir / candidate,
                base_dir / "clips" / candidate,
                root / candidate,
                root / "clips" / candidate,
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _parse_transcript_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if "\t" in stripped:
        utt_id, text = stripped.split("\t", maxsplit=1)
        return utt_id.strip(), text.strip()
    parts = stripped.split(maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _load_mls_splits(root: Path, include_splits: list[str]) -> dict[str, str]:
    split_by_id: dict[str, str] = {}
    split_names = include_splits or ["dev", "test", "train"]
    for split in split_names:
        for split_path in (root / "splits").glob(f"{split}*.txt"):
            with split_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    utt_id = line.strip()
                    if utt_id:
                        split_by_id[utt_id] = split_path.stem
    return split_by_id


def _build_audio_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for extension in AUDIO_EXTENSIONS:
        for path in root.rglob(f"*{extension}"):
            index.setdefault(path.stem, path)
    return index


def _build_mls_audio_index(root: Path, include_splits: list[str]) -> dict[str, Path]:
    if not include_splits:
        return _build_audio_index(root)
    index: dict[str, Path] = {}
    for split in include_splits:
        split_root = root / split
        if not split_root.is_dir():
            continue
        index.update(_build_audio_index(split_root))
    return index


def _infer_split(root: Path, path: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    split_markers = {
        "dev",
        "dev-clean",
        "dev-other",
        "test",
        "test-clean",
        "test-other",
        "validation",
        "valid",
        "train",
    }
    for part in parts:
        lowered = part.lower()
        if lowered in split_markers:
            return lowered
        if lowered.startswith(("dev-", "test-", "train-")):
            return lowered
    return path.stem


def _infer_locale(root: Path, path: Path, row: dict[str, str]) -> str:
    row_locale = row.get("locale")
    if row_locale:
        return row_locale
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    for part in parts:
        lowered = part.lower().replace("_", "-")
        if lowered in {"en", "en-us", "pt", "pt-br", "pt-pt"}:
            return lowered
    return "unknown"
