from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import unicodedata
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.experiment_a import (
    ctc_minimum_input_length,
    estimate_qwen_lengths,
    normalized_dictionary,
    read_audio_metadata,
)
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text


@dataclass(frozen=True)
class SourceRow:
    row_number: int
    audio_relative: str
    text: str


@dataclass(frozen=True)
class LabelAssembly:
    words: tuple[str, ...]
    phonemes: tuple[str, ...]
    phoneme_token_ids: tuple[int, ...]
    word_pronunciations: tuple[dict[str, object], ...]
    issues: tuple[dict[str, str | None], ...]


@dataclass(frozen=True)
class FullManifestSummary:
    tsv_path: str
    audio_root: str
    dictionary_path: str
    vocab_path: str
    language: str
    shard_size: int
    workers: int
    source_records: int
    ready_records: int
    review_records: int
    valid_audio_records: int
    total_audio_hours: float
    ready_audio_hours: float
    issue_counts: dict[str, int]
    completed_shards: int
    resumed_shards: int
    ready_manifest_path: str
    review_manifest_path: str
    shard_index_path: str
    elapsed_seconds: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assemble_full_label(
    text: str,
    dictionary: dict[str, tuple[str, ...]],
    vocab: PhonemeVocab,
) -> LabelAssembly:
    words = extract_word_tokens(text)
    issues: list[dict[str, str | None]] = []
    if not words:
        issues.append({"reason": "empty_token_sequence", "detail": None})

    sentence_phonemes: list[str] = []
    sentence_token_ids: list[int] = []
    word_pronunciations: list[dict[str, object]] = []
    for word in words:
        if word == "h":
            issues.append({"reason": "standalone_h", "detail": word})
        if "-" in word or "'" in word:
            issues.append({"reason": "unresolved_connector", "detail": word})

        pronunciations = dictionary.get(word)
        if not pronunciations:
            issues.append({"reason": "dictionary_missing", "detail": word})
            word_pronunciations.append({"word": word, "resolution": "missing"})
            continue
        if len(pronunciations) != 1:
            issues.append({"reason": "ambiguous_pronunciation", "detail": word})
            word_pronunciations.append(
                {
                    "word": word,
                    "resolution": "ambiguous",
                    "mfa_pronunciation_candidates": list(pronunciations),
                }
            )
            continue

        pronunciation = pronunciations[0]
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        phonemes = [unicodedata.normalize("NFC", token) for token in tokenized.tokens]
        oov_units = [unicodedata.normalize("NFC", unit) for unit in tokenized.oov_units]
        if oov_units:
            issues.append(
                {
                    "reason": "oov_phone",
                    "detail": f"{word}: {' '.join(oov_units)}",
                }
            )
        if not pronunciation.strip():
            issues.append({"reason": "empty_pronunciation", "detail": word})

        sentence_phonemes.extend(phonemes)
        sentence_token_ids.extend(tokenized.token_ids)
        word_pronunciations.append(
            {
                "word": word,
                "mfa_pronunciation": pronunciation,
                "phonemes": phonemes,
                "phoneme_token_ids": tokenized.token_ids,
                "oov_units": oov_units,
                "resolution": "exact" if not oov_units else "phone_oov",
            }
        )

    return LabelAssembly(
        words=tuple(words),
        phonemes=tuple(sentence_phonemes),
        phoneme_token_ids=tuple(sentence_token_ids),
        word_pronunciations=tuple(word_pronunciations),
        issues=tuple(_deduplicate_issues(issues)),
    )


def build_full_training_manifest(
    tsv_path: str | Path,
    audio_root: str | Path,
    dictionary_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    language: str = "pt-BR",
    audio_column: str = "audio",
    text_column: str = "text",
    shard_size: int = 5_000,
    workers: int = 16,
    resume: bool = True,
) -> FullManifestSummary:
    tsv = Path(tsv_path).expanduser()
    root = Path(audio_root).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (tsv, dictionary_file, vocab_file):
        if not path.is_file():
            raise FileNotFoundError(f"required file does not exist: {path}")
    if not root.is_dir():
        raise FileNotFoundError(f"audio root does not exist: {root}")
    if shard_size <= 0 or workers <= 0:
        raise ValueError("shard_size and workers must be positive")

    dictionary = normalized_dictionary(dictionary_file)
    vocab = load_phoneme_vocab(vocab_file)
    if not vocab.tokens or vocab.tokens[0] != "<blank>":
        raise ValueError("CTC vocabulary must place <blank> at token ID 0")

    destination.mkdir(parents=True, exist_ok=True)
    shard_dir = destination / "shards"
    report_dir = destination / "shard_reports"
    shard_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    build_config_path = destination / "build_config.json"
    build_config = _build_config(
        tsv,
        root,
        dictionary_file,
        vocab_file,
        language=language,
        audio_column=audio_column,
        text_column=text_column,
        shard_size=shard_size,
    )
    if build_config_path.exists():
        existing_config = json.loads(build_config_path.read_text(encoding="utf-8"))
        if existing_config != build_config:
            raise ValueError(
                "output directory contains shards from a different build configuration"
            )
        if not resume:
            raise ValueError("output directory already contains a resumable build")
    else:
        if any(shard_dir.iterdir()) or any(report_dir.iterdir()):
            raise ValueError("output directory has shards but no build_config.json")
        _write_json(build_config_path, build_config)

    started = time.monotonic()
    reports: list[dict[str, Any]] = []
    resumed_shards = 0
    rows: list[SourceRow] = []
    shard_index = 0

    def finish_shard(batch: list[SourceRow], index: int) -> None:
        nonlocal resumed_shards
        ready_path, review_path, report_path = _shard_paths(
            shard_dir,
            report_dir,
            index,
        )
        if resume and ready_path.is_file() and review_path.is_file() and report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("source_records") != len(batch):
                raise ValueError(f"resumed shard {index} has a different source row count")
            reports.append(report)
            resumed_shards += 1
            print(
                f"resumed shard={index:05d} rows={len(batch)} "
                f"ready={report['ready_records']} review={report['review_records']}",
                flush=True,
            )
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(
                executor.map(
                    lambda source: _process_source_row(
                        source,
                        root=root,
                        source_tsv=tsv,
                        language=language,
                        dictionary=dictionary,
                        vocab=vocab,
                    ),
                    batch,
                )
            )
        ready_records = [record for record in records if record["training_ready"]]
        review_records = [record for record in records if not record["training_ready"]]
        _write_jsonl(ready_path, ready_records)
        _write_jsonl(review_path, review_records)
        issue_counts: Counter[str] = Counter()
        for record in review_records:
            issue_counts.update(issue["reason"] for issue in record["issues"])
        report = {
            "shard_index": index,
            "source_records": len(records),
            "ready_records": len(ready_records),
            "review_records": len(review_records),
            "valid_audio_records": sum(record["audio_valid"] for record in records),
            "total_audio_seconds": sum(
                float(record.get("duration_seconds") or 0.0) for record in records
            ),
            "ready_audio_seconds": sum(
                float(record.get("duration_seconds") or 0.0) for record in ready_records
            ),
            "issue_counts": dict(sorted(issue_counts.items())),
            "first_row_number": batch[0].row_number,
            "last_row_number": batch[-1].row_number,
        }
        _write_json(report_path, report)
        reports.append(report)
        print(
            f"completed shard={index:05d} rows={len(records)} "
            f"ready={len(ready_records)} review={len(review_records)}",
            flush=True,
        )

    with tsv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        missing_columns = {
            column for column in (audio_column, text_column) if column not in fieldnames
        }
        if missing_columns:
            raise ValueError(f"TSV is missing required columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(reader, start=2):
            rows.append(
                SourceRow(
                    row_number=row_number,
                    audio_relative=str(row.get(audio_column) or "").strip(),
                    text=str(row.get(text_column) or "").strip(),
                )
            )
            if len(rows) == shard_size:
                finish_shard(rows, shard_index)
                rows = []
                shard_index += 1
        if rows:
            finish_shard(rows, shard_index)

    ready_shards = sorted(shard_dir.glob("ready-*.jsonl"))
    review_shards = sorted(shard_dir.glob("review-*.jsonl"))
    if not reports:
        raise ValueError("TSV contains no data rows")
    if len(ready_shards) != len(reports) or len(review_shards) != len(reports):
        raise RuntimeError("completed shard count does not match shard reports")
    ready_manifest_path = destination / "train_ready.jsonl"
    review_manifest_path = destination / "needs_review.jsonl"
    _merge_shards(ready_shards, ready_manifest_path)
    _merge_shards(review_shards, review_manifest_path)

    issue_counts: Counter[str] = Counter()
    for report in reports:
        issue_counts.update(report["issue_counts"])
    source_records = sum(int(report["source_records"]) for report in reports)
    ready_records = sum(int(report["ready_records"]) for report in reports)
    review_records = sum(int(report["review_records"]) for report in reports)
    index_path = destination / "shard_index.json"
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "ready_shards": [str(path) for path in ready_shards],
            "review_shards": [str(path) for path in review_shards],
            "reports": [
                str(report_dir / f"shard-{index:05d}.json")
                for index in range(len(reports))
            ],
        },
    )
    summary = FullManifestSummary(
        tsv_path=str(tsv),
        audio_root=str(root),
        dictionary_path=str(dictionary_file),
        vocab_path=str(vocab_file),
        language=language,
        shard_size=shard_size,
        workers=workers,
        source_records=source_records,
        ready_records=ready_records,
        review_records=review_records,
        valid_audio_records=sum(int(report["valid_audio_records"]) for report in reports),
        total_audio_hours=(
            sum(float(report["total_audio_seconds"]) for report in reports) / 3600
        ),
        ready_audio_hours=(
            sum(float(report["ready_audio_seconds"]) for report in reports) / 3600
        ),
        issue_counts=dict(sorted(issue_counts.items())),
        completed_shards=len(reports),
        resumed_shards=resumed_shards,
        ready_manifest_path=str(ready_manifest_path),
        review_manifest_path=str(review_manifest_path),
        shard_index_path=str(index_path),
        elapsed_seconds=time.monotonic() - started,
        status="pass" if source_records == ready_records + review_records else "fail",
    )
    _write_json(destination / "summary.json", summary.to_dict())
    return summary


def _process_source_row(
    source: SourceRow,
    *,
    root: Path,
    source_tsv: Path,
    language: str,
    dictionary: dict[str, tuple[str, ...]],
    vocab: PhonemeVocab,
) -> dict[str, Any]:
    issues: list[dict[str, str | None]] = []
    if not source.audio_relative:
        issues.append({"reason": "empty_audio", "detail": None})
    if not source.text:
        issues.append({"reason": "empty_text", "detail": None})

    label = assemble_full_label(source.text, dictionary, vocab)
    issues.extend(label.issues)
    relative_path = Path(source.audio_relative)
    audio_path = relative_path if relative_path.is_absolute() else root / relative_path
    metadata = None
    if source.audio_relative:
        if not audio_path.is_file():
            issues.append({"reason": "missing_audio_file", "detail": str(audio_path)})
        else:
            try:
                metadata = read_audio_metadata(audio_path)
            except (OSError, RuntimeError, ValueError, wave.Error) as error:
                issues.append(
                    {
                        "reason": "invalid_audio",
                        "detail": f"{type(error).__name__}: {error}",
                    }
                )

    record: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "noah_pt_full_500h",
        "split": "unsplit",
        "id": f"noah_pt_row_{source.row_number}",
        "source_tsv": str(source_tsv),
        "row_number": source.row_number,
        "audio_relative": source.audio_relative,
        "audio_path": str(audio_path),
        "text": source.text,
        "normalized_text": normalize_training_text(source.text),
        "language": language,
        "words": list(label.words),
        "phonemes": list(label.phonemes),
        "phoneme_token_ids": list(label.phoneme_token_ids),
        "label_length": len(label.phoneme_token_ids),
        "word_pronunciations": list(label.word_pronunciations),
        "audio_valid": metadata is not None,
    }
    if metadata is not None:
        feature_length, ctc_input_length = estimate_qwen_lengths(metadata)
        record.update(
            {
                "duration_seconds": round(metadata.duration_seconds, 6),
                "audio_frames": metadata.frames,
                "sample_rate": metadata.sample_rate,
                "estimated_feature_length": feature_length,
                "estimated_ctc_input_length": ctc_input_length,
            }
        )
        if label.phoneme_token_ids:
            minimum_length = ctc_minimum_input_length(label.phoneme_token_ids)
            record["ctc_minimum_input_length"] = minimum_length
            record["ctc_target_ratio"] = (
                round(minimum_length / ctc_input_length, 6)
                if ctc_input_length > 0
                else None
            )
            if ctc_input_length < minimum_length:
                issues.append(
                    {
                        "reason": "ctc_length_infeasible",
                        "detail": (
                            f"estimated_input={ctc_input_length}, minimum_target={minimum_length}"
                        ),
                    }
                )
        else:
            issues.append({"reason": "empty_ctc_target", "detail": None})
    elif not label.phoneme_token_ids:
        issues.append({"reason": "empty_ctc_target", "detail": None})

    deduplicated_issues = _deduplicate_issues(issues)
    record["issues"] = deduplicated_issues
    record["training_ready"] = not deduplicated_issues
    record["label_status"] = "ready" if record["training_ready"] else "needs_review"
    record["split_hash"] = _stable_fraction(source.audio_relative, source.text)
    return record


def _stable_fraction(audio_relative: str, text: str) -> float:
    digest = hashlib.sha256(f"{audio_relative}\0{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _deduplicate_issues(
    issues: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    seen: set[tuple[str | None, str | None]] = set()
    result: list[dict[str, str | None]] = []
    for issue in issues:
        key = (issue.get("reason"), issue.get("detail"))
        if key not in seen:
            seen.add(key)
            result.append(issue)
    return result


def _build_config(
    tsv: Path,
    audio_root: Path,
    dictionary: Path,
    vocab: Path,
    **settings: object,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "tsv": _file_identity(tsv),
        "audio_root": str(audio_root.resolve()),
        "dictionary": _file_identity(dictionary),
        "vocab": _file_identity(vocab),
        **settings,
    }


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _shard_paths(
    shard_dir: Path,
    report_dir: Path,
    index: int,
) -> tuple[Path, Path, Path]:
    return (
        shard_dir / f"ready-{index:05d}.jsonl",
        shard_dir / f"review-{index:05d}.jsonl",
        report_dir / f"shard-{index:05d}.json",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _merge_shards(shards: list[Path], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for shard in shards:
            with shard.open("rb") as source:
                shutil.copyfileobj(source, output)
    temporary.replace(destination)
