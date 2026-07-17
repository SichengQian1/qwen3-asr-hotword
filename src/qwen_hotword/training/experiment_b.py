from __future__ import annotations

import csv
import hashlib
import heapq
import json
import wave
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.experiment_a import (
    Candidate,
    ctc_minimum_input_length,
    estimate_qwen_lengths,
    normalized_dictionary,
    read_audio_metadata,
    resolve_clean_label,
)
from qwen_hotword.training.g2p_prep import normalize_training_text
from qwen_hotword.training.mfa_audit import load_word_counts

SPLIT_NAMES = ("train", "validation", "test")
RejectCallback = Callable[[str, int, str, str, str | None], None]


@dataclass(frozen=True)
class SplitSummary:
    requested_duration_seconds: float
    selected_duration_seconds: float
    selected_duration_hours: float
    selected_samples: int
    manifest_path: str
    review_path: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentBSummary:
    tsv_path: str
    audio_root: str
    dictionary_path: str
    word_counts_path: str
    vocab_path: str
    language: str
    seed: int
    split_strategy: str
    speaker_disjoint: bool
    exclusion_manifest_paths: list[str]
    excluded_audio_paths: int
    minimum_word_frequency: int
    maximum_ctc_target_ratio: float
    rows_scanned: int
    lexically_clean_rows: int
    candidate_pool_size: int
    rejection_counts: dict[str, int]
    rejection_examples_path: str
    split_summaries: dict[str, dict[str, object]]
    cross_split_audio_overlaps: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_experiment_b_manifests(
    tsv_path: str | Path,
    audio_root: str | Path,
    dictionary_path: str | Path,
    word_counts_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    train_hours: float = 8.0,
    validation_hours: float = 1.0,
    test_hours: float = 1.0,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 20_260_717,
    language: str = "pt-BR",
    minimum_word_frequency: int = 100,
    audio_column: str = "audio",
    text_column: str = "text",
    minimum_duration_seconds: float = 0.5,
    maximum_duration_seconds: float = 15.0,
    ctc_safety_margin: int = 2,
    maximum_ctc_target_ratio: float = 0.75,
    candidate_pool_size: int = 32_768,
    review_count: int = 20,
    max_rejection_examples: int = 500,
    exclusion_manifest_paths: tuple[str | Path, ...] = (),
) -> ExperimentBSummary:
    tsv = Path(tsv_path).expanduser()
    root = Path(audio_root).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    word_counts_file = Path(word_counts_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    exclusion_files = [Path(path).expanduser() for path in exclusion_manifest_paths]
    for path in (tsv, dictionary_file, word_counts_file, vocab_file):
        if not path.is_file():
            raise FileNotFoundError(f"required file does not exist: {path}")
    if not root.is_dir():
        raise FileNotFoundError(f"audio root does not exist: {root}")
    excluded_audio_paths = _load_excluded_audio_paths(exclusion_files)

    target_hours = {
        "train": train_hours,
        "validation": validation_hours,
        "test": test_hours,
    }
    split_fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    _validate_settings(
        target_hours,
        split_fractions,
        minimum_word_frequency=minimum_word_frequency,
        minimum_duration_seconds=minimum_duration_seconds,
        maximum_duration_seconds=maximum_duration_seconds,
        ctc_safety_margin=ctc_safety_margin,
        maximum_ctc_target_ratio=maximum_ctc_target_ratio,
        candidate_pool_size=candidate_pool_size,
    )

    dictionary = normalized_dictionary(dictionary_file)
    word_counts = load_word_counts(word_counts_file)
    vocab = load_phoneme_vocab(vocab_file)
    if not vocab.tokens or vocab.tokens[0] != "<blank>":
        raise ValueError("CTC vocabulary must place <blank> at token ID 0")

    capacities = _split_capacities(candidate_pool_size, split_fractions)
    candidate_heaps: dict[str, list[tuple[int, int, Candidate]]] = {
        split: [] for split in SPLIT_NAMES
    }
    rows_scanned = 0
    lexically_clean_rows = 0
    rejection_counts: Counter[str] = Counter()
    rejection_examples: list[dict[str, object]] = []

    def reject(
        reason: str,
        row_number: int,
        audio_relative: str,
        text: str,
        detail: str | None = None,
    ) -> None:
        rejection_counts[reason] += 1
        if len(rejection_examples) < max_rejection_examples:
            rejection_examples.append(
                {
                    "row_number": row_number,
                    "audio_relative": audio_relative,
                    "text": text,
                    "reason": reason,
                    "detail": detail,
                }
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
            rows_scanned += 1
            audio_relative = str(row.get(audio_column) or "").strip()
            text = str(row.get(text_column) or "").strip()
            if not audio_relative:
                reject("empty_audio", row_number, audio_relative, text)
                continue
            if not text:
                reject("empty_text", row_number, audio_relative, text)
                continue
            label, reason, detail = resolve_clean_label(
                text,
                dictionary,
                vocab,
                word_counts=word_counts,
                minimum_word_frequency=minimum_word_frequency,
            )
            if label is None:
                reject(reason or "unknown_label_failure", row_number, audio_relative, text, detail)
                continue
            lexically_clean_rows += 1

            split = assign_split(audio_relative, seed=seed, fractions=split_fractions)
            relative_path = Path(audio_relative)
            audio_path = relative_path if relative_path.is_absolute() else root / relative_path
            if str(audio_path) in excluded_audio_paths:
                reject("excluded_audio", row_number, audio_relative, text)
                continue
            candidate = Candidate(
                score=_selection_score(seed, split, audio_relative, text),
                row_number=row_number,
                audio_relative=audio_relative,
                audio_path=str(audio_path),
                text=text,
                label=label,
            )
            heap = candidate_heaps[split]
            entry = (-candidate.score, -row_number, candidate)
            if len(heap) < capacities[split]:
                heapq.heappush(heap, entry)
            elif entry > heap[0]:
                heapq.heapreplace(heap, entry)

    selected_by_split: dict[str, list[dict[str, object]]] = {}
    selected_audio_by_split: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        ranked = sorted(
            (entry[2] for entry in candidate_heaps[split]),
            key=lambda candidate: (candidate.score, candidate.row_number),
        )
        records: list[dict[str, object]] = []
        selected_audio: set[str] = set()
        selected_seconds = 0.0
        target_seconds = target_hours[split] * 3600.0
        for candidate in ranked:
            if selected_seconds >= target_seconds:
                break
            record = _candidate_to_record(
                candidate,
                split=split,
                source_tsv=tsv,
                language=language,
                minimum_duration_seconds=minimum_duration_seconds,
                maximum_duration_seconds=maximum_duration_seconds,
                ctc_safety_margin=ctc_safety_margin,
                maximum_ctc_target_ratio=maximum_ctc_target_ratio,
                selected_audio=selected_audio,
                reject=reject,
            )
            if record is None:
                continue
            selected_audio.add(candidate.audio_path)
            records.append(record)
            selected_seconds += cast(float, record["duration_seconds"])
        selected_by_split[split] = records
        selected_audio_by_split[split] = selected_audio

    destination.mkdir(parents=True, exist_ok=True)
    rejection_examples_path = destination / "rejection_examples.jsonl"
    _write_jsonl(rejection_examples_path, rejection_examples)
    split_summaries: dict[str, SplitSummary] = {}
    for split in SPLIT_NAMES:
        manifest_path = destination / f"experiment_b_{split}.jsonl"
        review_path = destination / f"experiment_b_{split}_review.txt"
        records = selected_by_split[split]
        _write_jsonl(manifest_path, records)
        _write_review(review_path, records[:review_count])
        selected_seconds = sum(cast(float, row["duration_seconds"]) for row in records)
        target_seconds = target_hours[split] * 3600.0
        split_summaries[split] = SplitSummary(
            requested_duration_seconds=target_seconds,
            selected_duration_seconds=selected_seconds,
            selected_duration_hours=selected_seconds / 3600.0,
            selected_samples=len(records),
            manifest_path=str(manifest_path),
            review_path=str(review_path),
            status="pass" if selected_seconds >= target_seconds else "fail",
        )

    overlaps = _count_cross_split_overlaps(selected_audio_by_split)
    status = "pass" if all(
        summary.status == "pass" for summary in split_summaries.values()
    ) and overlaps == 0 else "fail"
    summary = ExperimentBSummary(
        tsv_path=str(tsv),
        audio_root=str(root),
        dictionary_path=str(dictionary_file),
        word_counts_path=str(word_counts_file),
        vocab_path=str(vocab_file),
        language=language,
        seed=seed,
        split_strategy=(
            "stable_audio_path_hash_"
            f"{train_fraction:g}_{validation_fraction:g}_{test_fraction:g}"
        ),
        speaker_disjoint=False,
        exclusion_manifest_paths=[str(path) for path in exclusion_files],
        excluded_audio_paths=len(excluded_audio_paths),
        minimum_word_frequency=minimum_word_frequency,
        maximum_ctc_target_ratio=maximum_ctc_target_ratio,
        rows_scanned=rows_scanned,
        lexically_clean_rows=lexically_clean_rows,
        candidate_pool_size=sum(len(heap) for heap in candidate_heaps.values()),
        rejection_counts=dict(sorted(rejection_counts.items())),
        rejection_examples_path=str(rejection_examples_path),
        split_summaries={
            split: split_summary.to_dict()
            for split, split_summary in split_summaries.items()
        },
        cross_split_audio_overlaps=overlaps,
        status=status,
    )
    (destination / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def assign_split(
    audio_relative: str,
    *,
    seed: int,
    fractions: dict[str, float],
) -> str:
    digest = hashlib.blake2b(
        f"{seed}\0split\0{audio_relative}".encode(),
        digest_size=8,
    ).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    train_boundary = fractions["train"]
    validation_boundary = train_boundary + fractions["validation"]
    if fraction < train_boundary:
        return "train"
    if fraction < validation_boundary:
        return "validation"
    return "test"


def _candidate_to_record(
    candidate: Candidate,
    *,
    split: str,
    source_tsv: Path,
    language: str,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    ctc_safety_margin: int,
    maximum_ctc_target_ratio: float,
    selected_audio: set[str],
    reject: RejectCallback,
) -> dict[str, object] | None:
    def reject_row(reason: str, detail: str | None = None) -> None:
        reject(
            reason,
            candidate.row_number,
            candidate.audio_relative,
            candidate.text,
            detail,
        )

    audio_path = Path(candidate.audio_path)
    if not audio_path.is_file():
        reject_row("missing_audio_file")
        return None
    if candidate.audio_path in selected_audio:
        reject_row("duplicate_audio")
        return None
    try:
        metadata = read_audio_metadata(audio_path)
    except (OSError, RuntimeError, ValueError, wave.Error) as error:
        reject_row("invalid_audio", f"{type(error).__name__}: {error}")
        return None
    if metadata.duration_seconds < minimum_duration_seconds:
        reject_row("audio_too_short", f"duration={metadata.duration_seconds:.6f}")
        return None
    if metadata.duration_seconds > maximum_duration_seconds:
        reject_row("audio_too_long", f"duration={metadata.duration_seconds:.6f}")
        return None

    feature_length, ctc_input_length = estimate_qwen_lengths(metadata)
    minimum_ctc_length = ctc_minimum_input_length(candidate.label.phoneme_token_ids)
    if ctc_input_length < minimum_ctc_length + ctc_safety_margin:
        reject_row(
            "ctc_length_infeasible",
            f"estimated_input={ctc_input_length}, minimum_target={minimum_ctc_length}",
        )
        return None
    target_ratio = minimum_ctc_length / ctc_input_length
    if target_ratio > maximum_ctc_target_ratio:
        reject_row(
            "ctc_alignment_too_tight",
            f"ratio={target_ratio:.6f}, maximum={maximum_ctc_target_ratio:.6f}",
        )
        return None

    return {
        "schema_version": 1,
        "experiment": "B",
        "split": split,
        "id": f"noah_pt_row_{candidate.row_number}",
        "source_tsv": str(source_tsv),
        "row_number": candidate.row_number,
        "audio_relative": candidate.audio_relative,
        "audio_path": candidate.audio_path,
        "text": candidate.text,
        "normalized_text": normalize_training_text(candidate.text),
        "language": language,
        "duration_seconds": round(metadata.duration_seconds, 6),
        "audio_frames": metadata.frames,
        "sample_rate": metadata.sample_rate,
        "estimated_feature_length": feature_length,
        "estimated_ctc_input_length": ctc_input_length,
        "ctc_minimum_input_length": minimum_ctc_length,
        "ctc_safety_margin": ctc_safety_margin,
        "ctc_target_ratio": round(target_ratio, 6),
        "words": list(candidate.label.words),
        "phonemes": list(candidate.label.phonemes),
        "phoneme_token_ids": list(candidate.label.phoneme_token_ids),
        "label_length": len(candidate.label.phoneme_token_ids),
        "word_pronunciations": list(candidate.label.word_pronunciations),
        "selection_score": candidate.score,
    }


def _validate_settings(
    target_hours: dict[str, float],
    fractions: dict[str, float],
    *,
    minimum_word_frequency: int,
    minimum_duration_seconds: float,
    maximum_duration_seconds: float,
    ctc_safety_margin: int,
    maximum_ctc_target_ratio: float,
    candidate_pool_size: int,
) -> None:
    if any(hours <= 0 for hours in target_hours.values()):
        raise ValueError("all split target hours must be positive")
    if any(fraction <= 0 for fraction in fractions.values()):
        raise ValueError("all split fractions must be positive")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")
    if minimum_word_frequency <= 0:
        raise ValueError("minimum_word_frequency must be positive")
    if minimum_duration_seconds < 0 or maximum_duration_seconds <= minimum_duration_seconds:
        raise ValueError("invalid duration limits")
    if ctc_safety_margin < 0:
        raise ValueError("ctc_safety_margin must be non-negative")
    if not 0 < maximum_ctc_target_ratio <= 1:
        raise ValueError("maximum_ctc_target_ratio must be in (0, 1]")
    if candidate_pool_size < 100:
        raise ValueError("candidate_pool_size must be at least 100")


def _split_capacities(
    candidate_pool_size: int,
    fractions: dict[str, float],
) -> dict[str, int]:
    capacities = {
        split: max(1, int(candidate_pool_size * fractions[split]))
        for split in SPLIT_NAMES
    }
    capacities["train"] += candidate_pool_size - sum(capacities.values())
    return capacities


def _selection_score(seed: int, split: str, audio_relative: str, text: str) -> int:
    payload = f"{seed}\0select\0{split}\0{audio_relative}\0{text}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _count_cross_split_overlaps(audio_by_split: dict[str, set[str]]) -> int:
    return sum(
        len(audio_by_split[left] & audio_by_split[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    )


def _load_excluded_audio_paths(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"exclusion manifest does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                if not isinstance(row, dict) or not isinstance(row.get("audio_path"), str):
                    raise ValueError(f"exclusion row {path}:{line_number} has no audio_path")
                excluded.add(row["audio_path"])
    return excluded


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_review(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(f"[{index}] {row['id']}\n")
            handle.write(f"split: {row['split']}\n")
            handle.write(f"audio: {row['audio_path']}\n")
            handle.write(f"duration_seconds: {row['duration_seconds']}\n")
            handle.write(f"text: {row['text']}\n")
            handle.write(f"words: {' | '.join(cast(list[str], row['words']))}\n")
            handle.write(f"phonemes: {' '.join(cast(list[str], row['phonemes']))}\n")
            handle.write(f"phoneme_token_ids: {row['phoneme_token_ids']}\n")
            handle.write(
                "ctc_lengths: "
                f"estimated_input={row['estimated_ctc_input_length']}, "
                f"minimum_target={row['ctc_minimum_input_length']}\n\n"
            )
