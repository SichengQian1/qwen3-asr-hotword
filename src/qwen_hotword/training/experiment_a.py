from __future__ import annotations

import csv
import hashlib
import heapq
import importlib
import json
import unicodedata
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text
from qwen_hotword.training.mfa_audit import load_mfa_dictionary, load_word_counts

ALLOWED_SINGLE_LETTER_WORDS = {"a", "e", "o"}


@dataclass(frozen=True)
class AudioMetadata:
    frames: int
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class CleanLabel:
    words: tuple[str, ...]
    phonemes: tuple[str, ...]
    phoneme_token_ids: tuple[int, ...]
    word_pronunciations: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class Candidate:
    score: int
    row_number: int
    audio_relative: str
    audio_path: str
    text: str
    label: CleanLabel


@dataclass(frozen=True)
class ExperimentASummary:
    tsv_path: str
    audio_root: str
    dictionary_path: str
    word_counts_path: str
    vocab_path: str
    language: str
    seed: int
    minimum_word_frequency: int
    maximum_ctc_target_ratio: float
    requested_samples: int
    selected_samples: int
    rows_scanned: int
    lexically_clean_rows: int
    candidate_pool_size: int
    selected_duration_seconds: float
    selected_duration_minutes: float
    minimum_duration_seconds: float
    maximum_duration_seconds: float
    mean_duration_seconds: float
    rejection_counts: dict[str, int]
    manifest_path: str
    review_path: str
    rejection_examples_path: str
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def ctc_minimum_input_length(token_ids: tuple[int, ...] | list[int]) -> int:
    """Return target length plus blanks required between repeated adjacent labels."""
    repeated_adjacent_labels = sum(
        previous == current
        for previous, current in zip(token_ids, token_ids[1:], strict=False)
    )
    return len(token_ids) + repeated_adjacent_labels


def estimate_qwen_lengths(
    metadata: AudioMetadata,
    *,
    target_sample_rate: int = 16_000,
    hop_length: int = 160,
) -> tuple[int, int]:
    if metadata.frames <= 0 or metadata.sample_rate <= 0:
        raise ValueError("audio metadata must contain positive frame and sample-rate values")
    resampled_frames = round(metadata.frames * target_sample_rate / metadata.sample_rate)
    feature_length = resampled_frames // hop_length
    return feature_length, qwen3_asr_audio_output_length(feature_length)


def read_audio_metadata(path: str | Path) -> AudioMetadata:
    audio_path = Path(path)
    try:
        soundfile = importlib.import_module("soundfile")
    except ImportError:
        soundfile = None

    if soundfile is not None:
        info = soundfile.info(str(audio_path))
        frames = int(info.frames)
        sample_rate = int(info.samplerate)
        return AudioMetadata(
            frames=frames,
            sample_rate=sample_rate,
            duration_seconds=frames / sample_rate,
        )

    with wave.open(str(audio_path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
    return AudioMetadata(
        frames=frames,
        sample_rate=sample_rate,
        duration_seconds=frames / sample_rate,
    )


def normalized_dictionary(path: str | Path) -> dict[str, tuple[str, ...]]:
    raw_dictionary = load_mfa_dictionary(path)
    normalized: dict[str, list[str]] = defaultdict(list)
    for word, pronunciations in raw_dictionary.items():
        normalized_word = normalize_training_text(word)
        normalized[normalized_word].extend(pronunciations)
    return {
        word: tuple(dict.fromkeys(pronunciations))
        for word, pronunciations in normalized.items()
    }


def resolve_clean_label(
    text: str,
    dictionary: dict[str, tuple[str, ...]],
    vocab: PhonemeVocab,
    *,
    word_counts: Counter[str] | None = None,
    minimum_word_frequency: int = 1,
) -> tuple[CleanLabel | None, str | None, str | None]:
    words = extract_word_tokens(text)
    if not words:
        return None, "empty_token_sequence", None
    if "h" in words:
        return None, "standalone_h", "h"

    unsupported_single_letter = next(
        (
            word
            for word in words
            if len(word) == 1 and word not in ALLOWED_SINGLE_LETTER_WORDS
        ),
        None,
    )
    if unsupported_single_letter is not None:
        return None, "unsupported_single_letter", unsupported_single_letter

    connector_word = next((word for word in words if "-" in word or "'" in word), None)
    if connector_word is not None:
        return None, "unresolved_connector", connector_word

    if word_counts is not None and minimum_word_frequency > 1:
        low_frequency_word = next(
            (word for word in words if word_counts[word] < minimum_word_frequency),
            None,
        )
        if low_frequency_word is not None:
            return (
                None,
                "low_frequency_word",
                f"{low_frequency_word}: count={word_counts[low_frequency_word]}",
            )

    sentence_phonemes: list[str] = []
    sentence_token_ids: list[int] = []
    word_pronunciations: list[dict[str, object]] = []
    for word in words:
        pronunciations = dictionary.get(word)
        if not pronunciations:
            return None, "dictionary_missing", word
        if len(pronunciations) != 1:
            return None, "ambiguous_pronunciation", word

        pronunciation = pronunciations[0]
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        if tokenized.oov_units:
            return None, "oov_phone", f"{word}: {' '.join(tokenized.oov_units)}"
        if not tokenized.tokens:
            return None, "empty_pronunciation", word

        phonemes = [unicodedata.normalize("NFC", token) for token in tokenized.tokens]
        sentence_phonemes.extend(phonemes)
        sentence_token_ids.extend(tokenized.token_ids)
        word_pronunciations.append(
            {
                "word": word,
                "mfa_pronunciation": pronunciation,
                "phonemes": phonemes,
                "phoneme_token_ids": tokenized.token_ids,
                "resolution": "exact",
            }
        )

    return (
        CleanLabel(
            words=tuple(words),
            phonemes=tuple(sentence_phonemes),
            phoneme_token_ids=tuple(sentence_token_ids),
            word_pronunciations=tuple(word_pronunciations),
        ),
        None,
        None,
    )


def build_experiment_a_manifest(
    tsv_path: str | Path,
    audio_root: str | Path,
    dictionary_path: str | Path,
    word_counts_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    num_samples: int = 128,
    seed: int = 20_260_716,
    language: str = "pt-BR",
    minimum_word_frequency: int = 100,
    audio_column: str = "audio",
    text_column: str = "text",
    minimum_duration_seconds: float = 0.5,
    maximum_duration_seconds: float = 15.0,
    ctc_safety_margin: int = 2,
    maximum_ctc_target_ratio: float = 0.75,
    candidate_pool_size: int = 4096,
    review_count: int = 20,
    max_rejection_examples: int = 200,
) -> ExperimentASummary:
    tsv = Path(tsv_path).expanduser()
    root = Path(audio_root).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    word_counts_file = Path(word_counts_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    for path in (tsv, dictionary_file, word_counts_file, vocab_file):
        if not path.is_file():
            raise FileNotFoundError(f"required file does not exist: {path}")
    if not root.is_dir():
        raise FileNotFoundError(f"audio root does not exist: {root}")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if candidate_pool_size < num_samples:
        raise ValueError("candidate_pool_size must be at least num_samples")
    if minimum_duration_seconds < 0 or maximum_duration_seconds <= minimum_duration_seconds:
        raise ValueError("invalid duration limits")
    if ctc_safety_margin < 0:
        raise ValueError("ctc_safety_margin must be non-negative")
    if minimum_word_frequency <= 0:
        raise ValueError("minimum_word_frequency must be positive")
    if not 0.0 < maximum_ctc_target_ratio <= 1.0:
        raise ValueError("maximum_ctc_target_ratio must be in (0, 1]")

    dictionary = normalized_dictionary(dictionary_file)
    word_counts = load_word_counts(word_counts_file)
    vocab = load_phoneme_vocab(vocab_file)
    if not vocab.tokens or vocab.tokens[0] != "<blank>":
        raise ValueError("CTC vocabulary must place <blank> at token ID 0")

    rows_scanned = 0
    lexically_clean_rows = 0
    rejection_counts: Counter[str] = Counter()
    rejection_examples: list[dict[str, object]] = []
    candidate_heap: list[tuple[int, int, Candidate]] = []

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

            relative_path = Path(audio_relative)
            audio_path = relative_path if relative_path.is_absolute() else root / relative_path
            score = _selection_score(seed, audio_relative, text)
            candidate = Candidate(
                score=score,
                row_number=row_number,
                audio_relative=audio_relative,
                audio_path=str(audio_path),
                text=text,
                label=label,
            )
            heap_entry = (-score, -row_number, candidate)
            if len(candidate_heap) < candidate_pool_size:
                heapq.heappush(candidate_heap, heap_entry)
            elif heap_entry > candidate_heap[0]:
                heapq.heapreplace(candidate_heap, heap_entry)

    ranked_candidates = sorted(
        (entry[2] for entry in candidate_heap),
        key=lambda candidate: (candidate.score, candidate.row_number),
    )
    selected_records: list[dict[str, object]] = []
    selected_audio_paths: set[str] = set()
    for candidate in ranked_candidates:
        if len(selected_records) >= num_samples:
            break
        audio_path = Path(candidate.audio_path)
        if not audio_path.is_file():
            reject(
                "missing_audio_file",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
            )
            continue
        if candidate.audio_path in selected_audio_paths:
            reject(
                "duplicate_audio",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
            )
            continue
        try:
            metadata = read_audio_metadata(audio_path)
        except (OSError, RuntimeError, ValueError, wave.Error) as error:
            reject(
                "invalid_audio",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
                f"{type(error).__name__}: {error}",
            )
            continue
        if metadata.duration_seconds < minimum_duration_seconds:
            reject(
                "audio_too_short",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
                f"duration={metadata.duration_seconds:.6f}",
            )
            continue
        if metadata.duration_seconds > maximum_duration_seconds:
            reject(
                "audio_too_long",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
                f"duration={metadata.duration_seconds:.6f}",
            )
            continue

        feature_length, ctc_input_length = estimate_qwen_lengths(metadata)
        minimum_ctc_length = ctc_minimum_input_length(candidate.label.phoneme_token_ids)
        if ctc_input_length < minimum_ctc_length + ctc_safety_margin:
            reject(
                "ctc_length_infeasible",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
                (
                    f"estimated_input={ctc_input_length}, minimum_target={minimum_ctc_length}, "
                    f"margin={ctc_safety_margin}"
                ),
            )
            continue
        ctc_target_ratio = minimum_ctc_length / ctc_input_length
        if ctc_target_ratio > maximum_ctc_target_ratio:
            reject(
                "ctc_alignment_too_tight",
                candidate.row_number,
                candidate.audio_relative,
                candidate.text,
                (
                    f"minimum_target={minimum_ctc_length}, input={ctc_input_length}, "
                    f"ratio={ctc_target_ratio:.6f}, maximum={maximum_ctc_target_ratio:.6f}"
                ),
            )
            continue

        selected_audio_paths.add(candidate.audio_path)
        selected_records.append(
            {
                "schema_version": 1,
                "experiment": "A",
                "split": "train",
                "id": f"noah_pt_row_{candidate.row_number}",
                "source_tsv": str(tsv),
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
                "ctc_target_ratio": round(ctc_target_ratio, 6),
                "words": list(candidate.label.words),
                "phonemes": list(candidate.label.phonemes),
                "phoneme_token_ids": list(candidate.label.phoneme_token_ids),
                "label_length": len(candidate.label.phoneme_token_ids),
                "word_pronunciations": list(candidate.label.word_pronunciations),
                "selection_score": candidate.score,
            }
        )

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "experiment_a_train.jsonl"
    review_path = destination / "experiment_a_review.txt"
    rejection_examples_path = destination / "rejection_examples.jsonl"
    _write_jsonl(manifest_path, selected_records)
    _write_jsonl(rejection_examples_path, rejection_examples)
    _write_review(review_path, selected_records[:review_count])

    durations = [cast(float, record["duration_seconds"]) for record in selected_records]
    total_duration = sum(durations)
    summary = ExperimentASummary(
        tsv_path=str(tsv),
        audio_root=str(root),
        dictionary_path=str(dictionary_file),
        word_counts_path=str(word_counts_file),
        vocab_path=str(vocab_file),
        language=language,
        seed=seed,
        minimum_word_frequency=minimum_word_frequency,
        maximum_ctc_target_ratio=maximum_ctc_target_ratio,
        requested_samples=num_samples,
        selected_samples=len(selected_records),
        rows_scanned=rows_scanned,
        lexically_clean_rows=lexically_clean_rows,
        candidate_pool_size=len(ranked_candidates),
        selected_duration_seconds=total_duration,
        selected_duration_minutes=total_duration / 60.0,
        minimum_duration_seconds=min(durations, default=0.0),
        maximum_duration_seconds=max(durations, default=0.0),
        mean_duration_seconds=(total_duration / len(durations) if durations else 0.0),
        rejection_counts=dict(sorted(rejection_counts.items())),
        manifest_path=str(manifest_path),
        review_path=str(review_path),
        rejection_examples_path=str(rejection_examples_path),
        status="pass" if len(selected_records) == num_samples else "fail",
    )
    (destination / "summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _selection_score(seed: int, audio_relative: str, text: str) -> int:
    payload = f"{seed}\0{audio_relative}\0{text}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_review(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            handle.write(f"[{index}] {row['id']}\n")
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
