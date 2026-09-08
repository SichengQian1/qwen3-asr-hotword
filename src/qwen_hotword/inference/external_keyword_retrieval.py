from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import (
    HotwordScoringConfig,
    decode_ctc_posterior,
    profile_anchor_guided_decoded_hotwords,
)
from qwen_hotword.inference.hotword_prompt import normalize_match_words, strict_phrase_match
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab, tokenize_ipa_to_vocab

FINAL_FILES = (
    "run_config.json",
    "dataset_manifest.jsonl",
    "dataset_audit.json",
    "keyword_audit.json",
    "retrieval_output.json",
    "retrieval_details.jsonl",
    "evaluation_summary.json",
    "failure_cases.jsonl",
    "README.md",
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    audio_dir: Path
    transcript_path: Path


@dataclass(frozen=True)
class ExternalAudioRecord:
    sample_id: str
    source: str
    audio_path: Path
    reference_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "audio_path": str(self.audio_path),
            "reference_text": self.reference_text,
            "language": "pt-BR",
        }


@dataclass(frozen=True)
class KeywordBundle:
    entries: tuple[HotwordEntry, ...]
    phonemes: Mapping[str, str]
    audit: Mapping[str, object]


RetrieveFunction = Callable[[Any, Any, PhonemeVocab], Mapping[str, object]]


def parse_source_spec(value: str) -> SourceSpec:
    """Parse NAME=AUDIO_DIR,TRANSCRIPTS without baking work-zone paths into code."""
    try:
        name, paths = value.split("=", 1)
        audio_dir, transcript_path = paths.split(",", 1)
    except ValueError as error:
        raise ValueError("source must use NAME=AUDIO_DIR,TRANSCRIPTS") from error
    if not name.strip() or not audio_dir.strip() or not transcript_path.strip():
        raise ValueError("source name, audio directory, and transcript path must be non-empty")
    return SourceSpec(
        name=name.strip(),
        audio_dir=Path(audio_dir).expanduser(),
        transcript_path=Path(transcript_path).expanduser(),
    )


def load_keyword_bias(
    path: str | Path,
    *,
    vocab: PhonemeVocab,
    keyword_set: str = "hard_k266",
) -> KeywordBundle:
    source_path = Path(path).expanduser()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("keyword bias root must be an object")
    sets = raw.get("keyword_sets")
    phonemes = raw.get("keyword_phonemes")
    if not isinstance(sets, dict) or not isinstance(phonemes, dict):
        raise ValueError("keyword bias must contain keyword_sets and keyword_phonemes objects")
    surfaces = sets.get(keyword_set)
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError(f"keyword set is absent or empty: {keyword_set}")
    if any(not isinstance(surface, str) or not surface.strip() for surface in surfaces):
        raise ValueError("keyword set contains an invalid surface")

    entries: list[HotwordEntry] = []
    normalized_seen: dict[tuple[str, ...], str] = {}
    oov_by_surface: dict[str, list[str]] = {}
    missing_phonemes: list[str] = []
    selected_phonemes: dict[str, str] = {}
    for index, unstripped_surface in enumerate(surfaces):
        surface = unstripped_surface.strip()
        key = normalize_match_words(surface)
        if not key:
            raise ValueError(f"keyword has no normalized words: {surface!r}")
        previous = normalized_seen.get(key)
        if previous is not None:
            raise ValueError(f"duplicate normalized keyword: {previous!r} / {surface!r}")
        normalized_seen[key] = surface
        pronunciation = phonemes.get(surface)
        if not isinstance(pronunciation, str) or not pronunciation.strip():
            missing_phonemes.append(surface)
            continue
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        if tokenized.oov_units:
            oov_by_surface[surface] = tokenized.oov_units
            continue
        if not tokenized.token_ids:
            raise ValueError(f"keyword has no in-vocabulary phonemes: {surface!r}")
        selected_phonemes[surface] = pronunciation.strip()
        entries.append(
            HotwordEntry(
                hotword_id=f"external_pt_{index:04d}",
                language="pt-BR",
                surface=surface,
                normalized=" ".join(key),
                words=key,
                pronunciation=pronunciation.strip(),
                phoneme_tokens=tuple(tokenized.tokens),
                token_ids=tuple(tokenized.token_ids),
                source="keyword_bias:mfa",
                validation_occurrences=1,
            )
        )
    if missing_phonemes or oov_by_surface:
        raise ValueError(
            "keyword phoneme audit failed: "
            f"missing={missing_phonemes[:5]}, oov={list(oov_by_surface.items())[:5]}"
        )
    extra = sorted(set(phonemes) - set(surfaces))
    audit: dict[str, object] = {
        "status": "pass",
        "prompt_type": raw.get("prompt_type"),
        "language": raw.get("language"),
        "phoneme_source": raw.get("phoneme_source"),
        "keyword_set": keyword_set,
        "keyword_count": len(entries),
        "unique_normalized_keywords": len(normalized_seen),
        "missing_phoneme_count": 0,
        "oov_keyword_count": 0,
        "keywords_below_four_phonemes": [
            entry.surface for entry in entries if len(entry.token_ids) < 4
        ],
        "extra_phoneme_mapping_count": len(extra),
        "extra_phoneme_mappings": extra,
        "vocabulary_size": len(vocab.tokens),
        "blank_id": 0,
    }
    return KeywordBundle(entries=tuple(entries), phonemes=selected_phonemes, audit=audit)


def build_external_dataset(
    sources: Sequence[SourceSpec],
) -> tuple[tuple[ExternalAudioRecord, ...], dict[str, object]]:
    if not sources:
        raise ValueError("at least one source is required")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("source names must be unique")

    records: list[ExternalAudioRecord] = []
    source_audits: list[dict[str, object]] = []
    global_ids: dict[str, str] = {}
    for source in sources:
        if not source.audio_dir.is_dir():
            raise FileNotFoundError(f"audio directory does not exist: {source.audio_dir}")
        if not source.transcript_path.is_file():
            raise FileNotFoundError(f"transcript file does not exist: {source.transcript_path}")
        transcripts = _read_transcripts(source.transcript_path)
        audio_by_id: dict[str, Path] = {}
        for path in sorted(source.audio_dir.rglob("*")):
            if not path.is_file() or path.suffix.casefold() != ".flac":
                continue
            sample_id = path.stem
            if sample_id in audio_by_id:
                raise ValueError(f"duplicate FLAC stem in {source.name}: {sample_id!r}")
            audio_by_id[sample_id] = path
        if not audio_by_id:
            raise ValueError(f"source contains no FLAC files: {source.audio_dir}")
        missing_transcripts = sorted(set(audio_by_id) - set(transcripts))
        missing_audio = sorted(set(transcripts) - set(audio_by_id))
        if missing_transcripts or missing_audio:
            raise ValueError(
                f"source {source.name} audio/transcript mismatch: "
                f"missing_transcripts={missing_transcripts[:5]}, missing_audio={missing_audio[:5]}"
            )
        for sample_id in sorted(audio_by_id):
            previous_source = global_ids.get(sample_id)
            if previous_source is not None:
                raise ValueError(
                    f"cross-source duplicate sample ID {sample_id!r}: "
                    f"{previous_source} / {source.name}"
                )
            global_ids[sample_id] = source.name
            records.append(
                ExternalAudioRecord(
                    sample_id=sample_id,
                    source=source.name,
                    audio_path=audio_by_id[sample_id],
                    reference_text=transcripts[sample_id],
                )
            )
        source_audits.append(
            {
                "name": source.name,
                "audio_dir": str(source.audio_dir),
                "transcript_path": str(source.transcript_path),
                "flac_files": len(audio_by_id),
                "transcripts": len(transcripts),
                "matched_records": len(audio_by_id),
                "missing_transcripts": 0,
                "missing_audio": 0,
            }
        )
    return tuple(records), {
        "status": "pass",
        "language": "pt-BR",
        "audio_format": "FLAC",
        "sample_count": len(records),
        "unique_sample_ids": len(global_ids),
        "cross_source_duplicate_sample_ids": 0,
        "sources": source_audits,
        "external_test_set_used": True,
        "transcripts_used_for_candidate_generation": False,
    }


def run_external_keyword_retrieval(
    *,
    model_path: str | Path,
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    keyword_bias_path: str | Path,
    sources: Sequence[SourceSpec],
    output_dir: str | Path,
    keyword_set: str = "hard_k266",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    audit_only: bool = False,
    resume: bool = False,
    print_progress: bool = True,
    retrieve_function: RetrieveFunction | None = None,
) -> dict[str, object]:
    model = Path(model_path).expanduser()
    checkpoint = Path(checkpoint_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    keyword_file = Path(keyword_bias_path).expanduser()
    destination = Path(output_dir).expanduser()
    required_files = (model / "config.json", checkpoint, vocab_file, keyword_file)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")

    vocab = load_phoneme_vocab(vocab_file)
    bundle = load_keyword_bias(keyword_file, vocab=vocab, keyword_set=keyword_set)
    records, dataset_audit = build_external_dataset(sources)
    config = _run_config(
        model=model,
        checkpoint=checkpoint,
        vocab_path=vocab_file,
        keyword_path=keyword_file,
        sources=sources,
        keyword_set=keyword_set,
        device=device,
        dtype=dtype,
    )
    _prepare_output(destination, config, resume=resume)
    _write_or_verify_json(destination / "dataset_audit.json", dataset_audit)
    _write_or_verify_json(destination / "keyword_audit.json", bundle.audit)
    _write_or_verify_jsonl(
        destination / "dataset_manifest.jsonl", [record.to_dict() for record in records]
    )
    if audit_only:
        readme = _readme(config, status="audit_pass", sample_count=len(records))
        _atomic_write_text(destination / "README.md", readme)
        _write_hash_manifest(destination)
        return {
            "status": "audit_pass",
            "sample_count": len(records),
            "keyword_count": len(bundle.entries),
            "output_dir": str(destination),
        }

    if retrieve_function is None:
        detector = _load_detector(
            model=model,
            checkpoint=checkpoint,
            vocab=vocab,
            hotwords=bundle.entries,
            device=device,
            dtype=dtype,
        )

        def retrieve_function(
            waveform: Any, active_detector: Any, active_vocab: PhonemeVocab
        ) -> Mapping[str, object]:
            return _retrieve_waveform(active_detector, waveform, active_vocab)

    else:
        detector = None

    shard_dir = destination / "sample_shards"
    shard_dir.mkdir(exist_ok=True)
    for index, record in enumerate(records, start=1):
        shard = _shard_path(shard_dir, record)
        if shard.is_file():
            _validate_shard(_read_json(shard), record)
            continue
        waveform, audio_timing = _load_flac(record.audio_path)
        result = dict(retrieve_function(waveform, detector, vocab))
        row = _build_result_row(
            record,
            bundle=bundle,
            retrieval=result,
            audio_timing=audio_timing,
        )
        _write_json(shard, row)
        if print_progress:
            print(f"retrieved {index}/{len(records)}: {record.source}/{record.sample_id}")

    rows = [_read_json(_shard_path(shard_dir, record)) for record in records]
    for row, record in zip(rows, records, strict=True):
        _validate_shard(row, record)
    output, details, summary, failures = summarize_retrieval(rows, bundle=bundle)
    _write_json(destination / "retrieval_output.json", output)
    _write_jsonl(destination / "retrieval_details.jsonl", details)
    _write_json(destination / "evaluation_summary.json", summary)
    _write_jsonl(destination / "failure_cases.jsonl", failures)
    _atomic_write_text(
        destination / "README.md",
        _readme(config, status="pass", sample_count=len(records)),
    )
    _write_hash_manifest(destination)
    return summary


def summarize_retrieval(
    rows: Sequence[Mapping[str, Any]],
    *,
    bundle: KeywordBundle,
) -> tuple[
    dict[str, object],
    list[Mapping[str, object]],
    dict[str, object],
    list[Mapping[str, object]],
]:
    output: dict[str, object] = {}
    failures: list[Mapping[str, object]] = []
    for row in rows:
        selected = _list_of_mappings(row, "selected_matches")
        output[str(row["sample_id"])] = [
            {
                "word": str(item["surface"]),
                "phoneme": bundle.phonemes[str(item["surface"])],
            }
            for item in selected
        ]
        if (
            row.get("raw_missed_expected_hotword_ids")
            or row.get("final_missed_expected_hotword_ids")
            or row.get("wrong_selected_hotword_ids")
        ):
            failures.append(row)

    by_source: dict[str, dict[str, object]] = {}
    for source in sorted({str(row["source"]) for row in rows}):
        by_source[source] = _metric_summary([row for row in rows if row["source"] == source])
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "evaluation_scope": "complete_audio_to_ctc_anchor_retrieval_no_qwen_decoder",
        "external_test_set_used": True,
        "parameter_tuning_permitted": False,
        "transcripts_used_for_candidate_generation": False,
        "gate": {
            "threshold": 0.75,
            "top_k": 5,
            "maximum_edit_ratio": 0.35,
            "posterior_weight": 0.25,
            "minimum_posterior_confidence": 0.5,
            "minimum_top1_margin": 0.0,
            "minimum_phonemes": 1,
        },
        "retrieval_backend": {
            "name": "anchor_guided",
            "shortlist_size": 64,
            "anchor_ngram_sizes": [2, 3, 4],
            "anchors_per_entry": 24,
            "anchor_offset_tolerance": 1,
            "rerank_start_radius": 2,
        },
        "overall": _metric_summary(rows),
        "by_source": by_source,
    }
    return output, list(rows), summary, failures


def _build_result_row(
    record: ExternalAudioRecord,
    *,
    bundle: KeywordBundle,
    retrieval: Mapping[str, object],
    audio_timing: Mapping[str, object],
) -> dict[str, object]:
    raw = _list_of_mappings(retrieval, "raw_ranked_matches")
    selected = _list_of_mappings(retrieval, "selected_matches")
    expected_ids = tuple(
        entry.hotword_id
        for entry in bundle.entries
        if strict_phrase_match(record.reference_text, entry.surface)
    )
    raw_ids = tuple(str(item["hotword_id"]) for item in raw[:5])
    selected_ids = tuple(str(item["hotword_id"]) for item in selected)
    expected = set(expected_ids)
    timing = dict(_mapping(retrieval, "timing"))
    load_seconds = _required_number(audio_timing, "audio_load_seconds")
    model_seconds = _required_number(timing, "model_retrieval_seconds")
    timing["audio_load_seconds"] = load_seconds
    timing["audio_to_result_seconds"] = load_seconds + model_seconds
    return {
        "sample_id": record.sample_id,
        "source": record.source,
        "audio_path": str(record.audio_path),
        "reference_text": record.reference_text,
        "language": "pt-BR",
        "audio": dict(audio_timing),
        "expected_hotword_ids": list(expected_ids),
        "expected_hotwords": [
            entry.surface for entry in bundle.entries if entry.hotword_id in expected
        ],
        "raw_top5_hotword_ids": list(raw_ids),
        "raw_hit_expected_hotword_ids": sorted(expected.intersection(raw_ids)),
        "raw_missed_expected_hotword_ids": sorted(expected.difference(raw_ids)),
        "selected_matches": selected,
        "selected_hotword_ids": list(selected_ids),
        "final_hit_expected_hotword_ids": sorted(expected.intersection(selected_ids)),
        "final_missed_expected_hotword_ids": sorted(expected.difference(selected_ids)),
        "wrong_selected_hotword_ids": sorted(set(selected_ids).difference(expected)),
        "raw_ranked_matches": raw,
        "decoded_token_ids": retrieval.get("decoded_token_ids"),
        "decoded_tokens": retrieval.get("decoded_tokens"),
        "decoded_confidences": retrieval.get("decoded_confidences"),
        "suppressed_reason": retrieval.get("suppressed_reason"),
        "shortlist_candidates": retrieval.get("shortlist_candidates"),
        "postings_visited": retrieval.get("postings_visited"),
        "timing": timing,
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    expected = sum(len(_string_list(row, "expected_hotword_ids")) for row in rows)
    raw_hits = sum(len(_string_list(row, "raw_hit_expected_hotword_ids")) for row in rows)
    final_hits = sum(len(_string_list(row, "final_hit_expected_hotword_ids")) for row in rows)
    selected = sum(len(_string_list(row, "selected_hotword_ids")) for row in rows)
    negative_rows = [row for row in rows if not _string_list(row, "expected_hotword_ids")]
    negative_false_positives = sum(
        bool(_string_list(row, "selected_hotword_ids")) for row in negative_rows
    )
    timing_names = (
        "audio_load_seconds",
        "ctc_processor_seconds",
        "ctc_encoder_seconds",
        "ctc_head_seconds",
        "ctc_decode_seconds",
        "anchor_query_seconds",
        "hotword_matching_seconds",
        "hotword_sorting_seconds",
        "hotword_selection_seconds",
        "pure_retrieval_seconds",
        "model_retrieval_seconds",
        "audio_to_result_seconds",
    )
    timing = {
        name: _distribution([_required_number(_mapping(row, "timing"), name) for row in rows])
        for name in timing_names
    }
    pure_values = [
        _required_number(_mapping(row, "timing"), "pure_retrieval_seconds") for row in rows
    ]
    duration = sum(_required_number(_mapping(row, "audio"), "duration_seconds") for row in rows)
    total_wall = sum(
        _required_number(_mapping(row, "timing"), "audio_to_result_seconds") for row in rows
    )
    return {
        "sample_count": len(rows),
        "expected_hotwords": expected,
        "raw_retrieved_expected_hotwords_at_5": raw_hits,
        "raw_recall_at_5": _ratio(raw_hits, expected),
        "selected_hotwords": selected,
        "selected_true_positive_hotwords": final_hits,
        "final_retrieval_recall": _ratio(final_hits, expected),
        "final_retrieval_precision": _ratio(final_hits, selected),
        "negative_sample_count": len(negative_rows),
        "negative_sample_false_positives": negative_false_positives,
        "negative_sample_false_positive_rate": _ratio(negative_false_positives, len(negative_rows)),
        "mean_selected_hotwords_per_sample": selected / len(rows) if rows else None,
        "audio_duration_seconds": duration,
        "audio_to_result_real_time_factor": _ratio_float(total_wall, duration),
        "latency_scope": {
            "pure_retrieval_seconds": (
                "CTC greedy decode + Anchor query + Top-64 shortlist rerank/gate; "
                "excludes processor, encoder, and CTC Head"
            ),
            "model_retrieval_seconds": "processor + encoder + CTC Head + pure retrieval",
            "audio_to_result_seconds": "FLAC load/resample + model retrieval",
        },
        "latency_seconds": timing,
        "pure_retrieval_over_50ms_count": sum(value > 0.05 for value in pure_values),
        "pure_retrieval_over_50ms_rate": _ratio(
            sum(value > 0.05 for value in pure_values), len(pure_values)
        ),
    }


def _load_detector(
    *,
    model: Path,
    checkpoint: Path,
    vocab: PhonemeVocab,
    hotwords: tuple[HotwordEntry, ...],
    device: str,
    dtype: str,
) -> Any:
    from qwen_hotword.inference.streaming_backends import load_cumulative_ctc_detector

    config = HotwordScoringConfig(
        score_threshold=0.75,
        top_k=5,
        minimum_phonemes=1,
        maximum_edit_ratio=0.35,
        posterior_weight=0.25,
        minimum_posterior_confidence=0.5,
        minimum_top1_margin=0.0,
    )
    return load_cumulative_ctc_detector(
        model_path=model,
        checkpoint_path=checkpoint,
        vocab=vocab,
        hotwords=hotwords,
        language="Portuguese",
        device=device,
        dtype=dtype,
        scoring_config=config,
        retrieval_mode="operating",
        retrieval_backend="anchor_guided",
        anchor_shortlist_size=64,
        anchor_start_radius=2,
        anchor_ngram_sizes=(2, 3, 4),
        anchors_per_entry=24,
        anchor_offset_tolerance=1,
    )


def _retrieve_waveform(detector: Any, waveform: Any, vocab: PhonemeVocab) -> Mapping[str, object]:
    import torch

    from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post
    from qwen_hotword.training.ctc_overfit import build_audio_prompt

    if detector.anchor_index is None:
        raise RuntimeError("external retrieval requires an Anchor index")
    processor_started = time.perf_counter()
    prompt = build_audio_prompt(detector.wrapper.processor, "Portuguese")
    batch = detector.wrapper.processor(
        text=[prompt], audio=[waveform], return_tensors="pt", padding=True
    )
    processor_seconds = time.perf_counter() - processor_started
    features = batch["input_features"].to(
        device=detector.wrapper.model.device, dtype=detector.wrapper.model.dtype
    )
    mask = batch["feature_attention_mask"].to(device=detector.wrapper.model.device)
    _sync_cuda(torch, detector.device)
    encoder_started = time.perf_counter()
    encoder = extract_padded_ln_post(
        detector.wrapper.model.thinker.audio_tower, features, mask, no_grad=True
    )
    _sync_cuda(torch, detector.device)
    encoder_seconds = time.perf_counter() - encoder_started
    hidden = encoder.hidden_states.to(device=detector.device, dtype=torch.float32)
    lengths = encoder.input_lengths.to(device=detector.device)
    _sync_cuda(torch, detector.device)
    head_started = time.perf_counter()
    with torch.no_grad():
        logits = detector.head(hidden, input_lengths=lengths)
        effective = detector.head.output_lengths(lengths)
    _sync_cuda(torch, detector.device)
    head_seconds = time.perf_counter() - head_started

    retrieval_started = time.perf_counter()
    decode_started = time.perf_counter()
    effective_steps = int(effective[0].item())
    decoded = decode_ctc_posterior(logits[0], input_length=effective_steps, blank_id=0)
    decode_seconds = time.perf_counter() - decode_started
    ids = tuple(item.token_id for item in decoded)
    confidences = tuple(item.confidence for item in decoded)
    anchor_started = time.perf_counter()
    shortlist = detector.anchor_index.query(
        ids,
        confidences=confidences,
        maximum_candidates=detector.anchor_shortlist_size,
    )
    anchor_seconds = time.perf_counter() - anchor_started
    candidate_entries = tuple(
        detector.hotword_by_id[item.hotword_id] for item in shortlist.candidates
    )
    profiled = profile_anchor_guided_decoded_hotwords(
        decoded,
        effective_time_steps=effective_steps,
        hotwords=candidate_entries,
        start_hints={item.hotword_id: item.best_offset for item in shortlist.candidates},
        maximum_start_delta=detector.anchor_start_radius,
        config=detector.scoring_config,
    )
    pure_seconds = time.perf_counter() - retrieval_started
    model_seconds = processor_seconds + encoder_seconds + head_seconds + pure_seconds
    scored = profiled.result
    return {
        "decoded_token_ids": list(scored.decoded_token_ids),
        "decoded_tokens": [vocab.tokens[token_id] for token_id in scored.decoded_token_ids],
        "decoded_confidences": list(scored.decoded_confidences),
        "raw_ranked_matches": [match.to_dict() for match in scored.ranked_matches[:20]],
        "selected_matches": [match.to_dict() for match in scored.selected_matches],
        "suppressed_reason": scored.suppressed_reason,
        "shortlist_candidates": len(shortlist.candidates),
        "postings_visited": shortlist.postings_visited,
        "timing": {
            "ctc_processor_seconds": processor_seconds,
            "ctc_encoder_seconds": encoder_seconds,
            "ctc_head_seconds": head_seconds,
            "ctc_decode_seconds": decode_seconds,
            "anchor_query_seconds": anchor_seconds,
            "hotword_matching_seconds": profiled.matching_seconds,
            "hotword_sorting_seconds": profiled.sorting_seconds,
            "hotword_selection_seconds": profiled.selection_seconds,
            "pure_retrieval_seconds": pure_seconds,
            "model_retrieval_seconds": model_seconds,
        },
    }


def _load_flac(path: Path) -> tuple[Any, dict[str, object]]:
    try:
        librosa = importlib.import_module("librosa")
        np = importlib.import_module("numpy")
        sf = importlib.import_module("soundfile")
    except ImportError as error:
        raise RuntimeError("librosa, numpy, and soundfile are required for FLAC input") from error
    started = time.perf_counter()
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    channels = int(waveform.shape[1])
    waveform = waveform.mean(axis=1, dtype=np.float32)
    resampled = sample_rate != 16_000
    if resampled:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16_000)
    if waveform.size == 0 or not bool(np.isfinite(waveform).all()):
        raise ValueError(f"audio is empty or non-finite: {path}")
    seconds = time.perf_counter() - started
    return waveform, {
        "original_sample_rate": int(sample_rate),
        "target_sample_rate": 16_000,
        "channels": channels,
        "resampled": resampled,
        "samples": int(waveform.shape[0]),
        "duration_seconds": float(waveform.shape[0] / 16_000),
        "audio_load_seconds": seconds,
    }


def _read_transcripts(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "\t" in line:
                raw_id, text = line.split("\t", 1)
            elif "|" in line:
                raw_id, text = line.split("|", 1)
            else:
                pieces = line.split(maxsplit=1)
                if len(pieces) != 2:
                    raise ValueError(f"transcript row {line_number} has no text: {path}")
                raw_id, text = pieces
            sample_id = Path(raw_id.strip()).stem
            text = text.strip()
            if not sample_id or not text:
                raise ValueError(f"invalid transcript row {line_number}: {path}")
            if sample_id in rows:
                raise ValueError(f"duplicate transcript ID {sample_id!r}: {path}")
            rows[sample_id] = text
    if not rows:
        raise ValueError(f"transcript file is empty: {path}")
    return rows


def _run_config(
    *,
    model: Path,
    checkpoint: Path,
    vocab_path: Path,
    keyword_path: Path,
    sources: Sequence[SourceSpec],
    keyword_set: str,
    device: str,
    dtype: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "mode": "complete_audio_offline_retrieval",
        "model": {"path": str(model), "config": _identity(model / "config.json")},
        "ctc_checkpoint": _identity(checkpoint),
        "vocab": _identity(vocab_path),
        "keyword_bias": _identity(keyword_path),
        "keyword_set": keyword_set,
        "sources": [
            {
                "name": source.name,
                "audio_dir": str(source.audio_dir),
                "transcripts": _identity(source.transcript_path),
            }
            for source in sources
        ],
        "language": "Portuguese",
        "device": device,
        "dtype": dtype,
        "audio": {"format": "FLAC", "mono": True, "sample_rate": 16_000},
        "gate": {
            "threshold": 0.75,
            "top_k": 5,
            "minimum_phonemes": 1,
            "maximum_edit_ratio": 0.35,
            "posterior_weight": 0.25,
            "minimum_posterior_confidence": 0.5,
            "minimum_top1_margin": 0.0,
        },
        "retrieval": {
            "backend": "anchor_guided",
            "shortlist_size": 64,
            "anchor_ngram_sizes": [2, 3, 4],
            "anchors_per_entry": 24,
            "anchor_offset_tolerance": 1,
            "rerank_start_radius": 2,
            "saved_raw_rank_depth": 20,
        },
        "qwen_decoder_used": False,
        "transcripts_used_for_candidate_generation": False,
        "external_test_set_used": True,
        "parameter_tuning_permitted": False,
    }


def _prepare_output(destination: Path, config: Mapping[str, object], *, resume: bool) -> None:
    config_path = destination / "run_config.json"
    if destination.exists():
        if not resume:
            raise FileExistsError(f"output directory already exists: {destination}")
        if not config_path.is_file() or _read_json(config_path) != config:
            raise ValueError("resume run_config differs from the existing output")
        return
    destination.mkdir(parents=True)
    _write_json(config_path, config)


def _write_or_verify_json(path: Path, value: Mapping[str, object]) -> None:
    if path.is_file():
        if _read_json(path) != value:
            raise ValueError(f"resume artifact differs: {path}")
        return
    _write_json(path, value)


def _write_or_verify_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    expected = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"resume artifact differs: {path}")
        return
    _atomic_write_text(path, expected)


def _shard_path(root: Path, record: ExternalAudioRecord) -> Path:
    digest = hashlib.sha256(f"{record.source}\0{record.sample_id}".encode()).hexdigest()[:16]
    return root / f"{digest}.json"


def _validate_shard(row: Mapping[str, Any], record: ExternalAudioRecord) -> None:
    if row.get("sample_id") != record.sample_id or row.get("source") != record.source:
        raise ValueError(f"sample shard identity mismatch: {record.source}/{record.sample_id}")


def _readme(config: Mapping[str, object], *, status: str, sample_count: int) -> str:
    return (
        "# Portuguese external keyword retrieval\n\n"
        "Complete-audio offline CTC retrieval over two external FLAC test sources. "
        "The Qwen decoder is not run. All 266 keywords are active; Anchor search produces "
        "a Top-64 shortlist and the fixed 0.75 / posterior 0.5 / Top-5 gate produces the "
        "downstream keyword list. Transcripts are used only after retrieval for strict "
        "phrase-based evaluation.\n\n"
        f"Git commit: `{config['git_commit']}`\n\n"
        f"Status: `{status}`\n\n"
        f"Samples: `{sample_count}`\n"
    )


def _write_hash_manifest(destination: Path) -> None:
    lines = []
    for name in FINAL_FILES:
        path = destination / name
        if path.is_file():
            lines.append(f"{_sha256(path)}  {name}\n")
    _atomic_write_text(destination / "sha256.txt", "".join(lines))


def _write_json(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    midpoint = median(ordered)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "p50": midpoint,
        "median": midpoint,
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _ratio_float(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _mapping(row: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"field {key} must be an object")
    return value


def _list_of_mappings(row: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = row.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"field {key} must be a list of objects")
    return value


def _string_list(row: Mapping[str, Any], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"field {key} must be a list of strings")
    return value


def _required_number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"field {key} must be numeric")
    return float(value)


def _sync_cuda(torch: Any, device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
