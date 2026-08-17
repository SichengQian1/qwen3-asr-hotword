from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.hotwords.scoring import HotwordScoringConfig
from qwen_hotword.hotwords.simulation import load_simulated_cases
from qwen_hotword.inference.hotword_prompt import (
    DEFAULT_PT_BR_PROMPT_TEMPLATE,
    normalize_match_words,
    strict_phrase_match,
)
from qwen_hotword.inference.prompt_smoke import load_validation_manifest
from qwen_hotword.inference.streaming_backends import (
    load_cumulative_ctc_detector,
    load_official_vllm_streaming_backend,
)
from qwen_hotword.inference.streaming_core import (
    HotwordTiming,
    StreamingSample,
    run_streaming_sample,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

OUTPUT_FILES = (
    "run_config.json",
    "sample_results.jsonl",
    "chunk_timeline.jsonl",
    "summary.json",
    "boundary_summary.json",
    "latency_summary.json",
    "failure_cases.jsonl",
    "README.md",
    "sha256.txt",
)


def run_streaming_rag_evaluation(
    *,
    model_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    hotword_table_path: str | Path,
    cases_path: str | Path,
    offline_rag_dir: str | Path,
    ctc_checkpoint_path: str | Path,
    output_dir: str | Path,
    groups: tuple[str, ...] = ("A", "B", "C", "D", "E"),
    max_samples: int = 0,
    boundary_manifest_path: str | Path | None = None,
    chunk_size_sec: float = 2.0,
    unfixed_chunk_num: int = 2,
    unfixed_token_num: int = 5,
    threshold: float = 0.86,
    top_k: int = 3,
    maximum_edit_ratio: float = 0.35,
    posterior_weight: float = 0.25,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.0,
    prompt_template: str = DEFAULT_PT_BR_PROMPT_TEMPLATE,
    language: str = "Portuguese",
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    gpu_memory_utilization: float = 0.70,
    max_new_tokens: int = 128,
    seed: int = 20_260_817,
    resume: bool = False,
    print_progress: bool = True,
) -> dict[str, object]:
    group_set = tuple(dict.fromkeys(value.upper() for value in groups))
    if not group_set or any(value not in {"A", "B", "C", "D", "E"} for value in group_set):
        raise ValueError("groups must be a non-empty subset of A,B,C,D,E")
    if max_samples < 0:
        raise ValueError("max_samples must not be negative")
    if (chunk_size_sec, unfixed_chunk_num, unfixed_token_num) != (2.0, 2, 5):
        raise ValueError("first baseline is sealed to 2.0 sec / 2 chunks / 5 tokenizer tokens")

    paths = {
        "model": Path(model_path).expanduser(),
        "validation": Path(validation_manifest_path).expanduser(),
        "vocab": Path(vocab_path).expanduser(),
        "hotwords": Path(hotword_table_path).expanduser(),
        "cases": Path(cases_path).expanduser(),
        "offline": Path(offline_rag_dir).expanduser(),
        "checkpoint": Path(ctc_checkpoint_path).expanduser(),
        "output": Path(output_dir).expanduser(),
    }
    for name in ("validation", "vocab", "hotwords", "cases", "checkpoint"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"{name} input does not exist: {paths[name]}")
    if not paths["model"].is_dir() or not paths["offline"].is_dir():
        raise FileNotFoundError("model and offline RAG paths must be directories")
    boundary_path = Path(boundary_manifest_path).expanduser() if boundary_manifest_path else None
    if boundary_path is not None and not boundary_path.is_file():
        raise FileNotFoundError(f"boundary manifest does not exist: {boundary_path}")

    config = _build_config(
        paths=paths,
        boundary_path=boundary_path,
        groups=group_set,
        max_samples=max_samples,
        chunk_size_sec=chunk_size_sec,
        unfixed_chunk_num=unfixed_chunk_num,
        unfixed_token_num=unfixed_token_num,
        threshold=threshold,
        top_k=top_k,
        maximum_edit_ratio=maximum_edit_ratio,
        posterior_weight=posterior_weight,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
        prompt_template=prompt_template,
        language=language,
        dtype=dtype,
        device=device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    destination = paths["output"]
    shard_dir = destination / "sample_shards"
    _prepare_output(destination, config, resume=resume)
    shard_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_phoneme_vocab(paths["vocab"])
    hotwords = tuple(load_hotword_table(paths["hotwords"], vocab=vocab, blank_id=0))
    hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
    samples = (
        _load_boundary_samples(boundary_path, hotword_by_id)
        if boundary_path is not None
        else _load_offline_selection(
            paths["offline"],
            paths["validation"],
            paths["cases"],
            hotword_by_id,
        )
    )
    if max_samples:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("streaming evaluation selection is empty")

    results: list[dict[str, object]] = []
    timeline: list[dict[str, object]] = []
    for group in (value for value in group_set if value in {"A", "B"}):
        imported = _load_offline_predictions(paths["offline"], group, samples)
        for row in imported:
            _write_shard(shard_dir, group, str(row["case_id"]), row, ())
        results.extend(imported)

    streaming_groups = tuple(value for value in group_set if value in {"C", "D", "E"})
    missing_streaming_groups = {
        group
        for sample in samples
        for group in streaming_groups
        if not (resume and _shard_path(shard_dir, group, sample.case_id).is_file())
    }
    detector = None
    if "D" in missing_streaming_groups:
        scoring = HotwordScoringConfig(
            score_threshold=threshold,
            top_k=top_k,
            minimum_phonemes=4,
            maximum_edit_ratio=maximum_edit_ratio,
            posterior_weight=posterior_weight,
            minimum_posterior_confidence=minimum_posterior_confidence,
            minimum_top1_margin=minimum_top1_margin,
        )
        detector = load_cumulative_ctc_detector(
            model_path=paths["model"],
            checkpoint_path=paths["checkpoint"],
            vocab=vocab,
            hotwords=hotwords,
            language=language,
            device=device,
            dtype=dtype,
            scoring_config=scoring,
        )
    backend = None
    if missing_streaming_groups:
        backend = load_official_vllm_streaming_backend(
            paths["model"],
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=max_new_tokens,
        )
    for sample_index, sample in enumerate(samples, start=1):
        sample_missing_groups = {
            group
            for group in streaming_groups
            if not (resume and _shard_path(shard_dir, group, sample.case_id).is_file())
        }
        waveform = _load_waveform(sample) if sample_missing_groups else None
        for group in streaming_groups:
            shard = _shard_path(shard_dir, group, sample.case_id)
            if resume and shard.is_file():
                result, rows = _read_shard(shard)
            else:
                assert backend is not None and waveform is not None
                result, rows = run_streaming_sample(
                    backend=backend,
                    waveform=waveform,
                    sample=sample,
                    group=group,
                    hotword_surfaces={key: value.surface for key, value in hotword_by_id.items()},
                    ctc_detector=detector,
                    prompt_template=prompt_template,
                    chunk_size_sec=chunk_size_sec,
                    unfixed_chunk_num=unfixed_chunk_num,
                    unfixed_token_num=unfixed_token_num,
                )
                _write_shard(shard_dir, group, sample.case_id, result, rows)
            results.append(result)
            timeline.extend(rows)
            if print_progress:
                print(
                    f"streaming RAG sample={sample_index}/{len(samples)} "
                    f"group={group} case={sample.case_id}",
                    flush=True,
                )

    # Always re-read shards so resume and fresh runs have the same ordering.
    results, timeline = _collect_shards(shard_dir, group_set, samples)
    _apply_comparative_failure_classes(results)
    summary = _build_summary(results)
    boundary_summary = _build_boundary_summary(results)
    latency_summary = _build_latency_summary(results)
    failures = [row for row in results if row.get("failure_reason")]
    _write_jsonl(destination / "sample_results.jsonl", results)
    _write_jsonl(destination / "chunk_timeline.jsonl", timeline)
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "boundary_summary.json", boundary_summary)
    _write_json(destination / "latency_summary.json", latency_summary)
    _write_jsonl(destination / "failure_cases.jsonl", failures)
    (destination / "README.md").write_text(_report_readme(config, summary), encoding="utf-8")
    _write_hashes(destination)
    return summary


def _load_offline_selection(
    offline_dir: Path,
    validation_path: Path,
    cases_path: Path,
    hotword_by_id: Mapping[str, Any],
) -> tuple[StreamingSample, ...]:
    selection_path = offline_dir / "sample_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"offline sample selection does not exist: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    raw_samples = selection.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("offline sample selection has no samples list")
    records = load_validation_manifest(validation_path)
    cases = {case.case_id: case for case in load_simulated_cases(cases_path)}
    samples = []
    for raw in raw_samples:
        case_id = str(raw["case_id"])
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"offline selection references unknown case: {case_id}")
        record = records.get(case.sample_id)
        if record is None:
            raise ValueError(f"offline selection references unknown sample: {case.sample_id}")
        if raw.get("sample_id") != case.sample_id:
            raise ValueError(f"offline selection sample differs from case: {case_id}")
        expected = tuple(case.expected_hotword_ids)
        samples.append(
            StreamingSample(
                case_id=case_id,
                sample_id=case.sample_id,
                reference_text=record.reference_text,
                language=record.language,
                expected_hotword_ids=expected,
                expected_surfaces=tuple(hotword_by_id[item].surface for item in expected),
                active_hotword_ids=tuple(case.active_hotword_ids),
                audio_path=record.audio_path,
            )
        )
    return tuple(samples)


def _load_boundary_samples(
    path: Path,
    hotword_by_id: Mapping[str, Any],
) -> tuple[StreamingSample, ...]:
    rows = _read_jsonl(path)
    samples = []
    seen: set[str] = set()
    for line_number, raw in enumerate(rows, start=1):
        case_id = _required_string(raw, "case_id", line_number)
        if case_id in seen:
            raise ValueError(f"duplicate boundary case ID: {case_id}")
        seen.add(case_id)
        expected = _string_tuple(raw, "expected_hotword_ids", line_number, allow_empty=True)
        active = _string_tuple(raw, "active_hotword_ids", line_number)
        unknown = set(active) - hotword_by_id.keys()
        if unknown or not set(expected).issubset(active):
            raise ValueError(f"invalid boundary hotword IDs at row {line_number}: {unknown}")
        timing_rows = raw.get("hotword_timings", [])
        if not isinstance(timing_rows, list):
            raise ValueError(f"invalid hotword timings at row {line_number}")
        timings = tuple(
            HotwordTiming(
                hotword_id=_required_string(item, "hotword_id", line_number),
                start_sec=float(item["start_sec"]),
                end_sec=float(item["end_sec"]),
                timing_source=_required_string(item, "timing_source", line_number),
            )
            for item in timing_rows
        )
        if any(
            item.timing_source not in {"forced_alignment", "manual_confirmed"} for item in timings
        ):
            raise ValueError("boundary timings must be forced_alignment or manual_confirmed")
        if {item.hotword_id for item in timings} != set(expected):
            raise ValueError(f"boundary timings do not match expected IDs at row {line_number}")
        duration = raw.get("audio_duration_sec")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or duration <= 0:
            raise ValueError(f"boundary row {line_number} has invalid audio_duration_sec")
        if any(
            not 0 <= item.start_sec < item.end_sec <= float(duration) + 0.05 for item in timings
        ):
            raise ValueError(f"boundary timing is outside audio at row {line_number}")
        samples.append(
            StreamingSample(
                case_id=case_id,
                sample_id=_required_string(raw, "sample_id", line_number),
                reference_text=_required_string(raw, "reference_text", line_number),
                language=_required_string(raw, "language", line_number),
                expected_hotword_ids=expected,
                expected_surfaces=tuple(hotword_by_id[item].surface for item in expected),
                active_hotword_ids=active,
                timings=timings,
                boundary_bucket=_required_string(raw, "boundary_bucket", line_number),
                audio_path=_required_string(raw, "audio_path", line_number),
                leading_silence_sec=float(raw.get("leading_silence_sec", 0.0)),
            )
        )
    return tuple(samples)


def _load_waveform(sample: StreamingSample) -> Any:
    try:
        import librosa
        import numpy as np
    except ImportError as error:
        raise RuntimeError("librosa is required for real streaming evaluation") from error
    if not sample.audio_path:
        raise RuntimeError("streaming sample is missing audio_path")
    waveform, _ = librosa.load(sample.audio_path, sr=16_000, mono=True)
    if sample.leading_silence_sec < 0:
        raise ValueError("leading_silence_sec must not be negative")
    if sample.leading_silence_sec:
        silence = np.zeros(int(round(sample.leading_silence_sec * 16_000)), dtype=np.float32)
        waveform = np.concatenate((silence, waveform))
    return waveform


def _load_offline_predictions(
    offline_dir: Path,
    group: str,
    samples: Sequence[StreamingSample],
) -> list[dict[str, object]]:
    filename = "baseline_predictions.jsonl" if group == "A" else "retrieved_predictions.jsonl"
    lookup = {str(row["case_id"]): row for row in _read_jsonl(offline_dir / filename)}
    imported: list[dict[str, object]] = []
    for sample in samples:
        try:
            raw = lookup[sample.case_id]
        except KeyError as error:
            raise ValueError(f"offline {group} output misses case {sample.case_id}") from error
        if raw.get("sample_id") != sample.sample_id:
            raise ValueError(f"offline {group} sample differs for case {sample.case_id}")
        imported.append(
            {
                "case_id": sample.case_id,
                "sample_id": sample.sample_id,
                "experiment_group": group,
                "reference_text": sample.reference_text,
                "prediction": str(raw["prediction"]),
                "expected_hotword_ids": list(sample.expected_hotword_ids),
                "expected_hotwords": list(sample.expected_surfaces),
                "matched_expected_hotword_ids": [
                    hotword_id
                    for hotword_id, surface in zip(
                        sample.expected_hotword_ids,
                        sample.expected_surfaces,
                        strict=True,
                    )
                    if strict_phrase_match(str(raw["prediction"]), surface)
                ],
                "injected_hotword_ids": list(raw.get("injected_hotword_ids", [])),
                "injected_hotwords": list(raw.get("injected_hotwords", [])),
                "boundary_bucket": sample.boundary_bucket,
                "hotword_metrics": [],
                "partial_modification_count": None,
                "inference_seconds": float(raw.get("inference_seconds", 0.0)),
                "failure_reason": None,
                "source": f"imported_offline_{group}",
            }
        )
    return imported


def _build_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    groups: dict[str, dict[str, object]] = {}
    for group in ("A", "B", "C", "D", "E"):
        rows = [row for row in results if row["experiment_group"] == group]
        if not rows:
            continue
        expected_total = sum(len(row["expected_hotword_ids"]) for row in rows)
        matched_total = sum(len(row["matched_expected_hotword_ids"]) for row in rows)
        positive = [row for row in rows if row["expected_hotword_ids"]]
        word_errors = 0
        word_total = 0
        char_errors = 0
        char_total = 0
        for row in rows:
            ref_words = normalize_match_words(str(row["reference_text"]))
            hyp_words = normalize_match_words(str(row["prediction"]))
            word_errors += _edit_distance(ref_words, hyp_words)
            word_total += len(ref_words)
            ref_chars = tuple("".join(ref_words))
            hyp_chars = tuple("".join(hyp_words))
            char_errors += _edit_distance(ref_chars, hyp_chars)
            char_total += len(ref_chars)
        groups[group] = {
            "samples": len(rows),
            "positive_samples": len(positive),
            "hotword_exact_recall": _ratio(matched_total, expected_total),
            "sample_hotword_hit_rate": _ratio(
                sum(
                    set(row["expected_hotword_ids"]).issubset(row["matched_expected_hotword_ids"])
                    for row in positive
                ),
                len(positive),
            ),
            "wer": _ratio(word_errors, word_total),
            "cer": _ratio(char_errors, char_total),
            "mean_inference_seconds": mean(float(row["inference_seconds"]) for row in rows),
            "negative_wrong_hotword_injection_rate": _ratio(
                sum(
                    bool(row.get("injected_hotword_ids"))
                    for row in rows
                    if not row["expected_hotword_ids"]
                ),
                sum(not bool(row["expected_hotword_ids"]) for row in rows),
            ),
            "negative_hotword_hallucination_rate": _ratio(
                sum(
                    any(
                        strict_phrase_match(str(row["prediction"]), surface)
                        for surface in row.get("injected_hotwords", [])
                    )
                    for row in rows
                    if not row["expected_hotword_ids"]
                ),
                sum(not bool(row["expected_hotword_ids"]) for row in rows),
            ),
            "failure_counts": dict(
                Counter(row.get("failure_reason") for row in rows if row.get("failure_reason"))
            ),
        }
    offline_rag = groups.get("B", {}).get("hotword_exact_recall")
    streaming_rag = groups.get("D", {}).get("hotword_exact_recall")
    delta = None
    if isinstance(offline_rag, float) and isinstance(streaming_rag, float):
        delta = streaming_rag - offline_rag
    comparisons = {
        "streaming_baseline_minus_offline_baseline": _group_delta(groups, "C", "A"),
        "streaming_rag_minus_offline_rag": _group_delta(groups, "D", "B"),
        "streaming_oracle_minus_streaming_rag": _group_delta(groups, "E", "D"),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "groups": groups,
        "streaming_rag_minus_offline_rag_recall": delta,
        "comparisons": comparisons,
        "interpretation": _interpret(groups),
    }


def _group_delta(
    groups: Mapping[str, Mapping[str, object]],
    left: str,
    right: str,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in ("hotword_exact_recall", "sample_hotword_hit_rate", "wer", "cer"):
        left_value = groups.get(left, {}).get(metric)
        right_value = groups.get(right, {}).get(metric)
        result[metric] = (
            float(left_value) - float(right_value)
            if isinstance(left_value, (int, float))
            and not isinstance(left_value, bool)
            and isinstance(right_value, (int, float))
            and not isinstance(right_value, bool)
            else None
        )
    return result


def _apply_comparative_failure_classes(results: list[dict[str, object]]) -> None:
    by_case_group = {(str(row["case_id"]), str(row["experiment_group"])): row for row in results}
    for row in results:
        if row["experiment_group"] != "C" or not row["expected_hotword_ids"]:
            continue
        offline = by_case_group.get((str(row["case_id"]), "A"))
        if offline is None:
            if row.get("failure_reason") == "streaming_baseline_regression":
                row["failure_reason"] = "unknown_requires_review"
            continue
        offline_expected = set(cast(list[str], offline["expected_hotword_ids"]))
        offline_matched = set(cast(list[str], offline["matched_expected_hotword_ids"]))
        stream_matched = set(cast(list[str], row["matched_expected_hotword_ids"]))
        if offline_expected.issubset(offline_matched) and not offline_expected.issubset(
            stream_matched
        ):
            row["failure_reason"] = "streaming_baseline_regression"
        elif row.get("failure_reason") == "streaming_baseline_regression":
            row["failure_reason"] = "unknown_requires_review"


def _build_boundary_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    buckets: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in results:
        for item in row.get("hotword_metrics", []):
            bucket = item.get("boundary_bucket")
            if bucket:
                grouped[f"{row['experiment_group']}:{bucket}"].append(bool(item["final_correct"]))
    for key, correct in sorted(grouped.items()):
        buckets[key] = {
            "hotwords": len(correct),
            "hotword_exact_recall": _ratio(sum(correct), len(correct)),
        }
    return {"schema_version": 1, "buckets": buckets}


def _build_latency_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    metrics: dict[str, dict[str, object]] = {}
    names = (
        "ctc_first_detect_latency_sec",
        "first_correct_latency_sec",
        "stabilization_latency_sec",
        "chunks_from_injection_to_first_correct",
    )
    for group in ("C", "D", "E"):
        hotwords = [
            item
            for row in results
            if row["experiment_group"] == group
            for item in row.get("hotword_metrics", [])
        ]
        if not hotwords:
            continue
        metrics[group] = {
            name: _distribution(item.get(name) for item in hotwords) for name in names
        }
        mutable = [
            item["mutable_at_first_detect"]
            for item in hotwords
            if item.get("mutable_at_first_detect") is not None
        ]
        metrics[group]["mutable_at_first_detect_rate"] = _ratio(sum(mutable), len(mutable))
    return {"schema_version": 1, "groups": metrics}


def _interpret(groups: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    a = groups.get("A", {})
    b = groups.get("B", {})
    c = groups.get("C", {})
    d = groups.get("D", {})
    e = groups.get("E", {})
    d_recall = d.get("hotword_exact_recall")
    e_recall = e.get("hotword_exact_recall")
    c_wer = c.get("wer")
    d_wer = d.get("wer")
    a_wer = a.get("wer")
    b_wer = b.get("wer")
    if isinstance(d_recall, float) and isinstance(e_recall, float):
        if (
            isinstance(a_wer, float)
            and isinstance(b_wer, float)
            and isinstance(c_wer, float)
            and isinstance(d_wer, float)
            and c_wer > a_wer + 0.05
            and d_wer > b_wer + 0.05
        ):
            diagnosis = "streaming_asr_baseline"
        elif e_recall - d_recall >= 0.10:
            diagnosis = "ctc_detection_or_detection_timing"
        elif e_recall < 0.80:
            diagnosis = "streaming_prompt_or_fixed_unfixed_mechanism"
        else:
            diagnosis = "mixed_or_no_dominant_bottleneck"
    else:
        diagnosis = "insufficient_groups"
    return {"primary_diagnosis": diagnosis, "parameter_tuning_performed": False}


def _build_config(**values: Any) -> dict[str, object]:
    paths = values.pop("paths")
    boundary_path = values.pop("boundary_path")
    inputs = {
        key: _file_identity(path)
        for key, path in paths.items()
        if key in {"validation", "vocab", "hotwords", "cases", "checkpoint"}
    }
    inputs["offline_sample_selection"] = _file_identity(paths["offline"] / "sample_selection.json")
    model_config = paths["model"] / "config.json"
    tokenizer_config = paths["model"] / "tokenizer_config.json"
    if model_config.is_file():
        inputs["model_config"] = _file_identity(model_config)
    if tokenizer_config.is_file():
        inputs["tokenizer_config"] = _file_identity(tokenizer_config)
    if boundary_path is not None:
        inputs["boundary_manifest"] = _file_identity(boundary_path)
    try:
        version = importlib.metadata.version("qwen-asr")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed-local"
    if isinstance(values.get("groups"), tuple):
        values["groups"] = list(values["groups"])
    return {
        "schema_version": 1,
        "evaluation": "qwen3_asr_streaming_end_to_end_hotword_rag",
        "model_path": str(paths["model"]),
        "qwen_asr_version": version,
        "inference_backend": "official_qwen_vllm_streaming",
        "ctc_input_strategy": "causal_cumulative_audio_separate_transformers_encoder",
        "prompt_update_api": "experimental_state_prompt_refresh_via_public_initializer",
        "prompt_effect_policy": "same_step_state_refresh",
        "tokenizer": "qwen_processor.tokenizer",
        "git_commit": _git_commit(),
        "inputs": inputs,
        **values,
    }


def _prepare_output(destination: Path, config: Mapping[str, object], *, resume: bool) -> None:
    config_path = destination / "run_config.json"
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"output path is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        if not resume or not config_path.is_file():
            raise FileExistsError(
                "output directory is not empty; pass --resume for an identical run"
            )
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("resume config differs from existing run_config.json")
        return
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, config)


def _shard_path(root: Path, group: str, case_id: str) -> Path:
    digest = hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16]
    return root / f"{group}_{digest}.json"


def _write_shard(
    root: Path,
    group: str,
    case_id: str,
    result: Mapping[str, object],
    timeline: Sequence[Mapping[str, object]],
) -> None:
    _write_json(_shard_path(root, group, case_id), {"result": result, "timeline": list(timeline)})


def _read_shard(path: Path) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return dict(raw["result"]), tuple(dict(row) for row in raw["timeline"])


def _collect_shards(
    root: Path,
    groups: Sequence[str],
    samples: Sequence[StreamingSample],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results = []
    timeline: list[dict[str, object]] = []
    for sample in samples:
        for group in groups:
            path = _shard_path(root, group, sample.case_id)
            if not path.is_file():
                raise RuntimeError(f"missing completed sample shard: {path}")
            result, rows = _read_shard(path)
            results.append(result)
            timeline.extend(rows)
    return results, timeline


def _distribution(values: Iterable[Any]) -> dict[str, object]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + int(ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL input does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_hashes(destination: Path) -> None:
    lines = []
    for name in OUTPUT_FILES:
        if name == "sha256.txt":
            continue
        path = destination / name
        if path.is_file():
            lines.append(f"{_file_identity(path)['sha256']}  {path}\n")
    (destination / "sha256.txt").write_text("".join(lines), encoding="utf-8")


def _report_readme(config: Mapping[str, object], summary: Mapping[str, object]) -> str:
    return (
        "# Streaming end-to-end hotword RAG evaluation\n\n"
        "This directory is a separate, resumable 2 s / 2 chunk / 5 tokenizer-token "
        "baseline. Offline A/B rows are imported from the sealed offline RAG directory; "
        "C/D/E use Qwen's official vLLM streaming methods.\n\n"
        "Dynamic context has no official setter in the checked qwen-asr API. This run "
        "refreshes only prompt metadata using a temporary state created by the public "
        "initializer, and records that experimental mechanism in run_config.json and every "
        "timeline row.\n\n"
        f"Git commit: `{config['git_commit']}`\n\n"
        f"Status: `{summary['status']}`\n"
    )


def _required_string(raw: Mapping[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {key}")
    return value.strip()


def _string_tuple(
    raw: Mapping[str, Any],
    key: str,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"row {line_number} has invalid {key}")
    if not value and not allow_empty:
        raise ValueError(f"row {line_number} has empty {key}")
    return tuple(value)
