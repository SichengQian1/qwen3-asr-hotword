from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.config import EXPECTED_MODEL_NAME, ModelConfig
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.hotwords.simulation import SimulatedHotwordCase, load_simulated_cases
from qwen_hotword.inference.hotword_prompt import (
    DEFAULT_PT_BR_PROMPT_TEMPLATE,
    build_hotword_prompt,
    strict_matched_surfaces,
    strict_phrase_match,
)
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

DEFAULT_SELECTION_SEED = 20_260_729
OUTPUT_FILENAMES = (
    "sample_selection.json",
    "baseline_predictions.jsonl",
    "oracle_predictions.jsonl",
    "negative_prompt_predictions.jsonl",
    "prompt_smoke_report.json",
)


@dataclass(frozen=True)
class ValidationRecord:
    sample_id: str
    audio_path: str
    reference_text: str
    language: str


@dataclass(frozen=True)
class PromptSmokeSample:
    case_id: str
    sample_id: str
    audio_path: str
    reference_text: str
    language: str
    expected_hotword_ids: tuple[str, ...]
    expected_surfaces: tuple[str, ...]
    case_type: str
    selection_reason: str
    negative_control_hotword_id: str | None = None
    negative_control_surface: str | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["expected_hotword_ids"] = list(self.expected_hotword_ids)
        value["expected_surfaces"] = list(self.expected_surfaces)
        return value


@dataclass(frozen=True)
class PromptPrediction:
    case_id: str
    sample_id: str
    audio_path: str
    reference_text: str
    prediction: str
    language: str
    injected_hotword_ids: tuple[str, ...]
    injected_hotwords: tuple[str, ...]
    actual_prompt: str
    expected_hotword_ids: tuple[str, ...]
    expected_hotwords: tuple[str, ...]
    strict_matched_hotword_ids: tuple[str, ...]
    strict_matched_hotwords: tuple[str, ...]
    inference_seconds: float
    mode: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "injected_hotword_ids",
            "injected_hotwords",
            "expected_hotword_ids",
            "expected_hotwords",
            "strict_matched_hotword_ids",
            "strict_matched_hotwords",
        ):
            value[key] = list(value[key])
        return value


ModelLoader = Callable[[ModelConfig], Any]


def load_validation_manifest(path: str | Path) -> dict[str, ValidationRecord]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"validation manifest does not exist: {manifest_path}")
    records: dict[str, ValidationRecord] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid manifest JSON at {manifest_path}:{line_number}"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(f"manifest row {line_number} must be an object")
            split = _required_string(raw, "split", line_number)
            if split == "test":
                raise ValueError("sealed test data is forbidden in prompt smoke evaluation")
            if split != "validation":
                raise ValueError(
                    f"manifest row {line_number} is not formal validation data: {split!r}"
                )
            sample_id = _required_string(raw, "id", line_number)
            if sample_id in records:
                raise ValueError(f"duplicate validation sample ID: {sample_id}")
            records[sample_id] = ValidationRecord(
                sample_id=sample_id,
                audio_path=_required_string(raw, "audio_path", line_number),
                reference_text=_required_string(raw, "text", line_number),
                language=_required_string(raw, "language", line_number),
            )
    if not records:
        raise ValueError(f"validation manifest is empty: {manifest_path}")
    return records


def select_prompt_smoke_samples(
    records: Mapping[str, ValidationRecord],
    hotwords: Sequence[HotwordEntry],
    cases: Sequence[SimulatedHotwordCase],
    *,
    positive_count: int = 30,
    negative_count: int = 10,
    seed: int = DEFAULT_SELECTION_SEED,
) -> tuple[PromptSmokeSample, ...]:
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("positive and negative sample counts must be positive")
    hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
    if len(hotword_by_id) != len(hotwords):
        raise ValueError("hotword IDs must be unique")

    positives: list[tuple[SimulatedHotwordCase, ValidationRecord]] = []
    negatives: list[tuple[SimulatedHotwordCase, ValidationRecord]] = []
    for case in cases:
        record = records.get(case.sample_id)
        if record is None:
            raise ValueError(f"case {case.case_id} references unknown sample {case.sample_id}")
        if case.language != record.language:
            raise ValueError(f"case {case.case_id} language does not match its manifest row")
        unknown_ids = set(case.active_hotword_ids) - hotword_by_id.keys()
        if unknown_ids:
            raise ValueError(
                f"case {case.case_id} references unknown hotwords: {sorted(unknown_ids)}"
            )
        if case.expected_hotword_ids:
            for hotword_id in case.expected_hotword_ids:
                surface = hotword_by_id[hotword_id].surface
                if not strict_phrase_match(record.reference_text, surface):
                    raise ValueError(
                        f"positive case {case.case_id} reference does not strictly "
                        f"contain expected hotword {surface!r}"
                    )
            positives.append((case, record))
        else:
            negatives.append((case, record))

    if len(positives) < positive_count:
        raise ValueError(
            f"requested {positive_count} positive samples but only {len(positives)} exist"
        )
    if len(negatives) < negative_count:
        raise ValueError(
            f"requested {negative_count} negative samples but only {len(negatives)} exist"
        )

    selected_positive = _select_stratified_positives(
        positives,
        hotword_by_id,
        count=positive_count,
        seed=seed,
    )
    selected_negative = sorted(
        negatives,
        key=lambda item: _stable_rank(seed, f"negative:{item[0].case_id}"),
    )[:negative_count]

    samples: list[PromptSmokeSample] = []
    for case, record in selected_positive:
        expected = tuple(hotword_by_id[item] for item in case.expected_hotword_ids)
        bucket = _length_bucket(max(len(entry.token_ids) for entry in expected))
        multiplicity = "single_hotword" if len(expected) == 1 else "multi_hotword"
        samples.append(
            PromptSmokeSample(
                case_id=case.case_id,
                sample_id=case.sample_id,
                audio_path=record.audio_path,
                reference_text=record.reference_text,
                language=record.language,
                expected_hotword_ids=case.expected_hotword_ids,
                expected_surfaces=tuple(entry.surface for entry in expected),
                case_type="positive",
                selection_reason=(
                    f"deterministic_seed={seed}; length_bucket={bucket}; {multiplicity}"
                ),
            )
        )

    for case, record in selected_negative:
        available_ids = case.active_hotword_ids or tuple(hotword_by_id)
        candidates = [
            hotword_by_id[hotword_id]
            for hotword_id in available_ids
            if not strict_phrase_match(record.reference_text, hotword_by_id[hotword_id].surface)
        ]
        if not candidates:
            raise ValueError(f"negative case {case.case_id} has no strictly absent control hotword")
        injected = min(
            candidates,
            key=lambda entry: _stable_rank(
                seed,
                f"control:{case.case_id}:{entry.hotword_id}",
            ),
        )
        samples.append(
            PromptSmokeSample(
                case_id=case.case_id,
                sample_id=case.sample_id,
                audio_path=record.audio_path,
                reference_text=record.reference_text,
                language=record.language,
                expected_hotword_ids=(),
                expected_surfaces=(),
                case_type="negative",
                selection_reason=(f"deterministic_seed={seed}; negative_prompt_safety_control"),
                negative_control_hotword_id=injected.hotword_id,
                negative_control_surface=injected.surface,
            )
        )
    return tuple(samples)


def run_prompt_smoke(
    *,
    model_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    hotword_table_path: str | Path,
    cases_path: str | Path,
    output_dir: str | Path,
    positive_count: int = 30,
    negative_count: int = 10,
    seed: int = DEFAULT_SELECTION_SEED,
    prompt_template: str = DEFAULT_PT_BR_PROMPT_TEMPLATE,
    language: str = "Portuguese",
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    model_loader: ModelLoader | None = None,
    print_progress: bool = True,
) -> dict[str, object]:
    model = Path(model_path).expanduser()
    if model.name != EXPECTED_MODEL_NAME or not model.is_dir():
        raise ValueError(f"model path must be an existing {EXPECTED_MODEL_NAME} directory: {model}")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError("dtype must be bfloat16 or float16")
    if not language.strip():
        raise ValueError("language must not be empty")
    build_hotword_prompt(("probe",), template=prompt_template)

    destination = Path(output_dir).expanduser()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"prompt smoke output path is not a directory: {destination}")
    if destination.is_dir():
        existing_entries = sorted(str(path) for path in destination.iterdir())
        if existing_entries:
            raise FileExistsError(
                "prompt smoke output directory is not empty; refusing to mix or "
                "overwrite results: " + ", ".join(existing_entries)
            )
    output_paths = {name: destination / name for name in OUTPUT_FILENAMES}
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "prompt smoke output files already exist; refusing to overwrite: " + ", ".join(existing)
        )

    manifest_path = Path(validation_manifest_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    hotword_file = Path(hotword_table_path).expanduser()
    case_file = Path(cases_path).expanduser()
    records = load_validation_manifest(manifest_path)
    vocab = load_phoneme_vocab(vocab_file)
    hotwords = load_hotword_table(hotword_file, vocab=vocab, blank_id=0)
    cases = load_simulated_cases(case_file)
    samples = select_prompt_smoke_samples(
        records,
        hotwords,
        cases,
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
    )
    positives = tuple(sample for sample in samples if sample.case_type == "positive")
    negatives = tuple(sample for sample in samples if sample.case_type == "negative")

    config = ModelConfig(
        path=model,
        expected_name=EXPECTED_MODEL_NAME,
        dtype=dtype,
        device=device,
        local_files_only=True,
    )
    loader = model_loader or load_asr_model
    wrapper = loader(config)

    started = time.monotonic()
    total_calls = len(samples) + len(positives) + len(negatives)
    completed = 0
    baseline: list[PromptPrediction] = []
    baseline_by_case: dict[str, PromptPrediction] = {}
    for sample in samples:
        completed += 1
        prediction = _transcribe_one(
            wrapper,
            sample,
            mode="baseline",
            injected_ids=(),
            injected_surfaces=(),
            prompt="",
            language=language,
        )
        baseline.append(prediction)
        baseline_by_case[sample.case_id] = prediction
        _print_progress(
            enabled=print_progress,
            phase="baseline",
            completed=completed,
            total=total_calls,
            started=started,
        )

    oracle: list[PromptPrediction] = []
    for sample in positives:
        completed += 1
        prompt = build_hotword_prompt(sample.expected_surfaces, template=prompt_template)
        oracle.append(
            _transcribe_one(
                wrapper,
                sample,
                mode="oracle_prompt",
                injected_ids=sample.expected_hotword_ids,
                injected_surfaces=sample.expected_surfaces,
                prompt=prompt,
                language=language,
            )
        )
        _print_progress(
            enabled=print_progress,
            phase="oracle",
            completed=completed,
            total=total_calls,
            started=started,
        )

    negative_predictions: list[PromptPrediction] = []
    for sample in negatives:
        if sample.negative_control_hotword_id is None or sample.negative_control_surface is None:
            raise RuntimeError(f"negative case {sample.case_id} has no control hotword")
        completed += 1
        prompt = build_hotword_prompt(
            (sample.negative_control_surface,),
            template=prompt_template,
        )
        negative_predictions.append(
            _transcribe_one(
                wrapper,
                sample,
                mode="negative_prompt_control",
                injected_ids=(sample.negative_control_hotword_id,),
                injected_surfaces=(sample.negative_control_surface,),
                prompt=prompt,
                language=language,
            )
        )
        _print_progress(
            enabled=print_progress,
            phase="negative_control",
            completed=completed,
            total=total_calls,
            started=started,
        )

    report = _build_report(
        config=config,
        wrapper=wrapper,
        manifest_path=manifest_path,
        vocab_path=vocab_file,
        hotword_path=hotword_file,
        cases_path=case_file,
        samples=samples,
        baseline=baseline,
        oracle=oracle,
        negative=negative_predictions,
        baseline_by_case=baseline_by_case,
        prompt_template=prompt_template,
        language=language,
        seed=seed,
        elapsed=time.monotonic() - started,
        output_paths=output_paths,
    )
    selection = {
        "schema_version": 1,
        "evaluation_scope": "validation_prompt_smoke",
        "test_set_used": False,
        "seed": seed,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "samples": [sample.to_dict() for sample in samples],
    }
    _write_output_group(
        {
            output_paths["sample_selection.json"]: _json_bytes(selection),
            output_paths["baseline_predictions.jsonl"]: _jsonl_bytes(baseline),
            output_paths["oracle_predictions.jsonl"]: _jsonl_bytes(oracle),
            output_paths["negative_prompt_predictions.jsonl"]: _jsonl_bytes(negative_predictions),
            output_paths["prompt_smoke_report.json"]: _json_bytes(report),
        }
    )
    if print_progress:
        print(f"prompt smoke outputs written: {destination}", flush=True)
    return report


def _select_stratified_positives(
    positives: Sequence[tuple[SimulatedHotwordCase, ValidationRecord]],
    hotword_by_id: Mapping[str, HotwordEntry],
    *,
    count: int,
    seed: int,
) -> list[tuple[SimulatedHotwordCase, ValidationRecord]]:
    bucket_names = ("short_4_7", "medium_8_12", "long_13_plus")
    buckets: dict[str, list[tuple[SimulatedHotwordCase, ValidationRecord]]] = {
        name: [] for name in bucket_names
    }
    for item in positives:
        case = item[0]
        maximum_length = max(
            len(hotword_by_id[hotword_id].token_ids) for hotword_id in case.expected_hotword_ids
        )
        buckets[_length_bucket(maximum_length)].append(item)
    for name, values in buckets.items():
        ranked = sorted(
            values,
            key=lambda item: _stable_rank(
                seed,
                f"positive:{name}:{item[0].case_id}",
            ),
        )
        single = [item for item in ranked if len(item[0].expected_hotword_ids) == 1]
        multi = [item for item in ranked if len(item[0].expected_hotword_ids) > 1]
        balanced: list[tuple[SimulatedHotwordCase, ValidationRecord]] = []
        for index in range(max(len(single), len(multi))):
            if index < len(single):
                balanced.append(single[index])
            if index < len(multi):
                balanced.append(multi[index])
        buckets[name] = balanced

    quotas = {name: count // len(bucket_names) for name in bucket_names}
    for name in bucket_names[: count % len(bucket_names)]:
        quotas[name] += 1
    selected: list[tuple[SimulatedHotwordCase, ValidationRecord]] = []
    selected_ids: set[str] = set()
    for name in bucket_names:
        for item in buckets[name][: quotas[name]]:
            selected.append(item)
            selected_ids.add(item[0].case_id)
    if len(selected) < count:
        remaining = sorted(
            (item for item in positives if item[0].case_id not in selected_ids),
            key=lambda item: _stable_rank(seed, f"positive:fallback:{item[0].case_id}"),
        )
        selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise ValueError("unable to satisfy deterministic positive sample selection")
    return selected


def _transcribe_one(
    wrapper: Any,
    sample: PromptSmokeSample,
    *,
    mode: str,
    injected_ids: tuple[str, ...],
    injected_surfaces: tuple[str, ...],
    prompt: str,
    language: str,
) -> PromptPrediction:
    started = time.monotonic()
    results = wrapper.transcribe(
        audio=sample.audio_path,
        context=prompt,
        language=language,
        return_time_stamps=False,
    )
    inference_seconds = time.monotonic() - started
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise RuntimeError("Qwen3-ASR transcribe must return a sequence")
    if len(results) != 1:
        raise RuntimeError(f"single-audio Qwen3-ASR transcribe returned {len(results)} results")
    prediction = getattr(results[0], "text", None)
    if not isinstance(prediction, str):
        raise RuntimeError("Qwen3-ASR transcription result does not expose string text")
    match_surfaces = (
        injected_surfaces if mode == "negative_prompt_control" else sample.expected_surfaces
    )
    match_ids = injected_ids if mode == "negative_prompt_control" else sample.expected_hotword_ids
    matched_surfaces = strict_matched_surfaces(prediction, match_surfaces)
    matched_keys = set(matched_surfaces)
    matched_ids = tuple(
        hotword_id
        for hotword_id, surface in zip(match_ids, match_surfaces, strict=True)
        if surface in matched_keys
    )
    return PromptPrediction(
        case_id=sample.case_id,
        sample_id=sample.sample_id,
        audio_path=sample.audio_path,
        reference_text=sample.reference_text,
        prediction=prediction,
        language=sample.language,
        injected_hotword_ids=injected_ids,
        injected_hotwords=injected_surfaces,
        actual_prompt=prompt,
        expected_hotword_ids=sample.expected_hotword_ids,
        expected_hotwords=sample.expected_surfaces,
        strict_matched_hotword_ids=matched_ids,
        strict_matched_hotwords=matched_surfaces,
        inference_seconds=inference_seconds,
        mode=mode,
    )


def _build_report(
    *,
    config: ModelConfig,
    wrapper: Any,
    manifest_path: Path,
    vocab_path: Path,
    hotword_path: Path,
    cases_path: Path,
    samples: Sequence[PromptSmokeSample],
    baseline: Sequence[PromptPrediction],
    oracle: Sequence[PromptPrediction],
    negative: Sequence[PromptPrediction],
    baseline_by_case: Mapping[str, PromptPrediction],
    prompt_template: str,
    language: str,
    seed: int,
    elapsed: float,
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    positive_samples = [sample for sample in samples if sample.case_type == "positive"]
    expected_total = sum(len(sample.expected_hotword_ids) for sample in positive_samples)
    baseline_positive = [baseline_by_case[sample.case_id] for sample in positive_samples]
    baseline_correct = sum(
        len(prediction.strict_matched_hotword_ids) for prediction in baseline_positive
    )
    oracle_correct = sum(len(prediction.strict_matched_hotword_ids) for prediction in oracle)
    baseline_case_hits = sum(
        bool(prediction.strict_matched_hotword_ids) for prediction in baseline_positive
    )
    oracle_case_hits = sum(bool(prediction.strict_matched_hotword_ids) for prediction in oracle)
    baseline_recall = _safe_ratio(baseline_correct, expected_total)
    oracle_recall = _safe_ratio(oracle_correct, expected_total)
    absolute_improvement = oracle_recall - baseline_recall
    relative_improvement = absolute_improvement / baseline_recall if baseline_recall > 0.0 else None
    hallucinations = [
        prediction for prediction in negative if prediction.strict_matched_hotword_ids
    ]
    oracle_misses = [
        {
            "case_id": prediction.case_id,
            "sample_id": prediction.sample_id,
            "missing_hotword_ids": [
                hotword_id
                for hotword_id in prediction.expected_hotword_ids
                if hotword_id not in prediction.strict_matched_hotword_ids
            ],
            "missing_hotwords": [
                surface
                for surface in prediction.expected_hotwords
                if surface not in prediction.strict_matched_hotwords
            ],
        }
        for prediction in oracle
        if len(prediction.strict_matched_hotword_ids) < len(prediction.expected_hotword_ids)
    ]
    oracle_changed = sum(
        prediction.prediction != baseline_by_case[prediction.case_id].prediction
        for prediction in oracle
    )
    negative_changed = sum(
        prediction.prediction != baseline_by_case[prediction.case_id].prediction
        for prediction in negative
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "purpose": "qwen3_asr_oracle_hotword_prompt_injection_smoke",
        "evaluation_scope": "validation_prompt_smoke",
        "test_set_used": False,
        "ctc_retrieval_used": False,
        "model": {
            "path": str(config.path),
            "expected_name": config.expected_name,
            "dtype": config.dtype,
            "device": config.device,
            "local_files_only": config.local_files_only,
            "config": _file_identity(config.path / "config.json"),
            "config_metadata": _model_config_summary(config.path / "config.json"),
            "weight_index": _file_identity(config.path / "model.safetensors.index.json"),
            "wrapper_class": type(wrapper).__name__,
            "backend": getattr(wrapper, "backend", None),
            "max_new_tokens": getattr(wrapper, "max_new_tokens", None),
            "max_inference_batch_size": getattr(
                wrapper,
                "max_inference_batch_size",
                None,
            ),
            "load_count": 1,
        },
        "qwen_asr_version": _package_version("qwen-asr"),
        "prompt_interface": {
            "method": "Qwen3ASRModel.transcribe",
            "parameter_name": "context",
            "prompt_message_role": "system",
            "audio_message_role": "user",
            "template": prompt_template,
            "language": language,
        },
        "decoding": {
            "language": language,
            "return_time_stamps": False,
            "generate_overrides": {},
            "baseline_context": "",
        },
        "selection": {
            "seed": seed,
            "positive_cases": len(positive_samples),
            "negative_cases": len(samples) - len(positive_samples),
            "total_cases": len(samples),
        },
        "inputs": {
            "validation_manifest": _file_identity(manifest_path),
            "vocab": _file_identity(vocab_path),
            "hotword_table": _file_identity(hotword_path),
            "cases": _file_identity(cases_path),
        },
        "baseline_metrics": {
            "definition": (
                "strict normalized complete-word/phrase matching on positive "
                "validation cases; a case hit requires at least one expected hotword"
            ),
            "expected_hotwords": expected_total,
            "correct_hotwords": baseline_correct,
            "hotword_recall": baseline_recall,
            "positive_case_hits": baseline_case_hits,
            "positive_cases": len(positive_samples),
            "positive_case_hit_rate": _safe_ratio(
                baseline_case_hits,
                len(positive_samples),
            ),
        },
        "oracle_prompt_metrics": {
            "expected_hotwords": expected_total,
            "correct_hotwords": oracle_correct,
            "hotword_recall": oracle_recall,
            "positive_case_hits": oracle_case_hits,
            "positive_cases": len(positive_samples),
            "positive_case_hit_rate": _safe_ratio(
                oracle_case_hits,
                len(positive_samples),
            ),
            "additional_correct_hotwords_vs_baseline": (oracle_correct - baseline_correct),
            "absolute_recall_improvement": absolute_improvement,
            "relative_recall_improvement": relative_improvement,
            "cases_with_changed_text_vs_baseline": oracle_changed,
            "injected_but_missing": oracle_misses,
        },
        "negative_prompt_control_metrics": {
            "injected_wrong_hotwords": len(negative),
            "wrong_hotwords_written": len(hallucinations),
            "prompt_hallucination_rate": _safe_ratio(
                len(hallucinations),
                len(negative),
            ),
            "cases_with_changed_text_vs_baseline": negative_changed,
            "hallucination_cases": [
                {
                    "case_id": prediction.case_id,
                    "sample_id": prediction.sample_id,
                    "injected_hotword_ids": list(prediction.injected_hotword_ids),
                    "injected_hotwords": list(prediction.injected_hotwords),
                    "prediction": prediction.prediction,
                }
                for prediction in hallucinations
            ],
        },
        "inference": {
            "model_load_count": 1,
            "baseline_calls": len(baseline),
            "oracle_calls": len(oracle),
            "negative_control_calls": len(negative),
            "total_calls": len(baseline) + len(oracle) + len(negative),
            "wall_seconds": elapsed,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "limitations": [
            "Oracle validation smoke only; it does not measure retrieved RAG.",
            "No CTC case scores or sealed test data were read.",
            "Strict normalized complete-word/phrase matching has no aliases.",
            "No claim of prompt benefit is valid until work-zone outputs are inspected.",
        ],
        "next_step": "接入Retrieved RAG小规模评估",
    }


def _print_progress(
    *,
    enabled: bool,
    phase: str,
    completed: int,
    total: int,
    started: float,
) -> None:
    if not enabled:
        return
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0.0 else 0.0
    print(
        f"prompt_smoke phase={phase} completed={completed}/{total} "
        f"elapsed={elapsed:.1f}s rate={rate:.3f} cases/s eta={eta:.1f}s",
        flush=True,
    )


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[PromptPrediction]) -> bytes:
    return "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _write_output_group(outputs: Mapping[Path, bytes]) -> None:
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite prompt smoke output: {path}")
    temporary_paths: list[Path] = []
    try:
        for path, payload in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            if temporary.exists():
                raise FileExistsError(f"temporary output already exists: {temporary}")
            temporary.write_bytes(payload)
            temporary_paths.append(temporary)
        for path, temporary in zip(outputs, temporary_paths, strict=True):
            temporary.replace(path)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()


def _required_string(raw: Mapping[str, object], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest row {line_number} has invalid {key}")
    return value.strip()


def _stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _length_bucket(length: int) -> str:
    if 4 <= length <= 7:
        return "short_4_7"
    if 8 <= length <= 12:
        return "medium_8_12"
    if length >= 13:
        return "long_13_plus"
    raise ValueError(f"prompt smoke hotword has unsupported phoneme length: {length}")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required identity file does not exist: {path}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _model_config_summary(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"model config must contain a JSON object: {path}")
    summary: dict[str, object] = {}
    for key in ("architectures", "model_type", "torch_dtype"):
        value = raw.get(key)
        if isinstance(value, (str, int, float, bool, list)) or value is None:
            summary[key] = value
    thinker = raw.get("thinker_config")
    if isinstance(thinker, dict):
        audio = thinker.get("audio_config")
        if isinstance(audio, dict):
            summary["audio_config"] = {
                key: audio[key]
                for key in (
                    "d_model",
                    "encoder_layers",
                    "hidden_size",
                    "output_dim",
                )
                if key in audio
            }
        text = thinker.get("text_config")
        if isinstance(text, dict) and "hidden_size" in text:
            summary["text_hidden_size"] = text["hidden_size"]
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
