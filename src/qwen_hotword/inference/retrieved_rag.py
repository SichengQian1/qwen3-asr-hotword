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
    normalize_match_words,
    strict_matched_surfaces,
    strict_phrase_match,
)
from qwen_hotword.inference.prompt_smoke import ValidationRecord, load_validation_manifest
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

DEFAULT_SELECTION_SEED = 20_260_731
OUTPUT_FILENAMES = (
    "sample_selection.json",
    "baseline_predictions.jsonl",
    "retrieved_predictions.jsonl",
    "oracle_predictions.jsonl",
    "retrieved_rag_report.json",
)


@dataclass(frozen=True)
class RetrievedMatch:
    hotword_id: str
    surface: str
    score: float
    edit_ratio: float
    posterior_confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CtcCaseScore:
    case_id: str
    sample_id: str
    case_type: str
    active_hotword_ids: tuple[str, ...]
    expected_hotword_ids: tuple[str, ...]
    ranked_matches: tuple[RetrievedMatch, ...]


@dataclass(frozen=True)
class RetrievedRagSample:
    case_id: str
    sample_id: str
    audio_path: str
    reference_text: str
    language: str
    expected_hotword_ids: tuple[str, ...]
    expected_surfaces: tuple[str, ...]
    selected_matches: tuple[RetrievedMatch, ...]
    case_type: str
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["expected_hotword_ids"] = list(self.expected_hotword_ids)
        value["expected_surfaces"] = list(self.expected_surfaces)
        value["selected_matches"] = [match.to_dict() for match in self.selected_matches]
        return value


@dataclass(frozen=True)
class RetrievedRagPrediction:
    case_id: str
    sample_id: str
    audio_path: str
    reference_text: str
    prediction: str
    language: str
    injected_hotword_ids: tuple[str, ...]
    injected_hotwords: tuple[str, ...]
    retrieval_matches: tuple[RetrievedMatch, ...]
    actual_prompt: str
    expected_hotword_ids: tuple[str, ...]
    expected_hotwords: tuple[str, ...]
    strict_matched_expected_ids: tuple[str, ...]
    strict_matched_expected_hotwords: tuple[str, ...]
    strict_matched_injected_ids: tuple[str, ...]
    strict_matched_injected_hotwords: tuple[str, ...]
    word_errors: int
    reference_words: int
    inference_seconds: float
    inference_reused_from_baseline: bool
    mode: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in (
            "injected_hotword_ids",
            "injected_hotwords",
            "expected_hotword_ids",
            "expected_hotwords",
            "strict_matched_expected_ids",
            "strict_matched_expected_hotwords",
            "strict_matched_injected_ids",
            "strict_matched_injected_hotwords",
        ):
            value[key] = list(value[key])
        value["retrieval_matches"] = [match.to_dict() for match in self.retrieval_matches]
        return value


ModelLoader = Callable[[ModelConfig], Any]


def load_ctc_case_scores(path: str | Path) -> tuple[CtcCaseScore, ...]:
    score_path = Path(path).expanduser()
    if not score_path.is_file():
        raise FileNotFoundError(f"CTC case scores do not exist: {score_path}")
    rows: list[CtcCaseScore] = []
    seen_cases: set[str] = set()
    seen_samples: set[str] = set()
    with score_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid CTC score JSON at {score_path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"CTC score row {line_number} must be an object")
            case_id = _required_string(raw, "case_id", line_number)
            sample_id = _required_string(raw, "sample_id", line_number)
            if case_id in seen_cases or sample_id in seen_samples:
                raise ValueError("CTC case scores must have unique case and sample IDs")
            ranked_raw = raw.get("ranked_matches")
            if not isinstance(ranked_raw, list):
                raise ValueError(f"CTC score row {line_number} has invalid ranked_matches")
            matches = tuple(_parse_retrieved_match(value, line_number) for value in ranked_raw)
            if any(
                matches[index].score < matches[index + 1].score for index in range(len(matches) - 1)
            ):
                raise ValueError(f"CTC score row {line_number} is not score-ranked")
            row = CtcCaseScore(
                case_id=case_id,
                sample_id=sample_id,
                case_type=_required_string(raw, "case_type", line_number),
                active_hotword_ids=_string_tuple(
                    raw,
                    "active_hotword_ids",
                    line_number,
                ),
                expected_hotword_ids=_string_tuple(
                    raw,
                    "expected_hotword_ids",
                    line_number,
                    allow_empty=True,
                ),
                ranked_matches=matches,
            )
            if not set(row.expected_hotword_ids).issubset(row.active_hotword_ids):
                raise ValueError(f"CTC score row {line_number} expects an inactive hotword")
            if not {match.hotword_id for match in matches}.issubset(row.active_hotword_ids):
                raise ValueError(f"CTC score row {line_number} ranks an inactive hotword")
            seen_cases.add(case_id)
            seen_samples.add(sample_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"CTC case score table is empty: {score_path}")
    return tuple(rows)


def select_ctc_matches(
    score: CtcCaseScore,
    *,
    threshold: float,
    top_k: int,
    maximum_edit_ratio: float,
    minimum_posterior_confidence: float,
    minimum_top1_margin: float,
) -> tuple[RetrievedMatch, ...]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    for name, value in (
        ("maximum_edit_ratio", maximum_edit_ratio),
        ("minimum_posterior_confidence", minimum_posterior_confidence),
        ("minimum_top1_margin", minimum_top1_margin),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    qualified = tuple(
        match
        for match in score.ranked_matches
        if match.score >= threshold
        and match.edit_ratio <= maximum_edit_ratio
        and match.posterior_confidence >= minimum_posterior_confidence
    )
    if len(qualified) > 1 and qualified[0].score - qualified[1].score < minimum_top1_margin:
        return ()
    return qualified[:top_k]


def select_retrieved_rag_samples(
    records: Mapping[str, ValidationRecord],
    hotwords: Sequence[HotwordEntry],
    cases: Sequence[SimulatedHotwordCase],
    scores: Sequence[CtcCaseScore],
    *,
    positive_count: int = 60,
    negative_count: int = 40,
    seed: int = DEFAULT_SELECTION_SEED,
    threshold: float = 0.86,
    top_k: int = 3,
    maximum_edit_ratio: float = 0.35,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.0,
) -> tuple[RetrievedRagSample, ...]:
    if positive_count <= 0 or negative_count <= 0:
        raise ValueError("positive and negative sample counts must be positive")
    hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
    if len(hotword_by_id) != len(hotwords):
        raise ValueError("hotword IDs must be unique")
    case_by_id = {case.case_id: case for case in cases}
    score_by_id = {score.case_id: score for score in scores}
    if len(case_by_id) != len(cases) or len(score_by_id) != len(scores):
        raise ValueError("case IDs and CTC score case IDs must be unique")
    if case_by_id.keys() != score_by_id.keys():
        missing_scores = sorted(case_by_id.keys() - score_by_id.keys())
        extra_scores = sorted(score_by_id.keys() - case_by_id.keys())
        raise ValueError(
            "CTC scores and cases do not match: "
            f"missing_scores={missing_scores[:5]}, extra_scores={extra_scores[:5]}"
        )

    joined: list[
        tuple[
            SimulatedHotwordCase,
            ValidationRecord,
            CtcCaseScore,
            tuple[RetrievedMatch, ...],
        ]
    ] = []
    for case in cases:
        record = records.get(case.sample_id)
        if record is None:
            raise ValueError(f"case {case.case_id} references unknown sample {case.sample_id}")
        score = score_by_id[case.case_id]
        if (
            score.sample_id != case.sample_id
            or score.case_type != case.case_type
            or score.expected_hotword_ids != case.expected_hotword_ids
            or score.active_hotword_ids != case.active_hotword_ids
        ):
            raise ValueError(f"CTC score metadata differs from case {case.case_id}")
        if case.language != record.language:
            raise ValueError(f"case {case.case_id} language does not match validation")
        unknown_ids = set(case.active_hotword_ids) - hotword_by_id.keys()
        if unknown_ids:
            raise ValueError(
                f"case {case.case_id} references unknown hotwords: {sorted(unknown_ids)}"
            )
        for match in score.ranked_matches:
            if match.surface != hotword_by_id[match.hotword_id].surface:
                raise ValueError(
                    f"CTC score surface differs from hotword table: {match.hotword_id}"
                )
        for hotword_id in case.expected_hotword_ids:
            surface = hotword_by_id[hotword_id].surface
            if not strict_phrase_match(record.reference_text, surface):
                raise ValueError(f"positive case {case.case_id} does not contain {surface!r}")
        selected = select_ctc_matches(
            score,
            threshold=threshold,
            top_k=top_k,
            maximum_edit_ratio=maximum_edit_ratio,
            minimum_posterior_confidence=minimum_posterior_confidence,
            minimum_top1_margin=minimum_top1_margin,
        )
        joined.append((case, record, score, selected))

    positives = [item for item in joined if item[0].expected_hotword_ids]
    negatives = [item for item in joined if not item[0].expected_hotword_ids]
    if len(positives) < positive_count or len(negatives) < negative_count:
        raise ValueError(
            "not enough positive or negative cases for requested selection: "
            f"available={len(positives)}/{len(negatives)}, "
            f"requested={positive_count}/{negative_count}"
        )
    selected_positive = _select_stratified_positives(
        positives,
        hotword_by_id,
        count=positive_count,
        seed=seed,
    )
    triggered_negative = sorted(
        (item for item in negatives if item[3]),
        key=lambda item: _stable_rank(seed, f"negative_trigger:{item[0].case_id}"),
    )
    selected_negative = triggered_negative[:negative_count]
    selected_negative_ids = {item[0].case_id for item in selected_negative}
    if len(selected_negative) < negative_count:
        remaining = sorted(
            (item for item in negatives if item[0].case_id not in selected_negative_ids),
            key=lambda item: _stable_rank(seed, f"negative:{item[0].case_id}"),
        )
        selected_negative.extend(remaining[: negative_count - len(selected_negative)])

    samples: list[RetrievedRagSample] = []
    for case, record, _, selected in selected_positive:
        expected = tuple(hotword_by_id[item] for item in case.expected_hotword_ids)
        bucket = _length_bucket(max(len(entry.token_ids) for entry in expected))
        multiplicity = "single_hotword" if len(expected) == 1 else "multi_hotword"
        samples.append(
            _build_sample(
                case,
                record,
                expected,
                selected,
                case_type="positive",
                reason=(f"deterministic_seed={seed}; length_bucket={bucket}; {multiplicity}"),
            )
        )
    for case, record, _, selected in selected_negative:
        reason = (
            f"deterministic_seed={seed}; ctc_false_positive_stress"
            if selected
            else f"deterministic_seed={seed}; negative_no_ctc_trigger"
        )
        samples.append(
            _build_sample(
                case,
                record,
                (),
                selected,
                case_type="negative",
                reason=reason,
            )
        )
    return tuple(samples)


def run_retrieved_rag(
    *,
    model_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    hotword_table_path: str | Path,
    cases_path: str | Path,
    ctc_case_scores_path: str | Path,
    output_dir: str | Path,
    positive_count: int = 60,
    negative_count: int = 40,
    seed: int = DEFAULT_SELECTION_SEED,
    threshold: float = 0.86,
    top_k: int = 3,
    maximum_edit_ratio: float = 0.35,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.0,
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
        raise FileExistsError(f"Retrieved RAG output path is not a directory: {destination}")
    if destination.is_dir():
        existing_entries = sorted(str(path) for path in destination.iterdir())
        if existing_entries:
            raise FileExistsError(
                "Retrieved RAG output directory is not empty; refusing to mix or "
                "overwrite results: " + ", ".join(existing_entries)
            )
    output_paths = {name: destination / name for name in OUTPUT_FILENAMES}

    manifest_path = Path(validation_manifest_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    hotword_file = Path(hotword_table_path).expanduser()
    case_file = Path(cases_path).expanduser()
    score_file = Path(ctc_case_scores_path).expanduser()
    records = load_validation_manifest(manifest_path)
    vocab = load_phoneme_vocab(vocab_file)
    hotwords = load_hotword_table(hotword_file, vocab=vocab, blank_id=0)
    cases = load_simulated_cases(case_file)
    scores = load_ctc_case_scores(score_file)
    samples = select_retrieved_rag_samples(
        records,
        hotwords,
        cases,
        scores,
        positive_count=positive_count,
        negative_count=negative_count,
        seed=seed,
        threshold=threshold,
        top_k=top_k,
        maximum_edit_ratio=maximum_edit_ratio,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
    )
    hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
    positives = tuple(sample for sample in samples if sample.case_type == "positive")
    prompted = tuple(sample for sample in samples if sample.selected_matches)

    config = ModelConfig(
        path=model,
        expected_name=EXPECTED_MODEL_NAME,
        dtype=dtype,
        device=device,
        local_files_only=True,
    )
    wrapper = (model_loader or load_asr_model)(config)
    started = time.monotonic()
    total_calls = len(samples) + len(prompted) + len(positives)
    completed = 0

    baseline: list[RetrievedRagPrediction] = []
    baseline_by_case: dict[str, RetrievedRagPrediction] = {}
    for sample in samples:
        prediction = _transcribe_one(
            wrapper,
            sample,
            mode="baseline",
            injected_ids=(),
            injected_surfaces=(),
            retrieval_matches=(),
            prompt="",
            language=language,
        )
        baseline.append(prediction)
        baseline_by_case[sample.case_id] = prediction
        completed += 1
        _print_progress(
            enabled=print_progress,
            phase="baseline",
            completed=completed,
            total=total_calls,
            started=started,
        )

    retrieved: list[RetrievedRagPrediction] = []
    for sample in samples:
        if not sample.selected_matches:
            retrieved.append(
                _reuse_baseline(
                    baseline_by_case[sample.case_id],
                    mode="retrieved_prompt",
                )
            )
            continue
        injected_ids = tuple(match.hotword_id for match in sample.selected_matches)
        injected_surfaces = tuple(hotword_by_id[item].surface for item in injected_ids)
        prediction = _transcribe_one(
            wrapper,
            sample,
            mode="retrieved_prompt",
            injected_ids=injected_ids,
            injected_surfaces=injected_surfaces,
            retrieval_matches=sample.selected_matches,
            prompt=build_hotword_prompt(injected_surfaces, template=prompt_template),
            language=language,
        )
        retrieved.append(prediction)
        completed += 1
        _print_progress(
            enabled=print_progress,
            phase="retrieved",
            completed=completed,
            total=total_calls,
            started=started,
        )

    oracle: list[RetrievedRagPrediction] = []
    for sample in positives:
        prediction = _transcribe_one(
            wrapper,
            sample,
            mode="oracle_prompt",
            injected_ids=sample.expected_hotword_ids,
            injected_surfaces=sample.expected_surfaces,
            retrieval_matches=sample.selected_matches,
            prompt=build_hotword_prompt(
                sample.expected_surfaces,
                template=prompt_template,
            ),
            language=language,
        )
        oracle.append(prediction)
        completed += 1
        _print_progress(
            enabled=print_progress,
            phase="oracle",
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
        score_path=score_file,
        all_scores=scores,
        samples=samples,
        baseline=baseline,
        retrieved=retrieved,
        oracle=oracle,
        prompt_template=prompt_template,
        language=language,
        seed=seed,
        threshold=threshold,
        top_k=top_k,
        maximum_edit_ratio=maximum_edit_ratio,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
        elapsed=time.monotonic() - started,
        output_paths=output_paths,
    )
    selection = {
        "schema_version": 1,
        "evaluation_scope": "validation_retrieved_rag_smoke",
        "test_set_used": False,
        "seed": seed,
        "positive_count": len(positives),
        "negative_count": len(samples) - len(positives),
        "negative_ctc_false_positive_stress_cases": sum(
            sample.case_type == "negative" and bool(sample.selected_matches) for sample in samples
        ),
        "retrieval_config": {
            "threshold": threshold,
            "top_k": top_k,
            "maximum_edit_ratio": maximum_edit_ratio,
            "minimum_posterior_confidence": minimum_posterior_confidence,
            "minimum_top1_margin": minimum_top1_margin,
        },
        "samples": [sample.to_dict() for sample in samples],
    }
    _write_output_group(
        {
            output_paths["sample_selection.json"]: _json_bytes(selection),
            output_paths["baseline_predictions.jsonl"]: _jsonl_bytes(baseline),
            output_paths["retrieved_predictions.jsonl"]: _jsonl_bytes(retrieved),
            output_paths["oracle_predictions.jsonl"]: _jsonl_bytes(oracle),
            output_paths["retrieved_rag_report.json"]: _json_bytes(report),
        }
    )
    if print_progress:
        print(f"Retrieved RAG outputs written: {destination}", flush=True)
    return report


def _build_sample(
    case: SimulatedHotwordCase,
    record: ValidationRecord,
    expected: Sequence[HotwordEntry],
    selected: tuple[RetrievedMatch, ...],
    *,
    case_type: str,
    reason: str,
) -> RetrievedRagSample:
    return RetrievedRagSample(
        case_id=case.case_id,
        sample_id=case.sample_id,
        audio_path=record.audio_path,
        reference_text=record.reference_text,
        language=record.language,
        expected_hotword_ids=case.expected_hotword_ids,
        expected_surfaces=tuple(entry.surface for entry in expected),
        selected_matches=selected,
        case_type=case_type,
        selection_reason=reason,
    )


def _select_stratified_positives(
    positives: Sequence[
        tuple[
            SimulatedHotwordCase,
            ValidationRecord,
            CtcCaseScore,
            tuple[RetrievedMatch, ...],
        ]
    ],
    hotword_by_id: Mapping[str, HotwordEntry],
    *,
    count: int,
    seed: int,
) -> list[
    tuple[
        SimulatedHotwordCase,
        ValidationRecord,
        CtcCaseScore,
        tuple[RetrievedMatch, ...],
    ]
]:
    bucket_names = ("short_4_7", "medium_8_12", "long_13_plus")
    buckets: dict[
        str,
        list[
            tuple[
                SimulatedHotwordCase,
                ValidationRecord,
                CtcCaseScore,
                tuple[RetrievedMatch, ...],
            ]
        ],
    ] = {name: [] for name in bucket_names}
    for item in positives:
        maximum_length = max(
            len(hotword_by_id[hotword_id].token_ids) for hotword_id in item[0].expected_hotword_ids
        )
        buckets[_length_bucket(maximum_length)].append(item)
    for name, values in buckets.items():
        ranked = sorted(
            values,
            key=lambda item: _stable_rank(seed, f"positive:{name}:{item[0].case_id}"),
        )
        single = [item for item in ranked if len(item[0].expected_hotword_ids) == 1]
        multi = [item for item in ranked if len(item[0].expected_hotword_ids) > 1]
        balanced = []
        for index in range(max(len(single), len(multi))):
            if index < len(single):
                balanced.append(single[index])
            if index < len(multi):
                balanced.append(multi[index])
        buckets[name] = balanced
    quotas = {name: count // len(bucket_names) for name in bucket_names}
    for name in bucket_names[: count % len(bucket_names)]:
        quotas[name] += 1
    selected = [item for name in bucket_names for item in buckets[name][: quotas[name]]]
    selected_ids = {item[0].case_id for item in selected}
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
    sample: RetrievedRagSample,
    *,
    mode: str,
    injected_ids: tuple[str, ...],
    injected_surfaces: tuple[str, ...],
    retrieval_matches: tuple[RetrievedMatch, ...],
    prompt: str,
    language: str,
) -> RetrievedRagPrediction:
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
    return _prediction_record(
        sample,
        prediction=prediction,
        mode=mode,
        injected_ids=injected_ids,
        injected_surfaces=injected_surfaces,
        retrieval_matches=retrieval_matches,
        prompt=prompt,
        inference_seconds=inference_seconds,
        reused=False,
    )


def _prediction_record(
    sample: RetrievedRagSample,
    *,
    prediction: str,
    mode: str,
    injected_ids: tuple[str, ...],
    injected_surfaces: tuple[str, ...],
    retrieval_matches: tuple[RetrievedMatch, ...],
    prompt: str,
    inference_seconds: float,
    reused: bool,
) -> RetrievedRagPrediction:
    expected_matches = strict_matched_surfaces(prediction, sample.expected_surfaces)
    expected_keys = set(expected_matches)
    matched_expected_ids = tuple(
        hotword_id
        for hotword_id, surface in zip(
            sample.expected_hotword_ids,
            sample.expected_surfaces,
            strict=True,
        )
        if surface in expected_keys
    )
    injected_matches = strict_matched_surfaces(prediction, injected_surfaces)
    injected_keys = set(injected_matches)
    matched_injected_ids = tuple(
        hotword_id
        for hotword_id, surface in zip(injected_ids, injected_surfaces, strict=True)
        if surface in injected_keys
    )
    reference_words = normalize_match_words(sample.reference_text)
    prediction_words = normalize_match_words(prediction)
    return RetrievedRagPrediction(
        case_id=sample.case_id,
        sample_id=sample.sample_id,
        audio_path=sample.audio_path,
        reference_text=sample.reference_text,
        prediction=prediction,
        language=sample.language,
        injected_hotword_ids=injected_ids,
        injected_hotwords=injected_surfaces,
        retrieval_matches=retrieval_matches,
        actual_prompt=prompt,
        expected_hotword_ids=sample.expected_hotword_ids,
        expected_hotwords=sample.expected_surfaces,
        strict_matched_expected_ids=matched_expected_ids,
        strict_matched_expected_hotwords=expected_matches,
        strict_matched_injected_ids=matched_injected_ids,
        strict_matched_injected_hotwords=injected_matches,
        word_errors=_word_edit_distance(reference_words, prediction_words),
        reference_words=len(reference_words),
        inference_seconds=inference_seconds,
        inference_reused_from_baseline=reused,
        mode=mode,
    )


def _reuse_baseline(
    baseline: RetrievedRagPrediction,
    *,
    mode: str,
) -> RetrievedRagPrediction:
    return RetrievedRagPrediction(
        case_id=baseline.case_id,
        sample_id=baseline.sample_id,
        audio_path=baseline.audio_path,
        reference_text=baseline.reference_text,
        prediction=baseline.prediction,
        language=baseline.language,
        injected_hotword_ids=(),
        injected_hotwords=(),
        retrieval_matches=(),
        actual_prompt="",
        expected_hotword_ids=tuple(baseline.expected_hotword_ids),
        expected_hotwords=tuple(baseline.expected_hotwords),
        strict_matched_expected_ids=tuple(baseline.strict_matched_expected_ids),
        strict_matched_expected_hotwords=tuple(baseline.strict_matched_expected_hotwords),
        strict_matched_injected_ids=(),
        strict_matched_injected_hotwords=(),
        word_errors=baseline.word_errors,
        reference_words=baseline.reference_words,
        inference_seconds=0.0,
        inference_reused_from_baseline=True,
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
    score_path: Path,
    all_scores: Sequence[CtcCaseScore],
    samples: Sequence[RetrievedRagSample],
    baseline: Sequence[RetrievedRagPrediction],
    retrieved: Sequence[RetrievedRagPrediction],
    oracle: Sequence[RetrievedRagPrediction],
    prompt_template: str,
    language: str,
    seed: int,
    threshold: float,
    top_k: int,
    maximum_edit_ratio: float,
    minimum_posterior_confidence: float,
    minimum_top1_margin: float,
    elapsed: float,
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    sample_by_case = {sample.case_id: sample for sample in samples}
    baseline_by_case = {item.case_id: item for item in baseline}
    retrieved_by_case = {item.case_id: item for item in retrieved}
    oracle_by_case = {item.case_id: item for item in oracle}
    positive_cases = [sample.case_id for sample in samples if sample.expected_hotword_ids]
    expected_total = sum(
        len(sample_by_case[case_id].expected_hotword_ids) for case_id in positive_cases
    )
    baseline_metrics = _recognition_metrics(
        [baseline_by_case[case_id] for case_id in positive_cases],
        expected_total=expected_total,
    )
    retrieved_metrics = _recognition_metrics(
        [retrieved_by_case[case_id] for case_id in positive_cases],
        expected_total=expected_total,
    )
    oracle_metrics = _recognition_metrics(
        [oracle_by_case[case_id] for case_id in positive_cases],
        expected_total=expected_total,
    )
    baseline_correct = sum(
        len(baseline_by_case[case_id].strict_matched_expected_ids) for case_id in positive_cases
    )
    retrieved_correct = sum(
        len(retrieved_by_case[case_id].strict_matched_expected_ids) for case_id in positive_cases
    )
    oracle_correct = sum(
        len(oracle_by_case[case_id].strict_matched_expected_ids) for case_id in positive_cases
    )
    baseline_recall = _safe_ratio(baseline_correct, expected_total)
    retrieved_recall = _safe_ratio(retrieved_correct, expected_total)
    oracle_recall = _safe_ratio(oracle_correct, expected_total)

    selected_lookup = {
        sample.case_id: {match.hotword_id for match in sample.selected_matches}
        for sample in samples
    }
    retrieval_misses: list[dict[str, object]] = []
    decoder_misses: list[dict[str, object]] = []
    rescues: list[dict[str, object]] = []
    injected_expected = 0
    wrong_injected = 0
    wrong_written = 0
    newly_written_wrong = 0
    wrong_written_cases: list[dict[str, object]] = []
    newly_written_wrong_cases: list[dict[str, object]] = []
    for sample in samples:
        expected = set(sample.expected_hotword_ids)
        selected = selected_lookup[sample.case_id]
        injected_expected += len(selected & expected)
        wrong_ids = selected - expected
        wrong_injected += len(wrong_ids)
        retrieved_prediction = retrieved_by_case[sample.case_id]
        matched_injected = set(retrieved_prediction.strict_matched_injected_ids)
        written_wrong = wrong_ids & matched_injected
        baseline_prediction = baseline_by_case[sample.case_id].prediction
        baseline_written_wrong = {
            match.hotword_id
            for match in sample.selected_matches
            if match.hotword_id in wrong_ids
            and strict_phrase_match(baseline_prediction, match.surface)
        }
        new_written_wrong = written_wrong - baseline_written_wrong
        wrong_written += len(written_wrong)
        newly_written_wrong += len(new_written_wrong)
        if written_wrong:
            wrong_written_cases.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "wrong_hotword_ids": sorted(written_wrong),
                    "prediction": retrieved_prediction.prediction,
                }
            )
        if new_written_wrong:
            newly_written_wrong_cases.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "newly_written_wrong_hotword_ids": sorted(new_written_wrong),
                    "baseline_prediction": baseline_prediction,
                    "retrieved_prediction": retrieved_prediction.prediction,
                }
            )
        if not expected:
            continue
        missing_retrieval = expected - selected
        if missing_retrieval:
            retrieval_misses.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "missing_expected_hotword_ids": sorted(missing_retrieval),
                }
            )
        retrieved_expected = expected & selected
        final_expected = set(retrieved_prediction.strict_matched_expected_ids)
        missed_after_prompt = retrieved_expected - final_expected
        if missed_after_prompt:
            decoder_misses.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "retrieved_but_missing_hotword_ids": sorted(missed_after_prompt),
                }
            )
        baseline_expected = set(baseline_by_case[sample.case_id].strict_matched_expected_ids)
        rescued = (retrieved_expected - baseline_expected) & final_expected
        if rescued:
            rescues.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "rescued_hotword_ids": sorted(rescued),
                }
            )

    full_retrieval = _retrieval_metrics(
        all_scores,
        threshold=threshold,
        top_k=top_k,
        maximum_edit_ratio=maximum_edit_ratio,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
    )
    selected_scores = [score for score in all_scores if score.case_id in sample_by_case]
    selected_retrieval = _retrieval_metrics(
        selected_scores,
        threshold=threshold,
        top_k=top_k,
        maximum_edit_ratio=maximum_edit_ratio,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
    )
    retrieved_changed = sum(
        retrieved_by_case[case_id].prediction != baseline_by_case[case_id].prediction
        for case_id in sample_by_case
    )
    oracle_changed = sum(
        oracle_by_case[case_id].prediction != baseline_by_case[case_id].prediction
        for case_id in positive_cases
    )
    absolute_improvement = retrieved_recall - baseline_recall
    oracle_absolute_improvement = oracle_recall - baseline_recall
    prompted_count = sum(bool(sample.selected_matches) for sample in samples)
    return {
        "schema_version": 1,
        "status": "pass",
        "purpose": "qwen3_asr_ctc_retrieved_hotword_rag_smoke",
        "evaluation_scope": "validation_retrieved_rag_smoke",
        "test_set_used": False,
        "ctc_retrieval_used": True,
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
        "retrieval_config": {
            "threshold": threshold,
            "top_k": top_k,
            "maximum_edit_ratio": maximum_edit_ratio,
            "minimum_posterior_confidence": minimum_posterior_confidence,
            "minimum_top1_margin": minimum_top1_margin,
            "fixed_for_smoke_test": True,
            "tuning_performed": False,
        },
        "selection": {
            "seed": seed,
            "positive_cases": len(positive_cases),
            "negative_cases": len(samples) - len(positive_cases),
            "total_cases": len(samples),
            "prompted_cases": prompted_count,
            "negative_ctc_false_positive_stress_cases": sum(
                sample.case_type == "negative" and bool(sample.selected_matches)
                for sample in samples
            ),
            "negative_selection_note": (
                "all available threshold-triggered validation negatives are "
                "included before deterministic non-trigger negatives; selected "
                "negative rates are diagnostic, not an unbiased FPR estimate"
            ),
        },
        "inputs": {
            "validation_manifest": _file_identity(manifest_path),
            "vocab": _file_identity(vocab_path),
            "hotword_table": _file_identity(hotword_path),
            "cases": _file_identity(cases_path),
            "ctc_case_scores": _file_identity(score_path),
        },
        "full_validation_ctc_retrieval_metrics": full_retrieval,
        "selected_sample_ctc_retrieval_metrics": selected_retrieval,
        "baseline_metrics": baseline_metrics,
        "retrieved_prompt_metrics": {
            **retrieved_metrics,
            "additional_correct_hotwords_vs_baseline": (retrieved_correct - baseline_correct),
            "absolute_recall_improvement": absolute_improvement,
            "relative_recall_improvement": (
                absolute_improvement / baseline_recall if baseline_recall > 0.0 else None
            ),
            "cases_with_changed_text_vs_baseline": retrieved_changed,
        },
        "oracle_prompt_metrics": {
            **oracle_metrics,
            "additional_correct_hotwords_vs_baseline": (oracle_correct - baseline_correct),
            "absolute_recall_improvement": oracle_absolute_improvement,
            "relative_recall_improvement": (
                oracle_absolute_improvement / baseline_recall if baseline_recall > 0.0 else None
            ),
            "cases_with_changed_text_vs_baseline": oracle_changed,
        },
        "pipeline_attribution": {
            "expected_hotwords": expected_total,
            "expected_hotwords_retrieved_and_injected": injected_expected,
            "retrieval_miss_cases": retrieval_misses,
            "retrieved_but_decoder_miss_cases": decoder_misses,
            "hotword_rescue_cases_vs_baseline": rescues,
            "wrong_candidates_injected": wrong_injected,
            "wrong_injected_candidates_written": wrong_written,
            "wrong_candidate_write_rate": _safe_ratio(
                wrong_written,
                wrong_injected,
            ),
            "wrong_candidate_written_cases": wrong_written_cases,
            "newly_written_wrong_candidates_vs_baseline": newly_written_wrong,
            "new_wrong_candidate_hallucination_rate": _safe_ratio(
                newly_written_wrong,
                wrong_injected,
            ),
            "new_wrong_candidate_hallucination_cases": newly_written_wrong_cases,
        },
        "word_error_diagnostics": {
            "definition": (
                "corpus word edit distance after the same NFKC/lowercase/"
                "punctuation/space normalization used for strict hotword matching"
            ),
            "baseline": _word_error_metrics(baseline),
            "retrieved_prompt": _word_error_metrics(retrieved),
            "oracle_prompt_positive_cases": _word_error_metrics(oracle),
        },
        "inference": {
            "model_load_count": 1,
            "baseline_calls": len(baseline),
            "retrieved_prompt_calls": prompted_count,
            "retrieved_baseline_reuses": len(samples) - prompted_count,
            "oracle_calls": len(oracle),
            "total_model_calls": len(baseline) + prompted_count + len(oracle),
            "wall_seconds": elapsed,
            "baseline_inference_seconds": sum(item.inference_seconds for item in baseline),
            "retrieved_inference_seconds": sum(item.inference_seconds for item in retrieved),
            "oracle_inference_seconds": sum(item.inference_seconds for item in oracle),
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "limitations": [
            "Small validation-only diagnostic; it is not a production benchmark.",
            "The negative sample is deliberately enriched with CTC false positives.",
            "Simulated validation hotwords are common words/phrases and may have a high baseline.",
            "Strict complete-word/phrase matching has no aliases or morphology expansion.",
            "Threshold and top-k are fixed; no hyperparameter search is performed.",
            "No sealed test data are read.",
        ],
        "next_step": "inspect work-zone outputs, then tune threshold/top-k separately",
    }


def _retrieval_metrics(
    scores: Sequence[CtcCaseScore],
    *,
    threshold: float,
    top_k: int,
    maximum_edit_ratio: float,
    minimum_posterior_confidence: float,
    minimum_top1_margin: float,
) -> dict[str, object]:
    expected_total = 0
    selected_total = 0
    true_positive_total = 0
    positive_cases = 0
    positive_case_hits = 0
    negative_cases = 0
    negative_trigger_cases = 0
    for score in scores:
        expected = set(score.expected_hotword_ids)
        selected = select_ctc_matches(
            score,
            threshold=threshold,
            top_k=top_k,
            maximum_edit_ratio=maximum_edit_ratio,
            minimum_posterior_confidence=minimum_posterior_confidence,
            minimum_top1_margin=minimum_top1_margin,
        )
        selected_ids = {match.hotword_id for match in selected}
        true_positives = len(expected & selected_ids)
        expected_total += len(expected)
        selected_total += len(selected_ids)
        true_positive_total += true_positives
        if expected:
            positive_cases += 1
            positive_case_hits += bool(true_positives)
        else:
            negative_cases += 1
            negative_trigger_cases += bool(selected_ids)
    return {
        "cases": len(scores),
        "positive_cases": positive_cases,
        "negative_cases": negative_cases,
        "expected_hotwords": expected_total,
        "selected_hotwords": selected_total,
        "true_positive_hotwords": true_positive_total,
        "false_positive_hotwords": selected_total - true_positive_total,
        "precision": _safe_ratio(true_positive_total, selected_total),
        "recall": _safe_ratio(true_positive_total, expected_total),
        "positive_case_hits": positive_case_hits,
        "positive_case_hit_rate": _safe_ratio(
            positive_case_hits,
            positive_cases,
        ),
        "negative_false_positive_cases": negative_trigger_cases,
        "negative_case_false_positive_rate": _safe_ratio(
            negative_trigger_cases,
            negative_cases,
        ),
    }


def _recognition_metrics(
    predictions: Sequence[RetrievedRagPrediction],
    *,
    expected_total: int,
) -> dict[str, object]:
    correct = sum(len(item.strict_matched_expected_ids) for item in predictions)
    case_hits = sum(bool(item.strict_matched_expected_ids) for item in predictions)
    return {
        "definition": (
            "strict normalized complete-word/phrase matching on positive "
            "validation cases; a case hit requires at least one expected hotword"
        ),
        "expected_hotwords": expected_total,
        "correct_hotwords": correct,
        "hotword_recall": _safe_ratio(correct, expected_total),
        "positive_case_hits": case_hits,
        "positive_cases": len(predictions),
        "positive_case_hit_rate": _safe_ratio(case_hits, len(predictions)),
    }


def _word_error_metrics(
    predictions: Sequence[RetrievedRagPrediction],
) -> dict[str, object]:
    errors = sum(item.word_errors for item in predictions)
    reference_words = sum(item.reference_words for item in predictions)
    return {
        "cases": len(predictions),
        "word_errors": errors,
        "reference_words": reference_words,
        "word_error_rate": _safe_ratio(errors, reference_words),
    }


def _word_edit_distance(
    reference: tuple[str, ...],
    hypothesis: tuple[str, ...],
) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_word in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def _parse_retrieved_match(value: object, line_number: int) -> RetrievedMatch:
    if not isinstance(value, dict):
        raise ValueError(f"CTC score row {line_number} has a non-object match")
    return RetrievedMatch(
        hotword_id=_required_string(value, "hotword_id", line_number),
        surface=_required_string(value, "surface", line_number),
        score=_required_ratio(value, "score", line_number),
        edit_ratio=_required_ratio(value, "edit_ratio", line_number),
        posterior_confidence=_required_ratio(
            value,
            "posterior_confidence",
            line_number,
        ),
    )


def _required_string(
    raw: Mapping[str, object],
    key: str,
    line_number: int,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {key}")
    return value.strip()


def _required_ratio(
    raw: Mapping[str, object],
    key: str,
    line_number: int,
) -> float:
    value = raw.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"row {line_number} has invalid {key}")
    return float(value)


def _string_tuple(
    raw: Mapping[str, object],
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


def _length_bucket(length: int) -> str:
    if 4 <= length <= 7:
        return "short_4_7"
    if 8 <= length <= 12:
        return "medium_8_12"
    if length >= 13:
        return "long_13_plus"
    raise ValueError(f"Retrieved RAG hotword has unsupported phoneme length: {length}")


def _stable_rank(seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


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
        f"retrieved_rag phase={phase} completed={completed}/{total} "
        f"elapsed={elapsed:.1f}s rate={rate:.3f} calls/s eta={eta:.1f}s",
        flush=True,
    )


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(rows: Iterable[RetrievedRagPrediction]) -> bytes:
    return "".join(
        json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _write_output_group(outputs: Mapping[Path, bytes]) -> None:
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Retrieved RAG output: {path}")
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
                for key in ("d_model", "encoder_layers", "hidden_size", "output_dim")
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
