from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.config import EXPECTED_MODEL_NAME, ModelConfig
from qwen_hotword.hotwords.multi_nested import (
    CaseScore,
    MultiNestedCase,
    load_hotword_families,
    load_multi_nested_case_scores,
    load_multi_nested_cases,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.inference.hotword_prompt import (
    DEFAULT_PT_BR_PROMPT_TEMPLATE,
    build_hotword_prompt,
    strict_phrase_match,
)
from qwen_hotword.inference.prompt_smoke import ValidationRecord, load_validation_manifest
from qwen_hotword.inference.retrieved_rag import (
    RetrievedMatch,
    RetrievedRagPrediction,
    RetrievedRagSample,
    _json_bytes,
    _jsonl_bytes,
    _recognition_metrics,
    _reuse_baseline,
    _transcribe_one,
    _word_error_metrics,
    _write_output_group,
)
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

DEFAULT_MULTI_PROMPT_SEED = 20_260_805
DEFAULT_GROUP_QUOTAS: dict[str, int] = {
    "three_independent": 10,
    "nested_family_plus_two": 10,
    "nested_long_present": 8,
    "two_independent": 6,
    "nested_short_only": 3,
    "single_hotword": 3,
    "negative": 10,
}
FORMAL_100_GROUP_QUOTAS: dict[str, int] = {
    group: count * 2 for group, count in DEFAULT_GROUP_QUOTAS.items()
}
SELECTION_PROFILES: dict[str, dict[str, int]] = {
    "smoke50": DEFAULT_GROUP_QUOTAS,
    "formal100": FORMAL_100_GROUP_QUOTAS,
}
OUTPUT_FILENAMES = (
    "sample_selection.json",
    "baseline_predictions.jsonl",
    "retrieved_predictions.jsonl",
    "oracle_predictions.jsonl",
    "multi_nested_prompt_report.json",
)


@dataclass(frozen=True)
class MultiPromptSample:
    rag_sample: RetrievedRagSample
    primary_group: str
    containment_expected_ids: tuple[str, ...]
    redundant_family_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = self.rag_sample.to_dict()
        value["primary_group"] = self.primary_group
        value["containment_expected_ids"] = list(self.containment_expected_ids)
        value["redundant_family_ids"] = list(self.redundant_family_ids)
        return value


def select_multi_nested_prompt_samples(
    records: Mapping[str, ValidationRecord],
    hotwords: Sequence[HotwordEntry],
    cases: Sequence[MultiNestedCase],
    scores: Sequence[CaseScore],
    *,
    seed: int = DEFAULT_MULTI_PROMPT_SEED,
    group_quotas: Mapping[str, int] | None = None,
) -> tuple[MultiPromptSample, ...]:
    quotas = dict(group_quotas or DEFAULT_GROUP_QUOTAS)
    matching_profiles = [
        name for name, expected in SELECTION_PROFILES.items() if quotas == expected
    ]
    if not matching_profiles:
        raise ValueError(
            "multi-nested Prompt selection requires the fixed smoke50 or formal100 quotas"
        )
    expected_total = sum(quotas.values())
    hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
    score_by_id = {score.case_id: score for score in scores}
    if len(hotword_by_id) != len(hotwords) or len(score_by_id) != len(scores):
        raise ValueError("hotword and score IDs must be unique")
    grouped: dict[str, list[tuple[MultiNestedCase, CaseScore]]] = {group: [] for group in quotas}
    for case in cases:
        score = score_by_id.get(case.case_id)
        if score is None or score.sample_id != case.sample_id:
            raise ValueError(f"missing or mismatched CTC score for {case.case_id}")
        record = records.get(case.sample_id)
        if record is None or record.audio_path != case.audio_path:
            raise ValueError(f"case {case.case_id} does not match the validation manifest")
        if case.primary_group not in grouped:
            raise ValueError(f"unsupported v3 primary group: {case.primary_group}")
        grouped[case.primary_group].append((case, score))

    selected: list[MultiPromptSample] = []
    for group, quota in quotas.items():
        candidates = sorted(
            grouped[group],
            key=lambda item: (
                -_wrong_operating_count(item[0], item[1]),
                _stable_rank(seed, f"{group}:{item[0].case_id}"),
            ),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"group {group} has {len(candidates)} cases, fewer than required {quota}"
            )
        for case, score in candidates[:quota]:
            record = records[case.sample_id]
            expected_ids = case.longest_match_expected_ids
            expected_entries = tuple(hotword_by_id[item] for item in expected_ids)
            for entry in expected_entries:
                if not strict_phrase_match(record.reference_text, entry.surface):
                    raise ValueError(
                        f"case {case.case_id} reference does not contain {entry.surface!r}"
                    )
            matches = tuple(_retrieved_match(match) for match in score.operating_matches)
            rag_sample = RetrievedRagSample(
                case_id=case.case_id,
                sample_id=case.sample_id,
                audio_path=case.audio_path,
                reference_text=case.reference_text,
                language=case.language,
                expected_hotword_ids=expected_ids,
                expected_surfaces=tuple(entry.surface for entry in expected_entries),
                selected_matches=matches,
                case_type="negative" if group == "negative" else "positive",
                selection_reason=(
                    f"deterministic_seed={seed}; primary_group={group}; "
                    "prioritize_operating_wrong_candidate_stress"
                ),
            )
            selected.append(
                MultiPromptSample(
                    rag_sample=rag_sample,
                    primary_group=group,
                    containment_expected_ids=case.containment_expected_ids,
                    redundant_family_ids=tuple(
                        item for item in case.containment_expected_ids if item not in expected_ids
                    ),
                )
            )
    if len(selected) != expected_total or len(
        {item.rag_sample.audio_path for item in selected}
    ) != expected_total:
        raise RuntimeError(
            f"Prompt selection must contain {expected_total} audio-disjoint cases"
        )
    return tuple(selected)


def run_multi_nested_prompt_eval(
    *,
    model_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    hotword_table_path: str | Path,
    families_path: str | Path,
    cases_path: str | Path,
    ctc_case_scores_path: str | Path,
    ctc_report_path: str | Path,
    output_dir: str | Path,
    seed: int = DEFAULT_MULTI_PROMPT_SEED,
    selection_profile: str = "smoke50",
    prompt_template: str = DEFAULT_PT_BR_PROMPT_TEMPLATE,
    language: str = "Portuguese",
    dtype: str = "bfloat16",
    device: str = "cuda:0",
    model_loader: Any | None = None,
    print_progress: bool = True,
) -> dict[str, object]:
    model = Path(model_path).expanduser()
    if model.name != EXPECTED_MODEL_NAME or not model.is_dir():
        raise ValueError(f"model path must be an existing {EXPECTED_MODEL_NAME} directory: {model}")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError("dtype must be bfloat16 or float16")
    if selection_profile not in SELECTION_PROFILES:
        raise ValueError(f"unknown multi-nested selection profile: {selection_profile}")
    group_quotas = SELECTION_PROFILES[selection_profile]
    build_hotword_prompt(("probe",), template=prompt_template)
    destination = Path(output_dir).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Prompt output directory must be new and empty")
    output_paths = {name: destination / name for name in OUTPUT_FILENAMES}

    manifest = Path(validation_manifest_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    hotword_file = Path(hotword_table_path).expanduser()
    family_file = Path(families_path).expanduser()
    case_file = Path(cases_path).expanduser()
    score_file = Path(ctc_case_scores_path).expanduser()
    ctc_report_file = Path(ctc_report_path).expanduser()
    _validate_ctc_report(ctc_report_file)
    records = load_validation_manifest(manifest)
    vocab = load_phoneme_vocab(vocab_file)
    hotwords = load_hotword_table(hotword_file, vocab=vocab, blank_id=0)
    families = load_hotword_families(family_file)
    cases = load_multi_nested_cases(case_file)
    scores = load_multi_nested_case_scores(score_file)
    family_ids = {family.family_id for family in families}
    missing_family_ids = {
        family_id for case in cases for family_id in case.nested_family_ids
    } - family_ids
    if missing_family_ids:
        raise ValueError(f"cases reference unknown nested families: {sorted(missing_family_ids)}")
    samples = select_multi_nested_prompt_samples(
        records,
        hotwords,
        cases,
        scores,
        seed=seed,
        group_quotas=group_quotas,
    )
    hotword_by_id = {entry.hotword_id: entry for entry in hotwords}

    config = ModelConfig(
        path=model,
        expected_name=EXPECTED_MODEL_NAME,
        dtype=dtype,
        device=device,
        local_files_only=True,
    )
    wrapper = (model_loader or load_asr_model)(config)
    started = time.monotonic()
    positives = [sample for sample in samples if sample.rag_sample.expected_hotword_ids]
    prompted = [sample for sample in samples if sample.rag_sample.selected_matches]
    total_calls = len(samples) + len(prompted) + len(positives)
    completed = 0

    baseline: list[RetrievedRagPrediction] = []
    baseline_by_case: dict[str, RetrievedRagPrediction] = {}
    for item in samples:
        prediction = _transcribe_one(
            wrapper,
            item.rag_sample,
            mode="baseline",
            injected_ids=(),
            injected_surfaces=(),
            retrieval_matches=(),
            prompt="",
            language=language,
        )
        baseline.append(prediction)
        baseline_by_case[item.rag_sample.case_id] = prediction
        completed += 1
        _progress(print_progress, "baseline", completed, total_calls, started)

    retrieved: list[RetrievedRagPrediction] = []
    for item in samples:
        sample = item.rag_sample
        if not sample.selected_matches:
            retrieved.append(
                _reuse_baseline(baseline_by_case[sample.case_id], mode="retrieved_prompt")
            )
            continue
        injected_ids = tuple(match.hotword_id for match in sample.selected_matches)
        injected_surfaces = tuple(hotword_by_id[item_id].surface for item_id in injected_ids)
        retrieved.append(
            _transcribe_one(
                wrapper,
                sample,
                mode="retrieved_prompt",
                injected_ids=injected_ids,
                injected_surfaces=injected_surfaces,
                retrieval_matches=sample.selected_matches,
                prompt=build_hotword_prompt(injected_surfaces, template=prompt_template),
                language=language,
            )
        )
        completed += 1
        _progress(print_progress, "retrieved", completed, total_calls, started)

    oracle: list[RetrievedRagPrediction] = []
    for item in positives:
        sample = item.rag_sample
        oracle.append(
            _transcribe_one(
                wrapper,
                sample,
                mode="oracle_prompt",
                injected_ids=sample.expected_hotword_ids,
                injected_surfaces=sample.expected_surfaces,
                retrieval_matches=sample.selected_matches,
                prompt=build_hotword_prompt(sample.expected_surfaces, template=prompt_template),
                language=language,
            )
        )
        completed += 1
        _progress(print_progress, "oracle", completed, total_calls, started)

    report = _build_report(
        config=config,
        wrapper=wrapper,
        samples=samples,
        baseline=baseline,
        retrieved=retrieved,
        oracle=oracle,
        hotword_by_id=hotword_by_id,
        prompt_template=prompt_template,
        language=language,
        seed=seed,
        selection_profile=selection_profile,
        group_quotas=group_quotas,
        elapsed=time.monotonic() - started,
        inputs={
            "validation_manifest": manifest,
            "vocab": vocab_file,
            "hotword_table": hotword_file,
            "hotword_families": family_file,
            "cases": case_file,
            "ctc_case_scores": score_file,
            "ctc_report": ctc_report_file,
        },
        output_paths=output_paths,
    )
    selection = {
        "schema_version": 1,
        "evaluation_scope": "validation_multi_nested_prompt_eval",
        "test_set_used": False,
        "seed": seed,
        "selection_profile": selection_profile,
        "target_group_counts": group_quotas,
        "actual_group_counts": {
            group: sum(item.primary_group == group for item in samples)
            for group in group_quotas
        },
        "samples": [item.to_dict() for item in samples],
    }
    _write_output_group(
        {
            output_paths["sample_selection.json"]: _json_bytes(selection),
            output_paths["baseline_predictions.jsonl"]: _jsonl_bytes(baseline),
            output_paths["retrieved_predictions.jsonl"]: _jsonl_bytes(retrieved),
            output_paths["oracle_predictions.jsonl"]: _jsonl_bytes(oracle),
            output_paths["multi_nested_prompt_report.json"]: _json_bytes(report),
        }
    )
    if print_progress:
        print(f"Multi-nested Prompt outputs written: {destination}", flush=True)
    return report


def _build_report(
    *,
    config: ModelConfig,
    wrapper: Any,
    samples: Sequence[MultiPromptSample],
    baseline: Sequence[RetrievedRagPrediction],
    retrieved: Sequence[RetrievedRagPrediction],
    oracle: Sequence[RetrievedRagPrediction],
    hotword_by_id: Mapping[str, HotwordEntry],
    prompt_template: str,
    language: str,
    seed: int,
    selection_profile: str,
    group_quotas: Mapping[str, int],
    elapsed: float,
    inputs: Mapping[str, Path],
    output_paths: Mapping[str, Path],
) -> dict[str, object]:
    sample_by_case = {item.rag_sample.case_id: item for item in samples}
    baseline_by_case = {item.case_id: item for item in baseline}
    retrieved_by_case = {item.case_id: item for item in retrieved}
    oracle_by_case = {item.case_id: item for item in oracle}
    positive_ids = [
        item.rag_sample.case_id for item in samples if item.rag_sample.expected_hotword_ids
    ]
    expected_total = sum(
        len(sample_by_case[case_id].rag_sample.expected_hotword_ids) for case_id in positive_ids
    )
    baseline_positive = [baseline_by_case[item] for item in positive_ids]
    retrieved_positive = [retrieved_by_case[item] for item in positive_ids]
    oracle_positive = [oracle_by_case[item] for item in positive_ids]
    baseline_metrics = _recognition_metrics(baseline_positive, expected_total=expected_total)
    retrieved_metrics = _recognition_metrics(retrieved_positive, expected_total=expected_total)
    oracle_metrics = _recognition_metrics(oracle_positive, expected_total=expected_total)
    baseline_correct = sum(len(item.strict_matched_expected_ids) for item in baseline_positive)
    retrieved_correct = sum(len(item.strict_matched_expected_ids) for item in retrieved_positive)
    oracle_correct = sum(len(item.strict_matched_expected_ids) for item in oracle_positive)

    wrong_injected = wrong_written = newly_wrong_written = 0
    redundant_injected = redundant_written = 0
    rescue_cases: list[dict[str, object]] = []
    wrong_cases: list[dict[str, object]] = []
    for item in samples:
        sample = item.rag_sample
        retrieved_prediction = retrieved_by_case[sample.case_id]
        selected = {match.hotword_id for match in sample.selected_matches}
        expected = set(sample.expected_hotword_ids)
        redundant = set(item.redundant_family_ids)
        wrong = selected - expected - redundant
        matched = set(retrieved_prediction.strict_matched_injected_ids)
        written_wrong = wrong & matched
        baseline_prediction = baseline_by_case[sample.case_id]
        baseline_wrong = {
            hotword_id
            for hotword_id in wrong
            if strict_phrase_match(
                baseline_prediction.prediction, hotword_by_id[hotword_id].surface
            )
        }
        wrong_injected += len(wrong)
        wrong_written += len(written_wrong)
        newly_wrong_written += len(written_wrong - baseline_wrong)
        redundant_injected += len(selected & redundant)
        redundant_written += len(matched & redundant)
        baseline_hits = set(baseline_prediction.strict_matched_expected_ids)
        retrieved_hits = set(retrieved_prediction.strict_matched_expected_ids)
        rescued = (expected - baseline_hits) & retrieved_hits
        if rescued:
            rescue_cases.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "rescued_hotword_ids": sorted(rescued),
                }
            )
        if written_wrong:
            wrong_cases.append(
                {
                    "case_id": sample.case_id,
                    "sample_id": sample.sample_id,
                    "written_wrong_hotword_ids": sorted(written_wrong),
                }
            )

    by_group = {
        group: _group_recognition_metrics(
            [item for item in samples if item.primary_group == group],
            baseline_by_case,
            retrieved_by_case,
            oracle_by_case,
        )
        for group in group_quotas
    }
    baseline_recall = _safe_ratio(baseline_correct, expected_total)
    retrieved_recall = _safe_ratio(retrieved_correct, expected_total)
    oracle_recall = _safe_ratio(oracle_correct, expected_total)
    return {
        "schema_version": 1,
        "status": "pass",
        "purpose": "qwen3_asr_multi_nested_ctc_retrieved_prompt_evaluation",
        "evaluation_scope": "validation_multi_nested_prompt_eval",
        "test_set_used": False,
        "model": {
            "path": str(config.path),
            "expected_name": config.expected_name,
            "dtype": config.dtype,
            "device": config.device,
            "local_files_only": config.local_files_only,
            "wrapper_class": type(wrapper).__name__,
            "backend": getattr(wrapper, "backend", None),
            "max_new_tokens": getattr(wrapper, "max_new_tokens", None),
            "load_count": 1,
            "config": _file_identity(config.path / "config.json"),
            "weight_index": _file_identity(config.path / "model.safetensors.index.json"),
        },
        "qwen_asr_version": _package_version("qwen-asr"),
        "prompt_interface": {
            "method": "Qwen3ASRModel.transcribe",
            "parameter_name": "context",
            "template": prompt_template,
            "language": language,
        },
        "retrieval_config": {
            "threshold": 0.86,
            "top_k": 5,
            "maximum_edit_ratio": 0.35,
            "posterior_weight": 0.25,
            "minimum_posterior_confidence": 0.0,
            "minimum_top1_margin": 0.0,
            "fixed": True,
            "tuning_performed": False,
        },
        "selection": {
            "seed": seed,
            "profile": selection_profile,
            "total_cases": len(samples),
            "positive_cases": len(positive_ids),
            "negative_cases": len(samples) - len(positive_ids),
            "group_counts": {
                group: sum(item.primary_group == group for item in samples)
                for group in group_quotas
            },
            "prompted_cases": sum(bool(item.rag_sample.selected_matches) for item in samples),
            "prompted_positive_cases": sum(
                bool(item.rag_sample.selected_matches)
                and bool(item.rag_sample.expected_hotword_ids)
                for item in samples
            ),
            "prompted_negative_cases": sum(
                bool(item.rag_sample.selected_matches) and not item.rag_sample.expected_hotword_ids
                for item in samples
            ),
        },
        "inputs": {name: _file_identity(path) for name, path in inputs.items()},
        "baseline_metrics": baseline_metrics,
        "retrieved_prompt_metrics": {
            **retrieved_metrics,
            "additional_correct_hotwords_vs_baseline": retrieved_correct - baseline_correct,
            "absolute_recall_improvement": retrieved_recall - baseline_recall,
            "relative_recall_improvement": (
                (retrieved_recall - baseline_recall) / baseline_recall if baseline_recall else None
            ),
        },
        "oracle_prompt_metrics": {
            **oracle_metrics,
            "additional_correct_hotwords_vs_baseline": oracle_correct - baseline_correct,
            "absolute_recall_improvement": oracle_recall - baseline_recall,
        },
        "by_primary_group": by_group,
        "prompt_safety": {
            "wrong_candidates_injected": wrong_injected,
            "wrong_injected_candidates_written": wrong_written,
            "newly_written_wrong_candidates_vs_baseline": newly_wrong_written,
            "wrong_candidate_write_rate": _safe_ratio(wrong_written, wrong_injected),
            "new_wrong_candidate_hallucination_rate": _safe_ratio(
                newly_wrong_written, wrong_injected
            ),
            "wrong_candidate_written_cases": wrong_cases,
            "redundant_family_candidates_injected": redundant_injected,
            "redundant_family_candidates_written": redundant_written,
        },
        "hotword_rescue_cases_vs_baseline": rescue_cases,
        "word_error_diagnostics": {
            "baseline": _word_error_metrics(baseline),
            "retrieved_prompt": _word_error_metrics(retrieved),
            "oracle_prompt_positive_cases": _word_error_metrics(oracle),
        },
        "inference": {
            "model_load_count": 1,
            "baseline_calls": len(baseline),
            "retrieved_prompt_calls": sum(
                bool(item.rag_sample.selected_matches) for item in samples
            ),
            "oracle_calls": len(oracle),
            "total_model_calls": len(baseline)
            + sum(bool(item.rag_sample.selected_matches) for item in samples)
            + len(oracle),
            "wall_seconds": elapsed,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "limitations": [
            f"{len(samples)} deterministic validation cases; not a production benchmark.",
            "Longest-match is the final nested target; contained short hits are redundant.",
            "Fixed threshold and Top-5; no tuning is performed.",
            "No sealed test data are read.",
        ],
        "next_step": "inspect outputs before any threshold or Top-K tuning",
    }


def _group_recognition_metrics(
    samples: Sequence[MultiPromptSample],
    baseline: Mapping[str, RetrievedRagPrediction],
    retrieved: Mapping[str, RetrievedRagPrediction],
    oracle: Mapping[str, RetrievedRagPrediction],
) -> dict[str, object]:
    positives = [item for item in samples if item.rag_sample.expected_hotword_ids]
    expected = sum(len(item.rag_sample.expected_hotword_ids) for item in positives)
    ids = [item.rag_sample.case_id for item in positives]
    return {
        "case_count": len(samples),
        "expected_hotwords": expected,
        "baseline": _recognition_metrics([baseline[item] for item in ids], expected_total=expected),
        "retrieved": _recognition_metrics(
            [retrieved[item] for item in ids], expected_total=expected
        ),
        "oracle": _recognition_metrics([oracle[item] for item in ids], expected_total=expected),
    }


def _validate_ctc_report(path: Path) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("test_set_used") is not False:
        raise ValueError("CTC report is invalid or used sealed test data")
    config = raw.get("scoring_config")
    expected = {
        "threshold": 0.86,
        "top_k": 5,
        "maximum_edit_ratio": 0.35,
        "posterior_weight": 0.25,
        "minimum_posterior_confidence": 0.0,
        "minimum_top1_margin": 0.0,
        "minimum_phonemes": 4,
        "time_axis": "temporal_upsample_2x_only",
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("CTC report does not use the fixed v3 operating configuration")


def _retrieved_match(match: Any) -> RetrievedMatch:
    return RetrievedMatch(
        hotword_id=match.hotword_id,
        surface=match.surface,
        score=match.score,
        edit_ratio=match.edit_ratio,
        posterior_confidence=match.posterior_confidence,
    )


def _wrong_operating_count(case: MultiNestedCase, score: CaseScore) -> int:
    containment = set(case.containment_expected_ids)
    return sum(match.hotword_id not in containment for match in score.operating_matches)


def _stable_rank(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _progress(enabled: bool, phase: str, completed: int, total: int, started: float) -> None:
    if not enabled:
        return
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    print(
        f"multi_prompt phase={phase} completed={completed}/{total} "
        f"elapsed={elapsed:.1f}s rate={rate:.3f}/s eta={eta:.1f}s",
        flush=True,
    )


def _file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256_file(path)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
