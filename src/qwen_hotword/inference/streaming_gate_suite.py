from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StreamingGateProfile:
    name: str
    output_subdir: str
    group: str
    threshold: float
    top_k: int
    minimum_posterior_confidence: float


STREAMING_GATE_PROFILES = (
    StreamingGateProfile("no_rag", "baseline_oracle", "C", 0.86, 5, 0.0),
    StreamingGateProfile("conservative", "conservative", "D", 0.86, 5, 0.0),
    StreamingGateProfile("balanced", "balanced", "D", 0.82, 7, 0.5),
    StreamingGateProfile("recall_first_top5", "recall_first_top5", "D", 0.75, 5, 0.5),
    StreamingGateProfile("recall_first", "recall_first", "D", 0.75, 7, 0.5),
    StreamingGateProfile("oracle", "baseline_oracle", "E", 0.86, 5, 0.0),
)


def suite_resume_config_matches(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    """Accept an identical resume or the additive five-to-six-profile upgrade."""
    normalized_previous = dict(previous)
    normalized_current = dict(current)
    normalized_previous["status"] = "running"
    normalized_current["status"] = "running"
    if normalized_previous == normalized_current:
        return True

    previous_profiles = normalized_previous.pop("profiles", None)
    current_profiles = normalized_current.pop("profiles", None)
    if normalized_previous != normalized_current or not isinstance(current_profiles, list):
        return False
    legacy_profiles = [
        profile
        for profile in current_profiles
        if isinstance(profile, Mapping) and profile.get("name") != "recall_first_top5"
    ]
    return bool(previous_profiles == legacy_profiles)


def completed_profile_run(output_dir: str | Path, groups: tuple[str, ...]) -> bool:
    root = Path(output_dir).expanduser()
    required = (
        "run_config.json",
        "summary.json",
        "latency_summary.json",
        "sample_results.jsonl",
        "sha256.txt",
    )
    if not all((root / name).is_file() for name in required):
        return False
    try:
        summary = _read_mapping(root / "summary.json")
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    quality_groups = summary.get("groups")
    return (
        summary.get("status") == "pass"
        and isinstance(quality_groups, Mapping)
        and all(group in quality_groups for group in groups)
    )


def build_streaming_gate_suite_report(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir).expanduser()
    profiles: dict[str, dict[str, object]] = {}
    rows_by_profile: dict[str, list[dict[str, Any]]] = {}
    expected_selection: tuple[tuple[str, str, str], ...] | None = None
    for profile in STREAMING_GATE_PROFILES:
        run_dir = root / profile.output_subdir
        summary = _read_mapping(run_dir / "summary.json")
        latency = _read_mapping(run_dir / "latency_summary.json")
        sample_rows = _read_jsonl(run_dir / "sample_results.jsonl")
        selected_rows = [
            row for row in sample_rows if str(row.get("experiment_group")) == profile.group
        ]
        selection = tuple(
            sorted(
                (
                    str(row.get("case_id")),
                    str(row.get("sample_id")),
                    str(row.get("reference_text")),
                )
                for row in selected_rows
            )
        )
        if not selection:
            raise ValueError(f"profile {profile.name} has no group {profile.group} rows")
        if expected_selection is None:
            expected_selection = selection
        elif selection != expected_selection:
            raise ValueError(f"profile {profile.name} uses a different sample selection")
        quality_groups = summary.get("groups")
        latency_groups = latency.get("groups")
        if not isinstance(quality_groups, Mapping) or not isinstance(latency_groups, Mapping):
            raise ValueError(f"profile {profile.name} has invalid summary structure")
        quality = quality_groups.get(profile.group)
        timing = latency_groups.get(profile.group)
        if not isinstance(quality, Mapping) or not isinstance(timing, Mapping):
            raise ValueError(f"profile {profile.name} misses group {profile.group}")
        quality_with_prompt_recall = _with_prompt_hotword_recall(quality, selected_rows)
        rows_by_profile[profile.name] = selected_rows
        profiles[profile.name] = {
            "source_group": profile.group,
            "source_output_dir": str(run_dir),
            "source_summary_sha256": _sha256(run_dir / "summary.json"),
            "source_latency_sha256": _sha256(run_dir / "latency_summary.json"),
            "gate": (
                None
                if profile.group in {"C", "E"}
                else {
                    "threshold": profile.threshold,
                    "top_k": profile.top_k,
                    "maximum_edit_ratio": 0.35,
                    "posterior_weight": 0.25,
                    "minimum_posterior_confidence": (
                        profile.minimum_posterior_confidence
                    ),
                    "minimum_top1_margin": 0.0,
                }
            ),
            "quality": quality_with_prompt_recall,
            "latency": dict(timing),
        }
    baseline = profiles["no_rag"]["quality"]
    assert isinstance(baseline, Mapping)
    comparisons = {
        name: _quality_delta(profile["quality"], baseline)
        for name, profile in profiles.items()
        if name != "no_rag"
    }
    topk_cases = _topk_isolation_cases(rows_by_profile)
    return {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "evaluation": "streaming_4k_gate_prompt_filter_suite",
        "conceptual_groups": [profile.name for profile in STREAMING_GATE_PROFILES],
        "sample_count": len(expected_selection or ()),
        "profiles": profiles,
        "comparisons_vs_no_rag": comparisons,
        "topk_isolation": _topk_isolation_summary(profiles, topk_cases),
        "latency_scope": {
            "sample_inference_seconds": (
                "actual detector plus prompt refresh plus Qwen streaming wall clock; "
                "audio file loading excluded"
            ),
            "retrieval_seconds": (
                "actual CTC greedy decode plus Anchor query plus shortlist rerank; "
                "CTC encoder and Head excluded and reported separately"
            ),
            "step_total_seconds": "actual per-chunk detector plus prompt plus Qwen wall clock",
        },
    }


def write_streaming_gate_suite_report(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir).expanduser()
    report = build_streaming_gate_suite_report(root)
    rows_by_profile = {
        name: [
            row
            for row in _read_jsonl(root / profile.output_subdir / "sample_results.jsonl")
            if str(row.get("experiment_group")) == profile.group
        ]
        for name, profile in ((profile.name, profile) for profile in STREAMING_GATE_PROFILES)
        if name in {"recall_first_top5", "recall_first"}
    }
    topk_cases = _topk_isolation_cases(rows_by_profile)
    report_path = root / "suite_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "topk_isolation_summary.json").write_text(
        json.dumps(
            report["topk_isolation"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (root / "topk_isolation_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in topk_cases:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (root / "README.md").write_text(_readme(report), encoding="utf-8")
    files = (
        "suite_config.json",
        "suite_summary.json",
        "topk_isolation_summary.json",
        "topk_isolation_cases.jsonl",
        "README.md",
    )
    with (root / "sha256.txt").open("w", encoding="utf-8") as handle:
        for name in files:
            path = root / name
            if path.is_file():
                handle.write(f"{_sha256(path)}  {name}\n")
    return report


def _topk_isolation_summary(
    profiles: Mapping[str, Mapping[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    top5 = profiles.get("recall_first_top5")
    top7 = profiles.get("recall_first")
    if not isinstance(top5, Mapping) or not isinstance(top7, Mapping):
        raise ValueError("Top-K isolation profiles are missing")
    top5_gate = top5.get("gate")
    top7_gate = top7.get("gate")
    if not isinstance(top5_gate, Mapping) or not isinstance(top7_gate, Mapping):
        raise ValueError("Top-K isolation gates are missing")
    top5_invariants = {key: value for key, value in top5_gate.items() if key != "top_k"}
    top7_invariants = {key: value for key, value in top7_gate.items() if key != "top_k"}
    if top5_invariants != top7_invariants:
        raise ValueError("Top-K isolation profiles differ beyond top_k")
    if top5_gate.get("top_k") != 5 or top7_gate.get("top_k") != 7:
        raise ValueError("Top-K isolation profiles must compare Top-5 with Top-7")

    quality_fields = (
        "expected_hotwords",
        "correct_prompt_injected_hotwords",
        "prompt_hotword_recall",
        "correct_prompt_adopted_hotwords",
        "correct_prompt_adoption_rate",
        "wrong_injected_hotwords",
        "wrong_prompt_written_hotwords",
        "wrong_prompt_landing_rate",
        "final_hotword_recall",
        "final_hotword_precision",
        "sample_hotword_hit_rate",
        "wer",
        "cer",
        "negative_hotword_hallucination_rate",
        "mean_inference_seconds",
    )
    top5_quality = top5.get("quality")
    top7_quality = top7.get("quality")
    if not isinstance(top5_quality, Mapping) or not isinstance(top7_quality, Mapping):
        raise ValueError("Top-K isolation quality summaries are missing")
    selected_top5 = {name: top5_quality.get(name) for name in quality_fields}
    selected_top7 = {name: top7_quality.get(name) for name in quality_fields}
    return {
        "status": "pass",
        "comparison": "recall_first_top5_vs_recall_first_top7",
        "only_changed_parameter": "top_k",
        "shared_gate": top5_invariants,
        "top5": {
            "profile": "recall_first_top5",
            "top_k": 5,
            "quality": selected_top5,
            "latency": top5.get("latency"),
        },
        "top7": {
            "profile": "recall_first",
            "top_k": 7,
            "quality": selected_top7,
            "latency": top7.get("latency"),
        },
        "quality_delta_top5_minus_top7": {
            name: _numeric_delta(selected_top5.get(name), selected_top7.get(name))
            for name in quality_fields
        },
        "sample_comparison": {
            "changed_case_count": len(cases),
            "details_file": "topk_isolation_cases.jsonl",
        },
    }


def _topk_isolation_cases(
    rows_by_profile: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, object]]:
    top5_rows = rows_by_profile.get("recall_first_top5")
    top7_rows = rows_by_profile.get("recall_first")
    if top5_rows is None or top7_rows is None:
        raise ValueError("Top-K isolation sample rows are missing")
    top5_by_case = {str(row.get("case_id")): row for row in top5_rows}
    top7_by_case = {str(row.get("case_id")): row for row in top7_rows}
    if top5_by_case.keys() != top7_by_case.keys():
        raise ValueError("Top-K isolation profiles use different case IDs")

    differences = []
    for case_id in sorted(top5_by_case):
        top5 = top5_by_case[case_id]
        top7 = top7_by_case[case_id]
        expected = _string_list(top5.get("expected_hotword_ids"))
        top5_injected = _sample_injected_hotword_ids(top5)
        top7_injected = _sample_injected_hotword_ids(top7)
        top5_prediction = str(top5.get("prediction", ""))
        top7_prediction = str(top7.get("prediction", ""))
        if top5_injected == top7_injected and top5_prediction == top7_prediction:
            continue
        expected_set = set(expected)
        differences.append(
            {
                "case_id": case_id,
                "sample_id": str(top5.get("sample_id", "")),
                "primary_group": top5.get("primary_group"),
                "expected_hotword_ids": expected,
                "top5": {
                    "injected_hotword_ids": top5_injected,
                    "correct_injected_hotword_ids": sorted(
                        expected_set.intersection(top5_injected)
                    ),
                    "wrong_injected_hotword_ids": sorted(
                        set(top5_injected).difference(expected_set)
                    ),
                    "prediction": top5_prediction,
                },
                "top7": {
                    "injected_hotword_ids": top7_injected,
                    "correct_injected_hotword_ids": sorted(
                        expected_set.intersection(top7_injected)
                    ),
                    "wrong_injected_hotword_ids": sorted(
                        set(top7_injected).difference(expected_set)
                    ),
                    "prediction": top7_prediction,
                },
                "additional_top7_injected_hotword_ids": sorted(
                    set(top7_injected).difference(top5_injected)
                ),
                "prediction_changed": top5_prediction != top7_prediction,
            }
        )
    return differences


def _sample_injected_hotword_ids(row: Mapping[str, Any]) -> list[str]:
    direct = row.get("injected_hotword_ids")
    if isinstance(direct, list):
        return sorted(set(_string_list(direct)))
    candidates = row.get("injected_candidates")
    if not isinstance(candidates, list):
        return []
    return sorted(
        {
            str(candidate.get("hotword_id"))
            for candidate in candidates
            if isinstance(candidate, Mapping) and candidate.get("hotword_id") is not None
        }
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _quality_delta(current: object, baseline: Mapping[str, Any]) -> dict[str, float | None]:
    if not isinstance(current, Mapping):
        raise ValueError("profile quality is not a mapping")
    names = (
        "hotword_exact_recall",
        "prompt_hotword_recall",
        "final_hotword_recall",
        "final_hotword_precision",
        "correct_prompt_adoption_rate",
        "wrong_prompt_filter_rate",
        "wrong_prompt_landing_rate",
        "sample_hotword_hit_rate",
        "wer",
        "cer",
        "negative_hotword_hallucination_rate",
        "mean_inference_seconds",
    )
    return {name: _numeric_delta(current.get(name), baseline.get(name)) for name in names}


def _numeric_delta(left: object, right: object) -> float | None:
    if (
        isinstance(left, int | float)
        and not isinstance(left, bool)
        and isinstance(right, int | float)
        and not isinstance(right, bool)
    ):
        return float(left) - float(right)
    return None


def _with_prompt_hotword_recall(
    quality: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, object]:
    enriched = dict(quality)
    if "expected_hotwords" in enriched and "prompt_hotword_recall" in enriched:
        return enriched
    expected_total = sum(
        len(expected)
        for row in rows
        if isinstance((expected := row.get("expected_hotword_ids")), list)
    )
    correct_injected = enriched.get("correct_prompt_injected_hotwords")
    if not isinstance(correct_injected, int) or isinstance(correct_injected, bool):
        correct_injected = 0
        for row in rows:
            expected = row.get("expected_hotword_ids")
            candidates = row.get("injected_candidates")
            if not isinstance(expected, list) or not isinstance(candidates, list):
                continue
            expected_ids = {str(value) for value in expected}
            correct_injected += sum(
                isinstance(candidate, Mapping)
                and str(candidate.get("hotword_id", "")) in expected_ids
                for candidate in candidates
            )
    enriched["expected_hotwords"] = expected_total
    enriched["prompt_hotword_recall"] = (
        correct_injected / expected_total if expected_total else None
    )
    return enriched


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"suite input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"suite input is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"suite input does not exist: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid JSONL object at {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readme(report: Mapping[str, object]) -> str:
    return (
        "# Streaming 4k gate prompt-filter suite\n\n"
        "This suite compares no RAG, conservative Top-5, balanced Top-7, recall-first "
        "Top-5, recall-first Top-7, and Oracle streaming inference on the same sealed "
        "validation selection. Retrieval, CTC, Prompt/Qwen, per-step, per-sample, and "
        "real-time-factor latency are measured separately. Audio loading is outside "
        "sample inference timing.\n\n"
        f"Samples: {report['sample_count']}\n"
    )
