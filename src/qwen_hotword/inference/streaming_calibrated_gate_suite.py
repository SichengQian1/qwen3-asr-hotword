from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class CalibratedGateProfile:
    name: str
    output_subdir: str
    threshold: float
    top_k: int
    minimum_posterior_confidence: float


CALIBRATED_GATE_PROFILES = (
    CalibratedGateProfile(
        "precision_guarded_top7",
        "precision_guarded_top7",
        0.86,
        7,
        0.0,
    ),
    CalibratedGateProfile(
        "f1_with_fpr_guard_top7",
        "f1_with_fpr_guard_top7",
        0.83,
        7,
        0.0,
    ),
)

BASELINE_PROFILE = "conservative"
BASELINE_GATE = {
    "threshold": 0.86,
    "top_k": 5,
    "maximum_edit_ratio": 0.35,
    "posterior_weight": 0.25,
    "minimum_posterior_confidence": 0.0,
    "minimum_top1_margin": 0.0,
}
QUALITY_FIELDS = (
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
OUTPUT_FILES = (
    "suite_config.json",
    "calibrated_gate_summary.json",
    "calibrated_gate_cases.jsonl",
    "README.md",
)


def calibrated_suite_resume_config_matches(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> bool:
    normalized_previous = dict(previous)
    normalized_current = dict(current)
    normalized_previous["status"] = "running"
    normalized_current["status"] = "running"
    return normalized_previous == normalized_current


def validate_calibrated_gate_preflight(
    *,
    baseline_suite_dir: str | Path,
    calibration_summary_path: str | Path,
    ctc_report_path: str | Path,
    ctc_checkpoint_path: str | Path,
) -> dict[str, object]:
    baseline_root = Path(baseline_suite_dir).expanduser()
    calibration_path = Path(calibration_summary_path).expanduser()
    report_path = Path(ctc_report_path).expanduser()
    checkpoint_path = Path(ctc_checkpoint_path).expanduser()
    _verify_sha256_manifest(baseline_root / "sha256.txt")
    baseline_subdir = _baseline_subdir(baseline_root)
    _verify_sha256_manifest(baseline_root / baseline_subdir / "sha256.txt")
    _verify_sha256_manifest(calibration_path.parent / "sha256.txt")
    _verify_sha256_manifest(report_path.parent / "sha256.txt")

    calibration = _read_mapping(calibration_path)
    _validate_calibration_candidates(calibration)
    checkpoint_sha = _sha256(checkpoint_path)
    baseline_run_config = _read_mapping(baseline_root / baseline_subdir / "run_config.json")
    baseline_checkpoint_sha = _checkpoint_sha(baseline_run_config)
    if checkpoint_sha != baseline_checkpoint_sha:
        raise ValueError("requested checkpoint differs from the conservative D5 baseline")
    report = _read_mapping(report_path)
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("current CTC report is not bound to the requested checkpoint")

    calibration_config_path = calibration_path.parent / "calibration_config.json"
    calibration_config = _read_mapping(calibration_config_path)
    if calibration_config.get("candidate_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("calibration is not bound to the requested checkpoint")
    return {
        "baseline_subdir": baseline_subdir,
        "checkpoint_sha256": checkpoint_sha,
        "offline_selection_report_sha256": _offline_report_sha(baseline_run_config),
        "current_ctc_report_sha256": _sha256(report_path),
        "calibration_summary_sha256": _sha256(calibration_path),
        "calibration_config_sha256": _sha256(calibration_config_path),
    }


def build_calibrated_gate_suite_report(
    output_dir: str | Path,
    baseline_suite_dir: str | Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    root = Path(output_dir).expanduser()
    baseline_root = Path(baseline_suite_dir).expanduser()
    suite_config = _read_mapping(root / "suite_config.json")
    if suite_config.get("status") != "pass" or suite_config.get("test_set_used") is not False:
        raise ValueError("calibrated suite config is not a passed non-test run")
    _verify_sha256_manifest(baseline_root / "sha256.txt")
    baseline_subdir = _baseline_subdir(baseline_root)
    baseline_run = baseline_root / baseline_subdir
    _verify_sha256_manifest(baseline_run / "sha256.txt")
    for profile in CALIBRATED_GATE_PROFILES:
        _verify_sha256_manifest(root / profile.output_subdir / "sha256.txt")

    baseline_summary = _read_mapping(baseline_root / "suite_summary.json")
    baseline_profile = _profile(_profiles(baseline_summary), BASELINE_PROFILE)
    if baseline_profile.get("gate") != BASELINE_GATE:
        raise ValueError("baseline suite conservative profile is not sealed D5 0.86/posterior 0")
    baseline_quality_raw = _quality(baseline_profile)
    baseline_rows = _group_d_rows(baseline_run / "sample_results.jsonl")
    baseline_quality = _with_prompt_hotword_recall(baseline_quality_raw, baseline_rows)
    baseline_run_config = _read_mapping(baseline_run / "run_config.json")
    checkpoint_sha = _checkpoint_sha(baseline_run_config)
    baseline_selection = _selection(baseline_rows)

    candidates: dict[str, object] = {}
    changed_rows: list[dict[str, object]] = []
    offline_report_shas: set[str] = set()
    for profile in CALIBRATED_GATE_PROFILES:
        run = root / profile.output_subdir
        run_config = _read_mapping(run / "run_config.json")
        _validate_child_identity(
            baseline_run_config,
            run_config,
            expected_checkpoint_sha=checkpoint_sha,
        )
        offline_report_shas.add(_offline_report_sha(run_config))
        rows = _group_d_rows(run / "sample_results.jsonl")
        if _selection(rows) != baseline_selection:
            raise ValueError(f"profile {profile.name} uses a different D sample selection")
        summary = _read_mapping(run / "summary.json")
        latency = _read_mapping(run / "latency_summary.json")
        quality = _group_mapping(summary, "groups", "D", profile.name)
        timing = _group_mapping(latency, "groups", "D", profile.name)
        enriched_quality = _with_prompt_hotword_recall(quality, rows)
        expected_gate = _profile_gate(profile)
        _validate_run_gate(run_config, expected_gate, profile.name)
        candidates[profile.name] = {
            "source_output_dir": str(run),
            "source_summary_sha256": _sha256(run / "summary.json"),
            "source_latency_sha256": _sha256(run / "latency_summary.json"),
            "gate": expected_gate,
            "quality": _selected_quality(enriched_quality),
            "delta_from_d5_baseline": _quality_delta(enriched_quality, baseline_quality),
            "latency": dict(timing),
        }
        changed_rows.extend(
            _case_differences(
                baseline_rows,
                rows,
                comparison=profile.name + "_vs_conservative_d5",
            )
        )

    if len(offline_report_shas) != 1:
        raise ValueError("candidate runs use different offline selection report identities")
    offline_report_sha = next(iter(offline_report_shas))
    baseline_offline_report_sha = _offline_report_sha(baseline_run_config)
    if offline_report_sha != baseline_offline_report_sha:
        raise ValueError("candidate runs use a different offline selection report")
    preflight = suite_config.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("calibrated suite config has no preflight identity")
    if preflight.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("suite preflight checkpoint identity differs from child runs")
    current_ctc_report_sha = preflight.get("current_ctc_report_sha256")
    if not isinstance(current_ctc_report_sha, str) or len(current_ctc_report_sha) != 64:
        raise ValueError("suite preflight has no valid current CTC report identity")

    sample_count = len(baseline_selection)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "evaluation": "streaming_4k_multilingual_ctc_calibrated_d_only",
        "sample_count": sample_count,
        "baseline": {
            "profile": BASELINE_PROFILE,
            "source_suite_dir": str(baseline_root),
            "source_output_dir": str(baseline_run),
            "suite_summary_sha256": _sha256(baseline_root / "suite_summary.json"),
            "gate": BASELINE_GATE,
            "quality": _selected_quality(baseline_quality),
        },
        "candidates": candidates,
        "identity_checks": {
            "source_sha256_manifests_verified": True,
            "same_ctc_checkpoint_as_d5_baseline": True,
            "ctc_checkpoint_sha256": checkpoint_sha,
            "same_non_gate_child_run_config": True,
            "same_d_sample_selection": True,
            "same_offline_selection_report": True,
            "offline_selection_report_sha256": offline_report_sha,
            "full_rank_ctc_report_bound_to_checkpoint": True,
            "current_full_rank_ctc_report_sha256": current_ctc_report_sha,
            "calibration_summary_sha256": preflight.get("calibration_summary_sha256"),
            "calibration_config_sha256": preflight.get("calibration_config_sha256"),
        },
        "comparisons": {
            "topk_isolation": {
                "baseline": "conservative",
                "candidate": "precision_guarded_top7",
                "only_gate_change": "top_k_5_to_7",
                "quality_delta_candidate_minus_baseline": cast(
                    Mapping[str, object], candidates["precision_guarded_top7"]
                )["delta_from_d5_baseline"],
            },
            "recall_candidate_vs_d5": {
                "baseline": "conservative",
                "candidate": "f1_with_fpr_guard_top7",
                "gate_changes": ["threshold_0.86_to_0.83", "top_k_5_to_7"],
                "quality_delta_candidate_minus_baseline": cast(
                    Mapping[str, object], candidates["f1_with_fpr_guard_top7"]
                )["delta_from_d5_baseline"],
            },
        },
        "case_comparison": {
            "changed_profile_case_rows": len(changed_rows),
            "details_file": "calibrated_gate_cases.jsonl",
        },
        "latency_compared": False,
        "latency_note": (
            "Latency is preserved for diagnostics but separate H200 runs are not treated "
            "as a controlled latency comparison."
        ),
    }
    return result, changed_rows


def write_calibrated_gate_suite_report(
    output_dir: str | Path,
    baseline_suite_dir: str | Path,
) -> dict[str, object]:
    root = Path(output_dir).expanduser()
    report, changed_rows = build_calibrated_gate_suite_report(root, baseline_suite_dir)
    _write_json(root / "calibrated_gate_summary.json", report)
    _write_jsonl(root / "calibrated_gate_cases.jsonl", changed_rows)
    (root / "README.md").write_text(
        "# Calibrated D-only streaming gate suite\n\n"
        "This additive suite keeps the multilingual CTC checkpoint and formal100 D "
        "selection fixed. It compares the existing conservative 0.86/posterior 0/Top-5 "
        "baseline with exact full-rank calibration candidates at Top-7. It does not rerun "
        "C no-RAG or E Oracle and does not use the sealed test split.\n",
        encoding="utf-8",
    )
    _write_hashes(root)
    return report


def _validate_calibration_candidates(calibration: Mapping[str, Any]) -> None:
    if calibration.get("status") != "guarded_recall_gain_candidate_available":
        raise ValueError("calibration does not expose a guarded recall-gain candidate")
    if calibration.get("ranked_matches_complete") is not True:
        raise ValueError("calibration ranking is not complete")
    if calibration.get("non_exact_point_count") != 0:
        raise ValueError("calibration contains non-exact sweep points")
    raw_candidates = calibration.get("recommended_candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("calibration recommended_candidates is invalid")
    by_role = {
        str(value.get("role")): value for value in raw_candidates if isinstance(value, Mapping)
    }
    expected = {
        "precision_guarded": CALIBRATED_GATE_PROFILES[0],
        "f1_with_fpr_guard": CALIBRATED_GATE_PROFILES[1],
    }
    for role, profile in expected.items():
        candidate = by_role.get(role)
        if not isinstance(candidate, Mapping):
            raise ValueError(f"calibration has no {role} candidate")
        config = candidate.get("config")
        if not isinstance(config, Mapping):
            raise ValueError(f"calibration {role} config is invalid")
        expected_config = {
            "threshold": profile.threshold,
            "top_k": profile.top_k,
            "maximum_edit_ratio": 0.35,
            "minimum_posterior_confidence": profile.minimum_posterior_confidence,
            "minimum_top1_margin": 0.0,
        }
        if dict(config) != expected_config:
            raise ValueError(f"calibration {role} candidate differs from the sealed choice")


def _baseline_subdir(root: Path) -> str:
    config = _read_mapping(root / "suite_config.json")
    if config.get("status") != "pass" or config.get("test_set_used") is not False:
        raise ValueError("baseline suite is not a passed non-test run")
    profiles = config.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("baseline suite profiles are invalid")
    for profile in profiles:
        if isinstance(profile, Mapping) and profile.get("name") == BASELINE_PROFILE:
            subdir = profile.get("output_subdir")
            if isinstance(subdir, str) and subdir:
                return subdir
    raise ValueError("baseline suite has no conservative profile")


def _validate_child_identity(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_checkpoint_sha: str,
) -> None:
    if _checkpoint_sha(candidate) != expected_checkpoint_sha:
        raise ValueError("candidate child checkpoint differs from the D5 baseline")
    if _without_treatment(baseline) != _without_treatment(candidate):
        raise ValueError("candidate child differs from D5 baseline beyond gate/report/git/GPU")


def _without_treatment(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = cast(dict[str, Any], json.loads(json.dumps(value)))
    normalized.pop("git_commit", None)
    normalized.pop("gpu_memory_utilization", None)
    normalized.pop("threshold", None)
    normalized.pop("top_k", None)
    offline_control = normalized.get("offline_control")
    if not isinstance(offline_control, dict) or "report" not in offline_control:
        raise ValueError("child run config has no checkpoint-bound offline report")
    offline_control.pop("current_retrieval_config_not_compared", None)
    return normalized


def _validate_run_gate(
    config: Mapping[str, Any],
    expected: Mapping[str, object],
    profile_name: str,
) -> None:
    actual = {
        "threshold": config.get("threshold"),
        "top_k": config.get("top_k"),
        "maximum_edit_ratio": config.get("maximum_edit_ratio"),
        "posterior_weight": config.get("posterior_weight"),
        "minimum_posterior_confidence": config.get("minimum_posterior_confidence"),
        "minimum_top1_margin": config.get("minimum_top1_margin"),
    }
    if actual != expected:
        raise ValueError(f"profile {profile_name} run gate differs from the sealed choice")


def _profile_gate(profile: CalibratedGateProfile) -> dict[str, object]:
    return {
        "threshold": profile.threshold,
        "top_k": profile.top_k,
        "maximum_edit_ratio": 0.35,
        "posterior_weight": 0.25,
        "minimum_posterior_confidence": profile.minimum_posterior_confidence,
        "minimum_top1_margin": 0.0,
    }


def _checkpoint_sha(config: Mapping[str, Any]) -> str:
    inputs = config.get("inputs")
    checkpoint = inputs.get("checkpoint") if isinstance(inputs, Mapping) else None
    sha = checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("child run has no valid checkpoint SHA256")
    return sha


def _offline_report_sha(config: Mapping[str, Any]) -> str:
    control = config.get("offline_control")
    report = control.get("report") if isinstance(control, Mapping) else None
    sha = report.get("sha256") if isinstance(report, Mapping) else None
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("child run has no valid offline report SHA256")
    return sha


def _profiles(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    profiles = summary.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("suite summary profiles are invalid")
    return profiles


def _profile(profiles: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    profile = profiles.get(name)
    if not isinstance(profile, Mapping):
        raise ValueError(f"suite summary has no valid {name} profile")
    return profile


def _quality(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    quality = profile.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError("suite profile quality is invalid")
    return quality


def _group_mapping(
    value: Mapping[str, Any],
    collection: str,
    group: str,
    profile_name: str,
) -> Mapping[str, Any]:
    groups = value.get(collection)
    selected = groups.get(group) if isinstance(groups, Mapping) else None
    if not isinstance(selected, Mapping):
        raise ValueError(f"profile {profile_name} has no valid group {group} summary")
    return selected


def _group_d_rows(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in _read_jsonl(path) if row.get("experiment_group") == "D"]
    if not rows:
        raise ValueError(f"sample results contain no D rows: {path}")
    return rows


def _selection(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(row.get("case_id")),
                str(row.get("sample_id")),
                str(row.get("reference_text")),
            )
            for row in rows
        )
    )


def _selected_quality(quality: Mapping[str, Any]) -> dict[str, object]:
    return {name: quality.get(name) for name in QUALITY_FIELDS}


def _quality_delta(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        name: _numeric_delta(candidate.get(name), baseline.get(name)) for name in QUALITY_FIELDS
    }


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
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    enriched = dict(quality)
    expected_total = sum(
        len(expected)
        for row in rows
        if isinstance((expected := row.get("expected_hotword_ids")), list)
    )
    correct_injected = quality.get("correct_prompt_injected_hotwords")
    if not isinstance(correct_injected, int) or isinstance(correct_injected, bool):
        correct_injected = sum(
            len(
                set(_string_list(row.get("expected_hotword_ids"))).intersection(
                    _sample_injected_hotword_ids(row)
                )
            )
            for row in rows
        )
    enriched["expected_hotwords"] = expected_total
    enriched["prompt_hotword_recall"] = (
        correct_injected / expected_total if expected_total else None
    )
    return enriched


def _case_differences(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    comparison: str,
) -> list[dict[str, object]]:
    baseline_by_case = {str(row.get("case_id")): row for row in baseline_rows}
    candidate_by_case = {str(row.get("case_id")): row for row in candidate_rows}
    if baseline_by_case.keys() != candidate_by_case.keys():
        raise ValueError(f"{comparison} uses different case IDs")
    differences: list[dict[str, object]] = []
    for case_id in sorted(baseline_by_case):
        baseline = baseline_by_case[case_id]
        candidate = candidate_by_case[case_id]
        baseline_injected = _sample_injected_hotword_ids(baseline)
        candidate_injected = _sample_injected_hotword_ids(candidate)
        baseline_prediction = str(baseline.get("prediction", ""))
        candidate_prediction = str(candidate.get("prediction", ""))
        if baseline_injected == candidate_injected and baseline_prediction == candidate_prediction:
            continue
        expected = _string_list(baseline.get("expected_hotword_ids"))
        differences.append(
            {
                "comparison": comparison,
                "case_id": case_id,
                "sample_id": str(baseline.get("sample_id", "")),
                "primary_group": baseline.get("primary_group"),
                "expected_hotword_ids": expected,
                "baseline": {
                    "injected_hotword_ids": baseline_injected,
                    "prediction": baseline_prediction,
                },
                "candidate": {
                    "injected_hotword_ids": candidate_injected,
                    "prediction": candidate_prediction,
                },
                "added_injected_hotword_ids": sorted(
                    set(candidate_injected).difference(baseline_injected)
                ),
                "removed_injected_hotword_ids": sorted(
                    set(baseline_injected).difference(candidate_injected)
                ),
                "prediction_changed": baseline_prediction != candidate_prediction,
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
    return [str(item) for item in value] if isinstance(value, list) else []


def _verify_sha256_manifest(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"SHA256 manifest does not exist: {path}")
    entries = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, separator, raw_name = line.partition("  ")
        if not separator or len(digest) != 64 or not raw_name:
            raise ValueError(f"invalid SHA256 manifest row: {path}:{line_number}")
        local_target = path.parent / raw_name
        repository_target = Path(raw_name).expanduser()
        target = local_target if local_target.is_file() else repository_target
        if not target.is_file():
            raise FileNotFoundError(f"SHA256 target does not exist for {path}: {raw_name}")
        if _sha256(target) != digest:
            raise ValueError(f"SHA256 mismatch for {target} listed by {path}")
        entries += 1
    if entries == 0:
        raise ValueError(f"SHA256 manifest is empty: {path}")


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON input does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_hashes(root: Path) -> None:
    (root / "sha256.txt").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in OUTPUT_FILES),
        encoding="utf-8",
    )


def profile_dicts() -> list[dict[str, object]]:
    return [asdict(profile) for profile in CALIBRATED_GATE_PROFILES]
