from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from qwen_hotword.inference.streaming_gate_suite import STREAMING_GATE_PROFILES

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
)
CONTROL_PROFILES = ("no_rag", "oracle")
OUTPUT_FILES = (
    "checkpoint_regression_summary.json",
    "checkpoint_regression_cases.jsonl",
    "README.md",
    "sha256.txt",
)


def compare_streaming_checkpoint_suites(
    baseline_suite_dir: str | Path,
    candidate_suite_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Compare two sealed formal100 suites whose only input change is the CTC Head."""

    baseline_root = Path(baseline_suite_dir).expanduser()
    candidate_root = Path(candidate_suite_dir).expanduser()
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {destination}"
        )
    for root in (baseline_root, candidate_root):
        if not root.is_dir():
            raise FileNotFoundError(f"streaming suite directory does not exist: {root}")
        _verify_sha256_manifest(root / "sha256.txt")
        for subdir in sorted({profile.output_subdir for profile in STREAMING_GATE_PROFILES}):
            _verify_sha256_manifest(root / subdir / "sha256.txt")

    baseline_config = _read_mapping(baseline_root / "suite_config.json")
    candidate_config = _read_mapping(candidate_root / "suite_config.json")
    _validate_suite_config_pair(baseline_config, candidate_config)
    baseline_summary = _read_mapping(baseline_root / "suite_summary.json")
    candidate_summary = _read_mapping(candidate_root / "suite_summary.json")
    _validate_suite_summary_pair(baseline_summary, candidate_summary)

    profile_rows: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    baseline_checkpoint_sha: str | None = None
    candidate_checkpoint_sha: str | None = None
    checked_run_dirs: set[str] = set()
    gpu_allocations: dict[str, dict[str, float | None]] = {}
    for profile in STREAMING_GATE_PROFILES:
        baseline_run = baseline_root / profile.output_subdir
        candidate_run = candidate_root / profile.output_subdir
        baseline_run_config = _read_mapping(baseline_run / "run_config.json")
        candidate_run_config = _read_mapping(candidate_run / "run_config.json")
        gpu_allocations[profile.name] = {
            "baseline": _gpu_memory_utilization(baseline_run_config),
            "candidate": _gpu_memory_utilization(candidate_run_config),
        }
        if profile.output_subdir not in checked_run_dirs:
            baseline_sha, candidate_sha = _validate_run_config_pair(
                baseline_run_config,
                candidate_run_config,
            )
            if baseline_checkpoint_sha is None:
                baseline_checkpoint_sha = baseline_sha
                candidate_checkpoint_sha = candidate_sha
            elif (
                baseline_checkpoint_sha != baseline_sha
                or candidate_checkpoint_sha != candidate_sha
            ):
                raise ValueError("checkpoint identities differ between suite profile runs")
            checked_run_dirs.add(profile.output_subdir)

        baseline_rows = _group_rows(
            _read_jsonl(baseline_run / "sample_results.jsonl"),
            profile.group,
        )
        candidate_rows = _group_rows(
            _read_jsonl(candidate_run / "sample_results.jsonl"),
            profile.group,
        )
        _validate_same_selection(baseline_rows, candidate_rows, profile.name)
        profile_rows[profile.name] = (baseline_rows, candidate_rows)

    if baseline_checkpoint_sha is None or candidate_checkpoint_sha is None:
        raise ValueError("suite contains no checkpoint identities")
    if baseline_checkpoint_sha == candidate_checkpoint_sha:
        raise ValueError("baseline and candidate suites use the same CTC checkpoint")
    _validate_ctc_report_binding(baseline_config, baseline_checkpoint_sha)
    _validate_ctc_report_binding(candidate_config, candidate_checkpoint_sha)

    baseline_profiles = _profiles(baseline_summary)
    candidate_profiles = _profiles(candidate_summary)
    deltas = {
        profile.name: _quality_delta(
            _quality(candidate_profiles, profile.name),
            _quality(baseline_profiles, profile.name),
        )
        for profile in STREAMING_GATE_PROFILES
    }
    controls = {
        name: {
            "quality_equal": _selected_quality(
                _quality(baseline_profiles, name)
            )
            == _selected_quality(_quality(candidate_profiles, name)),
            "delta_candidate_minus_baseline": deltas[name],
        }
        for name in CONTROL_PROFILES
    }
    changed_cases = _changed_cases(profile_rows)
    sample_count = _required_int(baseline_summary, "sample_count")
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "portuguese_formal100_multilingual_ctc_checkpoint_swap",
        "only_intended_treatment": "ctc_checkpoint",
        "derived_input_change": "checkpoint-bound ctc_report",
        "test_set_used": False,
        "sample_count": sample_count,
        "baseline": {
            "suite_dir": str(baseline_root),
            "suite_config_sha256": _sha256(baseline_root / "suite_config.json"),
            "suite_summary_sha256": _sha256(baseline_root / "suite_summary.json"),
            "ctc_checkpoint_sha256": baseline_checkpoint_sha,
        },
        "candidate": {
            "suite_dir": str(candidate_root),
            "suite_config_sha256": _sha256(candidate_root / "suite_config.json"),
            "suite_summary_sha256": _sha256(candidate_root / "suite_summary.json"),
            "ctc_checkpoint_sha256": candidate_checkpoint_sha,
        },
        "identity_checks": {
            "root_config_equal_except_checkpoint_derived_report_and_gpu_allocation": True,
            "child_run_configs_equal_except_checkpoint_derived_report_git_and_gpu_allocation": True,
            "checkpoint_changed": True,
            "ctc_reports_bound_to_respective_checkpoints": True,
            "sample_selection_equal_for_every_profile": True,
            "source_sha256_manifests_verified": True,
        },
        "control_stability": {
            "all_control_quality_equal": all(
                bool(value["quality_equal"]) for value in controls.values()
            ),
            "profiles": controls,
            "interpretation": (
                "C no-RAG and E Oracle do not use CTC retrieval; any quality drift in "
                "these controls limits a strict causal interpretation of D-profile deltas."
            ),
        },
        "non_quality_runtime_context": {
            "gpu_memory_utilization_by_profile": gpu_allocations,
            "latency_compared": False,
            "reason": (
                "The historical suite used different vLLM memory allocations for some "
                "profiles; allocation and separate-run timing are not quality treatments."
            ),
        },
        "quality": {
            profile.name: {
                "gate": _profile(candidate_profiles, profile.name).get("gate"),
                "baseline": _selected_quality(_quality(baseline_profiles, profile.name)),
                "candidate": _selected_quality(_quality(candidate_profiles, profile.name)),
                "delta_candidate_minus_baseline": deltas[profile.name],
            }
            for profile in STREAMING_GATE_PROFILES
        },
        "case_comparison": {
            "changed_profile_case_rows": len(changed_cases),
            "details_file": "checkpoint_regression_cases.jsonl",
        },
    }

    destination.mkdir(parents=True)
    _write_json(destination / "checkpoint_regression_summary.json", result)
    _write_jsonl(destination / "checkpoint_regression_cases.jsonl", changed_cases)
    (destination / "README.md").write_text(
        "# Streaming checkpoint regression\n\n"
        "This report compares two Portuguese formal100 six-profile suites after "
        "verifying their SHA256 manifests, run contracts, and sample selections. "
        "The intended treatment is the CTC checkpoint; its checkpoint-bound CTC report "
        "is an expected derived input change. Timing is deliberately "
        "left in the source suites because separate H200 runs are not a controlled "
        "latency experiment.\n",
        encoding="utf-8",
    )
    _write_hashes(destination)
    return result


def _validate_suite_config_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        if value.get("status") != "pass" or value.get("test_set_used") is not False:
            raise ValueError(f"{name} suite config is not a completed non-test run")
    if _without_root_checkpoint(baseline) != _without_root_checkpoint(candidate):
        raise ValueError("suite configs differ beyond inputs.ctc_checkpoint")


def _without_root_checkpoint(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = cast(dict[str, Any], json.loads(json.dumps(value)))
    normalized["status"] = "pass"
    runtime = normalized.get("runtime")
    if not isinstance(runtime, dict) or "gpu_memory_utilization" not in runtime:
        raise ValueError("suite config has no runtime.gpu_memory_utilization")
    del runtime["gpu_memory_utilization"]
    inputs = normalized.get("inputs")
    if not isinstance(inputs, dict) or "ctc_checkpoint" not in inputs:
        raise ValueError("suite config has no inputs.ctc_checkpoint")
    del inputs["ctc_checkpoint"]
    if "ctc_report" not in inputs:
        raise ValueError("suite config has no inputs.ctc_report")
    del inputs["ctc_report"]
    return normalized


def _validate_suite_summary_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        if value.get("status") != "pass" or value.get("test_set_used") is not False:
            raise ValueError(f"{name} suite summary is not a passed non-test run")
    if _required_int(baseline, "sample_count") != _required_int(candidate, "sample_count"):
        raise ValueError("suite sample counts differ")
    if set(_profiles(baseline)) != set(_profiles(candidate)):
        raise ValueError("suite profile sets differ")


def _validate_run_config_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[str, str]:
    baseline_sha = _checkpoint_sha(baseline)
    candidate_sha = _checkpoint_sha(candidate)
    if _without_child_checkpoint(baseline) != _without_child_checkpoint(candidate):
        raise ValueError("child run configs differ beyond checkpoint and git_commit")
    return baseline_sha, candidate_sha


def _without_child_checkpoint(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = cast(dict[str, Any], json.loads(json.dumps(value)))
    normalized.pop("git_commit", None)
    normalized.pop("gpu_memory_utilization", None)
    inputs = normalized.get("inputs")
    if not isinstance(inputs, dict) or "checkpoint" not in inputs:
        raise ValueError("child run config has no inputs.checkpoint")
    del inputs["checkpoint"]
    offline_control = normalized.get("offline_control")
    if not isinstance(offline_control, dict) or "report" not in offline_control:
        raise ValueError("child run config has no checkpoint-bound offline report")
    del offline_control["report"]
    return normalized


def _gpu_memory_utilization(value: Mapping[str, Any]) -> float | None:
    raw = value.get("gpu_memory_utilization")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    return None


def _validate_ctc_report_binding(config: Mapping[str, Any], checkpoint_sha: str) -> None:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("suite config inputs are invalid")
    raw_path = inputs.get("ctc_report")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("suite config has no ctc_report path")
    report = _read_mapping(Path(raw_path).expanduser())
    if report.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("suite CTC report is not bound to its child checkpoint identity")


def _checkpoint_sha(value: Mapping[str, Any]) -> str:
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("child run config inputs are invalid")
    checkpoint = inputs.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("child run checkpoint identity is invalid")
    sha = checkpoint.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("child run checkpoint SHA256 is invalid")
    return sha


def _profiles(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    profiles = summary.get("profiles")
    if not isinstance(profiles, Mapping):
        raise ValueError("suite summary profiles are invalid")
    return profiles


def _profile(profiles: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = profiles.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"suite summary has no valid {name} profile")
    return value


def _quality(profiles: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _profile(profiles, name).get("quality")
    if not isinstance(value, Mapping):
        raise ValueError(f"suite summary has no valid {name} quality metrics")
    return value


def _selected_quality(quality: Mapping[str, Any]) -> dict[str, object]:
    return {name: quality.get(name) for name in QUALITY_FIELDS}


def _quality_delta(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        name: _numeric_delta(candidate.get(name), baseline.get(name))
        for name in QUALITY_FIELDS
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


def _group_rows(rows: Sequence[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if str(row.get("experiment_group")) == group]
    if not selected:
        raise ValueError(f"sample results contain no group {group}")
    return selected


def _validate_same_selection(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    profile_name: str,
) -> None:
    def selection(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
        return sorted(
            (
                str(row.get("case_id")),
                str(row.get("sample_id")),
                str(row.get("reference_text")),
            )
            for row in rows
        )

    if selection(baseline) != selection(candidate):
        raise ValueError(f"profile {profile_name} uses a different sample selection")


def _changed_cases(
    profile_rows: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for profile in STREAMING_GATE_PROFILES:
        baseline_rows, candidate_rows = profile_rows[profile.name]
        baseline_by_case = {str(row.get("case_id")): row for row in baseline_rows}
        candidate_by_case = {str(row.get("case_id")): row for row in candidate_rows}
        for case_id in sorted(baseline_by_case):
            baseline = baseline_by_case[case_id]
            candidate = candidate_by_case[case_id]
            baseline_injected = _injected_ids(baseline)
            candidate_injected = _injected_ids(candidate)
            baseline_prediction = str(baseline.get("prediction", ""))
            candidate_prediction = str(candidate.get("prediction", ""))
            if (
                baseline_injected == candidate_injected
                and baseline_prediction == candidate_prediction
            ):
                continue
            changes.append(
                {
                    "profile": profile.name,
                    "case_id": case_id,
                    "sample_id": str(baseline.get("sample_id", "")),
                    "expected_hotword_ids": _string_list(
                        baseline.get("expected_hotword_ids")
                    ),
                    "baseline": {
                        "injected_hotword_ids": baseline_injected,
                        "prediction": baseline_prediction,
                    },
                    "candidate": {
                        "injected_hotword_ids": candidate_injected,
                        "prediction": candidate_prediction,
                    },
                    "injection_changed": baseline_injected != candidate_injected,
                    "prediction_changed": baseline_prediction != candidate_prediction,
                }
            )
    return changes


def _injected_ids(row: Mapping[str, Any]) -> list[str]:
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


def _verify_sha256_manifest(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"SHA256 manifest does not exist: {path}")
    entries = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, separator, raw_name = line.partition("  ")
        if not separator or len(digest) != 64 or not raw_name:
            raise ValueError(f"invalid SHA256 manifest row: {path}:{line_number}")
        candidate = Path(raw_name).expanduser()
        target = candidate if candidate.is_file() else path.parent / raw_name
        if not target.is_file():
            raise FileNotFoundError(f"SHA256 target does not exist: {raw_name}")
        if _sha256(target) != digest:
            raise ValueError(f"SHA256 mismatch for {target}")
        entries += 1
    if entries == 0:
        raise ValueError(f"SHA256 manifest is empty: {path}")


def _required_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"{key} is not an integer")
    return result


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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


def _write_hashes(destination: Path) -> None:
    lines = []
    for name in OUTPUT_FILES:
        if name == "sha256.txt":
            continue
        path = destination / name
        lines.append(f"{_sha256(path)}  {name}\n")
    (destination / "sha256.txt").write_text("".join(lines), encoding="utf-8")
