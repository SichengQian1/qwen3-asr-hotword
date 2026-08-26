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
    StreamingGateProfile("recall_first", "recall_first", "D", 0.75, 7, 0.5),
    StreamingGateProfile("oracle", "baseline_oracle", "E", 0.86, 5, 0.0),
)


def build_streaming_gate_suite_report(output_dir: str | Path) -> dict[str, object]:
    root = Path(output_dir).expanduser()
    profiles: dict[str, dict[str, object]] = {}
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
            "quality": dict(quality),
            "latency": dict(timing),
        }
    baseline = profiles["no_rag"]["quality"]
    assert isinstance(baseline, Mapping)
    comparisons = {
        name: _quality_delta(profile["quality"], baseline)
        for name, profile in profiles.items()
        if name != "no_rag"
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "evaluation": "streaming_4k_gate_prompt_filter_suite",
        "conceptual_groups": [profile.name for profile in STREAMING_GATE_PROFILES],
        "sample_count": len(expected_selection or ()),
        "profiles": profiles,
        "comparisons_vs_no_rag": comparisons,
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
    report_path = root / "suite_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(_readme(report), encoding="utf-8")
    files = ("suite_config.json", "suite_summary.json", "README.md")
    with (root / "sha256.txt").open("w", encoding="utf-8") as handle:
        for name in files:
            path = root / name
            if path.is_file():
                handle.write(f"{_sha256(path)}  {name}\n")
    return report


def _quality_delta(current: object, baseline: Mapping[str, Any]) -> dict[str, float | None]:
    if not isinstance(current, Mapping):
        raise ValueError("profile quality is not a mapping")
    names = (
        "hotword_exact_recall",
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
        "This suite compares no RAG, conservative, balanced, recall-first, and Oracle "
        "streaming inference on the same sealed validation selection. Retrieval, CTC, "
        "Prompt/Qwen, per-step, per-sample, and real-time-factor latency are measured "
        "separately. Audio loading is outside sample inference timing.\n\n"
        f"Samples: {report['sample_count']}\n"
    )
