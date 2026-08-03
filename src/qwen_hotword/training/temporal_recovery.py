from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

CTC_LENGTH_ISSUE = "ctc_length_infeasible"
RATIO_BUCKETS = (
    "ratio_le_0_50",
    "ratio_0_50_to_0_75",
    "ratio_0_75_to_0_90",
    "ratio_0_90_to_1_00",
    "ratio_gt_1_00",
)
K = TypeVar("K")


@dataclass
class CountHours:
    records: int = 0
    duration_seconds: float = 0.0
    missing_duration_records: int = 0

    def add(self, duration_seconds: float | None) -> None:
        self.records += 1
        if duration_seconds is None:
            self.missing_duration_records += 1
        else:
            self.duration_seconds += duration_seconds

    def merge(self, other: CountHours) -> None:
        self.records += other.records
        self.duration_seconds += other.duration_seconds
        self.missing_duration_records += other.missing_duration_records

    def to_dict(self) -> dict[str, object]:
        return {
            "records": self.records,
            "hours": round(self.duration_seconds / 3600.0, 6),
            "duration_missing_records": self.missing_duration_records,
        }


@dataclass(frozen=True)
class CorpusAuditInput:
    name: str
    manifest_dir: Path


def parse_corpus_spec(value: str) -> CorpusAuditInput:
    if "=" not in value:
        raise ValueError("corpus must use NAME=MANIFEST_DIR")
    name, raw_path = value.split("=", maxsplit=1)
    name = name.strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(
            "corpus name must start with an alphanumeric character and contain "
            "only lowercase letters, digits, '_' or '-'"
        )
    if not raw_path.strip():
        raise ValueError("corpus manifest directory must not be empty")
    return CorpusAuditInput(name=name, manifest_dir=Path(raw_path).expanduser())


def audit_temporal_recovery_corpus(
    corpus: CorpusAuditInput,
    *,
    time_upsampling_factor: int = 2,
    release_max_effective_ratio: float = 0.90,
    progress_every: int = 50_000,
    print_progress: bool = True,
) -> dict[str, object]:
    if time_upsampling_factor <= 1:
        raise ValueError("time_upsampling_factor must exceed one for recovery audit")
    if not 0.0 < release_max_effective_ratio <= 1.0:
        raise ValueError("release_max_effective_ratio must be within (0, 1]")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    manifest_dir = corpus.manifest_dir
    summary_path = manifest_dir / "summary.json"
    ready_path = manifest_dir / "train_ready.jsonl"
    review_path = manifest_dir / "needs_review.jsonl"
    for path in (summary_path, ready_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(f"required manifest artifact does not exist: {path}")
    summary = _load_summary(summary_path)
    started = time.monotonic()

    original_review = CountHours()
    pure_temporal = CountHours()
    recoverable = CountHours()
    recommended = CountHours()
    high_pressure = CountHours()
    still_infeasible = CountHours()
    blocked_other = CountHours()
    blocked_with_time = CountHours()
    blocked_without_time = CountHours()
    issue_totals: dict[str, CountHours] = {}
    exact_intersections: dict[tuple[str, ...], CountHours] = {}
    pair_intersections: dict[tuple[str, str], CountHours] = {}
    all_ratio_buckets = {name: CountHours() for name in RATIO_BUCKETS}
    pure_ratio_buckets = {name: CountHours() for name in RATIO_BUCKETS}
    ratio_unavailable = CountHours()
    seen_ids: set[str] = set()
    review_digest = hashlib.sha256()

    with review_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            review_digest.update(raw_line)
            if not raw_line.strip():
                continue
            raw = _load_review_row(raw_line, review_path, line_number)
            record_id = _required_string(raw, "id", review_path, line_number)
            if record_id in seen_ids:
                raise ValueError(f"duplicate review record ID: {record_id}")
            seen_ids.add(record_id)
            if raw.get("training_ready") is not False:
                raise ValueError(
                    f"review row must have training_ready=false: {review_path}:{line_number}"
                )
            if raw.get("label_status") != "needs_review":
                raise ValueError(
                    f"review row must have label_status='needs_review': {review_path}:{line_number}"
                )
            duration = _optional_duration(raw, review_path, line_number)
            reasons = _issue_reasons(raw, review_path, line_number)
            original_review.add(duration)
            _add_metric(exact_intersections, reasons).add(duration)
            for reason in reasons:
                _add_metric(issue_totals, reason).add(duration)
            for pair in itertools.combinations(reasons, 2):
                _add_metric(pair_intersections, pair).add(duration)

            lengths = _optional_lengths(raw, review_path, line_number)
            effective_ratio: float | None = None
            if lengths is not None:
                estimated_input, minimum_target = lengths
                effective_input = estimated_input * time_upsampling_factor
                effective_ratio = (
                    minimum_target / effective_input if effective_input > 0 else math.inf
                )
                all_ratio_buckets[_ratio_bucket(effective_ratio)].add(duration)
            else:
                ratio_unavailable.add(duration)

            other_reasons = set(reasons) - {CTC_LENGTH_ISSUE}
            if reasons == (CTC_LENGTH_ISSUE,):
                if lengths is None:
                    raise ValueError(
                        f"pure temporal row lacks CTC lengths: {review_path}:{line_number}"
                    )
                estimated_input, minimum_target = lengths
                if estimated_input >= minimum_target:
                    raise ValueError(
                        f"temporal issue is inconsistent with original lengths: "
                        f"{review_path}:{line_number}"
                    )
                pure_temporal.add(duration)
                if effective_ratio is None:
                    raise RuntimeError("pure temporal effective ratio was not calculated")
                pure_ratio_buckets[_ratio_bucket(effective_ratio)].add(duration)
                if minimum_target <= estimated_input * time_upsampling_factor:
                    recoverable.add(duration)
                    if effective_ratio <= release_max_effective_ratio:
                        recommended.add(duration)
                    else:
                        high_pressure.add(duration)
                else:
                    still_infeasible.add(duration)
            elif other_reasons:
                blocked_other.add(duration)
                if CTC_LENGTH_ISSUE in reasons:
                    blocked_with_time.add(duration)
                else:
                    blocked_without_time.add(duration)
            else:
                raise ValueError(f"review row has no actionable issue: {review_path}:{line_number}")

            if print_progress and original_review.records % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"temporal2x_audit corpus={corpus.name} "
                    f"reviewed={original_review.records} elapsed={elapsed:.1f}s "
                    f"records_per_second={original_review.records / elapsed:.1f}",
                    flush=True,
                )

    _validate_partition(
        original_review,
        pure_temporal,
        blocked_other,
        context=f"{corpus.name} review",
    )
    _validate_partition(
        pure_temporal,
        recoverable,
        still_infeasible,
        context=f"{corpus.name} pure temporal",
    )
    _validate_partition(
        recoverable,
        recommended,
        high_pressure,
        context=f"{corpus.name} recoverable",
    )
    _validate_summary(summary, original_review, corpus.name)

    ready_records = _required_nonnegative_int(summary, "ready_records", summary_path)
    review_records = _required_nonnegative_int(summary, "review_records", summary_path)
    ready_hours = _required_nonnegative_number(summary, "ready_audio_hours", summary_path)
    total_hours = _required_nonnegative_number(summary, "total_audio_hours", summary_path)
    elapsed = time.monotonic() - started
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "read_only_temporal_upsample_recovery_audit",
        "corpus": corpus.name,
        "read_only": True,
        "ready_manifest_content_read": False,
        "sealed_test_content_read": False,
        "time_axis_policy": f"temporal_upsample_{time_upsampling_factor}x",
        "release_policy": {
            "pure_temporal_issue_only": True,
            "maximum_effective_ratio": release_max_effective_ratio,
            "high_pressure_interval": (f"({release_max_effective_ratio:.2f}, 1.00]"),
            "other_issue_records_released": False,
        },
        "inputs": {
            "manifest_dir": str(manifest_dir),
            "summary": _file_identity(summary_path),
            "ready_manifest": {
                "path": str(ready_path),
                "size": ready_path.stat().st_size,
                "content_read": False,
                "sha256": None,
            },
            "review_manifest": {
                "path": str(review_path),
                "size": review_path.stat().st_size,
                "sha256": review_digest.hexdigest(),
            },
        },
        "original": {
            "ready": {
                "records": ready_records,
                "hours": round(ready_hours, 6),
                "source": "summary.json; ready JSONL content was not read",
            },
            "review": original_review.to_dict(),
            "review_records_from_summary": review_records,
            "review_hours_from_summary_difference": round(
                max(0.0, total_hours - ready_hours),
                6,
            ),
        },
        "temporal_2x_results": {
            "pure_temporal_original_review": pure_temporal.to_dict(),
            "recoverable_total": recoverable.to_dict(),
            "recommended_ratio_le_limit": recommended.to_dict(),
            "high_pressure_deferred": high_pressure.to_dict(),
            "still_infeasible": still_infeasible.to_dict(),
            "blocked_by_other_issues": blocked_other.to_dict(),
            "blocked_by_other_issues_with_temporal_issue": blocked_with_time.to_dict(),
            "blocked_by_other_issues_without_temporal_issue": (blocked_without_time.to_dict()),
        },
        "effective_ratio_buckets": {
            "definition": (
                "ctc_minimum_input_length / "
                f"(estimated_ctc_input_length * {time_upsampling_factor})"
            ),
            "boundaries": {
                "ratio_le_0_50": "[0.00, 0.50]",
                "ratio_0_50_to_0_75": "(0.50, 0.75]",
                "ratio_0_75_to_0_90": "(0.75, 0.90]",
                "ratio_0_90_to_1_00": "(0.90, 1.00]",
                "ratio_gt_1_00": "(1.00, +inf]",
            },
            "pure_temporal_review": _metrics_map(pure_ratio_buckets),
            "all_review_with_length_metadata": _metrics_map(all_ratio_buckets),
            "review_without_length_metadata": ratio_unavailable.to_dict(),
            "warning": (
                "ratios on records with non-temporal label issues may use partial "
                "targets and are diagnostic only"
            ),
        },
        "issue_analysis": {
            "reason_totals": _metrics_map(issue_totals),
            "exact_issue_intersections": _intersection_map(exact_intersections),
            "pairwise_issue_intersections": _pair_map(pair_intersections),
            "counts_overlap": True,
        },
        "consistency": {
            "review_partition_records": (pure_temporal.records + blocked_other.records),
            "review_summary_records_match": original_review.records == review_records,
            "review_known_duration_hours": round(
                original_review.duration_seconds / 3600.0,
                6,
            ),
            "review_summary_difference_hours": round(
                max(0.0, total_hours - ready_hours),
                6,
            ),
        },
        "elapsed_seconds": elapsed,
        "limitations": [
            "This audit does not write or modify training manifests.",
            "Ready record-level ratios are not scanned; ready aggregates come from summary.json.",
            "Records with any non-temporal issue remain blocked even if 2x timing is feasible.",
            "Release manifests and corpus mixing weights are intentionally deferred.",
        ],
    }
    if print_progress:
        print(
            f"temporal2x_audit corpus={corpus.name} complete "
            f"reviewed={original_review.records} "
            f"recommended={recommended.records} "
            f"high_pressure={high_pressure.records} "
            f"still_infeasible={still_infeasible.records} "
            f"blocked_other={blocked_other.records} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return report


def run_temporal_recovery_audit(
    corpora: list[CorpusAuditInput],
    output_dir: str | Path,
    *,
    time_upsampling_factor: int = 2,
    release_max_effective_ratio: float = 0.90,
    progress_every: int = 50_000,
    print_progress: bool = True,
) -> dict[str, object]:
    if not corpora:
        raise ValueError("at least one corpus is required")
    names = [corpus.name for corpus in corpora]
    if len(set(names)) != len(names):
        raise ValueError("corpus names must be unique")
    destination = Path(output_dir).expanduser()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"audit output path is not a directory: {destination}")
    if destination.is_dir() and any(destination.iterdir()):
        raise FileExistsError(
            f"audit output directory is not empty; refusing to overwrite: {destination}"
        )

    started = time.monotonic()
    reports = [
        audit_temporal_recovery_corpus(
            corpus,
            time_upsampling_factor=time_upsampling_factor,
            release_max_effective_ratio=release_max_effective_ratio,
            progress_every=progress_every,
            print_progress=print_progress,
        )
        for corpus in corpora
    ]
    aggregate_keys = (
        "recoverable_total",
        "recommended_ratio_le_limit",
        "high_pressure_deferred",
        "still_infeasible",
        "blocked_by_other_issues",
    )
    aggregate: dict[str, CountHours] = {key: CountHours() for key in aggregate_keys}
    corpus_summaries: list[dict[str, object]] = []
    for report in reports:
        temporal = report["temporal_2x_results"]
        original = report["original"]
        if not isinstance(temporal, dict) or not isinstance(original, dict):
            raise RuntimeError("internal audit report structure is invalid")
        corpus_summary: dict[str, object] = {
            "corpus": report["corpus"],
            "original_ready": original["ready"],
            "original_review": original["review"],
        }
        for key in aggregate_keys:
            value = temporal[key]
            if not isinstance(value, dict):
                raise RuntimeError("internal temporal metric is invalid")
            corpus_summary[key] = value
            aggregate[key].records += int(value["records"])
            aggregate[key].duration_seconds += float(value["hours"]) * 3600.0
            aggregate[key].missing_duration_records += int(value["duration_missing_records"])
        corpus_summaries.append(corpus_summary)

    summary = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "read_only_temporal_upsample_recovery_audit",
        "read_only": True,
        "sealed_test_content_read": False,
        "time_upsampling_factor": time_upsampling_factor,
        "release_max_effective_ratio": release_max_effective_ratio,
        "corpus_order": names,
        "corpora": corpus_summaries,
        "aggregate": {key: value.to_dict() for key, value in aggregate.items()},
        "output_dir": str(destination),
        "elapsed_seconds": time.monotonic() - started,
        "next_step": ("inspect audit only; decide release volume before building any new manifest"),
    }
    outputs = {destination / f"{report['corpus']}.json": _json_bytes(report) for report in reports}
    outputs[destination / "summary.json"] = _json_bytes(summary)
    _write_output_group(outputs)
    if print_progress:
        print(f"Temporal 2x audit outputs written: {destination}", flush=True)
    return summary


def _load_summary(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid manifest summary JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"manifest summary must contain an object: {path}")
    if raw.get("status") != "pass":
        raise ValueError(f"manifest summary status is not pass: {path}")
    return raw


def _load_review_row(raw_line: bytes, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid review JSON at {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"review row must contain an object: {path}:{line_number}")
    return raw


def _issue_reasons(raw: dict[str, Any], path: Path, line_number: int) -> tuple[str, ...]:
    issues = raw.get("issues")
    if not isinstance(issues, list) or not issues:
        raise ValueError(f"review row has invalid issues: {path}:{line_number}")
    reasons: set[str] = set()
    for issue in issues:
        if not isinstance(issue, dict):
            raise ValueError(f"review row has non-object issue: {path}:{line_number}")
        reason = issue.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"review row has invalid issue reason: {path}:{line_number}")
        reasons.add(reason.strip())
    return tuple(sorted(reasons))


def _optional_lengths(raw: dict[str, Any], path: Path, line_number: int) -> tuple[int, int] | None:
    estimated = raw.get("estimated_ctc_input_length")
    minimum = raw.get("ctc_minimum_input_length")
    if estimated is None and minimum is None:
        return None
    if estimated is not None and (
        not isinstance(estimated, int) or isinstance(estimated, bool) or estimated < 0
    ):
        raise ValueError(f"review row has invalid CTC lengths: {path}:{line_number}")
    if minimum is not None and (
        not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0
    ):
        raise ValueError(f"review row has invalid CTC lengths: {path}:{line_number}")
    # Partial length metadata is expected when label assembly failed. Audio can
    # still provide an estimated input length while no valid minimum target
    # length exists. These rows remain blocked by their non-temporal issue and
    # have no meaningful effective ratio. Pure temporal rows are checked later
    # and still require both values.
    if estimated is None or minimum is None:
        return None
    return estimated, minimum


def _optional_duration(raw: dict[str, Any], path: Path, line_number: int) -> float | None:
    value = raw.get("duration_seconds")
    if value is None:
        return None
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"review row has invalid duration: {path}:{line_number}")
    return float(value)


def _required_string(raw: dict[str, Any], key: str, path: Path, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"review row has invalid {key}: {path}:{line_number}")
    return value.strip()


def _required_nonnegative_int(raw: dict[str, object], key: str, path: Path) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"summary has invalid {key}: {path}")
    return value


def _required_nonnegative_number(raw: dict[str, object], key: str, path: Path) -> float:
    value = raw.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"summary has invalid {key}: {path}")
    return float(value)


def _validate_summary(summary: dict[str, object], review: CountHours, corpus_name: str) -> None:
    summary_path = Path(f"{corpus_name}/summary.json")
    ready_records = _required_nonnegative_int(summary, "ready_records", summary_path)
    review_records = _required_nonnegative_int(summary, "review_records", summary_path)
    source_records = _required_nonnegative_int(summary, "source_records", summary_path)
    if ready_records + review_records != source_records:
        raise ValueError(f"{corpus_name} summary ready/review partition is inconsistent")
    if review.records != review_records:
        raise ValueError(
            f"{corpus_name} review row count differs from summary: "
            f"rows={review.records}, summary={review_records}"
        )
    total_hours = _required_nonnegative_number(summary, "total_audio_hours", summary_path)
    ready_hours = _required_nonnegative_number(summary, "ready_audio_hours", summary_path)
    if ready_hours > total_hours:
        raise ValueError(f"{corpus_name} ready audio hours exceed total audio hours")
    expected_review_hours = total_hours - ready_hours
    scanned_review_hours = review.duration_seconds / 3600.0
    if review.missing_duration_records == 0 and not math.isclose(
        scanned_review_hours,
        expected_review_hours,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise ValueError(
            f"{corpus_name} review audio hours differ from summary: "
            f"rows={scanned_review_hours:.6f}, summary={expected_review_hours:.6f}"
        )


def _validate_partition(
    total: CountHours, first: CountHours, second: CountHours, *, context: str
) -> None:
    if total.records != first.records + second.records:
        raise RuntimeError(f"{context} record partition is inconsistent")
    if total.missing_duration_records != (
        first.missing_duration_records + second.missing_duration_records
    ):
        raise RuntimeError(f"{context} duration coverage partition is inconsistent")
    if not math.isclose(
        total.duration_seconds,
        first.duration_seconds + second.duration_seconds,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(f"{context} duration partition is inconsistent")


def _ratio_bucket(ratio: float) -> str:
    if ratio <= 0.50:
        return "ratio_le_0_50"
    if ratio <= 0.75:
        return "ratio_0_50_to_0_75"
    if ratio <= 0.90:
        return "ratio_0_75_to_0_90"
    if ratio <= 1.00:
        return "ratio_0_90_to_1_00"
    return "ratio_gt_1_00"


def _add_metric(mapping: dict[K, CountHours], key: K) -> CountHours:
    if key not in mapping:
        mapping[key] = CountHours()
    return mapping[key]


def _metrics_map(values: dict[str, CountHours]) -> dict[str, dict[str, object]]:
    return {key: values[key].to_dict() for key in sorted(values)}


def _intersection_map(values: dict[tuple[str, ...], CountHours]) -> dict[str, dict[str, object]]:
    return {" + ".join(key): values[key].to_dict() for key in sorted(values)}


def _pair_map(values: dict[tuple[str, str], CountHours]) -> dict[str, dict[str, object]]:
    return {" & ".join(key): values[key].to_dict() for key in sorted(values)}


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _json_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_output_group(outputs: dict[Path, bytes]) -> None:
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite audit output: {path}")
    temporary_paths: list[Path] = []
    try:
        for path, payload in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            if temporary.exists():
                raise FileExistsError(f"temporary audit output already exists: {temporary}")
            temporary.write_bytes(payload)
            temporary_paths.append(temporary)
        for path, temporary in zip(outputs, temporary_paths, strict=True):
            temporary.replace(path)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
