from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

SELECTION_SCOPES = ("final", "any_step")


def sweep_operating_points(
    *,
    benchmark_dir: str | Path,
    output_dir: str | Path,
    profile: str = "representative",
    size: int = 4_000,
    window: str = "full_current",
    shortlist_size: int = 64,
    top_ks: Sequence[int] = (5, 7, 10),
    thresholds: Sequence[float] = (
        0.0,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
    ),
    maximum_edit_ratios: Sequence[float] = (0.35, 0.40, 0.45, 0.50, 0.60, 1.0),
    minimum_posterior_confidences: Sequence[float] = (0.0, 0.25, 0.50, 0.75),
    minimum_top1_margins: Sequence[float] = (0.0, 0.01, 0.02, 0.05),
    selection_scope: str = "final",
    target_recall: float = 0.90,
    diagnostic_precision_target: float = 0.85,
    deadline_seconds: float = 0.05,
) -> dict[str, object]:
    """Replay Operating gates exactly over a complete saved Anchor shortlist."""
    resolved_top_ks = _sorted_unique_ints(top_ks, name="top_ks")
    resolved_thresholds = _sorted_unique_probabilities(thresholds, name="thresholds")
    resolved_edit_ratios = _sorted_unique_probabilities(
        maximum_edit_ratios, name="maximum_edit_ratios"
    )
    resolved_posteriors = _sorted_unique_probabilities(
        minimum_posterior_confidences,
        name="minimum_posterior_confidences",
    )
    resolved_margins = _sorted_unique_probabilities(
        minimum_top1_margins, name="minimum_top1_margins"
    )
    if selection_scope not in SELECTION_SCOPES:
        raise ValueError(f"selection_scope must be one of {SELECTION_SCOPES}")
    for name, value in (
        ("target_recall", target_recall),
        ("diagnostic_precision_target", diagnostic_precision_target),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if size <= 0 or shortlist_size <= 0:
        raise ValueError("size and shortlist_size must be positive")
    if max(resolved_top_ks) > shortlist_size:
        raise ValueError("top_ks must not exceed shortlist_size")
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")

    source = Path(benchmark_dir).expanduser()
    destination = Path(output_dir).expanduser()
    query_path = source / "query_results.jsonl"
    source_config_path = source / "run_config.json"
    if not query_path.is_file() or not source_config_path.is_file():
        raise FileNotFoundError(f"incomplete Anchor benchmark directory: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Operating sweep output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    source_config = _read_json_object(source_config_path)
    if bool(source_config.get("test_set_used")):
        raise ValueError("Operating sweep must not tune on a sealed test set")
    retrieval_config = source_config.get("retrieval_config")
    if not isinstance(retrieval_config, Mapping):
        raise ValueError("source run_config is missing retrieval_config")
    posterior_weight = float(retrieval_config["posterior_weight"])

    rows = [
        row
        for row in _read_jsonl(query_path)
        if str(row.get("profile")) == profile
        and int(row.get("size", -1)) == size
        and str(row.get("window")) == window
        and int(row.get("shortlist_size", -1)) == shortlist_size
    ]
    if not rows:
        raise ValueError("no query rows match the requested profile/size/window/shortlist")
    _validate_rows(rows, maximum_top_k=max(resolved_top_ks))

    latency = _distribution([float(row["retrieval_seconds"]) for row in rows])
    latency_p95 = latency["p95"]
    latency_pass = isinstance(latency_p95, float) and latency_p95 <= deadline_seconds
    points: list[dict[str, Any]] = []
    for top_k, threshold, edit_ratio, posterior, margin in itertools.product(
        resolved_top_ks,
        resolved_thresholds,
        resolved_edit_ratios,
        resolved_posteriors,
        resolved_margins,
    ):
        point = _evaluate_point(
            rows,
            top_k=top_k,
            threshold=threshold,
            maximum_edit_ratio=edit_ratio,
            minimum_posterior_confidence=posterior,
            minimum_top1_margin=margin,
        )
        point["selection_metrics"] = point[selection_scope]
        metrics = point[selection_scope]
        point["target_checks"] = {
            "retrieval_p95_le_deadline": latency_pass,
            "recall_ge_target": _metric(metrics, "recall") >= target_recall,
            "precision_ge_diagnostic_target": (
                _metric(metrics, "precision") >= diagnostic_precision_target
            ),
        }
        points.append(point)

    baseline = _find_source_baseline(points, retrieval_config=retrieval_config)
    frontier = _pareto_frontier(points, scope=selection_scope, latency_pass=latency_pass)
    recommendation = _recommend(
        points,
        scope=selection_scope,
        target_recall=target_recall,
        diagnostic_precision_target=diagnostic_precision_target,
        latency_pass=latency_pass,
    )
    recommendation["source_gates_by_top_k"] = {
        str(top_k): _find_source_gate_point(
            points, retrieval_config=retrieval_config, top_k=top_k
        )
        for top_k in resolved_top_ks
    }
    recommendation["best_by_top_k"] = {
        str(top_k): _recommend(
            [point for point in points if int(point["config"]["top_k"]) == top_k],
            scope=selection_scope,
            target_recall=target_recall,
            diagnostic_precision_target=diagnostic_precision_target,
            latency_pass=latency_pass,
        )
        for top_k in resolved_top_ks
    }
    recommended_point = recommendation["recall_first"]
    if isinstance(recommended_point, Mapping) and baseline is not None:
        recommendation["delta_from_source_baseline"] = _metric_delta(
            recommended_point[selection_scope], baseline[selection_scope]
        )

    run_config = {
        "schema_version": 1,
        "purpose": "exact_operating_gate_replay_over_complete_anchor_shortlist",
        "source_benchmark": _file_identity(query_path),
        "source_run_config": _file_identity(source_config_path),
        "variant": {
            "profile": profile,
            "size": size,
            "window": window,
            "shortlist_size": shortlist_size,
        },
        "fixed_ranking": {
            "posterior_weight": posterior_weight,
            "note": "posterior weight is frozen because changing it would rerank the shortlist",
        },
        "grid": {
            "top_ks": list(resolved_top_ks),
            "thresholds": list(resolved_thresholds),
            "maximum_edit_ratios": list(resolved_edit_ratios),
            "minimum_posterior_confidences": list(resolved_posteriors),
            "minimum_top1_margins": list(resolved_margins),
        },
        "selection_scope": selection_scope,
        "target_recall": target_recall,
        "diagnostic_precision_target": diagnostic_precision_target,
        "precision_target_is_blocking": False,
        "deadline_seconds": deadline_seconds,
        "test_set_used": False,
    }
    recommendation_output = {
        "schema_version": 1,
        "status": recommendation["status"],
        "selection_scope": selection_scope,
        "source_baseline": baseline,
        **recommendation,
        "test_set_used": False,
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": "operating_gate_sweep_fixed_anchor_ranking",
        "source_query_rows": len(rows),
        "final_queries": sum(bool(row["is_final"]) for row in rows),
        "sweep_points": len(points),
        "pareto_points": len(frontier),
        "source_retrieval_latency_seconds": latency,
        "deadline_seconds": deadline_seconds,
        "retrieval_p95_le_deadline": latency_pass,
        "selection_scope": selection_scope,
        "target_recall": target_recall,
        "diagnostic_precision_target": diagnostic_precision_target,
        "precision_target_is_blocking": False,
        "recommendation_status": recommendation["status"],
        "strict_recall_and_precision_point_count": recommendation[
            "strict_recall_and_precision_point_count"
        ],
        "test_set_used": False,
    }
    _write_jsonl(destination / "sweep_results.jsonl", points)
    _write_jsonl(destination / "pareto_frontier.jsonl", frontier)
    _write_json(destination / "recommended_config.json", recommendation_output)
    _write_json(destination / "run_config.json", run_config)
    _write_json(destination / "summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def _evaluate_point(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    threshold: float,
    maximum_edit_ratio: float,
    minimum_posterior_confidence: float,
    minimum_top1_margin: float,
) -> dict[str, Any]:
    selected_by_row: list[tuple[Mapping[str, Any], tuple[str, ...], str | None]] = []
    for row in rows:
        qualified = [
            match
            for match in row["top_matches"]
            if float(match["score"]) >= threshold
            and float(match["edit_ratio"]) <= maximum_edit_ratio
            and float(match["posterior_confidence"]) >= minimum_posterior_confidence
        ]
        reason: str | None = None
        if not qualified:
            reason = "below_threshold"
            selected: tuple[str, ...] = ()
        elif (
            len(qualified) > 1
            and float(qualified[0]["score"]) - float(qualified[1]["score"])
            < minimum_top1_margin
        ):
            reason = "ambiguous_top_matches"
            selected = ()
        else:
            selected = tuple(str(match["hotword_id"]) for match in qualified[:top_k])
        selected_by_row.append((row, selected, reason))
    config = {
        "top_k": top_k,
        "threshold": threshold,
        "maximum_edit_ratio": maximum_edit_ratio,
        "minimum_posterior_confidence": minimum_posterior_confidence,
        "minimum_top1_margin": minimum_top1_margin,
    }
    return {
        "schema_version": 1,
        "point_id": _point_id(config),
        "config": config,
        "final": _final_metrics(selected_by_row),
        "any_step": _any_step_metrics(selected_by_row),
        "suppression_reason_counts": {
            reason: sum(item_reason == reason for _, _, item_reason in selected_by_row)
            for reason in ("below_threshold", "ambiguous_top_matches")
        },
        "test_set_used": False,
    }


def _final_metrics(
    selected_by_row: Sequence[tuple[Mapping[str, Any], tuple[str, ...], str | None]],
) -> dict[str, int | float | None]:
    final = [item for item in selected_by_row if bool(item[0]["is_final"])]
    positive = [item for item in final if item[0]["expected_hotword_ids"]]
    negative = [item for item in final if not item[0]["expected_hotword_ids"]]
    expected = sum(len(item[0]["expected_hotword_ids"]) for item in final)
    correct = sum(
        len(set(row["expected_hotword_ids"]) & set(selected))
        for row, selected, _ in final
    )
    returned = sum(len(selected) for _, selected, _ in final)
    wrong = returned - correct
    return {
        "expected_hotwords": expected,
        "correct": correct,
        "returned": returned,
        "wrong": wrong,
        "recall": _ratio(correct, expected),
        "precision": _ratio(correct, returned),
        "positive_cases": len(positive),
        "positive_case_hit_rate": _ratio(
            sum(
                bool(set(row["expected_hotword_ids"]) & set(selected))
                for row, selected, _ in positive
            ),
            len(positive),
        ),
        "negative_cases": len(negative),
        "negative_case_false_positive_rate": _ratio(
            sum(bool(selected) for _, selected, _ in negative), len(negative)
        ),
        "mean_returned_per_query": returned / len(final) if final else None,
    }


def _any_step_metrics(
    selected_by_row: Sequence[tuple[Mapping[str, Any], tuple[str, ...], str | None]],
) -> dict[str, int | float | None]:
    final = [item for item in selected_by_row if bool(item[0]["is_final"])]
    expected_pairs = {
        (str(row["case_id"]), str(hotword_id))
        for row, _, _ in final
        for hotword_id in row["expected_hotword_ids"]
    }
    positive_cases = {
        str(row["case_id"]) for row, _, _ in final if row["expected_hotword_ids"]
    }
    negative_cases = {
        str(row["case_id"]) for row, _, _ in final if not row["expected_hotword_ids"]
    }
    returned_pairs = {
        (str(row["case_id"]), hotword_id)
        for row, selected, _ in selected_by_row
        for hotword_id in selected
    }
    correct_pairs = expected_pairs & returned_pairs
    negative_hit_cases = {
        case_id
        for case_id in negative_cases
        if any(returned_case == case_id for returned_case, _ in returned_pairs)
    }
    return {
        "expected_hotwords": len(expected_pairs),
        "correct": len(correct_pairs),
        "returned": len(returned_pairs),
        "wrong": len(returned_pairs - expected_pairs),
        "recall": _ratio(len(correct_pairs), len(expected_pairs)),
        "precision": _ratio(len(correct_pairs), len(returned_pairs)),
        "positive_cases": len(positive_cases),
        "positive_case_hit_rate": _ratio(
            len({case_id for case_id, _ in correct_pairs}), len(positive_cases)
        ),
        "negative_cases": len(negative_cases),
        "negative_case_false_positive_rate": _ratio(
            len(negative_hit_cases), len(negative_cases)
        ),
        "mean_returned_per_query": (
            len(returned_pairs) / len({str(row["case_id"]) for row, _, _ in final})
            if final
            else None
        ),
    }


def _recommend(
    points: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    target_recall: float,
    diagnostic_precision_target: float,
    latency_pass: bool,
) -> dict[str, Any]:
    recall_eligible = [
        point
        for point in points
        if latency_pass and _metric(point[scope], "recall") >= target_recall
    ]
    strict = [
        point
        for point in recall_eligible
        if _metric(point[scope], "precision") >= diagnostic_precision_target
    ]
    pool = recall_eligible if recall_eligible else list(points)
    recall_first = min(pool, key=lambda point: _recommendation_key(point, scope=scope))
    return {
        "status": "target_recall_met" if recall_eligible else "target_recall_not_met",
        "selection_policy": (
            "among points meeting recall and latency, maximize precision then minimize "
            "negative FPR; diagnostic precision target is reported but non-blocking"
        ),
        "recall_eligible_point_count": len(recall_eligible),
        "strict_recall_and_precision_point_count": len(strict),
        "recall_first": recall_first,
        "best_strict_point": (
            min(strict, key=lambda point: _recommendation_key(point, scope=scope))
            if strict
            else None
        ),
    }


def _recommendation_key(point: Mapping[str, Any], *, scope: str) -> tuple[float, ...]:
    metrics = point[scope]
    config = point["config"]
    return (
        -_metric(metrics, "precision"),
        _metric(metrics, "negative_case_false_positive_rate"),
        -_metric(metrics, "recall"),
        _metric(metrics, "mean_returned_per_query"),
        float(config["top_k"]),
        -float(config["threshold"]),
        float(config["maximum_edit_ratio"]),
        -float(config["minimum_posterior_confidence"]),
        -float(config["minimum_top1_margin"]),
    )


def _pareto_frontier(
    points: Sequence[Mapping[str, Any]], *, scope: str, latency_pass: bool
) -> list[Mapping[str, Any]]:
    if not latency_pass:
        return []
    unique_by_metrics: dict[tuple[float, float, float], Mapping[str, Any]] = {}
    for point in points:
        metrics = point[scope]
        key = (
            _metric(metrics, "recall"),
            _metric(metrics, "precision"),
            _metric(metrics, "negative_case_false_positive_rate"),
        )
        current = unique_by_metrics.get(key)
        if current is None or _recommendation_key(
            point, scope=scope
        ) < _recommendation_key(current, scope=scope):
            unique_by_metrics[key] = point
    unique_points = tuple(unique_by_metrics.values())
    frontier = []
    for point in unique_points:
        metrics = point[scope]
        dominated = any(
            other is not point and _dominates(other[scope], metrics)
            for other in unique_points
        )
        if not dominated:
            frontier.append(point)
    return sorted(
        frontier,
        key=lambda point: (
            -_metric(point[scope], "recall"),
            -_metric(point[scope], "precision"),
            _metric(point[scope], "negative_case_false_positive_rate"),
            int(point["config"]["top_k"]),
        ),
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_recall = _metric(left, "recall")
    right_recall = _metric(right, "recall")
    left_precision = _metric(left, "precision")
    right_precision = _metric(right, "precision")
    left_fpr = _metric(left, "negative_case_false_positive_rate")
    right_fpr = _metric(right, "negative_case_false_positive_rate")
    weak = (
        left_recall >= right_recall
        and left_precision >= right_precision
        and left_fpr <= right_fpr
    )
    strict = (
        left_recall > right_recall
        or left_precision > right_precision
        or left_fpr < right_fpr
    )
    return weak and strict


def _find_source_baseline(
    points: Sequence[Mapping[str, Any]], *, retrieval_config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    target = {
        "top_k": int(retrieval_config["top_k"]),
        "threshold": float(retrieval_config["threshold"]),
        "maximum_edit_ratio": float(retrieval_config["maximum_edit_ratio"]),
        "minimum_posterior_confidence": float(
            retrieval_config["minimum_posterior_confidence"]
        ),
        "minimum_top1_margin": float(retrieval_config["minimum_top1_margin"]),
    }
    return next((point for point in points if point["config"] == target), None)


def _find_source_gate_point(
    points: Sequence[Mapping[str, Any]],
    *,
    retrieval_config: Mapping[str, Any],
    top_k: int,
) -> Mapping[str, Any] | None:
    target = {
        "top_k": top_k,
        "threshold": float(retrieval_config["threshold"]),
        "maximum_edit_ratio": float(retrieval_config["maximum_edit_ratio"]),
        "minimum_posterior_confidence": float(
            retrieval_config["minimum_posterior_confidence"]
        ),
        "minimum_top1_margin": float(retrieval_config["minimum_top1_margin"]),
    }
    return next((point for point in points if point["config"] == target), None)


def _metric_delta(
    selected: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {
        name: _metric(selected, name) - _metric(baseline, name)
        for name in ("recall", "precision", "negative_case_false_positive_rate")
    }


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, maximum_top_k: int) -> None:
    final_counts: dict[str, int] = {}
    case_ids = {str(row.get("case_id")) for row in rows}
    for row in rows:
        case_id = str(row.get("case_id"))
        if bool(row.get("is_final")):
            final_counts[case_id] = final_counts.get(case_id, 0) + 1
        matches = row.get("top_matches")
        if not isinstance(matches, list):
            raise ValueError("query row is missing top_matches")
        if not bool(row.get("ranked_matches_complete")):
            raise ValueError(
                "query rows do not contain the complete ranked shortlist; rerun the Anchor "
                "benchmark with --saved-ranked-matches equal to the shortlist size"
            )
        available = int(row.get("ranked_matches_available", -1))
        if available != len(matches) or len(matches) > int(row["candidate_count"]):
            raise ValueError("query row ranked-match completeness metadata is inconsistent")
        if len(matches) < maximum_top_k and len(matches) != available:
            raise ValueError("query row cannot support the requested maximum top-k")
        hotword_ids = [str(match["hotword_id"]) for match in matches]
        if len(set(hotword_ids)) != len(hotword_ids):
            raise ValueError("query row contains duplicate ranked hotwords")
        scores = [float(match["score"]) for match in matches]
        if any(left < right for left, right in zip(scores, scores[1:], strict=False)):
            raise ValueError("query row matches are not sorted by descending score")
    if set(final_counts) != case_ids or any(count != 1 for count in final_counts.values()):
        raise ValueError("selected query rows must contain exactly one final row per case")


def _point_id(config: Mapping[str, Any]) -> str:
    return (
        f"k{int(config['top_k'])}"
        f"_t{float(config['threshold']):.4f}"
        f"_e{float(config['maximum_edit_ratio']):.4f}"
        f"_p{float(config['minimum_posterior_confidence']):.4f}"
        f"_m{float(config['minimum_top1_margin']):.4f}"
    )


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        if name == "negative_case_false_positive_rate":
            return 0.0
        return -1.0
    return float(value)


def _sorted_unique_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain positive integers")
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError(f"{name} must be unique and strictly increasing")
    return resolved


def _sorted_unique_probabilities(
    values: Sequence[float], *, name: str
) -> tuple[float, ...]:
    resolved = tuple(values)
    if not resolved or any(not 0.0 <= value <= 1.0 for value in resolved):
        raise ValueError(f"{name} must contain values in [0, 1]")
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError(f"{name} must be unique and strictly increasing")
    return resolved


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


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_sha256_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "sha256.txt"
    )
    (directory / "sha256.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )
