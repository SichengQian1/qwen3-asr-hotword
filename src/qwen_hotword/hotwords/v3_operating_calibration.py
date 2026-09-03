from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_hotword.hotwords.multi_nested import (
    CaseScore,
    evaluate_multi_nested_case_scores,
    load_hotword_families,
    load_multi_nested_case_scores,
    load_multi_nested_cases,
)
from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.phonemes.coverage import load_phoneme_vocab


def calibrate_v3_operating_points(
    *,
    vocab_path: str | Path,
    hotword_path: str | Path,
    families_path: str | Path,
    cases_path: str | Path,
    case_scores_path: str | Path,
    candidate_report_path: str | Path,
    output_dir: str | Path,
    reference_report_path: str | Path | None = None,
    top_ks: Sequence[int] = (1, 3, 5),
    thresholds: Sequence[float] = (
        0.70,
        0.71,
        0.72,
        0.73,
        0.74,
        0.75,
        0.76,
        0.77,
        0.78,
        0.79,
        0.80,
        0.81,
        0.82,
        0.83,
        0.84,
        0.85,
        0.86,
        0.87,
        0.88,
        0.89,
        0.90,
    ),
    minimum_posterior_confidences: Sequence[float] = (0.0, 0.25, 0.50, 0.75),
) -> dict[str, object]:
    """Replay v3 gates over saved ranks, recommending only provably exact points."""
    resolved_top_ks = _sorted_unique_ints(top_ks, name="top_ks")
    resolved_thresholds = _sorted_unique_probabilities(thresholds, name="thresholds")
    resolved_posteriors = _sorted_unique_probabilities(
        minimum_posterior_confidences,
        name="minimum_posterior_confidences",
    )
    if max(resolved_top_ks) > 5:
        raise ValueError("saved v3 ranking_top5 cannot replay top_k greater than 5")

    paths = {
        "vocab": Path(vocab_path).expanduser(),
        "hotwords": Path(hotword_path).expanduser(),
        "families": Path(families_path).expanduser(),
        "cases": Path(cases_path).expanduser(),
        "case_scores": Path(case_scores_path).expanduser(),
        "candidate_report": Path(candidate_report_path).expanduser(),
    }
    if reference_report_path is not None:
        paths["reference_report"] = Path(reference_report_path).expanduser()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    destination = Path(output_dir).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"v3 calibration output must be new and empty: {destination}")

    candidate_report = _read_json_object(paths["candidate_report"])
    reference_report = (
        _read_json_object(paths["reference_report"]) if "reference_report" in paths else None
    )
    _validate_report(candidate_report, paths=paths, label="candidate")
    if reference_report is not None:
        _validate_report(reference_report, paths=paths, label="reference")

    vocab = load_phoneme_vocab(paths["vocab"])
    hotwords = load_hotword_table(paths["hotwords"], vocab=vocab, blank_id=0)
    families = load_hotword_families(paths["families"])
    cases = load_multi_nested_cases(paths["cases"])
    source_scores = load_multi_nested_case_scores(paths["case_scores"])
    _validate_saved_scores(source_scores)

    source_metrics = evaluate_multi_nested_case_scores(cases, hotwords, families, source_scores)
    _validate_report_metrics(candidate_report, source_metrics)
    scoring_config = _mapping(candidate_report, "scoring_config")
    maximum_edit_ratio = float(scoring_config["maximum_edit_ratio"])
    minimum_top1_margin = float(scoring_config["minimum_top1_margin"])
    if minimum_top1_margin != 0.0:
        raise ValueError("v3 calibration currently requires minimum_top1_margin=0")

    source_config = {
        "top_k": int(scoring_config["top_k"]),
        "threshold": float(scoring_config["threshold"]),
        "maximum_edit_ratio": maximum_edit_ratio,
        "minimum_posterior_confidence": float(scoring_config["minimum_posterior_confidence"]),
        "minimum_top1_margin": minimum_top1_margin,
    }
    candidate_baseline = _compact_metrics(source_metrics)
    reference_baseline = (
        _compact_metrics(_mapping(reference_report, "metrics"))
        if reference_report is not None
        else None
    )

    points: list[dict[str, Any]] = []
    for top_k, threshold, posterior in itertools.product(
        resolved_top_ks, resolved_thresholds, resolved_posteriors
    ):
        config = {
            "top_k": top_k,
            "threshold": threshold,
            "maximum_edit_ratio": maximum_edit_ratio,
            "minimum_posterior_confidence": posterior,
            "minimum_top1_margin": minimum_top1_margin,
        }
        scores, incomplete_case_ids = _replay_scores(
            source_scores,
            top_k=top_k,
            threshold=threshold,
            maximum_edit_ratio=maximum_edit_ratio,
            minimum_posterior_confidence=posterior,
        )
        is_source = config == source_config
        if is_source:
            scores = source_scores
            incomplete_case_ids = ()
        metrics = evaluate_multi_nested_case_scores(cases, hotwords, families, scores)
        points.append(
            {
                "schema_version": 1,
                "point_id": _point_id(config),
                "config": config,
                "is_source_operating_point": is_source,
                "replay_exact": not incomplete_case_ids,
                "replay_complete_cases": len(cases) - len(incomplete_case_ids),
                "replay_total_cases": len(cases),
                "incomplete_case_count": len(incomplete_case_ids),
                "incomplete_case_ids": list(incomplete_case_ids),
                "metrics": _compact_metrics(metrics),
                "delta_from_candidate_baseline": _metric_delta(
                    _compact_metrics(metrics), candidate_baseline
                ),
                "test_set_used": False,
            }
        )

    exact_points = [point for point in points if bool(point["replay_exact"])]
    frontier = _pareto_frontier(exact_points)
    candidates = _select_candidates(exact_points, baseline=candidate_baseline)
    if reference_baseline is not None:
        for candidate in candidates:
            candidate["delta_from_reference_baseline"] = _metric_delta(
                _mapping(candidate, "metrics"), reference_baseline
            )
    precision_guarded_recall_gain = any(
        candidate["role"] == "precision_guarded"
        and float(candidate["delta_from_candidate_baseline"]["recall"]) > 1e-12
        for candidate in candidates
    )
    limitation = (
        "all requested points were exactly replayable from ranking_top5"
        if len(exact_points) == len(points)
        else (
            "ranking_top5 is truncated before Operating guards; non-exact points are "
            "diagnostic only and excluded from recommendations"
        )
    )
    if not candidates:
        status = "full_shortlist_required"
    elif precision_guarded_recall_gain:
        status = "guarded_recall_gain_candidate_available"
    else:
        status = "exact_sweep_complete_no_guarded_recall_gain"

    destination.mkdir(parents=True, exist_ok=True)
    identities = {name: _file_identity(path) for name, path in paths.items()}
    run_config = {
        "schema_version": 1,
        "purpose": "portuguese_v3_multilingual_ctc_operating_point_calibration",
        "inputs": identities,
        "candidate_checkpoint_sha256": candidate_report.get("checkpoint_sha256"),
        "reference_checkpoint_sha256": (
            reference_report.get("checkpoint_sha256") if reference_report is not None else None
        ),
        "fixed_ranking": {
            "posterior_weight": float(scoring_config["posterior_weight"]),
            "maximum_edit_ratio": maximum_edit_ratio,
            "minimum_top1_margin": minimum_top1_margin,
            "saved_rank_depth": 5,
        },
        "grid": {
            "top_ks": list(resolved_top_ks),
            "thresholds": list(resolved_thresholds),
            "minimum_posterior_confidences": list(resolved_posteriors),
        },
        "test_set_used": False,
        "ctc_inference_performed": False,
        "qwen_inference_performed": False,
    }
    summary = {
        "schema_version": 1,
        "status": status,
        "case_count": len(cases),
        "hotword_count": len(hotwords),
        "sweep_point_count": len(points),
        "exact_point_count": len(exact_points),
        "non_exact_point_count": len(points) - len(exact_points),
        "pareto_point_count": len(frontier),
        "guarded_recall_gain_candidate_available": precision_guarded_recall_gain,
        "candidate_baseline": candidate_baseline,
        "reference_baseline": reference_baseline,
        "recommended_candidates": candidates,
        "selection_policy": {
            "precision_guarded": (
                "maximize recall among exact points with precision no lower and negative "
                "FPR no higher than the candidate source operating point"
            ),
            "f1_with_fpr_guard": (
                "maximize F1 among exact points with negative FPR no higher than the "
                "candidate source operating point"
            ),
            "quality_threshold_is_predeclared": False,
        },
        "replay_limitation": limitation,
        "next_action": (
            "review one or two exact candidates before any end-to-end rerun"
            if precision_guarded_recall_gain
            else (
                "do not rerun end-to-end yet; inspect the exact frontier and, if recall "
                "cannot improve, save a complete ranked shortlist in one CTC-only pass"
            )
            if candidates
            else (
                "rerun CTC-only v3 scoring once with a saved complete ranked shortlist; "
                "do not choose a deployment gate from truncated estimates"
            )
        ),
        "test_set_used": False,
        "ctc_inference_performed": False,
        "qwen_inference_performed": False,
    }
    _write_json(destination / "calibration_config.json", run_config)
    _write_jsonl(destination / "operating_point_sweep.jsonl", points)
    _write_jsonl(destination / "exact_pareto_frontier.jsonl", frontier)
    _write_json(destination / "candidate_summary.json", summary)
    _write_readme(destination / "README.md", summary)
    _write_sha256_manifest(destination)
    return summary


def _replay_scores(
    scores: Sequence[CaseScore],
    *,
    top_k: int,
    threshold: float,
    maximum_edit_ratio: float,
    minimum_posterior_confidence: float,
) -> tuple[tuple[CaseScore, ...], tuple[str, ...]]:
    replayed: list[CaseScore] = []
    incomplete: list[str] = []
    for score in scores:
        eligible = tuple(
            match
            for match in score.ranked_matches
            if match.score >= threshold
            and match.edit_ratio <= maximum_edit_ratio
            and match.posterior_confidence >= minimum_posterior_confidence
        )
        selected = eligible[:top_k]
        last_saved_score = score.ranked_matches[-1].score
        complete = len(selected) == top_k or last_saved_score < threshold
        if not complete:
            incomplete.append(score.case_id)
        replayed.append(
            CaseScore(
                case_id=score.case_id,
                sample_id=score.sample_id,
                primary_group=score.primary_group,
                ranked_matches=score.ranked_matches,
                operating_matches=selected,
                effective_time_steps=score.effective_time_steps,
                decoded_token_count=score.decoded_token_count,
            )
        )
    return tuple(replayed), tuple(incomplete)


def _validate_saved_scores(scores: Sequence[CaseScore]) -> None:
    for score in scores:
        if len(score.ranked_matches) != 5:
            raise ValueError(f"case {score.case_id} must contain exactly five saved ranked matches")
        ids = [match.hotword_id for match in score.ranked_matches]
        if len(ids) != len(set(ids)):
            raise ValueError(f"case {score.case_id} contains duplicate ranked hotwords")
        values = [match.score for match in score.ranked_matches]
        if any(left < right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError(f"case {score.case_id} ranked matches are not score-sorted")


def _validate_report(report: Mapping[str, Any], *, paths: Mapping[str, Path], label: str) -> None:
    if bool(report.get("test_set_used")):
        raise ValueError(f"{label} report used the sealed test set")
    checkpoint_sha = report.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError(f"{label} report has no valid checkpoint SHA256")
    for report_key, path_key in (
        ("vocab_sha256", "vocab"),
        ("hotword_table_sha256", "hotwords"),
        ("families_sha256", "families"),
        ("cases_sha256", "cases"),
    ):
        expected = report.get(report_key)
        actual = _sha256(paths[path_key])
        if expected != actual:
            raise ValueError(f"{label} report {report_key} does not match {path_key}")


def _validate_report_metrics(report: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    reported = _compact_metrics(_mapping(report, "metrics"))["overall"]
    actual = _compact_metrics(recomputed)["overall"]
    for name in (
        "expected_hotwords",
        "selected_hotwords",
        "true_positive_hotwords",
        "precision",
        "recall",
        "f1",
        "positive_case_hit_rate",
        "negative_case_false_positive_rate",
    ):
        if not _same_number(reported[name], actual[name]):
            raise ValueError(
                f"candidate report metrics do not match case scores for overall.{name}"
            )


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    overall = _mapping(_mapping(metrics, "overall"), "operating")
    groups = _mapping(metrics, "by_primary_group")
    nested = _mapping(metrics, "nested")
    return {
        "overall": dict(overall),
        "by_primary_group": {
            name: dict(_mapping(value, "operating"))
            for name, value in groups.items()
            if isinstance(value, Mapping)
        },
        "nested": {
            name: nested.get(name)
            for name in (
                "short_only_short_operating_recall",
                "short_only_long_operating_false_trigger_rate",
                "long_present_long_operating_recall",
                "long_present_short_operating_recall",
            )
        },
    }


def _select_candidates(
    points: Sequence[Mapping[str, Any]], *, baseline: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not points:
        return []
    baseline_overall = _mapping(baseline, "overall")
    baseline_precision = float(baseline_overall["precision"])
    baseline_fpr = float(baseline_overall["negative_case_false_positive_rate"])
    guarded = [
        point
        for point in points
        if _overall(point, "precision") >= baseline_precision
        and _overall(point, "negative_case_false_positive_rate") <= baseline_fpr
    ]
    selected: list[tuple[str, Mapping[str, Any]]] = []
    if guarded:
        selected.append(
            (
                "precision_guarded",
                min(
                    guarded,
                    key=lambda point: (
                        -_overall(point, "recall"),
                        -_overall(point, "f1"),
                        float(_mapping(point, "config")["top_k"]),
                        -float(_mapping(point, "config")["threshold"]),
                    ),
                ),
            )
        )
    fpr_guarded = [
        point
        for point in points
        if _overall(point, "negative_case_false_positive_rate") <= baseline_fpr
    ]
    if fpr_guarded:
        f1_best = min(
            fpr_guarded,
            key=lambda point: (
                -_overall(point, "f1"),
                -_overall(point, "recall"),
                -_overall(point, "precision"),
                float(_mapping(point, "config")["top_k"]),
                -float(_mapping(point, "config")["threshold"]),
            ),
        )
        if not any(point["point_id"] == f1_best["point_id"] for _, point in selected):
            selected.append(("f1_with_fpr_guard", f1_best))
    return [
        {
            "role": role,
            "point_id": point["point_id"],
            "config": point["config"],
            "metrics": point["metrics"],
            "delta_from_candidate_baseline": point["delta_from_candidate_baseline"],
        }
        for role, point in selected[:2]
    ]


def _pareto_frontier(points: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    frontier = [
        point
        for point in points
        if not any(other is not point and _dominates(other, point) for other in points)
    ]
    unique: dict[tuple[float, float, float], Mapping[str, Any]] = {}
    for point in frontier:
        key = (
            _overall(point, "recall"),
            _overall(point, "precision"),
            _overall(point, "negative_case_false_positive_rate"),
        )
        current = unique.get(key)
        if current is None or _config_key(point) < _config_key(current):
            unique[key] = point
    return sorted(
        unique.values(),
        key=lambda point: (
            -_overall(point, "recall"),
            -_overall(point, "precision"),
            _overall(point, "negative_case_false_positive_rate"),
            _config_key(point),
        ),
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_values = (
        _overall(left, "recall"),
        _overall(left, "precision"),
        -_overall(left, "negative_case_false_positive_rate"),
    )
    right_values = (
        _overall(right, "recall"),
        _overall(right, "precision"),
        -_overall(right, "negative_case_false_positive_rate"),
    )
    return all(a >= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a > b for a, b in zip(left_values, right_values, strict=True)
    )


def _config_key(point: Mapping[str, Any]) -> tuple[float, float, float]:
    config = _mapping(point, "config")
    return (
        float(config["top_k"]),
        -float(config["threshold"]),
        -float(config["minimum_posterior_confidence"]),
    )


def _metric_delta(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    current = _mapping(metrics, "overall")
    source = _mapping(baseline, "overall")
    return {
        name: float(current[name]) - float(source[name])
        for name in (
            "recall",
            "precision",
            "f1",
            "positive_case_hit_rate",
            "negative_case_false_positive_rate",
        )
    }


def _overall(point: Mapping[str, Any], name: str) -> float:
    metrics = _mapping(_mapping(point, "metrics"), "overall")
    return float(metrics[name])


def _point_id(config: Mapping[str, Any]) -> str:
    return (
        f"k{int(config['top_k'])}"
        f"_t{float(config['threshold']):.4f}"
        f"_p{float(config['minimum_posterior_confidence']):.4f}"
    )


def _validate_probability_sequence(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    resolved = tuple(values)
    if not resolved or any(not 0.0 <= value <= 1.0 for value in resolved):
        raise ValueError(f"{name} must contain values in [0, 1]")
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError(f"{name} must be unique and strictly increasing")
    return resolved


def _sorted_unique_probabilities(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    return _validate_probability_sequence(values, name=name)


def _sorted_unique_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain positive integers")
    if tuple(sorted(set(resolved))) != resolved:
        raise ValueError(f"{name} must be unique and strictly increasing")
    return resolved


def _mapping(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(f"missing object: {name}")
    child = value.get(name)
    if not isinstance(child, Mapping):
        raise ValueError(f"missing object: {name}")
    return child


def _same_number(left: object, right: object) -> bool:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return left == right
    return abs(float(left) - float(right)) <= 1e-12


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_readme(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(
        "# Portuguese v3 operating-point calibration\n\n"
        f"Status: `{summary['status']}`.\n\n"
        "This directory replays threshold, posterior-confidence, and Top-K gates "
        "without loading the CTC Head or Qwen. Recommendations include only points "
        "whose result is provably complete from the saved `ranking_top5`. Non-exact "
        "rows remain diagnostics and must not be used as deployment settings.\n",
        encoding="utf-8",
    )


def _write_sha256_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "sha256.txt"
    )
    (directory / "sha256.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )
