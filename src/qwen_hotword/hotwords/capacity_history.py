from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

OBSERVATION_KS = (5, 7, 10)


def build_hotword_capacity_history(
    *,
    stages: Sequence[tuple[str, str | Path]],
    output_dir: str | Path,
    profiles: Sequence[str] = ("representative",),
    sizes: Sequence[int] | None = None,
) -> dict[str, object]:
    if not stages:
        raise ValueError("at least one capacity history stage is required")
    labels = [label for label, _ in stages]
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("capacity history stage labels must be non-empty and unique")
    resolved_profiles = tuple(dict.fromkeys(profiles))
    if not resolved_profiles:
        raise ValueError("capacity history profiles must not be empty")
    resolved_sizes = None if sizes is None else tuple(sorted(set(sizes)))
    if resolved_sizes is not None and (
        not resolved_sizes or any(size <= 0 for size in resolved_sizes)
    ):
        raise ValueError("capacity history sizes must be positive")

    destination = Path(output_dir).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"capacity history output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    history_rows: list[dict[str, object]] = []
    stage_inputs: list[dict[str, object]] = []
    for stage_order, (label, directory_value) in enumerate(stages, start=1):
        directory = Path(directory_value).expanduser()
        summary_path = directory / "summary.json"
        query_path = directory / "query_results.jsonl"
        if not summary_path.is_file() or not query_path.is_file():
            raise FileNotFoundError(
                f"capacity stage requires summary.json and query_results.jsonl: {directory}"
            )
        summary = _load_json(summary_path)
        if bool(summary.get("test_set_used")):
            raise ValueError(f"capacity history refuses test-set stage: {label}")
        rows = _load_jsonl(query_path)
        stage_inputs.append(
            {
                "stage": label,
                "directory": str(directory),
                "summary_sha256": _sha256(summary_path),
                "query_results_sha256": _sha256(query_path),
            }
        )
        grouped: dict[tuple[str, int, str | None, int | None], list[dict[str, Any]]] = {}
        for row in rows:
            profile = str(row["profile"])
            size = int(row["size"])
            if profile not in resolved_profiles:
                continue
            if resolved_sizes is not None and size not in resolved_sizes:
                continue
            window = str(row["window"]) if "window" in row else None
            shortlist_size = int(row["shortlist_size"]) if "shortlist_size" in row else None
            grouped.setdefault((profile, size, window, shortlist_size), []).append(row)
        if not grouped:
            raise ValueError(f"capacity stage {label} has no rows after profile/size filters")
        for (profile, size, window, shortlist_size), level_rows in sorted(
            grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
        ):
            history_rows.extend(
                _normalize_stage_rows(
                    stage=label,
                    stage_order=stage_order,
                    directory=directory,
                    summary=summary,
                    profile=profile,
                    size=size,
                    window=window,
                    shortlist_size=shortlist_size,
                    rows=level_rows,
                )
            )

    report = {
        "schema_version": 1,
        "status": "pass",
        "metric_definition": {
            "raw_recall_at_k": "correct expected hotword hits divided by expected hotwords",
            "raw_precision_at_k": (
                "correct expected hotword hits divided by actual returned raw candidates"
            ),
            "operating_precision_at_k": (
                "correct expected hotword hits divided by actually selected Operating@K candidates"
            ),
            "top7_top10_usage": (
                "Operating observation only; runtime prompt injection remains Top-5"
            ),
        },
        "profiles": list(resolved_profiles),
        "sizes": None if resolved_sizes is None else list(resolved_sizes),
        "stage_count": len(stages),
        "ranking_rows": len(history_rows),
        "stages": stage_inputs,
        "history": history_rows,
        "test_set_used": False,
    }
    _write_json(destination / "run_config.json", {
        "schema_version": 1,
        "stages": stage_inputs,
        "profiles": list(resolved_profiles),
        "sizes": None if resolved_sizes is None else list(resolved_sizes),
        "test_set_used": False,
    })
    _write_json(destination / "optimization_history.json", report)
    _write_tsv(destination / "optimization_history.tsv", history_rows)
    _write_sha256_manifest(destination)
    return report


def _normalize_stage_rows(
    *,
    stage: str,
    stage_order: int,
    directory: Path,
    summary: Mapping[str, Any],
    profile: str,
    size: int,
    window: str | None,
    shortlist_size: int | None,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    final = [row for row in rows if bool(row["is_final"])]
    if not final:
        raise ValueError(f"capacity stage {stage} has no final rows for {profile}/{size}")
    common = {
        "schema_version": 1,
        "stage_order": stage_order,
        "stage": stage,
        "output_dir": str(directory),
        "mode": str(summary.get("mode", "full_scan_edit_distance")),
        "profile": profile,
        "size": size,
        "window": window,
        "shortlist_size": shortlist_size,
        "expected_hotwords": sum(len(row["expected_hotword_ids"]) for row in final),
        "positive_cases": sum(bool(row["expected_hotword_ids"]) for row in final),
        "negative_cases": sum(not bool(row["expected_hotword_ids"]) for row in final),
        "timing_protocol": summary.get("timing_protocol"),
        "gc_policy": summary.get("gc_policy"),
        "rerank_mode": summary.get("rerank_mode"),
        "anchor_start_radius": summary.get("anchor_start_radius"),
        "test_set_used": False,
    }
    normalized: list[dict[str, object]] = []
    first = final[0]
    if "candidate_ids" in first and "raw_top5_ids" in first:
        normalized.append(
            _ranking_row(
                common,
                ranking="anchor_rerank",
                rows=rows,
                id_key=lambda k: f"raw_top{k}_ids",
                latency_key="retrieval_seconds",
                latency_scope="anchor_query_plus_shortlist_rerank",
                operating_key="operating_ids",
            )
        )
    elif "raw_top5_ids" in first:
        normalized.append(
            _ranking_row(
                common,
                ranking="full_scan_raw",
                rows=rows,
                id_key=lambda k: f"raw_top{k}_ids",
                latency_key="retrieval_seconds",
                latency_scope="full_scan_retrieval",
                operating_key="operating_ids",
            )
        )
    elif "exact_top5_ids" in first:
        normalized.append(
            _ranking_row(
                common,
                ranking="aho_corasick_exact",
                rows=rows,
                id_key=lambda k: f"exact_top{k}_ids",
                latency_key="retrieval_seconds",
                latency_scope="exact_aho_corasick_retrieval",
                operating_key=None,
            )
        )
    elif "candidate_ids_at_64" in first or "anchor_top5_ids" in first:
        normalized.append(
            _ranking_row(
                common,
                ranking="anchor_shortlist",
                rows=rows,
                id_key=lambda k: (
                    f"anchor_top{k}_ids"
                    if f"anchor_top{k}_ids" in first
                    else _anchor_source_key(first)
                ),
                latency_key="anchor_retrieval_seconds",
                latency_scope="anchor_index_query_only",
                operating_key=None,
                slice_to_k=True,
            )
        )
        normalized.append(
            _ranking_row(
                common,
                ranking="full_scan_reference",
                rows=rows,
                id_key=lambda k: f"reference_raw_top{k}_ids",
                latency_key="full_scan_reference_seconds",
                latency_scope="full_scan_reference_only",
                operating_key=(
                    "reference_operating_ids" if "reference_operating_ids" in first else None
                ),
            )
        )
    else:
        raise ValueError(f"capacity stage {stage} query schema is not recognized")
    return normalized


def _anchor_source_key(row: Mapping[str, Any]) -> str:
    candidate_sizes = sorted(
        int(key.rsplit("_", 1)[1])
        for key in row
        if key.startswith("candidate_ids_at_")
    )
    if not candidate_sizes:
        raise ValueError("anchor capacity row has no candidate shortlist")
    return f"candidate_ids_at_{candidate_sizes[0]}"


def _ranking_row(
    common: Mapping[str, object],
    *,
    ranking: str,
    rows: Sequence[Mapping[str, Any]],
    id_key: Any,
    latency_key: str,
    latency_scope: str,
    operating_key: str | None,
    slice_to_k: bool = False,
) -> dict[str, object]:
    final = [row for row in rows if bool(row["is_final"])]
    expected_total = cast(int, common["expected_hotwords"])
    positive = [row for row in final if row["expected_hotword_ids"]]
    negative = [row for row in final if not row["expected_hotword_ids"]]
    result = dict(common)
    result.update(
        {
            "ranking": ranking,
            "top7_top10_observation_only": True,
            "latency_scope": latency_scope,
        }
    )
    for k in OBSERVATION_KS:
        key = id_key(k)
        if any(key not in row for row in final):
            result[f"raw_correct_at_{k}"] = None
            result[f"raw_recall_at_{k}"] = None
            result[f"raw_precision_at_{k}"] = None
            result[f"positive_case_hit_rate_at_{k}"] = None
            continue
        selected = [list(row[key])[:k] if slice_to_k else list(row[key]) for row in final]
        hits = sum(
            len(set(row["expected_hotword_ids"]) & set(ids))
            for row, ids in zip(final, selected, strict=True)
        )
        selected_total = sum(len(ids) for ids in selected)
        result[f"raw_correct_at_{k}"] = hits
        result[f"raw_recall_at_{k}"] = _ratio(hits, expected_total)
        result[f"raw_precision_at_{k}"] = _ratio(hits, selected_total)
        result[f"positive_case_hit_rate_at_{k}"] = _ratio(
            sum(
                bool(set(row["expected_hotword_ids"]) & set(ids))
                for row, ids in zip(final, selected, strict=True)
                if row["expected_hotword_ids"]
            ),
            len(positive),
        )
        _add_any_step_ranking_metrics(
            result,
            rows=rows,
            final=final,
            key=key,
            k=k,
            slice_to_k=slice_to_k,
        )
    for k in OBSERVATION_KS:
        resolved_operating_key = _resolve_operating_key(
            final[0], base_key=operating_key, k=k
        )
        if resolved_operating_key is None or any(
            resolved_operating_key not in row for row in final
        ):
            result[f"operating_correct_at_{k}"] = None
            result[f"operating_recall_at_{k}"] = None
            result[f"operating_precision_at_{k}"] = None
            result[f"operating_positive_case_hit_rate_at_{k}"] = None
            result[f"negative_case_false_positive_rate_at_{k}"] = None
            result[f"any_step_operating_correct_at_{k}"] = None
            result[f"any_step_operating_recall_at_{k}"] = None
            result[f"any_step_operating_precision_at_{k}"] = None
            result[f"any_step_operating_positive_case_hit_rate_at_{k}"] = None
            result[f"any_step_negative_case_false_positive_rate_at_{k}"] = None
            continue
        operating_hits = sum(
            len(
                set(row["expected_hotword_ids"])
                & set(row[resolved_operating_key])
            )
            for row in final
        )
        operating_selected = sum(len(row[resolved_operating_key]) for row in final)
        result[f"operating_correct_at_{k}"] = operating_hits
        result[f"operating_recall_at_{k}"] = _ratio(operating_hits, expected_total)
        result[f"operating_precision_at_{k}"] = _ratio(
            operating_hits, operating_selected
        )
        result[f"operating_positive_case_hit_rate_at_{k}"] = _ratio(
            sum(
                bool(
                    set(row["expected_hotword_ids"])
                    & set(row[resolved_operating_key])
                )
                for row in positive
            ),
            len(positive),
        )
        result[f"negative_case_false_positive_rate_at_{k}"] = _ratio(
            sum(bool(row[resolved_operating_key]) for row in negative), len(negative)
        )
        _add_any_step_operating_metrics(
            result,
            rows=rows,
            final=final,
            operating_key=resolved_operating_key,
            k=k,
        )
    result["negative_case_false_positive_rate"] = result[
        "negative_case_false_positive_rate_at_5"
    ]
    result["any_step_negative_case_false_positive_rate"] = result[
        "any_step_negative_case_false_positive_rate_at_5"
    ]
    latency = [float(row[latency_key]) for row in rows]
    result.update({f"latency_{key}": value for key, value in _distribution(latency).items()})
    return result


def _add_any_step_ranking_metrics(
    result: dict[str, object],
    *,
    rows: Sequence[Mapping[str, Any]],
    final: Sequence[Mapping[str, Any]],
    key: str,
    k: int,
    slice_to_k: bool,
) -> None:
    if any(key not in row for row in rows):
        result[f"any_step_raw_correct_at_{k}"] = None
        result[f"any_step_raw_recall_at_{k}"] = None
        result[f"any_step_positive_case_hit_rate_at_{k}"] = None
        return
    expected_pairs, positive_case_ids, _ = _case_truth(final)
    rows_by_case = _rows_by_case(rows)
    detected = {
        (case_id, hotword_id)
        for case_id, hotword_id in expected_pairs
        if any(
            hotword_id in (list(row[key])[:k] if slice_to_k else row[key])
            for row in rows_by_case[case_id]
        )
    }
    returned = {
        (str(row["case_id"]), str(hotword_id))
        for row in rows
        for hotword_id in (list(row[key])[:k] if slice_to_k else row[key])
    }
    result[f"any_step_raw_correct_at_{k}"] = len(detected)
    result[f"any_step_raw_recall_at_{k}"] = _ratio(len(detected), len(expected_pairs))
    result[f"any_step_raw_precision_at_{k}"] = _ratio(len(detected), len(returned))
    result[f"any_step_positive_case_hit_rate_at_{k}"] = _ratio(
        len({case_id for case_id, _ in detected}), len(positive_case_ids)
    )


def _add_any_step_operating_metrics(
    result: dict[str, object],
    *,
    rows: Sequence[Mapping[str, Any]],
    final: Sequence[Mapping[str, Any]],
    operating_key: str,
    k: int,
) -> None:
    if any(operating_key not in row for row in rows):
        result[f"any_step_operating_correct_at_{k}"] = None
        result[f"any_step_operating_recall_at_{k}"] = None
        result[f"any_step_operating_precision_at_{k}"] = None
        result[f"any_step_operating_positive_case_hit_rate_at_{k}"] = None
        result[f"any_step_negative_case_false_positive_rate_at_{k}"] = None
        return
    expected_pairs, positive_case_ids, negative_case_ids = _case_truth(final)
    rows_by_case = _rows_by_case(rows)
    detected = {
        (case_id, hotword_id)
        for case_id, hotword_id in expected_pairs
        if any(hotword_id in row[operating_key] for row in rows_by_case[case_id])
    }
    returned = {
        (str(row["case_id"]), str(hotword_id))
        for row in rows
        for hotword_id in row[operating_key]
    }
    result[f"any_step_operating_correct_at_{k}"] = len(detected)
    result[f"any_step_operating_recall_at_{k}"] = _ratio(
        len(detected), len(expected_pairs)
    )
    result[f"any_step_operating_precision_at_{k}"] = _ratio(
        len(detected), len(returned)
    )
    result[f"any_step_operating_positive_case_hit_rate_at_{k}"] = _ratio(
        len({case_id for case_id, _ in detected}), len(positive_case_ids)
    )
    result[f"any_step_negative_case_false_positive_rate_at_{k}"] = _ratio(
        sum(
            any(row[operating_key] for row in rows_by_case[case_id])
            for case_id in negative_case_ids
        ),
        len(negative_case_ids),
    )


def _resolve_operating_key(
    row: Mapping[str, Any], *, base_key: str | None, k: int
) -> str | None:
    for candidate in (
        f"operating_top{k}_ids",
        f"reference_operating_top{k}_ids",
    ):
        if candidate in row:
            return candidate
    if k == 5 and base_key is not None and base_key in row:
        return base_key
    return None


def _case_truth(
    final: Sequence[Mapping[str, Any]],
) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    expected_pairs = {
        (str(row["case_id"]), str(hotword_id))
        for row in final
        for hotword_id in row["expected_hotword_ids"]
    }
    positive_case_ids = {
        str(row["case_id"]) for row in final if row["expected_hotword_ids"]
    }
    negative_case_ids = {
        str(row["case_id"]) for row in final if not row["expected_hotword_ids"]
    }
    return expected_pairs, positive_case_ids, negative_case_ids


def _rows_by_case(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    return grouped


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "p50": median(ordered),
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"capacity history JSON must contain an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"capacity history JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    if not rows:
        raise ValueError(f"capacity history query results are empty: {path}")
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_manifest(directory: Path) -> None:
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.name != "sha256.txt"
    )
    (directory / "sha256.txt").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in paths),
        encoding="utf-8",
    )
