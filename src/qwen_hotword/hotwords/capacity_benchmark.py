from __future__ import annotations

import gc
import hashlib
import json
import math
import resource
import sys
import time
import tracemalloc
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from qwen_hotword.hotwords.capacity_assets import (
    PROFILE_NAMES,
    CapacityCase,
    load_capacity_base_cases,
    parse_capacity_sizes,
)
from qwen_hotword.hotwords.capacity_replay import (
    load_capacity_replay,
    replay_decoded_phonemes,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.hotwords.scoring import (
    HotwordScoringConfig,
    profile_decoded_hotwords,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.edit_distance import sequence_edit_distance_backend


def benchmark_hotword_capacity(
    *,
    assets_root: str | Path,
    replay_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    profiles: Sequence[str] = PROFILE_NAMES,
    sizes: Sequence[int] = (100, 500, 1_000, 2_000, 5_000, 10_000),
    threshold: float = 0.86,
    top_k: int = 5,
    minimum_phonemes: int = 4,
    maximum_edit_ratio: float = 0.35,
    posterior_weight: float = 0.25,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.0,
    warmup_queries: int = 1,
    stop_retrieval_p95_seconds: float = 2.0,
    continue_after_deadline_failure: bool = False,
    print_progress: bool = True,
) -> dict[str, object]:
    resolved_profiles = tuple(dict.fromkeys(profiles))
    if not resolved_profiles or any(profile not in PROFILE_NAMES for profile in resolved_profiles):
        raise ValueError(f"profiles must be a non-empty subset of {PROFILE_NAMES}")
    resolved_sizes = parse_capacity_sizes(sizes)
    if top_k != 5:
        raise ValueError("capacity v1 is sealed to Top-5 retrieval")
    if warmup_queries < 0:
        raise ValueError("warmup_queries must not be negative")
    if stop_retrieval_p95_seconds <= 0:
        raise ValueError("stop_retrieval_p95_seconds must be positive")
    paths = {
        "assets": Path(assets_root).expanduser(),
        "replay": Path(replay_path).expanduser(),
        "vocab": Path(vocab_path).expanduser(),
        "output": Path(output_dir).expanduser(),
    }
    if not paths["assets"].is_dir():
        raise FileNotFoundError(f"capacity assets root does not exist: {paths['assets']}")
    for key in ("replay", "vocab"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"capacity benchmark input does not exist: {paths[key]}")
    destination = paths["output"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"capacity benchmark output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    vocab = load_phoneme_vocab(paths["vocab"])
    replay_rows = load_capacity_replay(paths["replay"])
    config = HotwordScoringConfig(
        score_threshold=threshold,
        top_k=top_k,
        minimum_phonemes=minimum_phonemes,
        maximum_edit_ratio=maximum_edit_ratio,
        posterior_weight=posterior_weight,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
    )
    config.validate()
    edit_distance_backend = sequence_edit_distance_backend()
    query_rows: list[dict[str, Any]] = []
    level_summaries: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for profile in resolved_profiles:
        profile_summaries: dict[str, Any] = {}
        for size in resolved_sizes:
            level_dir = paths["assets"] / profile / f"size_{size}"
            hotword_path = level_dir / "hotwords.jsonl"
            cases_path = level_dir / "cases.jsonl"
            if not hotword_path.is_file() or not cases_path.is_file():
                raise FileNotFoundError(f"capacity level is incomplete: {level_dir}")
            load_metrics, hotwords = _profile_registry_load(hotword_path, vocab=vocab)
            hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
            cases = load_capacity_base_cases(cases_path)
            case_by_id = {case.case_id: case for case in cases}
            _validate_level(size, cases, hotword_by_id, replay_rows)

            for warmup in replay_rows[:warmup_queries]:
                case = case_by_id[str(warmup["case_id"])]
                active = tuple(hotword_by_id[item] for item in case.active_hotword_ids)
                profile_decoded_hotwords(
                    replay_decoded_phonemes(warmup),
                    effective_time_steps=int(warmup["effective_time_steps"]),
                    hotwords=active,
                    config=config,
                )

            level_rows: list[dict[str, object]] = []
            previous_top5: dict[str, tuple[str, ...]] = {}
            for query_index, replay in enumerate(replay_rows, start=1):
                case = case_by_id[str(replay["case_id"])]
                active = tuple(hotword_by_id[item] for item in case.active_hotword_ids)
                profiled = profile_decoded_hotwords(
                    replay_decoded_phonemes(replay),
                    effective_time_steps=int(replay["effective_time_steps"]),
                    hotwords=active,
                    config=config,
                )
                result = profiled.result
                expected = set(case.expected_hotword_ids)
                ranking_ids = tuple(match.hotword_id for match in result.ranked_matches)
                raw_top5 = ranking_ids[:top_k]
                operating_ids = tuple(match.hotword_id for match in result.selected_matches)
                ranks = {
                    hotword_id: ranking_ids.index(hotword_id) + 1
                    for hotword_id in expected
                    if hotword_id in ranking_ids
                }
                expected_scores = {
                    hotword_id: result.ranked_matches[rank - 1].score
                    for hotword_id, rank in ranks.items()
                }
                top5_floor_score = (
                    result.ranked_matches[min(top_k, len(result.ranked_matches)) - 1].score
                    if result.ranked_matches
                    else None
                )
                expected_top5_margins = {
                    hotword_id: score - top5_floor_score
                    for hotword_id, score in expected_scores.items()
                    if top5_floor_score is not None and ranks[hotword_id] <= top_k
                }
                previous = previous_top5.get(case.case_id)
                churn = None if previous is None else 1.0 - _jaccard(previous, raw_top5)
                previous_top5[case.case_id] = raw_top5
                source_seconds = _source_ctc_seconds(replay.get("source_timings"))
                row = {
                    "schema_version": 1,
                    "profile": profile,
                    "size": size,
                    "case_id": case.case_id,
                    "sample_id": case.sample_id,
                    "primary_group": case.primary_group,
                    "chunk_id": int(replay["chunk_id"]),
                    "cumulative_audio_sec": float(replay["cumulative_audio_sec"]),
                    "is_final": bool(replay["is_final"]),
                    "is_tail_flush": bool(replay["is_tail_flush"]),
                    "active_hotwords": len(active),
                    "expected_hotword_ids": sorted(expected),
                    "expected_ranks": ranks,
                    "expected_scores": expected_scores,
                    "expected_top5_margins": expected_top5_margins,
                    "top5_floor_score": top5_floor_score,
                    "raw_top5_ids": list(raw_top5),
                    "operating_ids": list(operating_ids),
                    "raw_expected_hits_at_5": len(expected & set(raw_top5)),
                    "operating_expected_hits_at_5": len(expected & set(operating_ids)),
                    "negative_false_positive": not expected and bool(operating_ids),
                    "raw_top5_churn": churn,
                    "matching_seconds": profiled.matching_seconds,
                    "sorting_seconds": profiled.sorting_seconds,
                    "selection_seconds": profiled.selection_seconds,
                    "retrieval_seconds": profiled.retrieval_seconds,
                    "source_ctc_seconds": source_seconds,
                    "source_timings": replay.get("source_timings", {}),
                    "ctc_plus_retrieval_seconds": (
                        source_seconds + profiled.retrieval_seconds
                        if source_seconds is not None
                        else None
                    ),
                    "top_matches": [
                        match.to_dict() for match in result.ranked_matches[: max(top_k, 20)]
                    ],
                }
                level_rows.append(row)
                query_rows.append(row)
                if print_progress and (query_index == len(replay_rows) or query_index % 50 == 0):
                    print(
                        f"capacity benchmark profile={profile} size={size} "
                        f"queries={query_index}/{len(replay_rows)}",
                        flush=True,
                    )
            summary = _summarize_level(
                profile=profile,
                size=size,
                rows=level_rows,
                registry_load=load_metrics,
                threshold=threshold,
                top_k=top_k,
            )
            profile_summaries[str(size)] = summary
            del hotwords, hotword_by_id, cases, case_by_id, active, profiled, result
            gc.collect()
            retrieval_p95 = float(summary["performance"]["retrieval_seconds"]["p95"])
            if retrieval_p95 > stop_retrieval_p95_seconds and not continue_after_deadline_failure:
                summary["progressive_stop_triggered"] = True
                summary["progressive_stop_reason"] = (
                    f"retrieval p95 {retrieval_p95:.6f}s exceeds {stop_retrieval_p95_seconds:.6f}s"
                )
                break
        level_summaries[profile] = profile_summaries

    recommendation = _capacity_recommendation(level_summaries, resolved_sizes)
    query_path = destination / "query_results.jsonl"
    quality_path = destination / "quality_summary.json"
    performance_path = destination / "performance_summary.json"
    recommendation_path = destination / "capacity_recommendation.json"
    run_config_path = destination / "run_config.json"
    _write_jsonl(query_path, query_rows)
    quality_summary = {
        profile: {size: value["quality"] for size, value in summaries.items()}
        for profile, summaries in level_summaries.items()
    }
    performance_summary = {
        profile: {
            size: {
                "performance": value["performance"],
                "registry_load": value["registry_load"],
                "progressive_stop_triggered": value.get("progressive_stop_triggered", False),
                "progressive_stop_reason": value.get("progressive_stop_reason"),
            }
            for size, value in summaries.items()
        }
        for profile, summaries in level_summaries.items()
    }
    run_config = {
        "schema_version": 1,
        "purpose": "portuguese_hotword_capacity_replay_benchmark",
        "profiles": list(resolved_profiles),
        "sizes": list(resolved_sizes),
        "retrieval_config": {
            "threshold": threshold,
            "top_k": top_k,
            "minimum_phonemes": minimum_phonemes,
            "maximum_edit_ratio": maximum_edit_ratio,
            "posterior_weight": posterior_weight,
            "minimum_posterior_confidence": minimum_posterior_confidence,
            "minimum_top1_margin": minimum_top1_margin,
            "edit_distance_backend": edit_distance_backend,
        },
        "warmup_queries": warmup_queries,
        "stop_retrieval_p95_seconds": stop_retrieval_p95_seconds,
        "continue_after_deadline_failure": continue_after_deadline_failure,
        "inputs": {
            "asset_summary": _file_identity(paths["assets"] / "asset_summary.json"),
            "replay": _file_identity(paths["replay"]),
            "vocab": _file_identity(paths["vocab"]),
        },
        "test_set_used": False,
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "elapsed_seconds": time.monotonic() - started,
        "query_rows": len(query_rows),
        "edit_distance_backend": edit_distance_backend,
        "levels": level_summaries,
        "recommendation": recommendation,
        "test_set_used": False,
    }
    _write_json(run_config_path, run_config)
    _write_json(quality_path, quality_summary)
    _write_json(performance_path, performance_summary)
    _write_json(recommendation_path, recommendation)
    _write_json(destination / "summary.json", report)
    _write_sha256_manifest(destination)
    return report


def _profile_registry_load(
    path: Path, *, vocab: Any
) -> tuple[dict[str, object], list[HotwordEntry]]:
    gc.collect()
    rss_before = _current_rss_bytes()
    started = time.perf_counter()
    entries = load_hotword_table(path, vocab=vocab, blank_id=0)
    load_seconds = time.perf_counter() - started
    rss_after = _current_rss_bytes()
    del entries
    gc.collect()
    tracemalloc.start()
    profiled_started = time.perf_counter()
    entries = load_hotword_table(path, vocab=vocab, blank_id=0)
    profiled_reload_seconds = time.perf_counter() - profiled_started
    heap_current, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (
        {
            "entries": len(entries),
            "file_bytes": path.stat().st_size,
            "load_seconds": load_seconds,
            "tracemalloc_profiled_reload_seconds": profiled_reload_seconds,
            "python_heap_current_bytes": heap_current,
            "python_heap_peak_bytes": heap_peak,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": (
                rss_after - rss_before if rss_before is not None and rss_after is not None else None
            ),
            "rss_measurement": (
                "linux_proc_status_current"
                if sys.platform.startswith("linux")
                else "ru_maxrss_peak"
            ),
        },
        entries,
    )


def _validate_level(
    size: int,
    cases: Sequence[CapacityCase],
    hotword_by_id: Mapping[str, HotwordEntry],
    replay_rows: Sequence[Mapping[str, Any]],
) -> None:
    case_by_id = {case.case_id: case for case in cases}
    case_ids = set(case_by_id)
    replay_case_ids = {str(row["case_id"]) for row in replay_rows}
    missing_replay_cases = replay_case_ids - case_ids
    if missing_replay_cases:
        raise ValueError(
            f"capacity replay refers to unknown level cases: {sorted(missing_replay_cases)}"
        )
    final_counts: dict[str, int] = {}
    for row in replay_rows:
        case_id = str(row["case_id"])
        case = case_by_id[case_id]
        replay_expected = row.get("expected_hotword_ids")
        if not isinstance(replay_expected, list) or tuple(replay_expected) != tuple(
            case.expected_hotword_ids
        ):
            raise ValueError(f"capacity replay truth differs for case {case_id}")
        if bool(row.get("is_final")):
            final_counts[case_id] = final_counts.get(case_id, 0) + 1
    invalid_final = {
        case_id: final_counts.get(case_id, 0)
        for case_id in replay_case_ids
        if final_counts.get(case_id, 0) != 1
    }
    if invalid_final:
        raise ValueError(f"capacity replay must have one final row per case: {invalid_final}")
    for case in cases:
        if len(case.active_hotword_ids) != size or len(set(case.active_hotword_ids)) != size:
            raise ValueError(f"capacity case {case.case_id} does not contain exactly {size} IDs")
        missing = set(case.active_hotword_ids) - set(hotword_by_id)
        if missing:
            raise ValueError(f"capacity case {case.case_id} has unknown active IDs")
        if not set(case.expected_hotword_ids).issubset(case.active_hotword_ids):
            raise ValueError(f"capacity case {case.case_id} expects inactive hotwords")


def _summarize_level(
    *,
    profile: str,
    size: int,
    rows: Sequence[Mapping[str, Any]],
    registry_load: Mapping[str, object],
    threshold: float,
    top_k: int,
) -> dict[str, Any]:
    final = [row for row in rows if bool(row["is_final"])]
    if not final:
        raise ValueError("capacity replay contains no final rows")
    expected_total = sum(len(row["expected_hotword_ids"]) for row in final)
    operating_hits = sum(int(row["operating_expected_hits_at_5"]) for row in final)
    positive = [row for row in final if row["expected_hotword_ids"]]
    negative = [row for row in final if not row["expected_hotword_ids"]]
    reciprocal_ranks: list[float] = []
    all_ranks: list[float] = []
    missing_ranks = 0
    top5_margins: list[float] = []
    for row in final:
        ranks = row["expected_ranks"]
        if not isinstance(ranks, dict):
            raise ValueError("capacity query has invalid expected ranks")
        expected_ids = row["expected_hotword_ids"]
        if not isinstance(expected_ids, list):
            raise ValueError("capacity query has invalid expected hotword IDs")
        for hotword_id in expected_ids:
            rank = ranks.get(hotword_id)
            if rank is None:
                missing_ranks += 1
                reciprocal_ranks.append(0.0)
            else:
                value = float(rank)
                all_ranks.append(value)
                reciprocal_ranks.append(1.0 / value)
        margins = row["expected_top5_margins"]
        if not isinstance(margins, dict):
            raise ValueError("capacity query has invalid Top-5 margins")
        top5_margins.extend(float(value) for value in margins.values())
    churn_values = [
        float(row["raw_top5_churn"]) for row in rows if row["raw_top5_churn"] is not None
    ]
    retrieval = [float(row["retrieval_seconds"]) for row in rows]
    matching = [float(row["matching_seconds"]) for row in rows]
    sorting = [float(row["sorting_seconds"]) for row in rows]
    selection = [float(row["selection_seconds"]) for row in rows]
    detector = [
        float(row["ctc_plus_retrieval_seconds"])
        for row in rows
        if row["ctc_plus_retrieval_seconds"] is not None
    ]
    rank_hits = {
        k: sum(sum(float(rank) <= k for rank in row["expected_ranks"].values()) for row in final)
        for k in (1, 3, 5, 10, 20)
    }
    quality = {
        "expected_hotwords": expected_total,
        **{f"raw_correct_at_{k}": rank_hits[k] for k in (1, 3, 5, 10, 20)},
        **{f"raw_recall_at_{k}": _ratio(rank_hits[k], expected_total) for k in (1, 3, 5, 10, 20)},
        "operating_correct_at_5": operating_hits,
        "operating_recall_at_5": _ratio(operating_hits, expected_total),
        "positive_cases": len(positive),
        "raw_positive_case_hit_rate": _ratio(
            sum(int(row["raw_expected_hits_at_5"]) > 0 for row in positive),
            len(positive),
        ),
        "operating_positive_case_hit_rate": _ratio(
            sum(int(row["operating_expected_hits_at_5"]) > 0 for row in positive),
            len(positive),
        ),
        "negative_cases": len(negative),
        "negative_case_false_positive_rate": _ratio(
            sum(bool(row["negative_false_positive"]) for row in negative),
            len(negative),
        ),
        "mean_reciprocal_rank": mean(reciprocal_ranks) if reciprocal_ranks else None,
        "expected_rank": _distribution(all_ranks),
        "expected_rank_missing": missing_ranks,
        "expected_top5_score_margin": _distribution(top5_margins),
        "raw_top5_churn": _distribution(churn_values),
    }
    duration_buckets: dict[str, list[float]] = {}
    for row in rows:
        bucket = _duration_bucket(float(row["cumulative_audio_sec"]))
        duration_buckets.setdefault(bucket, []).append(float(row["retrieval_seconds"]))
    performance = {
        "retrieval_seconds": _distribution(retrieval),
        "matching_seconds": _distribution(matching),
        "sorting_seconds": _distribution(sorting),
        "selection_seconds": _distribution(selection),
        "ctc_plus_retrieval_seconds": _distribution(detector),
        "retrieval_over_100ms_rate": _ratio(
            sum(value > 0.1 for value in retrieval), len(retrieval)
        ),
        "retrieval_over_200ms_rate": _ratio(
            sum(value > 0.2 for value in retrieval), len(retrieval)
        ),
        "retrieval_over_1s_rate": _ratio(sum(value > 1.0 for value in retrieval), len(retrieval)),
        "retrieval_over_2s_rate": _ratio(sum(value > 2.0 for value in retrieval), len(retrieval)),
        "detector_over_2s_rate": _ratio(sum(value > 2.0 for value in detector), len(detector)),
        "retrieval_seconds_by_cumulative_audio": {
            key: _distribution(values) for key, values in sorted(duration_buckets.items())
        },
        "source_gpu_memory": _source_gpu_memory(rows),
    }
    return {
        "profile": profile,
        "size": size,
        "threshold": threshold,
        "top_k": top_k,
        "queries": len(rows),
        "final_queries": len(final),
        "quality": quality,
        "performance": performance,
        "registry_load": dict(registry_load),
        "test_set_used": False,
        "status": "pass",
    }


def _capacity_recommendation(
    levels: Mapping[str, Mapping[str, Any]], sizes: Sequence[int]
) -> dict[str, object]:
    representative = levels.get("representative", {})
    if "100" not in representative:
        return {
            "status": "insufficient_data",
            "reason": "representative 100-hotword baseline was not completed",
        }
    baseline = representative["100"]
    if not isinstance(baseline, dict):
        raise ValueError("invalid representative baseline summary")
    base_quality = baseline["quality"]
    verified = 0
    decisions: dict[str, dict[str, object]] = {}
    for size in sizes:
        raw = representative.get(str(size))
        if not isinstance(raw, dict):
            break
        quality = raw["quality"]
        performance = raw["performance"]
        raw_drop = float(base_quality["raw_recall_at_5"]) - float(quality["raw_recall_at_5"])
        operating_drop = float(base_quality["operating_recall_at_5"]) - float(
            quality["operating_recall_at_5"]
        )
        retrieval_p95 = float(performance["retrieval_seconds"]["p95"])
        retrieval_p99 = float(performance["retrieval_seconds"]["p99"])
        detector_values = performance["ctc_plus_retrieval_seconds"]
        detector_p95 = detector_values["p95"]
        checks = {
            "raw_recall_drop_le_1pp": raw_drop <= 0.01 + 1e-12,
            "operating_recall_drop_le_1pp": operating_drop <= 0.01 + 1e-12,
            "negative_fpr_le_3pct": float(quality["negative_case_false_positive_rate"]) <= 0.03,
            "retrieval_p95_le_100ms": retrieval_p95 <= 0.1,
            "retrieval_p99_le_200ms": retrieval_p99 <= 0.2,
            "detector_p95_lt_2s": detector_p95 is None or float(detector_p95) < 2.0,
        }
        passed = all(checks.values())
        decisions[str(size)] = {
            "passed": passed,
            "checks": checks,
            "raw_recall_drop_pp": raw_drop * 100.0,
            "operating_recall_drop_pp": operating_drop * 100.0,
        }
        if passed:
            verified = size
        else:
            break
    previous = 0
    for size in sizes:
        if size >= verified:
            break
        previous = size
    recommended = previous or verified
    return {
        "status": "pass" if verified else "no_supported_capacity",
        "verified_maximum": verified,
        "recommended_online_cap": recommended,
        "headroom_policy": (
            "recommend the previous passing tested level to retain approximately "
            "one scale step of engineering headroom"
        ),
        "decisions": decisions,
        "hard_negative_profile_is_diagnostic": True,
        "test_set_used": False,
    }


def _source_ctc_seconds(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    keys = ("processor_seconds", "encoder_seconds", "ctc_head_seconds", "ctc_decode_seconds")
    if not all(isinstance(value.get(key), int | float) for key in keys):
        return None
    return sum(float(value[key]) for key in keys)


def _source_gpu_memory(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    values: dict[str, list[int]] = {
        "gpu_allocated_bytes": [],
        "gpu_reserved_bytes": [],
        "gpu_peak_allocated_bytes": [],
    }
    for row in rows:
        timings = row.get("source_timings")
        if not isinstance(timings, dict):
            continue
        for key in values:
            value = timings.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                values[key].append(value)
    return {key: max(items) if items else None for key, items in values.items()}


def _duration_bucket(seconds: float) -> str:
    for boundary in (2, 4, 6, 10, 20, 30):
        if seconds <= boundary + 1e-9:
            return f"le_{boundary}s"
    return "gt_30s"


def _current_rss_bytes() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": mean(ordered),
        "median": median(ordered),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 1.0


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_sha256_manifest(destination: Path) -> None:
    paths = sorted(
        path for path in destination.rglob("*") if path.is_file() and path.name != "sha256.txt"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(destination)}" for path in paths]
    (destination / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
