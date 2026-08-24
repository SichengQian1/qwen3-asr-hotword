from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, median
from typing import Any

from qwen_hotword.hotwords.capacity_assets import (
    PROFILE_NAMES,
    CapacityCase,
    load_capacity_base_cases,
    parse_capacity_sizes,
)
from qwen_hotword.hotwords.capacity_replay import load_capacity_replay
from qwen_hotword.hotwords.exact_automaton import (
    IntegerAhoCorasick,
    filter_longest_exact_matches,
    rank_unique_exact_matches,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

OBSERVATION_KS = (5, 7, 10)


def benchmark_exact_hotword_capacity(
    *,
    assets_root: str | Path,
    replay_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    profiles: Sequence[str] = PROFILE_NAMES,
    sizes: Sequence[int] = (100, 500, 1_000, 2_000, 4_000),
    warmup_queries: int = 3,
    deadline_seconds: float = 0.05,
    print_progress: bool = True,
) -> dict[str, object]:
    resolved_profiles = tuple(dict.fromkeys(profiles))
    if not resolved_profiles or any(profile not in PROFILE_NAMES for profile in resolved_profiles):
        raise ValueError(f"profiles must be a non-empty subset of {PROFILE_NAMES}")
    resolved_sizes = parse_capacity_sizes(sizes)
    if warmup_queries < 0:
        raise ValueError("warmup_queries must not be negative")
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
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
            raise FileNotFoundError(f"exact capacity input does not exist: {paths[key]}")
    destination = paths["output"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"exact capacity output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    vocab = load_phoneme_vocab(paths["vocab"])
    replay_rows = load_capacity_replay(paths["replay"])
    query_rows: list[dict[str, object]] = []
    level_summaries: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for profile in resolved_profiles:
        profile_summaries: dict[str, Any] = {}
        for size in resolved_sizes:
            level_dir = paths["assets"] / profile / f"size_{size}"
            hotword_path = level_dir / "hotwords.jsonl"
            cases_path = level_dir / "cases.jsonl"
            if not hotword_path.is_file() or not cases_path.is_file():
                raise FileNotFoundError(f"exact capacity level is incomplete: {level_dir}")
            entries = load_hotword_table(hotword_path, vocab=vocab, blank_id=0)
            cases = load_capacity_base_cases(cases_path)
            _validate_level(size, cases, entries, replay_rows)
            matcher, index_metrics = _profile_index(entries)
            case_by_id = {case.case_id: case for case in cases}
            for warmup in replay_rows[:warmup_queries]:
                case = case_by_id[str(warmup["case_id"])]
                matcher.find(
                    _decoded_ids(warmup),
                    confidences=_decoded_confidences(warmup),
                    active_hotword_ids=case.active_hotword_ids,
                )

            level_rows: list[dict[str, object]] = []
            for query_index, replay in enumerate(replay_rows, start=1):
                case = case_by_id[str(replay["case_id"])]
                scan_started = time.perf_counter()
                raw_occurrences = matcher.find(
                    _decoded_ids(replay),
                    confidences=_decoded_confidences(replay),
                    active_hotword_ids=case.active_hotword_ids,
                    longest_only=False,
                )
                occurrences = filter_longest_exact_matches(raw_occurrences)
                ranked = rank_unique_exact_matches(occurrences)
                retrieval_seconds = time.perf_counter() - scan_started
                ranked_ids = tuple(match.hotword_id for match in ranked)
                expected = set(case.expected_hotword_ids)
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
                    "expected_hotword_ids": sorted(expected),
                    "raw_exact_occurrences": len(raw_occurrences),
                    "longest_exact_occurrences": len(occurrences),
                    "longest_filter_removed": len(raw_occurrences) - len(occurrences),
                    "exact_unique_matches": len(ranked),
                    "exact_found_ids": list(ranked_ids),
                    "exact_expected_hits": len(expected & set(ranked_ids)),
                    **{
                        f"exact_top{k}_ids": list(ranked_ids[:k])
                        for k in OBSERVATION_KS
                    },
                    **{
                        f"exact_expected_hits_at_{k}": len(expected & set(ranked_ids[:k]))
                        for k in OBSERVATION_KS
                    },
                    "negative_false_positive": not expected and bool(ranked_ids),
                    "retrieval_seconds": retrieval_seconds,
                    "ranked_matches": [match.to_dict() for match in ranked[:10]],
                }
                level_rows.append(row)
                query_rows.append(row)
                if print_progress and (
                    query_index == len(replay_rows) or query_index % 50 == 0
                ):
                    print(
                        f"exact capacity profile={profile} size={size} "
                        f"queries={query_index}/{len(replay_rows)}",
                        flush=True,
                    )
            summary = _summarize_level(
                profile=profile,
                size=size,
                rows=level_rows,
                index_metrics=index_metrics,
                deadline_seconds=deadline_seconds,
            )
            profile_summaries[str(size)] = summary
        level_summaries[profile] = profile_summaries

    quality_summary = {
        profile: {size: value["quality"] for size, value in levels.items()}
        for profile, levels in level_summaries.items()
    }
    performance_summary = {
        profile: {
            size: {
                "performance": value["performance"],
                "index": value["index"],
            }
            for size, value in levels.items()
        }
        for profile, levels in level_summaries.items()
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "mode": "aho_corasick_exact_full_current_sequence",
        "elapsed_seconds": time.monotonic() - started,
        "query_rows": len(query_rows),
        "levels": level_summaries,
        "deadline_seconds": deadline_seconds,
        "test_set_used": False,
    }
    _write_jsonl(destination / "query_results.jsonl", query_rows)
    _write_json(destination / "quality_summary.json", quality_summary)
    _write_json(destination / "performance_summary.json", performance_summary)
    _write_json(
        destination / "run_config.json",
        {
            "schema_version": 1,
            "purpose": "portuguese_exact_aho_corasick_capacity_benchmark",
            "profiles": list(resolved_profiles),
            "sizes": list(resolved_sizes),
            "warmup_queries": warmup_queries,
            "deadline_seconds": deadline_seconds,
            "longest_match_filter": True,
            "ranking": (
                "mean_confidence_desc_phone_count_desc_minimum_confidence_desc_"
                "start_token_asc_hotword_id_asc"
            ),
            "inputs": {
                "asset_summary": _file_identity(paths["assets"] / "asset_summary.json"),
                "replay": _file_identity(paths["replay"]),
                "vocab": _file_identity(paths["vocab"]),
            },
            "test_set_used": False,
        },
    )
    _write_json(destination / "summary.json", report)
    _write_sha256_manifest(destination)
    return report


def _profile_index(
    entries: Sequence[HotwordEntry],
) -> tuple[IntegerAhoCorasick, dict[str, object]]:
    rss_before = _current_rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    matcher = IntegerAhoCorasick(entries)
    build_seconds = time.perf_counter() - started
    heap_current, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _current_rss_bytes()
    return matcher, {
        "patterns": matcher.pattern_count,
        "nodes": matcher.node_count,
        "transitions": matcher.transition_count,
        "build_seconds": build_seconds,
        "python_heap_current_bytes": heap_current,
        "python_heap_peak_bytes": heap_peak,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": (
            rss_after - rss_before if rss_before is not None and rss_after is not None else None
        ),
    }


def _summarize_level(
    *,
    profile: str,
    size: int,
    rows: Sequence[Mapping[str, Any]],
    index_metrics: Mapping[str, object],
    deadline_seconds: float,
) -> dict[str, object]:
    final = [row for row in rows if bool(row["is_final"])]
    if not final:
        raise ValueError("exact capacity replay contains no final rows")
    positive = [row for row in final if row["expected_hotword_ids"]]
    negative = [row for row in final if not row["expected_hotword_ids"]]
    expected_total = sum(len(row["expected_hotword_ids"]) for row in final)
    exact_hits = sum(int(row["exact_expected_hits"]) for row in final)
    retrieval = [float(row["retrieval_seconds"]) for row in rows]
    quality = {
        "expected_hotwords": expected_total,
        "exact_correct": exact_hits,
        "exact_recall": _ratio(exact_hits, expected_total),
        **{
            f"exact_correct_at_{k}": sum(
                int(row[f"exact_expected_hits_at_{k}"]) for row in final
            )
            for k in OBSERVATION_KS
        },
        **{
            f"exact_recall_at_{k}": _ratio(
                sum(int(row[f"exact_expected_hits_at_{k}"]) for row in final),
                expected_total,
            )
            for k in OBSERVATION_KS
        },
        **{
            f"exact_precision_at_{k}": _ratio(
                sum(int(row[f"exact_expected_hits_at_{k}"]) for row in final),
                sum(len(row[f"exact_top{k}_ids"]) for row in final),
            )
            for k in OBSERVATION_KS
        },
        "positive_cases": len(positive),
        "positive_case_hit_rate": _ratio(
            sum(int(row["exact_expected_hits"]) > 0 for row in positive),
            len(positive),
        ),
        **{
            f"positive_case_hit_rate_at_{k}": _ratio(
                sum(
                    int(row[f"exact_expected_hits_at_{k}"]) > 0
                    for row in positive
                ),
                len(positive),
            )
            for k in OBSERVATION_KS
        },
        "negative_cases": len(negative),
        "negative_case_false_positive_rate": _ratio(
            sum(bool(row["negative_false_positive"]) for row in negative),
            len(negative),
        ),
        "exact_unique_matches": _distribution(
            [float(row["exact_unique_matches"]) for row in rows]
        ),
        "longest_filter_removed": _distribution(
            [float(row["longest_filter_removed"]) for row in rows]
        ),
    }
    performance = {
        "retrieval_seconds": _distribution(retrieval),
        "retrieval_over_deadline_rate": _ratio(
            sum(value > deadline_seconds for value in retrieval), len(retrieval)
        ),
        "deadline_seconds": deadline_seconds,
    }
    return {
        "profile": profile,
        "size": size,
        "quality": quality,
        "performance": performance,
        "index": dict(index_metrics),
        "queries": len(rows),
        "final_queries": len(final),
        "status": "pass",
        "test_set_used": False,
    }


def _validate_level(
    size: int,
    cases: Sequence[CapacityCase],
    entries: Sequence[HotwordEntry],
    replay_rows: Sequence[Mapping[str, Any]],
) -> None:
    entry_ids = {entry.hotword_id for entry in entries}
    case_by_id = {case.case_id: case for case in cases}
    replay_case_ids = {str(row["case_id"]) for row in replay_rows}
    if replay_case_ids - set(case_by_id):
        raise ValueError("exact replay refers to cases absent from the capacity level")
    final_counts: dict[str, int] = {}
    for replay in replay_rows:
        case_id = str(replay["case_id"])
        case = case_by_id[case_id]
        if tuple(replay.get("expected_hotword_ids", ())) != case.expected_hotword_ids:
            raise ValueError(f"exact replay truth differs for case {case_id}")
        if bool(replay.get("is_final")):
            final_counts[case_id] = final_counts.get(case_id, 0) + 1
    if any(final_counts.get(case_id, 0) != 1 for case_id in replay_case_ids):
        raise ValueError("exact replay must have one final row per replay case")
    for case in cases:
        if len(case.active_hotword_ids) != size or len(set(case.active_hotword_ids)) != size:
            raise ValueError(f"capacity case {case.case_id} does not contain exactly {size} IDs")
        if not set(case.active_hotword_ids).issubset(entry_ids):
            raise ValueError(f"capacity case {case.case_id} contains unknown hotword IDs")


def _decoded_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    decoded = row.get("decoded")
    if not isinstance(decoded, list):
        raise ValueError("exact replay row has invalid decoded sequence")
    return tuple(int(item["token_id"]) for item in decoded)


def _decoded_confidences(row: Mapping[str, Any]) -> tuple[float, ...]:
    decoded = row.get("decoded")
    if not isinstance(decoded, list):
        raise ValueError("exact replay row has invalid decoded sequence")
    return tuple(float(item["confidence"]) for item in decoded)


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
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _current_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"exact capacity identity input does not exist: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
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
        path
        for path in directory.iterdir()
        if path.is_file() and path.name != "sha256.txt"
    )
    lines = [f"{_sha256_file(path)}  {path.name}\n" for path in paths]
    (directory / "sha256.txt").write_text("".join(lines), encoding="utf-8")
