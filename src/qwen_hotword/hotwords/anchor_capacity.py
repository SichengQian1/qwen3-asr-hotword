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

from qwen_hotword.hotwords.anchor_index import AnchorIndexConfig, PhonemeAnchorIndex
from qwen_hotword.hotwords.capacity_assets import (
    PROFILE_NAMES,
    CapacityCase,
    load_capacity_base_cases,
)
from qwen_hotword.hotwords.capacity_replay import (
    load_capacity_replay,
    replay_decoded_phonemes,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.hotwords.scoring import HotwordScoringConfig, profile_decoded_hotwords
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

DEFAULT_SHORTLIST_SIZES = (64, 128, 256)


def benchmark_anchor_hotword_capacity(
    *,
    assets_root: str | Path,
    replay_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    profiles: Sequence[str] = ("representative",),
    sizes: Sequence[int] = (4_000,),
    shortlist_sizes: Sequence[int] = DEFAULT_SHORTLIST_SIZES,
    ngram_sizes: Sequence[int] = (2, 3, 4),
    anchors_per_entry: int = 24,
    offset_tolerance: int = 1,
    threshold: float = 0.86,
    top_k: int = 5,
    minimum_phonemes: int = 4,
    maximum_edit_ratio: float = 0.35,
    posterior_weight: float = 0.25,
    minimum_posterior_confidence: float = 0.0,
    minimum_top1_margin: float = 0.0,
    warmup_queries: int = 3,
    deadline_seconds: float = 0.05,
    print_progress: bool = True,
) -> dict[str, object]:
    resolved_profiles = tuple(dict.fromkeys(profiles))
    if not resolved_profiles or any(profile not in PROFILE_NAMES for profile in resolved_profiles):
        raise ValueError(f"profiles must be a non-empty subset of {PROFILE_NAMES}")
    resolved_sizes = _positive_sorted_unique(sizes, name="capacity sizes", minimum=100)
    resolved_shortlists = _positive_sorted_unique(
        shortlist_sizes, name="shortlist sizes", minimum=1
    )
    resolved_ngrams = _positive_sorted_unique(ngram_sizes, name="n-gram sizes", minimum=1)
    if top_k != 5:
        raise ValueError("anchor capacity baseline is sealed to Top-5 operating retrieval")
    if warmup_queries < 0:
        raise ValueError("warmup_queries must not be negative")
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    anchor_config = AnchorIndexConfig(
        ngram_sizes=resolved_ngrams,
        anchors_per_entry=anchors_per_entry,
        offset_tolerance=offset_tolerance,
    )
    anchor_config.validate()
    scoring_config = HotwordScoringConfig(
        score_threshold=threshold,
        top_k=top_k,
        minimum_phonemes=minimum_phonemes,
        maximum_edit_ratio=maximum_edit_ratio,
        posterior_weight=posterior_weight,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
    )
    scoring_config.validate()
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
            raise FileNotFoundError(f"anchor capacity input does not exist: {paths[key]}")
    destination = paths["output"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"anchor capacity output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    vocab = load_phoneme_vocab(paths["vocab"])
    replay_rows = load_capacity_replay(paths["replay"])
    all_query_rows: list[dict[str, object]] = []
    level_summaries: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    for profile in resolved_profiles:
        profile_summaries: dict[str, Any] = {}
        for size in resolved_sizes:
            level_dir = paths["assets"] / profile / f"size_{size}"
            hotword_path = level_dir / "hotwords.jsonl"
            cases_path = level_dir / "cases.jsonl"
            if not hotword_path.is_file() or not cases_path.is_file():
                raise FileNotFoundError(f"anchor capacity level is incomplete: {level_dir}")
            entries = load_hotword_table(hotword_path, vocab=vocab, blank_id=0)
            cases = load_capacity_base_cases(cases_path)
            _validate_level(size, cases, entries, replay_rows)
            index, index_metrics = _profile_index(entries, config=anchor_config)
            entry_by_id = {entry.hotword_id: entry for entry in entries}
            case_by_id = {case.case_id: case for case in cases}
            maximum_candidates = max(resolved_shortlists)
            for warmup in replay_rows[:warmup_queries]:
                case = case_by_id[str(warmup["case_id"])]
                decoded = replay_decoded_phonemes(warmup)
                index.query(
                    tuple(item.token_id for item in decoded),
                    confidences=tuple(item.confidence for item in decoded),
                    active_hotword_ids=case.active_hotword_ids,
                    maximum_candidates=maximum_candidates,
                )

            level_rows: list[dict[str, object]] = []
            for query_number, replay in enumerate(replay_rows, start=1):
                case = case_by_id[str(replay["case_id"])]
                active_entries = tuple(entry_by_id[item] for item in case.active_hotword_ids)
                decoded = replay_decoded_phonemes(replay)

                reference = profile_decoded_hotwords(
                    decoded,
                    effective_time_steps=int(replay["effective_time_steps"]),
                    hotwords=active_entries,
                    config=scoring_config,
                )
                reference_top5 = tuple(
                    match.hotword_id for match in reference.result.ranked_matches[:top_k]
                )
                anchor_started = time.perf_counter()
                result = index.query(
                    tuple(item.token_id for item in decoded),
                    confidences=tuple(item.confidence for item in decoded),
                    active_hotword_ids=case.active_hotword_ids,
                    maximum_candidates=maximum_candidates,
                )
                anchor_seconds = time.perf_counter() - anchor_started
                candidate_ids = tuple(candidate.hotword_id for candidate in result.candidates)
                expected = set(case.expected_hotword_ids)
                candidate_ranks = {
                    hotword_id: rank
                    for rank, hotword_id in enumerate(candidate_ids, start=1)
                }
                row: dict[str, object] = {
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
                    "active_hotwords": len(active_entries),
                    "expected_hotword_ids": sorted(expected),
                    "expected_candidate_ranks": {
                        hotword_id: candidate_ranks[hotword_id]
                        for hotword_id in sorted(expected)
                        if hotword_id in candidate_ranks
                    },
                    "reference_raw_top5_ids": list(reference_top5),
                    "reference_top5_candidate_ranks": {
                        hotword_id: candidate_ranks[hotword_id]
                        for hotword_id in reference_top5
                        if hotword_id in candidate_ranks
                    },
                    "reference_expected_hits_at_5": len(expected & set(reference_top5)),
                    "exact_hotword_ids": list(result.exact_hotword_ids),
                    "exact_expected_hits": len(expected & set(result.exact_hotword_ids)),
                    "anchored_hotwords": len(result.anchored_hotword_ids),
                    "no_anchor": result.no_anchor,
                    "postings_visited": result.postings_visited,
                    "candidate_count": result.total_candidate_count,
                    "anchor_retrieval_seconds": anchor_seconds,
                    "full_scan_reference_seconds": reference.retrieval_seconds,
                    "top_anchor_candidates": [
                        candidate.to_dict() for candidate in result.candidates[:20]
                    ],
                }
                for shortlist_size in resolved_shortlists:
                    shortlist = candidate_ids[:shortlist_size]
                    row[f"candidate_ids_at_{shortlist_size}"] = list(shortlist)
                    row[f"candidate_count_at_{shortlist_size}"] = len(shortlist)
                    row[f"expected_hits_at_{shortlist_size}"] = len(expected & set(shortlist))
                    row[f"reference_top5_hits_at_{shortlist_size}"] = len(
                        set(reference_top5) & set(shortlist)
                    )
                level_rows.append(row)
                all_query_rows.append(row)
                if print_progress and (
                    query_number == len(replay_rows) or query_number % 20 == 0
                ):
                    print(
                        f"anchor capacity profile={profile} size={size} "
                        f"queries={query_number}/{len(replay_rows)}",
                        flush=True,
                    )
            profile_summaries[str(size)] = _summarize_level(
                profile=profile,
                size=size,
                rows=level_rows,
                shortlist_sizes=resolved_shortlists,
                index_metrics=index_metrics,
                deadline_seconds=deadline_seconds,
            )
        level_summaries[profile] = profile_summaries

    quality_summary = {
        profile: {size: value["quality"] for size, value in levels.items()}
        for profile, levels in level_summaries.items()
    }
    performance_summary = {
        profile: {
            size: {"performance": value["performance"], "index": value["index"]}
            for size, value in levels.items()
        }
        for profile, levels in level_summaries.items()
    }
    report = {
        "schema_version": 1,
        "status": "pass",
        "mode": "aho_corasick_exact_union_positional_rare_ngram_anchor_shortlist",
        "elapsed_seconds": time.monotonic() - started,
        "query_rows": len(all_query_rows),
        "levels": level_summaries,
        "shortlist_sizes": list(resolved_shortlists),
        "deadline_seconds": deadline_seconds,
        "latency_scope": "anchor_index_query_only_excludes_full_scan_reference",
        "test_set_used": False,
    }
    diagnostic_rows, diagnostic_summary = _build_diagnostics(
        all_query_rows,
        maximum_shortlist=max(resolved_shortlists),
        deadline_seconds=deadline_seconds,
    )
    _write_jsonl(destination / "query_results.jsonl", all_query_rows)
    _write_jsonl(destination / "diagnostic_cases.jsonl", diagnostic_rows)
    _write_json(destination / "diagnostic_summary.json", diagnostic_summary)
    _write_json(destination / "quality_summary.json", quality_summary)
    _write_json(destination / "performance_summary.json", performance_summary)
    _write_json(
        destination / "run_config.json",
        {
            "schema_version": 1,
            "purpose": "portuguese_anchor_shortlist_capacity_benchmark",
            "profiles": list(resolved_profiles),
            "sizes": list(resolved_sizes),
            "shortlist_sizes": list(resolved_shortlists),
            "anchor_config": {
                "ngram_sizes": list(resolved_ngrams),
                "anchors_per_entry": anchors_per_entry,
                "offset_tolerance": offset_tolerance,
                "exact_union": True,
                "full_current_ctc_sequence": True,
            },
            "reference_scoring_config": {
                "threshold": threshold,
                "top_k": top_k,
                "minimum_phonemes": minimum_phonemes,
                "maximum_edit_ratio": maximum_edit_ratio,
                "posterior_weight": posterior_weight,
                "minimum_posterior_confidence": minimum_posterior_confidence,
                "minimum_top1_margin": minimum_top1_margin,
            },
            "warmup_queries": warmup_queries,
            "deadline_seconds": deadline_seconds,
            "latency_scope": "anchor_index_query_only_excludes_full_scan_reference",
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


def _build_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_shortlist: int,
    deadline_seconds: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    final = [row for row in rows if bool(row["is_final"])]
    for row in final:
        candidate_ids = set(row[f"candidate_ids_at_{maximum_shortlist}"])
        for hotword_id in row["expected_hotword_ids"]:
            if hotword_id not in candidate_ids:
                diagnostics.append(
                    _diagnostic_row(
                        row,
                        reason=f"expected_not_in_anchor_top_{maximum_shortlist}",
                        hotword_id=str(hotword_id),
                    )
                )
        for hotword_id in row["reference_raw_top5_ids"]:
            if hotword_id not in candidate_ids:
                diagnostics.append(
                    _diagnostic_row(
                        row,
                        reason=f"reference_top5_not_in_anchor_top_{maximum_shortlist}",
                        hotword_id=str(hotword_id),
                    )
                )
    for row in rows:
        if float(row["anchor_retrieval_seconds"]) > deadline_seconds:
            diagnostics.append(
                _diagnostic_row(row, reason="anchor_retrieval_over_deadline", hotword_id=None)
            )
    reason_counts: dict[str, int] = {}
    for row in diagnostics:
        reason = str(row["reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return diagnostics, {
        "schema_version": 1,
        "status": "pass",
        "maximum_shortlist": maximum_shortlist,
        "deadline_seconds": deadline_seconds,
        "diagnostic_rows": len(diagnostics),
        "reason_counts": dict(sorted(reason_counts.items())),
        "test_set_used": False,
    }


def _diagnostic_row(
    row: Mapping[str, Any], *, reason: str, hotword_id: str | None
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "reason": reason,
        "profile": row["profile"],
        "size": row["size"],
        "case_id": row["case_id"],
        "sample_id": row["sample_id"],
        "primary_group": row["primary_group"],
        "chunk_id": row["chunk_id"],
        "cumulative_audio_sec": row["cumulative_audio_sec"],
        "is_final": row["is_final"],
        "hotword_id": hotword_id,
        "anchor_retrieval_seconds": row["anchor_retrieval_seconds"],
        "postings_visited": row["postings_visited"],
        "candidate_count": row["candidate_count"],
        "exact_hotword_ids": row["exact_hotword_ids"],
        "expected_hotword_ids": row["expected_hotword_ids"],
        "expected_candidate_ranks": row["expected_candidate_ranks"],
        "reference_raw_top5_ids": row["reference_raw_top5_ids"],
        "reference_top5_candidate_ranks": row["reference_top5_candidate_ranks"],
    }


def _profile_index(
    entries: Sequence[HotwordEntry], *, config: AnchorIndexConfig
) -> tuple[PhonemeAnchorIndex, dict[str, object]]:
    rss_before = _current_rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    index = PhonemeAnchorIndex(entries, config=config)
    build_seconds = time.perf_counter() - started
    heap_current, heap_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _current_rss_bytes()
    return index, {
        "entries": index.entry_count,
        "unique_anchor_ngrams": index.unique_anchor_ngrams,
        "selected_anchors": index.selected_anchor_count,
        "postings": index.posting_count,
        "entries_without_anchors": index.entries_without_anchors,
        "build_seconds": build_seconds,
        "python_heap_current_bytes": heap_current,
        "python_heap_peak_bytes": heap_peak,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
    }


def _summarize_level(
    *,
    profile: str,
    size: int,
    rows: Sequence[Mapping[str, Any]],
    shortlist_sizes: Sequence[int],
    index_metrics: Mapping[str, object],
    deadline_seconds: float,
) -> dict[str, object]:
    final = [row for row in rows if bool(row["is_final"])]
    if not final:
        raise ValueError("anchor capacity replay contains no final rows")
    positive = [row for row in final if row["expected_hotword_ids"]]
    expected_total = sum(len(row["expected_hotword_ids"]) for row in final)
    reference_top5_total = sum(len(row["reference_raw_top5_ids"]) for row in final)
    positive_reference_top5_total = sum(
        len(row["reference_raw_top5_ids"]) for row in positive
    )
    quality: dict[str, object] = {
        "expected_hotwords": expected_total,
        "positive_cases": len(positive),
        "reference_raw_correct_at_5": sum(
            int(row["reference_expected_hits_at_5"]) for row in final
        ),
        "reference_raw_recall_at_5": _ratio(
            sum(int(row["reference_expected_hits_at_5"]) for row in final), expected_total
        ),
        "exact_correct": sum(int(row["exact_expected_hits"]) for row in final),
        "exact_recall": _ratio(
            sum(int(row["exact_expected_hits"]) for row in final), expected_total
        ),
        "no_anchor_rate_all_queries": _ratio(
            sum(bool(row["no_anchor"]) for row in rows), len(rows)
        ),
        "no_anchor_rate_final_queries": _ratio(
            sum(bool(row["no_anchor"]) for row in final), len(final)
        ),
    }
    for shortlist_size in shortlist_sizes:
        expected_hits = sum(int(row[f"expected_hits_at_{shortlist_size}"]) for row in final)
        reference_hits = sum(
            int(row[f"reference_top5_hits_at_{shortlist_size}"]) for row in final
        )
        quality[f"expected_correct_at_{shortlist_size}"] = expected_hits
        quality[f"expected_recall_at_{shortlist_size}"] = _ratio(
            expected_hits, expected_total
        )
        quality[f"positive_case_hit_rate_at_{shortlist_size}"] = _ratio(
            sum(int(row[f"expected_hits_at_{shortlist_size}"]) > 0 for row in positive),
            len(positive),
        )
        quality[f"reference_top5_coverage_at_{shortlist_size}"] = _ratio(
            reference_hits, reference_top5_total
        )
        quality[f"reference_top5_positive_coverage_at_{shortlist_size}"] = _ratio(
            sum(
                int(row[f"reference_top5_hits_at_{shortlist_size}"])
                for row in positive
            ),
            positive_reference_top5_total,
        )
        quality[f"candidate_count_at_{shortlist_size}"] = _distribution(
            [float(row[f"candidate_count_at_{shortlist_size}"]) for row in rows]
        )
    retrieval = [float(row["anchor_retrieval_seconds"]) for row in rows]
    performance = {
        "anchor_retrieval_seconds": _distribution(retrieval),
        "anchor_retrieval_over_deadline_rate": _ratio(
            sum(value > deadline_seconds for value in retrieval), len(retrieval)
        ),
        "full_scan_reference_seconds": _distribution(
            [float(row["full_scan_reference_seconds"]) for row in rows]
        ),
        "postings_visited": _distribution(
            [float(row["postings_visited"]) for row in rows]
        ),
        "deadline_seconds": deadline_seconds,
        "latency_scope": "anchor_index_query_only_excludes_full_scan_reference",
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
        raise ValueError("anchor replay refers to cases absent from the capacity level")
    final_counts: dict[str, int] = {}
    for replay in replay_rows:
        case_id = str(replay["case_id"])
        case = case_by_id[case_id]
        if tuple(replay.get("expected_hotword_ids", ())) != case.expected_hotword_ids:
            raise ValueError(f"anchor replay truth differs for case {case_id}")
        if bool(replay.get("is_final")):
            final_counts[case_id] = final_counts.get(case_id, 0) + 1
    if any(final_counts.get(case_id, 0) != 1 for case_id in replay_case_ids):
        raise ValueError("anchor replay must have one final row per replay case")
    for case in cases:
        if len(case.active_hotword_ids) != size or len(set(case.active_hotword_ids)) != size:
            raise ValueError(f"capacity case {case.case_id} does not contain exactly {size} IDs")
        if not set(case.active_hotword_ids).issubset(entry_ids):
            raise ValueError(f"capacity case {case.case_id} contains unknown hotword IDs")


def _positive_sorted_unique(
    values: Sequence[int], *, name: str, minimum: int
) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved or any(value < minimum for value in resolved):
        raise ValueError(f"{name} must be non-empty and at least {minimum}")
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


def _current_rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage if sys.platform == "darwin" else usage * 1024)


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"anchor capacity identity input does not exist: {path}")
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
