from __future__ import annotations

import gc
import hashlib
import json
import math
import time
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
from qwen_hotword.hotwords.scoring import (
    DecodedPhoneme,
    HotwordScoringConfig,
    profile_anchor_guided_decoded_hotwords,
    profile_decoded_hotwords,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

DEFAULT_SHORTLIST_SIZES = (64, 128, 256)
DEFAULT_LOOKBACK_SECONDS: tuple[float | None, ...] = (None, 2.0, 4.0, 6.0)
OBSERVATION_KS = (5, 7, 10)
GC_POLICIES = ("normal", "defer_during_retrieval_pass")
RERANK_MODES = ("full_search", "anchor_guided")


def crop_decoded_to_lookback(
    decoded: Sequence[DecodedPhoneme],
    *,
    effective_time_steps: int,
    cumulative_audio_sec: float,
    lookback_sec: float | None,
) -> tuple[tuple[DecodedPhoneme, ...], int, int]:
    """Crop a causal CTC decode by its current frame axis, never by text units."""
    if effective_time_steps <= 0:
        raise ValueError("effective_time_steps must be positive")
    if cumulative_audio_sec <= 0:
        raise ValueError("cumulative_audio_sec must be positive")
    if lookback_sec is not None and lookback_sec <= 0:
        raise ValueError("lookback_sec must be positive or None")
    if lookback_sec is None or lookback_sec >= cumulative_audio_sec:
        cutoff = 0
    else:
        retained_fraction = lookback_sec / cumulative_audio_sec
        retained_steps = max(1, math.ceil(effective_time_steps * retained_fraction))
        cutoff = effective_time_steps - retained_steps
    cropped = tuple(
        DecodedPhoneme(
            token_id=item.token_id,
            confidence=item.confidence,
            start_step=max(item.start_step, cutoff) - cutoff,
            end_step=item.end_step - cutoff,
        )
        for item in decoded
        if item.end_step > cutoff
    )
    return cropped, effective_time_steps - cutoff, cutoff


def benchmark_anchor_rerank_capacity(
    *,
    assets_root: str | Path,
    replay_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    profiles: Sequence[str] = ("representative",),
    sizes: Sequence[int] = (4_000,),
    shortlist_sizes: Sequence[int] = DEFAULT_SHORTLIST_SIZES,
    lookback_seconds: Sequence[float | None] = DEFAULT_LOOKBACK_SECONDS,
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
    gc_policy: str = "defer_during_retrieval_pass",
    rerank_mode: str = "full_search",
    anchor_start_radius: int = 2,
    print_progress: bool = True,
) -> dict[str, object]:
    resolved_profiles = tuple(dict.fromkeys(profiles))
    if not resolved_profiles or any(profile not in PROFILE_NAMES for profile in resolved_profiles):
        raise ValueError(f"profiles must be a non-empty subset of {PROFILE_NAMES}")
    resolved_sizes = _positive_sorted_unique(sizes, name="capacity sizes", minimum=100)
    resolved_shortlists = _positive_sorted_unique(
        shortlist_sizes, name="shortlist sizes", minimum=top_k
    )
    resolved_lookbacks = _validate_lookbacks(lookback_seconds)
    resolved_ngrams = _positive_sorted_unique(ngram_sizes, name="n-gram sizes", minimum=1)
    if top_k != 5:
        raise ValueError("rerank baseline is sealed to Top-5 operating retrieval")
    if warmup_queries < 0:
        raise ValueError("warmup_queries must not be negative")
    if deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    if gc_policy not in GC_POLICIES:
        raise ValueError(f"gc_policy must be one of {GC_POLICIES}")
    if rerank_mode not in RERANK_MODES:
        raise ValueError(f"rerank_mode must be one of {RERANK_MODES}")
    if anchor_start_radius < 0:
        raise ValueError("anchor_start_radius must not be negative")

    anchor_config = AnchorIndexConfig(
        ngram_sizes=resolved_ngrams,
        anchors_per_entry=anchors_per_entry,
        offset_tolerance=offset_tolerance,
    )
    scoring_config = HotwordScoringConfig(
        score_threshold=threshold,
        top_k=top_k,
        minimum_phonemes=minimum_phonemes,
        maximum_edit_ratio=maximum_edit_ratio,
        posterior_weight=posterior_weight,
        minimum_posterior_confidence=minimum_posterior_confidence,
        minimum_top1_margin=minimum_top1_margin,
    )
    anchor_config.validate()
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
            raise FileNotFoundError(f"anchor rerank input does not exist: {paths[key]}")
    destination = paths["output"]
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"anchor rerank output must be new and empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    vocab = load_phoneme_vocab(paths["vocab"])
    replay_rows = load_capacity_replay(paths["replay"])
    query_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, Any]] = {}
    index_summaries: dict[str, dict[str, object]] = {}
    started = time.monotonic()
    gc_enabled_before = gc.isenabled()
    pre_gc_started = time.perf_counter()
    pre_gc_collected = gc.collect()
    pre_gc_seconds = time.perf_counter() - pre_gc_started
    try:
        for profile in resolved_profiles:
            summaries[profile] = {}
            index_summaries[profile] = {}
            for size in resolved_sizes:
                level_dir = paths["assets"] / profile / f"size_{size}"
                entries, cases = _load_level(level_dir, vocab=vocab)
                _validate_level(size, cases, entries, replay_rows)
                index_started = time.perf_counter()
                index = PhonemeAnchorIndex(entries, config=anchor_config)
                index_seconds = time.perf_counter() - index_started
                index_summaries[profile][str(size)] = {
                    "entries": index.entry_count,
                    "unique_anchor_ngrams": index.unique_anchor_ngrams,
                    "selected_anchors": index.selected_anchor_count,
                    "postings": index.posting_count,
                    "entries_without_anchors": index.entries_without_anchors,
                    "build_seconds": index_seconds,
                }
                entry_by_id = {entry.hotword_id: entry for entry in entries}
                case_by_id = {case.case_id: case for case in cases}
                _warm_up(
                    replay_rows[:warmup_queries],
                    case_by_id=case_by_id,
                    entry_by_id=entry_by_id,
                    index=index,
                    shortlist_size=max(resolved_shortlists),
                    scoring_config=scoring_config,
                    rerank_mode=rerank_mode,
                    anchor_start_radius=anchor_start_radius,
                )
                level_rows: list[dict[str, object]] = []
                total_variants = len(resolved_lookbacks) * len(resolved_shortlists)
                variant_number = 0
                if gc_policy == "defer_during_retrieval_pass" and gc_enabled_before:
                    gc.disable()
                try:
                    for lookback_sec in resolved_lookbacks:
                        for shortlist_size in resolved_shortlists:
                            variant_number += 1
                            window_name = _window_name(lookback_sec)
                            for query_number, replay in enumerate(replay_rows, start=1):
                                row = _run_query(
                                    replay,
                                    case=case_by_id[str(replay["case_id"])],
                                    entry_by_id=entry_by_id,
                                    index=index,
                                    shortlist_size=shortlist_size,
                                    lookback_sec=lookback_sec,
                                    scoring_config=scoring_config,
                                    deadline_seconds=deadline_seconds,
                                    profile=profile,
                                    size=size,
                                    rerank_mode=rerank_mode,
                                    anchor_start_radius=anchor_start_radius,
                                )
                                level_rows.append(row)
                                query_rows.append(row)
                                if print_progress and (
                                    query_number == len(replay_rows)
                                    or query_number % 50 == 0
                                ):
                                    print(
                                        f"anchor rerank profile={profile} size={size} "
                                        f"window={window_name} shortlist={shortlist_size} "
                                        f"queries={query_number}/{len(replay_rows)} "
                                        f"variant={variant_number}/{total_variants}",
                                        flush=True,
                                    )
                finally:
                    if gc_policy == "defer_during_retrieval_pass" and gc_enabled_before:
                        gc.enable()
                summaries[profile][str(size)] = _summarize_level(
                    level_rows,
                    shortlist_sizes=resolved_shortlists,
                    lookback_seconds=resolved_lookbacks,
                    deadline_seconds=deadline_seconds,
                )
    finally:
        if not gc.isenabled() and gc_enabled_before:
            gc.enable()
    post_gc_started = time.perf_counter()
    post_gc_collected = gc.collect()
    post_gc_seconds = time.perf_counter() - post_gc_started

    quality_summary = {
        profile: {
            size: {
                window: {
                    shortlist: variant["quality"]
                    for shortlist, variant in windows.items()
                }
                for window, windows in level.items()
            }
            for size, level in sizes_by_profile.items()
        }
        for profile, sizes_by_profile in summaries.items()
    }
    performance_summary = {
        profile: {
            size: {
                window: {
                    shortlist: variant["performance"]
                    for shortlist, variant in windows.items()
                }
                for window, windows in level.items()
            }
            for size, level in sizes_by_profile.items()
        }
        for profile, sizes_by_profile in summaries.items()
    }
    run_config = {
        "schema_version": 1,
        "purpose": "anchor_shortlist_approximate_rerank_and_causal_window_ablation",
        "profiles": list(resolved_profiles),
        "sizes": list(resolved_sizes),
        "shortlist_sizes": list(resolved_shortlists),
        "lookback_seconds": [value for value in resolved_lookbacks],
        "window_method": "current_cumulative_ctc_frame_axis_causal_crop",
        "anchor_config": {
            "ngram_sizes": list(resolved_ngrams),
            "anchors_per_entry": anchors_per_entry,
            "offset_tolerance": offset_tolerance,
        },
        "retrieval_config": {
            "threshold": threshold,
            "top_k": top_k,
            "minimum_phonemes": minimum_phonemes,
            "maximum_edit_ratio": maximum_edit_ratio,
            "posterior_weight": posterior_weight,
            "minimum_posterior_confidence": minimum_posterior_confidence,
            "minimum_top1_margin": minimum_top1_margin,
            "rerank_mode": rerank_mode,
            "anchor_start_radius": anchor_start_radius,
        },
        "gc_policy": gc_policy,
        "deadline_seconds": deadline_seconds,
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
        "mode": "anchor_shortlist_then_existing_approximate_phoneme_scorer",
        "rerank_mode": rerank_mode,
        "query_rows": len(query_rows),
        "source_replay_rows": len(replay_rows),
        "levels": summaries,
        "index": index_summaries,
        "gc": {
            "policy": gc_policy,
            "enabled_before": gc_enabled_before,
            "enabled_during_retrieval_pass": (
                False
                if gc_policy == "defer_during_retrieval_pass" and gc_enabled_before
                else gc_enabled_before
            ),
            "enabled_restored": gc.isenabled() == gc_enabled_before,
            "pre_collect_seconds": pre_gc_seconds,
            "pre_collected": pre_gc_collected,
            "post_collect_seconds_outside_latency": post_gc_seconds,
            "post_collected": post_gc_collected,
        },
        "deadline_seconds": deadline_seconds,
        "latency_scope": "anchor_query_plus_shortlist_rerank",
        "test_set_used": False,
    }
    _write_jsonl(destination / "query_results.jsonl", query_rows)
    _write_json(destination / "quality_summary.json", quality_summary)
    _write_json(destination / "performance_summary.json", performance_summary)
    _write_json(destination / "run_config.json", run_config)
    _write_json(destination / "summary.json", report)
    _write_sha256_manifest(destination)
    return report


def _run_query(
    replay: Mapping[str, Any],
    *,
    case: CapacityCase,
    entry_by_id: Mapping[str, HotwordEntry],
    index: PhonemeAnchorIndex,
    shortlist_size: int,
    lookback_sec: float | None,
    scoring_config: HotwordScoringConfig,
    deadline_seconds: float,
    profile: str,
    size: int,
    rerank_mode: str,
    anchor_start_radius: int,
) -> dict[str, object]:
    decoded, window_steps, cutoff_step = crop_decoded_to_lookback(
        replay_decoded_phonemes(replay),
        effective_time_steps=int(replay["effective_time_steps"]),
        cumulative_audio_sec=float(replay["cumulative_audio_sec"]),
        lookback_sec=lookback_sec,
    )
    anchor_started = time.perf_counter()
    candidates = index.query(
        tuple(item.token_id for item in decoded),
        confidences=tuple(item.confidence for item in decoded),
        active_hotword_ids=case.active_hotword_ids,
        maximum_candidates=shortlist_size,
    )
    anchor_seconds = time.perf_counter() - anchor_started
    candidate_entries = tuple(entry_by_id[item.hotword_id] for item in candidates.candidates)
    if rerank_mode == "anchor_guided":
        rerank = profile_anchor_guided_decoded_hotwords(
            decoded,
            effective_time_steps=window_steps,
            hotwords=candidate_entries,
            start_hints={item.hotword_id: item.best_offset for item in candidates.candidates},
            maximum_start_delta=anchor_start_radius,
            config=scoring_config,
        )
    else:
        rerank = profile_decoded_hotwords(
            decoded,
            effective_time_steps=window_steps,
            hotwords=candidate_entries,
            config=scoring_config,
        )
    total_seconds = anchor_seconds + rerank.retrieval_seconds
    ranking_ids = tuple(match.hotword_id for match in rerank.result.ranked_matches)
    operating_ids = tuple(match.hotword_id for match in rerank.result.selected_matches)
    expected = set(case.expected_hotword_ids)
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
        "window": _window_name(lookback_sec),
        "lookback_sec": lookback_sec,
        "window_cutoff_step": cutoff_step,
        "window_effective_time_steps": window_steps,
        "window_decoded_phonemes": len(decoded),
        "shortlist_size": shortlist_size,
        "rerank_mode": rerank_mode,
        "anchor_start_radius": anchor_start_radius,
        "candidate_count": len(candidates.candidates),
        "candidate_expected_hits": len(
            expected & {item.hotword_id for item in candidates.candidates}
        ),
        "postings_visited": candidates.postings_visited,
        "exact_hotword_ids": list(candidates.exact_hotword_ids),
        "exact_expected_hits": len(expected & set(candidates.exact_hotword_ids)),
        "candidate_ids": [item.hotword_id for item in candidates.candidates],
        "expected_hotword_ids": sorted(expected),
        "operating_ids": list(operating_ids),
        "operating_expected_hits_at_5": len(expected & set(operating_ids)),
        "negative_false_positive": not expected and bool(operating_ids),
        "anchor_seconds": anchor_seconds,
        "rerank_matching_seconds": rerank.matching_seconds,
        "rerank_sorting_seconds": rerank.sorting_seconds,
        "rerank_selection_seconds": rerank.selection_seconds,
        "rerank_seconds": rerank.retrieval_seconds,
        "retrieval_seconds": total_seconds,
        "over_deadline": total_seconds > deadline_seconds,
        "top_matches": [match.to_dict() for match in rerank.result.ranked_matches[:20]],
    }
    for k in OBSERVATION_KS:
        selected = ranking_ids[:k]
        row[f"raw_top{k}_ids"] = list(selected)
        row[f"raw_expected_hits_at_{k}"] = len(expected & set(selected))
    return row


def _warm_up(
    replay_rows: Sequence[Mapping[str, Any]],
    *,
    case_by_id: Mapping[str, CapacityCase],
    entry_by_id: Mapping[str, HotwordEntry],
    index: PhonemeAnchorIndex,
    shortlist_size: int,
    scoring_config: HotwordScoringConfig,
    rerank_mode: str,
    anchor_start_radius: int,
) -> None:
    for replay in replay_rows:
        case = case_by_id[str(replay["case_id"])]
        decoded = replay_decoded_phonemes(replay)
        result = index.query(
            tuple(item.token_id for item in decoded),
            confidences=tuple(item.confidence for item in decoded),
            active_hotword_ids=case.active_hotword_ids,
            maximum_candidates=shortlist_size,
        )
        hotwords = tuple(entry_by_id[item.hotword_id] for item in result.candidates)
        if rerank_mode == "anchor_guided":
            profile_anchor_guided_decoded_hotwords(
                decoded,
                effective_time_steps=int(replay["effective_time_steps"]),
                hotwords=hotwords,
                start_hints={item.hotword_id: item.best_offset for item in result.candidates},
                maximum_start_delta=anchor_start_radius,
                config=scoring_config,
            )
        else:
            profile_decoded_hotwords(
                decoded,
                effective_time_steps=int(replay["effective_time_steps"]),
                hotwords=hotwords,
                config=scoring_config,
            )


def _summarize_level(
    rows: Sequence[Mapping[str, Any]],
    *,
    shortlist_sizes: Sequence[int],
    lookback_seconds: Sequence[float | None],
    deadline_seconds: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for lookback_sec in lookback_seconds:
        window = _window_name(lookback_sec)
        output[window] = {}
        for shortlist_size in shortlist_sizes:
            selected = [
                row
                for row in rows
                if row["window"] == window and int(row["shortlist_size"]) == shortlist_size
            ]
            final = [row for row in selected if bool(row["is_final"])]
            positive = [row for row in final if row["expected_hotword_ids"]]
            negative = [row for row in final if not row["expected_hotword_ids"]]
            expected_total = sum(len(row["expected_hotword_ids"]) for row in final)
            quality: dict[str, object] = {
                "expected_hotwords": expected_total,
                "positive_cases": len(positive),
                "negative_cases": len(negative),
                "candidate_correct": sum(
                    int(row["candidate_expected_hits"]) for row in final
                ),
                "candidate_recall": _ratio(
                    sum(int(row["candidate_expected_hits"]) for row in final),
                    expected_total,
                ),
                "candidate_positive_case_hit_rate": _ratio(
                    sum(int(row["candidate_expected_hits"]) > 0 for row in positive),
                    len(positive),
                ),
                "exact_correct": sum(int(row["exact_expected_hits"]) for row in final),
                "exact_recall": _ratio(
                    sum(int(row["exact_expected_hits"]) for row in final),
                    expected_total,
                ),
            }
            _add_any_step_quality(quality, selected, final=final)
            for k in OBSERVATION_KS:
                hits = sum(int(row[f"raw_expected_hits_at_{k}"]) for row in final)
                returned = sum(len(row[f"raw_top{k}_ids"]) for row in final)
                quality[f"raw_correct_at_{k}"] = hits
                quality[f"raw_recall_at_{k}"] = _ratio(hits, expected_total)
                quality[f"raw_precision_at_{k}"] = _ratio(hits, returned)
                quality[f"raw_positive_case_hit_rate_at_{k}"] = _ratio(
                    sum(int(row[f"raw_expected_hits_at_{k}"]) > 0 for row in positive),
                    len(positive),
                )
            operating_hits = sum(
                int(row["operating_expected_hits_at_5"]) for row in final
            )
            operating_returned = sum(len(row["operating_ids"]) for row in final)
            quality.update(
                {
                    "operating_correct_at_5": operating_hits,
                    "operating_recall_at_5": _ratio(operating_hits, expected_total),
                    "operating_precision_at_5": _ratio(
                        operating_hits, operating_returned
                    ),
                    "operating_positive_case_hit_rate_at_5": _ratio(
                        sum(
                            int(row["operating_expected_hits_at_5"]) > 0
                            for row in positive
                        ),
                        len(positive),
                    ),
                    "negative_case_false_positive_rate": _ratio(
                        sum(bool(row["negative_false_positive"]) for row in negative),
                        len(negative),
                    ),
                }
            )
            retrieval = [float(row["retrieval_seconds"]) for row in selected]
            performance = {
                "anchor_seconds": _distribution(
                    [float(row["anchor_seconds"]) for row in selected]
                ),
                "rerank_seconds": _distribution(
                    [float(row["rerank_seconds"]) for row in selected]
                ),
                "retrieval_seconds": _distribution(retrieval),
                "retrieval_over_deadline_rate": _ratio(
                    sum(value > deadline_seconds for value in retrieval), len(retrieval)
                ),
                "deadline_seconds": deadline_seconds,
                "postings_visited": _distribution(
                    [float(row["postings_visited"]) for row in selected]
                ),
                "candidate_count": _distribution(
                    [float(row["candidate_count"]) for row in selected]
                ),
            }
            output[window][str(shortlist_size)] = {
                "quality": quality,
                "performance": performance,
                "queries": len(selected),
                "final_queries": len(final),
                "status": "pass",
                "test_set_used": False,
            }
    return output


def _add_any_step_quality(
    quality: dict[str, object],
    rows: Sequence[Mapping[str, Any]],
    *,
    final: Sequence[Mapping[str, Any]],
) -> None:
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
    rows_by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        rows_by_case.setdefault(str(row["case_id"]), []).append(row)

    def detected_pairs(key: str) -> set[tuple[str, str]]:
        return {
            (case_id, hotword_id)
            for case_id, hotword_id in expected_pairs
            if any(hotword_id in row[key] for row in rows_by_case[case_id])
        }

    def case_hit_rate(pairs: set[tuple[str, str]]) -> float | None:
        hit_cases = {case_id for case_id, _ in pairs}
        return _ratio(len(hit_cases), len(positive_case_ids))

    candidate_pairs = detected_pairs("candidate_ids")
    quality["any_step_candidate_correct"] = len(candidate_pairs)
    quality["any_step_candidate_recall"] = _ratio(
        len(candidate_pairs), len(expected_pairs)
    )
    quality["any_step_candidate_positive_case_hit_rate"] = case_hit_rate(candidate_pairs)
    for k in OBSERVATION_KS:
        key = f"raw_top{k}_ids"
        pairs = detected_pairs(key)
        returned_pairs = {
            (str(row["case_id"]), str(hotword_id))
            for row in rows
            for hotword_id in row[key]
        }
        quality[f"any_step_raw_correct_at_{k}"] = len(pairs)
        quality[f"any_step_raw_recall_at_{k}"] = _ratio(len(pairs), len(expected_pairs))
        quality[f"any_step_raw_precision_at_{k}"] = _ratio(
            len(pairs), len(returned_pairs)
        )
        quality[f"any_step_raw_positive_case_hit_rate_at_{k}"] = case_hit_rate(pairs)
    operating_pairs = detected_pairs("operating_ids")
    operating_returned_pairs = {
        (str(row["case_id"]), str(hotword_id))
        for row in rows
        for hotword_id in row["operating_ids"]
    }
    quality["any_step_operating_correct_at_5"] = len(operating_pairs)
    quality["any_step_operating_recall_at_5"] = _ratio(
        len(operating_pairs), len(expected_pairs)
    )
    quality["any_step_operating_precision_at_5"] = _ratio(
        len(operating_pairs), len(operating_returned_pairs)
    )
    quality["any_step_operating_positive_case_hit_rate_at_5"] = case_hit_rate(
        operating_pairs
    )
    quality["any_step_negative_case_false_positive_rate"] = _ratio(
        sum(
            any(row["operating_ids"] for row in rows_by_case[case_id])
            for case_id in negative_case_ids
        ),
        len(negative_case_ids),
    )


def _load_level(
    level_dir: Path, *, vocab: Any
) -> tuple[list[HotwordEntry], tuple[CapacityCase, ...]]:
    hotword_path = level_dir / "hotwords.jsonl"
    cases_path = level_dir / "cases.jsonl"
    if not hotword_path.is_file() or not cases_path.is_file():
        raise FileNotFoundError(f"anchor rerank capacity level is incomplete: {level_dir}")
    return (
        load_hotword_table(hotword_path, vocab=vocab, blank_id=0),
        load_capacity_base_cases(cases_path),
    )


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
        raise ValueError("anchor rerank replay refers to cases absent from the capacity level")
    final_counts: dict[str, int] = {}
    for replay in replay_rows:
        case_id = str(replay["case_id"])
        case = case_by_id[case_id]
        if tuple(replay.get("expected_hotword_ids", ())) != case.expected_hotword_ids:
            raise ValueError(f"anchor rerank replay truth differs for case {case_id}")
        if bool(replay.get("is_final")):
            final_counts[case_id] = final_counts.get(case_id, 0) + 1
    if any(final_counts.get(case_id, 0) != 1 for case_id in replay_case_ids):
        raise ValueError("anchor rerank replay must have one final row per replay case")
    for case in cases:
        if len(case.active_hotword_ids) != size or len(set(case.active_hotword_ids)) != size:
            raise ValueError(f"capacity case {case.case_id} does not contain exactly {size} IDs")
        if not set(case.active_hotword_ids).issubset(entry_ids):
            raise ValueError(f"capacity case {case.case_id} contains unknown hotword IDs")


def _validate_lookbacks(values: Sequence[float | None]) -> tuple[float | None, ...]:
    resolved = tuple(values)
    if not resolved or any(value is not None and value <= 0 for value in resolved):
        raise ValueError("lookback seconds must contain None/full or positive values")
    names = tuple(_window_name(value) for value in resolved)
    if len(set(names)) != len(names):
        raise ValueError("lookback seconds must be unique")
    return resolved


def _window_name(value: float | None) -> str:
    return "full_current" if value is None else f"recent_{value:g}s"


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


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"anchor rerank identity input does not exist: {path}")
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
