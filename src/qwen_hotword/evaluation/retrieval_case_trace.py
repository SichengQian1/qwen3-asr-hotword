"""Explain saved candidates on CPU using unchanged Anchor/scoring implementations."""

from __future__ import annotations

import math
from typing import Any

from qwen_hotword.evaluation.joint_diagnostics import gate_failures
from qwen_hotword.hotwords.anchor_index import AnchorIndexConfig, PhonemeAnchorIndex
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import (
    DecodedPhoneme,
    HotwordScoringConfig,
    _score_candidate,
    profile_anchor_guided_decoded_hotwords,
)
from qwen_hotword.training.edit_distance import sequence_editops


def make_index(run: dict[str, Any], config: dict[str, Any]) -> PhonemeAnchorIndex:
    retrieval = config["retrieval"]
    return PhonemeAnchorIndex(
        run["variants"],
        config=AnchorIndexConfig(
            ngram_sizes=tuple(retrieval["anchor_ngram_sizes"]),
            anchors_per_entry=retrieval["anchors_per_entry"],
            offset_tolerance=retrieval["anchor_offset_tolerance"],
        ),
    )


def scoring_config(gate: dict[str, Any]) -> HotwordScoringConfig:
    return HotwordScoringConfig(
        score_threshold=gate["threshold"],
        top_k=7,
        minimum_phonemes=gate["minimum_phonemes"],
        maximum_edit_ratio=gate["maximum_edit_ratio"],
        posterior_weight=gate["posterior_weight"],
        minimum_posterior_confidence=gate["minimum_posterior_confidence"],
        minimum_top1_margin=gate["minimum_top1_margin"],
    )


def decoded_sequence(row: dict[str, Any]) -> tuple[DecodedPhoneme, ...]:
    # Original frame intervals are not saved for every token. Ordinal steps are
    # sufficient for matching, but must NEVER be exported as acoustic timestamps.
    return tuple(
        DecodedPhoneme(t, p, i, i + 1)
        for i, (t, p) in enumerate(
            zip(row["decoded_token_ids"], row["decoded_confidences"], strict=True)
        )
    )


def match_evidence(
    match: dict[str, Any], entry: HotwordEntry, row: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    start, end = match["decoded_start"], match["decoded_end"]
    ids = tuple(row["decoded_token_ids"][start:end])
    phones = row["decoded_tokens"][start:end]
    edits = sequence_editops(entry.token_ids, ids)
    if len(edits) != match["edit_distance"]:
        raise ValueError("saved/local edit distance does not reproduce")
    weight = gate["posterior_weight"]
    factor = 1 - weight + weight * match["posterior_confidence"]
    return {
        **{k: v for k, v in match.items() if k not in {"start_step", "end_step"}},
        "query_phonemes": list(entry.phoneme_tokens),
        "ctc_span": phones,
        "ctc_span_confidences": row["decoded_confidences"][start:end],
        "ctc_context_start": max(0, start - 5),
        "ctc_context": row["decoded_tokens"][max(0, start - 5) : end + 5],
        "failed_gates": gate_failures(match, gate),
        "gate_margins": {
            "score_minus_minimum": match["score"] - gate["threshold"],
            "maximum_edit_minus_actual": gate["maximum_edit_ratio"] - match["edit_ratio"],
            "posterior_minus_minimum": (
                match["posterior_confidence"] - gate["minimum_posterior_confidence"]
            ),
        },
        "score_components": {
            "edit_similarity": 1 - match["edit_ratio"],
            "confidence_factor": factor,
        },
        "edits_table_to_ctc": [
            {
                "operation": op,
                "query_index": a,
                "span_index": b,
                "query_phone": entry.phoneme_tokens[a] if op != "insert" else None,
                "ctc_phone": phones[b] if op != "delete" else None,
            }
            for op, a, b in edits
        ],
    }


def trace_case(
    run: dict[str, Any],
    row: dict[str, Any],
    identifier: str,
    config: dict[str, Any],
    gate: dict[str, Any],
    index: PhonemeAnchorIndex,
) -> dict[str, Any]:
    decoded = decoded_sequence(row)
    retrieval = config["retrieval"]
    limit = retrieval["shortlist_size"]
    query = index.query(
        row["decoded_token_ids"],
        confidences=row["decoded_confidences"],
        maximum_candidates=index.entry_count,
    )
    limited = index.query(
        row["decoded_token_ids"],
        confidences=row["decoded_confidences"],
        maximum_candidates=limit,
    )
    if limited.candidates != query.candidates[:limit]:
        raise ValueError("uncapped Anchor query does not preserve the saved cap prefix")
    candidates = {c.hotword_id: c for c in query.candidates}
    scorer = scoring_config(gate)
    reproduced = profile_anchor_guided_decoded_hotwords(
        decoded,
        effective_time_steps=max(1, len(decoded)),
        hotwords=tuple(run["index"][c.hotword_id] for c in limited.candidates),
        start_hints={c.hotword_id: c.best_offset for c in limited.candidates},
        maximum_start_delta=retrieval["rerank_start_radius"],
        config=scorer,
    ).result.ranked_matches
    saved = row["raw_variant_ranked_matches"]
    if [m.hotword_id for m in reproduced] != [m["hotword_id"] for m in saved]:
        raise ValueError("Anchor/scorer replay does not reproduce saved variant ranking")
    for actual, original in zip(reproduced, saved, strict=True):
        for key in (
            "score",
            "edit_distance",
            "edit_ratio",
            "posterior_confidence",
            "decoded_start",
            "decoded_end",
        ):
            if not math.isclose(getattr(actual, key), original[key], abs_tol=1e-10, rel_tol=1e-10):
                raise ValueError(f"saved scorer result does not reproduce: {key}")
    ranks = {c.hotword_id: i for i, c in enumerate(query.candidates, 1)}
    variants = []
    for entry in run["variants"]:
        if run["mapping"][entry.hotword_id].hotword_id != identifier:
            continue
        candidate = candidates.get(entry.hotword_id)
        rank = ranks.get(entry.hotword_id)
        saved_match = next((m for m in saved if m["hotword_id"] == entry.hotword_id), None)
        full = _score_candidate(entry, decoded, scorer).to_dict()
        variants.append(
            {
                "variant_id": entry.hotword_id,
                "language": entry.language,
                "pronunciation": entry.pronunciation,
                "phonemes": list(entry.phoneme_tokens),
                "anchor_state": "no_anchor_and_no_exact_hit"
                if rank is None
                else ("outside_shortlist_cap" if rank > limit else "inside_shortlist_cap"),
                "anchor_rank_before_cap": rank,
                "anchor_evidence": candidate.to_dict() if candidate else None,
                "saved_guided_match": (
                    match_evidence(saved_match, entry, row, gate) if saved_match else None
                ),
                "diagnostic_full_start_search": match_evidence(full, entry, row, gate),
            }
        )
    saved_surface = next(
        (
            {"raw_rank": i, **match_evidence(m, run["index"][m["matched_variant_id"]], row, gate)}
            for i, m in enumerate(row["raw_ranked_matches"], 1)
            if m["hotword_id"] == identifier
        ),
        None,
    )
    return {
        "saved_surface_match": saved_surface,
        "variants": variants,
        "anchor_total_candidates_before_cap": query.total_candidate_count,
        "shortlist_cap": limit,
        "saved_ranking_and_scores_reproduced": True,
        "full_search_scope": "target_only_all_starts_existing_width_limits_not_actual_retrieval",
        "position_unit": "decoded_token_index_not_audio_time",
    }
