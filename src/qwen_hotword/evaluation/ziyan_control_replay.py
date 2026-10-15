"""CPU replay of the production Anchor index from sealed greedy CTC sequences."""

from __future__ import annotations

import math
from typing import Any

from qwen_hotword.evaluation.retrieval_case_trace import (
    decoded_sequence,
    make_index,
    scoring_config,
)
from qwen_hotword.hotwords.scoring import profile_anchor_guided_decoded_hotwords
from qwen_hotword.inference.external_keyword_retrieval import _string_list
from qwen_hotword.inference.external_keyword_topk_replay import (
    _select,
    _validated_source_gate,
    replay_rows_top7,
)
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.wave_retrieval import _view_metrics


def prepare_runtime(runtime: dict[str, Any], config: dict[str, Any]) -> None:
    runtime["index"] = make_index(runtime, config)
    runtime["by_id"] = {e.hotword_id: e for e in runtime["variants"]}


def retrieve(
    row: dict[str, Any], runtime: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Ordinal positions are internal only; acoustic frame intervals are exported as null."""
    decoded = decoded_sequence(row)
    retrieval = config["retrieval"]
    gate = _validated_source_gate(config)
    shortlist = runtime["index"].query(
        tuple(row["decoded_token_ids"]),
        confidences=tuple(row["decoded_confidences"]),
        maximum_candidates=retrieval["shortlist_size"],
    )
    scored = profile_anchor_guided_decoded_hotwords(
        decoded,
        effective_time_steps=max(1, len(decoded)),
        hotwords=tuple(runtime["by_id"][c.hotword_id] for c in shortlist.candidates),
        start_hints={c.hotword_id: c.best_offset for c in shortlist.candidates},
        maximum_start_delta=retrieval["rerank_start_radius"],
        config=scoring_config(gate),
    ).result
    raw = [{**m.to_dict(), "start_step": None, "end_step": None} for m in scored.ranked_matches]
    merged = collapse_variants(
        {
            "raw_ranked_matches": raw,
            "shortlist_candidates": len(raw),
            # Required scratch keys for the unchanged surface-merging function.
            # No synthetic model timings or frame positions leave this helper.
            "timing": {"pure_retrieval_seconds": 0.0, "model_retrieval_seconds": 0.0},
        },
        runtime["variants"],
        runtime["mapping"],
        gate,
    )
    return {
        **{k: v for k, v in merged.items() if k != "timing"},
        "postings_visited": shortlist.postings_visited,
        "anchor_candidates_before_cap": shortlist.total_candidate_count,
    }


def verify_reproduction(original: dict[str, Any], replay: dict[str, Any]) -> None:
    for field in ("raw_variant_ranked_matches", "raw_ranked_matches", "selected_matches"):
        saved, actual = original[field], replay[field]
        if len(saved) != len(actual):
            raise ValueError(f"baseline replay length differs: {original['sample_id']} {field}")
        for a, b in zip(saved, actual, strict=True):
            keys = set(a) - {"start_step", "end_step"}
            if keys != set(b) - {"start_step", "end_step"}:
                raise ValueError("baseline replay match schema differs")
            for key in keys:
                equal = (
                    math.isclose(a[key], b[key], abs_tol=1e-10, rel_tol=1e-10)
                    if isinstance(a[key], (float, int))
                    else a[key] == b[key]
                )
                if not equal:
                    raise ValueError(
                        f"baseline replay differs: {original['sample_id']} {field} {key}"
                    )
    if any(
        original[key] != replay[key]
        for key in ("shortlist_candidates", "variant_shortlist_candidates", "postings_visited")
    ):
        raise ValueError("baseline replay candidate counts differ")


def make_row(
    source: dict[str, Any], replay: dict[str, Any], runtime: dict[str, Any]
) -> dict[str, Any]:
    expected = [
        e
        for e in runtime["bundle"].entries
        if strict_phrase_match(source["reference_text"], e.surface)
    ]
    return {
        **{
            k: source[k]
            for k in (
                "sample_id",
                "source",
                "audio_path",
                "reference_text",
                "language",
                "audio_sha256",
                "decoded_token_ids",
                "decoded_tokens",
                "decoded_confidences",
            )
        },
        "source_row_sha256": source["row_sha256"],
        **replay,
        "expected_hotword_ids": [e.hotword_id for e in expected],
        "expected_hotwords": [e.surface for e in expected],
        "selected_hotword_ids": [m["hotword_id"] for m in replay["selected_matches"]],
        "position_unit": "decoded_token_index_not_audio_time",
        "acoustic_frame_intervals_available": False,
    }


def metrics(
    rows: list[dict[str, Any]], runtime: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    replayed = replay_rows_top7(rows, gate=gate, bundle=runtime["bundle"])
    result = {}
    targets = set(runtime["target_ids"])
    for k in (5, 7):
        view = _view_metrics(replayed, targets, top_k=k)
        view["full_active_table_view"] = view.pop("full_4000_view")
        view["declared_target_count"] = view.pop("declared_wave_target_count")
        negative = [
            r for r in replayed if not targets.intersection(_string_list(r, "expected_hotword_ids"))
        ]
        false_target = sum(
            bool(targets.intersection(_string_list(r, f"top{k}_selected_hotword_ids")))
            for r in negative
        )
        view["target_negative_sample_count"] = len(negative)
        view["target_negative_samples_with_false_target"] = false_target
        view["target_negative_false_positive_rate"] = (
            false_target / len(negative) if negative else None
        )
        view["any_output_on_target_negative_samples"] = sum(
            bool(r[f"top{k}_selected_hotword_ids"]) for r in negative
        )
        result[f"top{k}"] = view
    return result


def paired_change(
    old: dict[str, Any], new: dict[str, Any], gate: dict[str, Any], target_ids: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": old["sample_id"],
        "target_count": len(target_ids.intersection(old["expected_hotword_ids"])),
        "anchor_candidates_before_cap": {
            "old": old["anchor_candidates_before_cap"],
            "new": new["anchor_candidates_before_cap"],
        },
        "variant_shortlist_count": {
            "old": old["variant_shortlist_candidates"],
            "new": new["variant_shortlist_candidates"],
        },
        "surface_shortlist_count": {
            "old": old["shortlist_candidates"],
            "new": new["shortlist_candidates"],
        },
    }
    for k in (5, 7):
        a = _select(old["raw_ranked_matches"], gate=gate, top_k=k)
        b = _select(new["raw_ranked_matches"], gate=gate, top_k=k)
        expected = set(old["expected_hotword_ids"]) & target_ids
        a_hits = {m["hotword_id"] for m in a} & expected
        b_hits = {m["hotword_id"] for m in b} & expected
        result[f"top{k}"] = {
            "gained_target_ids": sorted(b_hits - a_hits),
            "lost_target_ids": sorted(a_hits - b_hits),
            "old_selected": [m["surface"] for m in a],
            "new_selected": [m["surface"] for m in b],
            "old_reference_absent": [
                m["surface"]
                for m in a
                if not strict_phrase_match(old["reference_text"], str(m["surface"]))
            ],
            "new_reference_absent": [
                m["surface"]
                for m in b
                if not strict_phrase_match(new["reference_text"], str(m["surface"]))
            ],
        }
    return result
