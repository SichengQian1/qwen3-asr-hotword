"""Adapt multilingual pronunciation variants to one Top-K slot per reviewed surface."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import LANGUAGES
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import (
    _metric_summary,
    _select,
    _validated_source_gate,
    _verify_source_top5_metrics,
    replay_rows_top7,
)
from qwen_hotword.inference.wave_retrieval import _view_metrics


def joint_bundle(
    bundles: dict[str, external.KeywordBundle],
) -> tuple[external.KeywordBundle, tuple[HotwordEntry, ...], dict[str, HotwordEntry]]:
    canonical: dict[str, HotwordEntry] = {}
    indexed: dict[tuple[str, tuple[int, ...]], HotwordEntry] = {}
    by_variant = {}
    for lang in LANGUAGES:
        for entry in bundles[lang].entries:
            if entry.normalized not in canonical:
                canonical[entry.normalized] = replace(
                    entry,
                    language="mul",
                    hotword_id="joint_"
                    + hashlib.sha256(entry.normalized.encode()).hexdigest()[:16],
                )
            by_variant[entry.hotword_id] = canonical[entry.normalized]
            indexed.setdefault((entry.normalized, entry.token_ids), entry)
    entries = tuple(canonical[k] for k in sorted(canonical))
    return (
        external.KeywordBundle(entries, {e.surface: e.pronunciation for e in entries}, {}),
        tuple(sorted(indexed.values(), key=lambda e: (e.normalized, e.hotword_id))),
        by_variant,
    )


def collapse_variants(
    retrieval: Mapping[str, Any],
    variants: tuple[HotwordEntry, ...],
    by_variant: dict[str, HotwordEntry],
    gate: Mapping[str, object],
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = retrieval["raw_ranked_matches"]
    if len(raw) != retrieval["shortlist_candidates"]:
        raise ValueError("joint-word merging requires the complete scored variant shortlist")
    lookup = {e.hotword_id: e for e in variants}
    seen, merged = set(), []
    for match in raw:
        variant = lookup[match["hotword_id"]]
        word = by_variant[variant.hotword_id]
        if word.hotword_id in seen:
            continue
        seen.add(word.hotword_id)
        merged.append(
            {
                **match,
                "hotword_id": word.hotword_id,
                "surface": word.surface,
                "matched_variant_id": variant.hotword_id,
                "matched_pronunciation": variant.pronunciation,
                "matched_language": variant.language,
            }
        )
    # Keep the highest-ranked pronunciation per surface, then apply the SAME fixed gate.
    selected = _select(merged, gate=gate, top_k=5)
    elapsed = time.perf_counter() - started
    timing = dict(retrieval["timing"])
    timing["joint_word_merge_seconds"] = elapsed
    for key in ("pure_retrieval_seconds", "model_retrieval_seconds"):
        timing[key] += elapsed
    return {
        **retrieval,
        "raw_ranked_matches": merged,
        "selected_matches": selected,
        "shortlist_candidates": len(merged),
        "timing": timing,
        "raw_variant_ranked_matches": raw,
        "variant_shortlist_candidates": len(raw),
    }


def export_joint_group(
    output: Path,
    name: str,
    group: dict[str, Any],
    bundle: external.KeywordBundle,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # Legacy summaries use the same surface-presence metric; downstream IPA comes
    # from the winning pronunciation for THIS audio, not a fixed language default.
    _, _, summary, _ = external.summarize_retrieval(rows, bundle=bundle)
    summary["candidate_scope"] = "joint_multilingual"
    replay = replay_rows_top7(rows, gate=_validated_source_gate(group["config"]), bundle=bundle)
    _verify_source_top5_metrics(
        {
            "overall": _metric_summary(replay, top_k=5),
            "by_source": {name: _metric_summary(replay, top_k=5)},
        },
        summary,
    )
    metrics = {}
    for k in (5, 7):
        delivery = {}
        for original, replayed in zip(rows, replay, strict=True):
            matches = {r["hotword_id"]: r for r in original["raw_ranked_matches"]}
            delivery[original["sample_id"]] = [
                {"word": matches[i]["surface"], "phoneme": matches[i]["matched_pronunciation"]}
                for i in external._string_list(replayed, f"top{k}_selected_hotword_ids")
            ]
        external._write_or_verify_json(output / "delivery" / f"{name}_top{k}.json", delivery)
        view = _view_metrics(replay, set(group["target_ids"]), top_k=k)
        view["full_reviewed_joint_view"] = view.pop("full_4000_view")
        view["declared_retained_target_count"] = view.pop("declared_wave_target_count")
        metrics[f"top{k}"] = view
    external._write_or_verify_jsonl(output / name / "retrieval_details.jsonl", rows)
    external._write_or_verify_json(output / name / "evaluation_summary.json", summary)
    external._write_or_verify_json(output / name / "topk_metrics.json", metrics)
    return metrics
