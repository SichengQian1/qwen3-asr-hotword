"""Frozen AISHELL selection and explicit Chinese target metrics; no Latin word matching."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import (
    HotwordMatch,
    HotwordScoringConfig,
    select_operating_matches,
)
from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.phonemes.mfa_v03_assets import pin
from qwen_hotword.training.zh_en_common import rows


def select_dataset(dataset: Path, policy: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Use reviewed or unchanged original labels, never infer extra targets."""
    for name, expected in policy["source_sha256"].items():
        pin(dataset / name, expected)
    with (dataset / "kw/kw_candidates.tsv").open(encoding="utf-8") as stream:
        candidates = [r["word"] for r in csv.DictReader(stream, delimiter="\t")]
    mode = policy.get("selection_mode", "reviewed")
    if mode not in ("reviewed", "original"):
        raise ValueError("unknown AISHELL selection mode")
    if mode == "original" and (
        "candidate_surfaces" in policy or policy.get("excluded_target_pairs")
    ):
        raise ValueError("original evaluation cannot use a whitelist or target exclusions")
    surfaces = sorted(candidates) if mode == "original" else policy["candidate_surfaces"]
    if len(surfaces) != len(set(surfaces)) or not all(isinstance(w, str) and w for w in surfaces):
        raise ValueError("duplicate/empty candidate surfaces")
    if len(candidates) != len(set(candidates)) or not set(surfaces) <= set(candidates):
        raise ValueError("reviewed candidates do not match source table")
    excluded_rows = policy.get("excluded_target_pairs", [])
    excluded = {(r["utt_id"], r["word"]) for r in excluded_rows}
    if len(excluded) != len(excluded_rows):
        raise ValueError("duplicate exclusion decisions")
    seen: set[str] = set()
    observed_exclusions: set[tuple[str, str]] = set()
    selected = []
    allowed = set(surfaces)
    for row in rows(dataset / "val/kw_subset/kw.jsonl"):
        sid, text = row["utt_id"], row["text"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sid) or sid in seen or not text:
            raise ValueError("invalid/duplicate utterance ID or empty reference")
        seen.add(sid)
        words = [k["word"] for k in row["keywords"]]
        if len(words) != len(set(words)) or any(w not in text for w in words):
            raise ValueError(f"invalid original targets: {sid}")
        if mode == "original" and (not words or not set(words) <= allowed):
            raise ValueError(f"original row has no targets or targets outside full table: {sid}")
        targets = []
        for word in words:
            if (sid, word) in excluded:
                observed_exclusions.add((sid, word))
            elif word in allowed:
                targets.append(word)
        if targets:
            selected.append({"sample_id": sid, "reference_text": text, "targets": targets})
    if observed_exclusions != excluded:
        raise ValueError("an excluded pair was not found in the pinned source")
    counts = {
        "candidates": len(surfaces),
        "utterances": len(selected),
        "targets": len({w for r in selected for w in r["targets"]}),
        "target_pairs": sum(len(r["targets"]) for r in selected),
    }
    if counts != policy["expected_counts"]:
        raise ValueError(f"approved denominator changed: {counts}")
    return list(surfaces), selected


def check_result_counts(result: dict[str, Any], expected: dict[str, int]) -> None:
    actual = (result["sample_count"], result["target_words"], result["target_pairs"])
    required = (expected["utterances"], expected["targets"], expected["target_pairs"])
    if actual != required:
        raise ValueError(f"evaluation denominator changed: {actual}; expected {required}")


def assemble_entries(
    prepared: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    vocab: PhonemeVocab,
) -> tuple[list[HotwordEntry], list[dict[str, Any]]]:
    """Top1 only. Never keep partial pronunciations or collapse homophone surfaces."""
    lookup = {(r["language"], r["word"]): r for r in predictions}
    if len(lookup) != len(predictions):
        raise ValueError("duplicate MFA predictions")
    requested = {(u["language"], u["word"]) for r in prepared for u in r["units"]}
    if set(lookup) - requested:
        raise ValueError("unexpected MFA prediction")
    entries, failures = [], []
    for number, row in enumerate(prepared):
        ids: list[int] = []
        errors = list(row["errors"])
        for unit in row["units"]:
            result = lookup.get((unit["language"], unit["word"]))
            if not result or result.get("status") != "ok" or not result.get("candidates"):
                errors.append("missing_or_failed_g2p:" + unit["word"])
                continue
            top1 = result["candidates"][0]
            tokens = top1.get("token_ids", [])
            if (
                not top1.get("usable")
                or not tokens
                or any(type(i) is not int or i <= 1 or i >= len(vocab.tokens) for i in tokens)
            ):
                errors.append("unusable_top1:" + unit["word"])
            else:
                ids.extend(tokens)
        if errors or not ids:
            failures.append({"surface": row["surface"], "errors": errors or ["empty_label"]})
            continue
        phones = tuple(vocab.tokens[i] for i in ids)
        entries.append(
            HotwordEntry(
                hotword_id=f"aishell_zh_{number:04d}",
                language="zh",
                surface=row["surface"],
                normalized=row["surface"],
                words=(row["surface"],),
                pronunciation=" ".join(phones),
                phoneme_tokens=phones,
                token_ids=tuple(ids),
                source="mandarin_china_mfa:3.0.0:pinned_training_frontend_top1",
                validation_occurrences=0,  # Unknown here; targets never influence index weights.
            )
        )
    return entries, failures


def selections(result: dict[str, Any], scoring: dict[str, Any]) -> dict[str, list[str]]:
    matches = tuple(HotwordMatch(**r) for r in result["raw_ranked_matches"])
    if len({m.surface for m in matches}) != len(matches):
        raise ValueError("duplicate surfaces in ranking; this index has one pronunciation per word")
    if any(not math.isfinite(m.score) for m in matches):
        raise ValueError("nonfinite ranking score")
    config = HotwordScoringConfig(**scoring)
    out = {}
    for k in (5, 7):
        gated, _ = select_operating_matches(matches, config=config, top_k=k)
        out[f"raw_top{k}"] = [m.surface for m in matches[:k]]
        out[f"gated_top{k}"] = [m.surface for m in gated]
    return out


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter(w for r in details for w in r["targets"])
    denominator = sum(totals.values())
    if not denominator:
        raise ValueError("no targets")
    metrics = {}
    for mode in ("raw_top5", "raw_top7", "gated_top5", "gated_top7"):
        hits: Counter[str] = Counter()
        absent = unannotated = selected = full = 0
        for row in details:
            words, targets = set(row["selections"][mode]), set(row["targets"])
            hits.update(words & targets)
            selected += len(words)
            full += targets <= words
            # This is a textual diagnostic, not an entity-correctness/false-positive label.
            absent += sum(w not in row["reference_text"] for w in words - targets)
            unannotated += sum(w in row["reference_text"] for w in words - targets)
        metrics[mode] = {
            "target_hits": sum(hits.values()),
            "target_pairs": denominator,
            "recall": sum(hits.values()) / denominator,
            "macro_word_recall": sum(hits[w] / n for w, n in totals.items()) / len(totals),
            "all_targets_hit_utterances": full,
            "selected_word_sample_pairs": selected,
            "non_target_words_absent_from_reference": absent,
            "non_target_words_present_but_unannotated_or_excluded": unannotated,
            "by_word": {w: {"hits": hits[w], "targets": n} for w, n in sorted(totals.items())},
        }
    return {
        "sample_count": len(details),
        "target_words": len(totals),
        "target_pairs": denominator,
        "metrics": metrics,
    }
