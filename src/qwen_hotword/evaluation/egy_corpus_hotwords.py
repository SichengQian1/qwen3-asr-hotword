"""Corpus-conditioned Arabic retrieval benchmark; no reference enters the detector."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from qwen_hotword.evaluation.egy_hotword_review import BUCKETS, tokens

RATIOS = dict(zip(BUCKETS, (0.8, 0.1, 0.1), strict=True))


def order_key(value: str) -> str:
    return hashlib.sha256(("egy-retrieval-20261123:" + value).encode()).hexdigest()


def select_audio(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row["language"] != "ar" or row["split"] != "test":
            raise ValueError("requires sealed Arabic test records")
        group = row["source_group"]
        if group not in RATIOS or row["audio_sha256"] in seen:
            raise ValueError("unknown source or duplicate test audio")
        seen.add(row["audio_sha256"])
        if not math.isfinite(row["duration_seconds"]) or row["duration_seconds"] <= 0:
            raise ValueError("invalid duration")
        buckets[group].append(row)
    capacity = {g: sum(r["duration_seconds"] for r in buckets[g]) for g in RATIOS}
    target = min(capacity[g] / RATIOS[g] for g in RATIOS)
    if target <= 0:
        raise ValueError("all three source groups need held-out audio")
    selected, actual = [], {}
    for group, ratio in RATIOS.items():
        ordered = sorted(buckets[group], key=lambda r: order_key(r["id"]))
        total = 0.0
        for row in ordered:
            if total >= target * ratio - 1e-7:
                break
            selected.append(row)
            total += row["duration_seconds"]
        actual[group] = {
            "samples": sum(r["source_group"] == group for r in selected),
            "hours": total / 3600,
            "target_hours": target * ratio / 3600,
        }
    return selected, {
        "target_hours": target / 3600,
        "by_source": actual,
        "rounding": "whole audio; overshoot less than one selected audio per bucket",
        "selection_uses_keywords": False,
    }


def assert_disjoint(selected: list[dict[str, Any]], protected: Any) -> None:
    ids = {r["id"] for r in selected}
    hashes = {r["audio_sha256"] for r in selected}
    paths = {r["audio_path"] for r in selected}
    recordings = {
        r["recording_key"] for r in selected if r.get("grouping_basis") == "recognized_recording"
    }
    for row in protected:
        if (
            row["id"] in ids
            or row.get("audio_sha256") in hashes
            or row["audio_path"] in paths
            or (
                row.get("grouping_basis") == "recognized_recording"
                and row.get("recording_key") in recordings
            )
        ):
            raise ValueError("selected test overlaps train/validation: " + row["id"])


def valid_surface(surface: str) -> bool:
    parts = tokens(surface)
    return (
        1 <= len(parts) <= 6
        and sum(map(len, parts)) >= 3
        and all(
            all(c.isalpha() and "ARABIC" in unicodedata.name(c, "") for c in word) for word in parts
        )
    )


def entity_surfaces(text: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in predictions:
        start, end = int(item["start"]), int(item["end"])
        if not 0 <= start < end <= len(text):
            raise ValueError("NER offset outside reference")
        surface = text[start:end]
        # Refuse partial-word pieces, including attached conjunction/preposition splits.
        if (start and (text[start - 1].isalnum() or unicodedata.combining(text[start - 1]))) or (
            end < len(text) and (text[end].isalnum() or unicodedata.combining(text[end]))
        ):
            continue
        if (
            item["entity_group"] in {"PERS", "LOC", "ORG"}
            and float(item["score"]) >= 0.9
            and valid_surface(surface)
        ):
            result.append(
                {
                    "surface": surface,
                    "category": item["entity_group"],
                    "origin": "CAMeLBERT_NER",
                    "ner_score": float(item["score"]),
                }
            )
    return result


def phrase_present(text: str, surface: str) -> bool:
    haystack, needle = tokens(text), tokens(surface)
    return bool(needle) and any(
        haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1)
    )


def build_table(
    evaluation: list[dict[str, Any]],
    training: list[dict[str, Any]],
    extracted: dict[str, list[dict[str, Any]]],
    terms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, ...], dict[str, Any]] = {}
    evaluation_ngrams: set[tuple[str, ...]] = set()
    for row in evaluation:
        words = tokens(row["text"])
        for length in range(1, 7):
            evaluation_ngrams.update(words[i : i + length] for i in range(len(words) - length + 1))
    for role, rows in [("target", evaluation), ("distractor", training)]:
        for row in rows:
            found = list(extracted[row["id"]])
            found += [
                {
                    "surface": t["surface"],
                    "category": t["category"],
                    "origin": "observed_technical_term",
                    "gloss_zh": t["gloss_zh"],
                }
                for t in terms
                if phrase_present(row["text"], t["surface"])
            ]
            for item in found:
                form = tokens(item["surface"])
                if not valid_surface(item["surface"]):
                    continue
                if role == "distractor" and form in evaluation_ngrams:
                    continue
                if form not in candidates:
                    candidates[form] = {
                        **item,
                        "id": "egy_hw_" + order_key(" ".join(form))[:16],
                        "role": role,
                        "examples": [],
                        "human_semantic_review_completed": False,
                    }
                entry = candidates[form]
                if len(entry["examples"]) < 2:
                    entry["examples"].append(
                        {k: row[k] for k in ("id", "source_group", "text", "source", "source_row")}
                    )
    targets = sorted(
        (v for v in candidates.values() if v["role"] == "target"), key=lambda r: r["id"]
    )
    distractors = sorted(
        (v for v in candidates.values() if v["role"] == "distractor"), key=lambda r: r["id"]
    )[: len(targets)]
    if not targets:
        raise ValueError("no observed entity/term targets; no meaningful recall denominator")
    return targets + distractors


def expected_ids(text: str, table: list[dict[str, Any]]) -> set[str]:
    """Longest first non-overlapping phrases; repeated mentions count once per audio."""
    words = tokens(text)
    spans = []
    for entry in table:
        form = tokens(entry["surface"])
        for start in range(len(words) - len(form) + 1):
            if words[start : start + len(form)] == form:
                spans.append((-len(form), start, entry["id"]))
    occupied: set[int] = set()
    found: set[str] = set()
    for negative_size, start, identifier in sorted(spans):
        positions = set(range(start, start - negative_size))
        if not positions & occupied:
            occupied |= positions
            found.add(identifier)
    return found


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = sum(len(r["expected_ids"]) for r in rows)
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "negative_sample_count": sum(not r["expected_ids"] for r in rows),
        "expected_target_sample_pairs": denominator,
        "targets_per_audio": dict(Counter(str(len(r["expected_ids"])) for r in rows)),
    }
    for mode in ("operating", "raw_ranked"):
        result[mode] = {}
        for k in (5, 7):
            hits = selected = absent = negative_prompts = 0
            for row in rows:
                gold = set(row["expected_ids"])
                matches = row["raw_ranked_matches"]
                if mode == "operating":
                    matches = [
                        m
                        for m in matches
                        if m["score"] >= 0.75
                        and m["edit_ratio"] <= 0.35
                        and m["posterior_confidence"] >= 0.5
                    ]
                ids = {m["hotword_id"] for m in matches[:k]}
                hits += len(gold & ids)
                selected += len(ids)
                # Nested shorter matches can be true in reference but not separate targets.
                absent += len(ids - set(row["reference_present_ids"]))
                negative_prompts += int(not gold and bool(ids))
            result[mode][f"top{k}"] = {
                "target_hits": hits,
                "target_recall": hits / denominator if denominator else None,
                "selected_word_sample_pairs": selected,
                "words_absent_from_reference": absent,
                "reference_presence_precision": (selected - absent) / selected
                if selected
                else None,
                "negative_audio_false_prompt_rate": negative_prompts
                / result["negative_sample_count"]
                if result["negative_sample_count"]
                else None,
                "negative_audio_with_candidates": negative_prompts,
                "capacity_upper_bound": sum(min(k, len(r["expected_ids"])) for r in rows)
                / denominator
                if denominator
                else None,
            }
    return result
