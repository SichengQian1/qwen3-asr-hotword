"""Post-hoc analysis of saved English rankings (no change to retrieval policy)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

from qwen_hotword.evaluation.zh_en_delivery import GATE, canonical
from qwen_hotword.training.edit_distance import sequence_edit_distance


def neighbors(table: list[dict[str, Any]], source: str) -> dict[str, set[str]]:
    """Compare English source pronunciations against every distinct joint surface."""
    entries = [
        (canonical(row["surface"]), tuple(p["token_ids"]), p["sources"])
        for row in table
        for p in row["pronunciations"]
    ]
    result: dict[str, set[str]] = {}
    for word, ids, sources in entries:
        if source not in sources:
            continue
        close = result.setdefault(word, set())
        for other, alt, _ in entries:
            cap = max(len(ids), len(alt)) // 5
            if (
                other != word
                and abs(len(ids) - len(alt)) <= cap
                and sequence_edit_distance(ids, alt) <= cap
            ):
                close.add(other)
    return result


def count_probe(
    word: str, gold: set[str], ranked: list[str], near: set[str], counts: Counter[str]
) -> list[str]:
    counts["pairs"] += 1
    if word in ranked:
        counts["hits"] += 1
        return []
    counts["misses"] += 1
    rivals = [w for w in ranked if w in near and w not in gold]
    if rivals:
        counts["miss_with_non_gold_neighbor"] += 1
    elif near.intersection(ranked):
        counts["miss_only_other_gold_neighbor"] += 1
    else:
        counts["miss_without_ranked_neighbor"] += 1
    return rivals


def text_contains(text: str, word: str) -> bool:
    def tokens(value: str) -> str:
        return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))

    token_word = tokens(word)
    return bool(token_word) and f" {token_word} " in f" {tokens(text)} "


def choose_cases(cases: Iterable[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    # Deterministic variety sample per split, not a population estimate.
    selected = []
    seen: set[tuple[str, str]] = set()
    used_utterances: set[str] = set()
    counts: Counter[str] = Counter()
    for item in sorted(cases, key=lambda r: (r["group"], r["sample_id"], r["rank"])):
        group, word, uid = item["group"], item["target"], item["sample_id"]
        if counts[group] < limit and (group, word) not in seen and uid not in used_utterances:
            selected.append(item)
            seen.add((group, word))
            used_utterances.add(uid)
            counts[group] += 1
    return selected


def failed_gates(match: dict[str, Any]) -> list[str]:
    failures = []
    if match["score"] < GATE["threshold"]:
        failures.append("score")
    if match["edit_ratio"] > GATE["maximum_edit_ratio"]:
        failures.append("edit_ratio")
    if match["posterior_confidence"] < GATE["minimum_posterior_confidence"]:
        failures.append("posterior_confidence")
    return failures
