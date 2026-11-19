"""English subset scoring on fixed final-prefix lists. No retrieval or re-ranking."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.zh_en_delivery import canonical

VIEWS = ("raw_top5_last", "gated_top5_last", "raw_top7_last", "gated_top7_last")
GROUPS = {
    "regular": "常规英文",
    "cross": "跨语言专名",
    "review": "边界词及其他品牌",
    "navigation": "导航实体",
    "multi": "数字／缩写及列明多读法（重叠）",
}


@dataclass(frozen=True)
class Label:
    surface: str
    group: str
    alias: str
    multi: bool


def policy(path: Path, table: list[dict[str, Any]]) -> dict[str, Label]:
    pool = {canonical(r["surface"]) for r in table}
    with path.open(encoding="utf-8", newline="") as stream:
        rules = list(csv.DictReader(stream, delimiter="\t"))
    labels = {
        canonical(r["surface"]): Label(
            r["surface"],
            r["group"],
            canonical(r["registered_alias"]),
            r["multiple_reading_tag"] == "1",
        )
        for r in rules
    }
    members: dict[str, set[str]] = defaultdict(set)
    for r in table:
        for p in r["pronunciations"]:
            for src in p["sources"]:
                members[src].add(canonical(r["surface"]))
    if (
        len(labels) != len(rules)
        or labels.keys() != members["en-smarthome-val"]
        or members["en-tts-583"] & labels.keys()
    ):
        raise ValueError("incomplete or conflicting smart-home grouping")
    aliases = [v.alias for v in labels.values() if v.alias]
    if (
        not set(aliases) <= pool
        or len(set(aliases)) != len(aliases)
        or set(aliases) & labels.keys()
        or any(v.group not in ("regular", "cross", "review") for v in labels.values())
    ):
        raise ValueError("invalid groups or ambiguous in-table aliases")
    for row in table:
        name = row["surface"]
        if canonical(name) in members["en-tts-583"]:
            tag = bool(re.search(r"\d|\b[A-Z]{2,}\b", name))
            labels[canonical(name)] = Label(name, "navigation", "", tag)
    return labels


def views(body: dict[str, Any]) -> list[list[str]]:
    raw = [canonical(w) for w in body["raw_last"]]
    final = [canonical(m["word"]) for m in body["steps"][-1]]
    return [raw[:5], final[:5], raw[:7], final[:7]]


@dataclass
class Word:
    n: int = 0
    strict: Counter[str] = field(default_factory=Counter)
    entity: Counter[str] = field(default_factory=Counter)

    def add(self, target: str, alias: str, lists: list[list[str]]) -> None:
        self.n += 1
        for view, ranked in zip(VIEWS, lists, strict=True):
            self.strict[view] += target in ranked
            self.entity[view] += target in ranked or bool(alias and alias in ranked)


def summary(words: list[Word], metric: str, macro: bool) -> list[float | None]:
    if not words:
        return [None] * len(VIEWS)
    n = sum(w.n for w in words)
    return [
        sum(getattr(w, metric)[v] / w.n for w in words) / len(words)
        if macro
        else sum(getattr(w, metric)[v] for w in words) / n
        for v in VIEWS
    ]


def worst(words: dict[str, Word], labels: dict[str, Label]) -> list[str]:
    return sorted(
        (w for w in words if labels[w].group == "cross"),
        key=lambda w: (Fraction(words[w].strict[VIEWS[-1]], words[w].n), -words[w].n, w),
    )[:10]
