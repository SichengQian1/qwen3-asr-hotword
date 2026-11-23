"""Joint-table expansion and gold coverage checks for the Chinese light-hotword set."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation import zh_en_eleven as sets

LIGHT = "zh-light-hotwords"
SUMMARY = "中文汇总（AISHELL完整集计一次）"
GROUPS = (sets.FULL, *(g for g in base.COUNTS if g.startswith("zh-")), LIGHT)
REUSED = {"繁花", "哪吒之魔童闹海", "灯带"}
# Pronunciation input only. Preserve the original spelling in every retrieved candidate.
SPOKEN = {
    "芒果TV": "芒果 tee vee",
    "QQ音乐": "cue cue 音乐",
    "甄嬛传": "甄环撰",
    "如懿传": "如懿撰",
    "芈月传": "芈月撰",
}


def additions(words: list[str], previous: list[dict[str, Any]]) -> list[str]:
    existing = {base.canonical(r["surface"]) for r in previous}
    if len(words) != 63 or len({base.canonical(w) for w in words}) != 63:
        raise ValueError("expected 63 distinct requested words")
    if {w for w in words if base.canonical(w) in existing} != REUSED:
        raise ValueError("joint table overlap changed")
    return [w for w in words if base.canonical(w) not in existing]


def metadata(wavs: Path, explicit: Path | None) -> Path:
    if explicit:
        if not explicit.is_file():
            raise ValueError(f"找不到指定标注 {explicit}")
        return explicit
    # Prefer metadata for the actual 16 kHz renders over a source synthesis manifest.
    for names in (
        ("meta_16k.jsonl", "metadata_16k.jsonl"),
        ("metadata.jsonl", "meta.jsonl", "manifest.jsonl", "labels.jsonl", "labels.tsv"),
    ):
        found = [d / n for d in (wavs, wavs.parent) for n in names if (d / n).is_file()]
        if len(found) == 1:
            return found[0]
        if found:
            raise ValueError(f"存在多份标注，用 --labels 指定其中一份：{found}")
    raise ValueError("需要逐 WAV 标注；请查看 light_layout.json 或用 --labels 指定 JSONL/TSV")


def label_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as f:
        if path.suffix.lower() == ".tsv":
            yield from csv.DictReader(f, delimiter="\t")
        elif path.suffix.lower() == ".jsonl":
            for line in f:
                if line.strip():
                    yield json.loads(line)
        else:
            raise ValueError("标注格式须为 JSONL 或 TSV")


def light_gold(
    root: Path, files: list[Path], labels: Path, words: list[str]
) -> list[tuple[str, str, Path, list[str]]]:
    available = {p.resolve(): p for p in files}
    if not files or len(available) != len(files):
        raise ValueError(f"empty directory or duplicate path aliases: {root}")
    vocabulary = {base.canonical(w): w for w in words}
    result = []
    seen: set[Path] = set()
    for line, row in enumerate(label_rows(labels), 1):
        if row.get("status", "ok") != "ok":
            raise ValueError(f"{labels}:{line}: non-ok synthesis status")
        location = row.get("wav") or row.get("audio_path")
        if not isinstance(location, str) or not location.strip():
            raise ValueError(f"{labels}:{line}: missing wav/audio_path")
        paths = {(d / location).resolve() for d in (root, labels.parent)} & available.keys()
        if len(paths) != 1:
            raise ValueError(f"{labels}:{line}: WAV path missing or ambiguous: {location}")
        path = paths.pop()
        if path in seen:
            raise ValueError(f"{labels}:{line}: duplicate WAV: {location}")
        seen.add(path)
        targets = next(
            (row[k] for k in ("keywords", "targets", "keyword", "hotword") if k in row), None
        )
        if isinstance(targets, str) and targets.strip().startswith("["):
            targets = json.loads(targets)
        if isinstance(targets, str):
            targets = [targets]
        if targets is None:
            # Exact whole-utterance label only, never substring-match a sentence.
            ref = row.get("text")
            if isinstance(ref, str) and base.canonical(ref) in vocabulary:
                targets = [ref]
        if not isinstance(targets, list) or any(
            not isinstance(x, str) or base.canonical(x) not in vocabulary for x in targets
        ):
            raise ValueError(f"{labels}:{line}: missing/unknown keyword; explicit [] for negatives")
        result.append(
            (
                LIGHT,
                available[path].relative_to(root).as_posix(),
                available[path],
                sorted({vocabulary[base.canonical(x)] for x in targets}),
            )
        )
    missing = available.keys() - seen
    if missing:
        raise ValueError(f"{len(missing)} WAVs missing annotations, e.g. {min(missing)}")
    if not any(target for _, _, _, target in result):
        raise ValueError("light test has no positive annotations")
    return sorted(result)


def aggregate(totals: Mapping[str, Mapping[str, Counter[str]]]) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = {}
    for group in GROUPS:
        if group == base.AIS:
            continue
        for view, counts in totals[group].items():
            result.setdefault(view, Counter()).update(counts)
    return result
