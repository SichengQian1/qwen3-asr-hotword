"""Score delivered contexts after retrieval, using sealed full-utterance references."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.inference.stream_context_contract import MODES


def reference_matcher(entries: Iterable[HotwordEntry]) -> Callable[[str], set[str]]:
    # Same contiguous normalized word tuples as production strict_phrase_match;
    # normalize each reference once instead of once for each of 4,000 entries.
    phrases: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for entry in entries:
        phrase = normalize_match_words(entry.surface)
        if not phrase:
            raise ValueError("empty normalized hotword phrase")
        phrases[phrase].add(entry.hotword_id)
    widths = {len(p) for p in phrases}

    def match(text: str) -> set[str]:
        words = normalize_match_words(text)
        result: set[str] = set()
        for width in widths:
            for offset in range(len(words) - width + 1):
                result.update(phrases.get(words[offset : offset + width], ()))
        return result

    return match


def _counts(expected: set[str], present: set[str], selected: set[str]) -> dict[str, int]:
    return {
        "sample_count": 1,
        "expected_target_sample_pairs": len(expected),
        "target_hits": len(expected & selected),
        "selected_word_sample_pairs": len(selected),
        "words_absent_from_full_reference": len(selected - present),
    }


def _sum_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {k: sum(r[k] for r in rows) for k in rows[0]}
    denominator = result["expected_target_sample_pairs"]
    result["target_recall"] = result["target_hits"] / denominator if denominator else None
    return result


def score_stream_contexts(
    contexts: dict[str, Any],
    samples: list[dict[str, Any]],
    evaluation: dict[str, Any],
    entries: tuple[HotwordEntry, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    match = reference_matcher(entries)
    by_surface = {e.surface: e.hotword_id for e in entries}
    details: list[dict[str, Any]] = []
    accum: dict[str, Any] = {}
    for sample in samples:
        group, key, utt = sample["group"], sample["key"], sample["sample_id"]
        specification = evaluation[group]
        present = match(specification["references"][utt])
        expected = present & set(specification["target_ids"])
        detail: dict[str, Any] = {
            "group": group,
            "key": key,
            "sample_id": utt,
            "expected_target_ids": sorted(expected),
            "modes": {},
        }
        for mode in MODES:
            steps = contexts[mode][key][utt]
            sets = [{by_surface[c["word"]] for c in candidates} for candidates in steps]
            selected_views = {"last_context": sets[-1], "ever_delivered": set().union(*sets)}
            views = {view: _counts(expected, present, ids) for view, ids in selected_views.items()}
            detail["modes"][mode] = {
                view: {"target_hit_ids": sorted(expected & ids), "selected_ids": sorted(ids)}
                for view, ids in selected_views.items()
            }
            group_counts = accum.setdefault(group, {}).setdefault(mode, {})
            for view, counts in views.items():
                group_counts.setdefault(view, []).append(counts)
        details.append(detail)
    groups = {
        group: {
            mode: {view: _sum_counts(rows) for view, rows in views.items()}
            for mode, views in modes.items()
        }
        for group, modes in accum.items()
    }
    for group, modes in groups.items():
        for views in modes.values():
            if any(
                counts["expected_target_sample_pairs"]
                != evaluation[group]["expected_target_sample_pairs"]
                for counts in views.values()
            ):
                raise ValueError(f"measured target denominator changed: {group}")
    languages = {}
    for lang in ("en", "es", "pt"):
        names = [g for g in groups if evaluation[g]["language"] == lang]
        if not names:
            raise ValueError(f"no measured data for language: {lang}")
        languages[lang] = {
            mode: {
                view: _sum_counts(
                    [
                        {k: v for k, v in groups[g][mode][view].items() if k != "target_recall"}
                        for g in names
                    ]
                )
                for view in ("last_context", "ever_delivered")
            }
            for mode in MODES
        }
    return {"by_language": languages, "by_dataset": groups}, details
