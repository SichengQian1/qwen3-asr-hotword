"""Paired final-ASR measures; reference-relative errors are not audited hallucinations."""

from __future__ import annotations

from collections import Counter
from typing import Any

from qwen_hotword.evaluation.prompt_order import GROUPS, PAIRS, key, rank
from qwen_hotword.inference.hotword_prompt import normalize_match_words, strict_phrase_match
from qwen_hotword.training.edit_distance import sequence_editops


def fraction(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def score(
    sample: dict[str, Any],
    condition: dict[str, Any],
    prediction: str,
    targets: list[dict[str, str]],
) -> dict[str, Any]:
    ref, hyp = normalize_match_words(sample["reference"]), normalize_match_words(prediction)
    ids = {w: n for n, w in enumerate(sorted(set(ref) | set(hyp)))}
    edits = Counter(
        op[0] for op in sequence_editops(tuple(ids[w] for w in ref), tuple(ids[w] for w in hyp))
    )
    expected = {key(t["word"]) for t in sample["oracle"]}
    matched = {key(t["word"]) for t in targets if strict_phrase_match(prediction, t["word"])}
    candidates = {key(t["word"]): t["word"] for t in condition["candidates"]}
    correct = {k for k, w in candidates.items() if strict_phrase_match(sample["reference"], w)}
    wrong = set(candidates) - correct
    written = {k for k, w in candidates.items() if strict_phrase_match(prediction, w)}
    high = {key(t["word"]) for t in condition["candidates"] if t.get("priority") == "high"}
    return {
        "expected_targets": sorted(expected),
        "matched_targets": sorted(matched & expected),
        "absent_declared_targets_written": sorted(matched - expected),
        "correct_injected": sorted(correct),
        "correct_injected_written": sorted(correct & written),
        "wrong_injected": sorted(wrong),
        "wrong_injected_written": sorted(wrong & written),
        "high_correct": sorted(high & correct),
        "high_correct_written": sorted(high & correct & written),
        "high_wrong": sorted(high & wrong),
        "high_wrong_written": sorted(high & wrong & written),
        "reference_words": len(ref),
        "word_errors": sum(edits.values()),
        "substitutions": edits["replace"],
        "deletions": edits["delete"],
        "insertions": edits["insert"],
        "normalized_prediction": " ".join(hyp),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [r["metrics"] for r in rows]

    def total(field: str) -> int:
        return sum(len(m[field]) for m in metrics)

    negative = [m for m in metrics if not m["expected_targets"]]
    return {
        "sample_count": len(rows),
        "candidate_count_histogram": dict(Counter(str(len(r["candidates"])) for r in rows)),
        "target_recall": fraction(total("matched_targets"), total("expected_targets")),
        "correct_prompt_adoption": fraction(
            total("correct_injected_written"), total("correct_injected")
        ),
        "wrong_prompt_landing": fraction(total("wrong_injected_written"), total("wrong_injected")),
        "high_priority_correct_adoption": fraction(
            total("high_correct_written"), total("high_correct")
        ),
        "high_priority_wrong_landing": fraction(total("high_wrong_written"), total("high_wrong")),
        "no_target_samples_with_declared_target_written": fraction(
            sum(bool(m["absent_declared_targets_written"]) for m in negative), len(negative)
        ),
        "no_target_samples_with_wrong_injected_written": fraction(
            sum(bool(m["wrong_injected_written"]) for m in negative), len(negative)
        ),
        "no_target_wrong_injected_landing": fraction(
            sum(len(m["wrong_injected_written"]) for m in negative),
            sum(len(m["wrong_injected"]) for m in negative),
        ),
        "wer": fraction(
            sum(m["word_errors"] for m in metrics), sum(m["reference_words"] for m in metrics)
        ),
        "substitutions": sum(m["substitutions"] for m in metrics),
        "deletions": sum(m["deletions"] for m in metrics),
        "insertions": sum(m["insertions"] for m in metrics),
        "zero_reference_word_samples": sum(m["reference_words"] == 0 for m in metrics),
        "generated_token_cap_reached": sum(r.get("generation_limit_reached", False) for r in rows),
    }


def compare(sample: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    # Compare both texts on a shared wrong-candidate universe, even when one group is no-prompt.
    universe = {
        key(t["word"]): t["word"]
        for r in (left, right)
        for t in r["candidates"]
        if not strict_phrase_match(sample["reference"], t["word"])
    }
    wrong = [
        {k for k, w in universe.items() if strict_phrase_match(r["prediction"], w)}
        for r in (left, right)
    ]
    hits = [set(r["metrics"]["matched_targets"]) for r in (left, right)]
    a, b = left["group"], right["group"]
    ordering = (a, b) in (("B", "C"), ("B", "D"), ("C", "D"), ("G", "H"))
    prompt_changed = left["actual_context"] != right["actual_context"]
    eligible = (len(left["candidates"]) >= 2 and prompt_changed) if ordering else prompt_changed
    priority_probe = None
    if (a, b) == ("E", "F") and sample["top7"]:
        word = sample["top7"][0]["word"]
        priority_probe = {
            "word": word,
            "reference_present": strict_phrase_match(sample["reference"], word),
            "left_written": strict_phrase_match(left["prediction"], word),
            "right_written": strict_phrase_match(right["prediction"], word),
        }
    return {
        "uid": sample["uid"],
        "raw_text_changed": left["prediction"] != right["prediction"],
        "normalized_text_changed": left["metrics"]["normalized_prediction"]
        != right["metrics"]["normalized_prediction"],
        "prompt_changed": prompt_changed,
        "effect_test_eligible": eligible,
        "ordering_comparison": ordering,
        "less_than_two_candidates": ordering and len(left["candidates"]) < 2,
        "correct_targets_added": sorted(hits[1] - hits[0]),
        "correct_targets_lost": sorted(hits[0] - hits[1]),
        "expected_target_count": len(sample["oracle"]),
        "wrong_candidate_universe": sorted(universe),
        "wrong_candidates_added": sorted(wrong[1] - wrong[0]),
        "wrong_candidates_disappeared": sorted(wrong[0] - wrong[1]),
        "word_error_delta": right["metrics"]["word_errors"] - left["metrics"]["word_errors"],
        "no_target_sample": not sample["oracle"],
        "priority_probe": priority_probe,
    }


def aggregate_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    informative = [r for r in rows if r["effect_test_eligible"]]
    priority = [r["priority_probe"] for r in rows if r["priority_probe"] is not None]
    priority_rates = {}
    for present in (True, False):
        subset = [r for r in priority if r["reference_present"] == present]
        label = "correct" if present else "reference_absent"
        priority_rates[label] = {
            side: fraction(sum(r[f"{side}_written"] for r in subset), len(subset))
            for side in ("left", "right")
        }
    return {
        "sample_count": len(rows),
        "priority_promoted_term_landing": priority_rates,
        "prompt_changed_samples": sum(r["prompt_changed"] for r in rows),
        "less_than_two_candidates": sum(r["less_than_two_candidates"] for r in rows),
        "identical_prompt_samples": sum(not r["prompt_changed"] for r in rows),
        "raw_text_changes": fraction(sum(r["raw_text_changed"] for r in rows), len(rows)),
        "normalized_text_changes": fraction(
            sum(r["normalized_text_changed"] for r in rows), len(rows)
        ),
        "eligible_normalized_text_changes": fraction(
            sum(r["normalized_text_changed"] for r in informative), len(informative)
        ),
        "correct_targets_added": sum(len(r["correct_targets_added"]) for r in rows),
        "correct_targets_lost": sum(len(r["correct_targets_lost"]) for r in rows),
        "correct_target_comparison_denominator": sum(r["expected_target_count"] for r in rows),
        "wrong_candidates_added": sum(len(r["wrong_candidates_added"]) for r in rows),
        "wrong_candidates_disappeared": sum(len(r["wrong_candidates_disappeared"]) for r in rows),
        "wrong_candidate_comparison_denominator": sum(
            len(r["wrong_candidate_universe"]) for r in rows
        ),
        "word_error_delta": sum(r["word_error_delta"] for r in rows),
        "wer_improved_samples": sum(r["word_error_delta"] < 0 for r in rows),
        "wer_worsened_samples": sum(r["word_error_delta"] > 0 for r in rows),
        "no_target_wrong_candidates_added": sum(
            len(r["wrong_candidates_added"]) for r in rows if r["no_target_sample"]
        ),
    }


def summarize(plan: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[Any]]:
    lookup = {(r["uid"], r["group"]): r for r in rows}
    paired = []
    for sample in plan["samples"]:
        conditions = {g: lookup[sample["uid"], g] for g in GROUPS}
        comparisons = {
            f"{a}_vs_{b}": compare(sample, conditions[a], conditions[b]) for a, b in PAIRS
        }
        paired.append(
            {
                "uid": sample["uid"],
                "dataset": sample["dataset"],
                "stratum": sample["stratum"],
                "reference": sample["reference"],
                "conditions": conditions,
                "comparisons": comparisons,
            }
        )
    views: dict[str, Any] = {"overall": paired}
    for field in ("dataset", "stratum"):
        for value in sorted({s[field] for s in paired}):
            views[f"by_{field}/{value}"] = [s for s in paired if s[field] == value]
    for dataset in sorted({s["dataset"] for s in paired}):
        for stratum in ("target_hit", "target_miss", "no_target"):
            views[f"by_dataset_stratum/{dataset}/{stratum}"] = [
                s for s in paired if s["dataset"] == dataset and s["stratum"] == stratum
            ]
    reports = {
        name: {
            "groups": {g: aggregate([s["conditions"][g] for s in cases]) for g in GROUPS},
            "comparisons": {
                f"{a}_vs_{b}": aggregate_pairs([s["comparisons"][f"{a}_vs_{b}"] for s in cases])
                for a, b in PAIRS
            },
        }
        for name, cases in views.items()
    }
    examples = []
    for a, b in PAIRS:
        label = f"{a}_vs_{b}"
        for category in ("unchanged", "improved", "worsened", "changed_equal_wer"):
            choices = []
            for sample in paired:
                c = sample["comparisons"][label]
                actual = (
                    "unchanged"
                    if not c["normalized_text_changed"]
                    else "improved"
                    if c["word_error_delta"] < 0
                    else "worsened"
                    if c["word_error_delta"] > 0
                    else "changed_equal_wer"
                )
                if c["effect_test_eligible"] and actual == category:
                    choices.append(sample)
            chosen = (
                min(
                    choices,
                    key=lambda s: rank(plan["configuration"]["seed"], label, category, s["uid"]),
                )
                if choices
                else None
            )
            examples.append(
                {
                    "comparison": label,
                    "category": category,
                    "eligible_count": len(choices),
                    "uid": chosen["uid"] if chosen else None,
                }
            )
    repeats = []
    for s in plan["samples"]:
        if s["repeat_b"]:
            first, repeat = lookup[s["uid"], "B"], lookup[s["uid"], "B_repeat"]
            repeats.append(compare(s, first, repeat))
    return {
        "views": reports,
        "repeatability": aggregate_pairs(repeats),
        "repeat_details": repeats,
        "representative_selection": examples,
        "representative_rule": "seeded case per eligible comparison/outcome; missing explicit",
        "interpretation_limits": [
            "Stratified exploratory sample; not an estimate for all 6000 audios.",
            "Changes need not improve; reference-relative errors are not audited hallucinations.",
            "G/H are Oracle, including targets outside Top7 and possibly more than seven.",
            "B/G mix correctness, coverage, count and order; they are not pure order effects.",
            "Oracle is unavailable online and is not a strict mathematical upper bound.",
            "<2 candidates or identical prompts do not test order sensitivity.",
            "Adoption is associative; use paired no-prompt controls for attribution.",
        ],
    }, paired
