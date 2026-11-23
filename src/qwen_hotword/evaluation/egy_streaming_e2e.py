"""Causal replay and paired Arabic transcript metrics; no inference dependencies."""

from __future__ import annotations

from collections import Counter
from typing import Any

from qwen_hotword.evaluation.casablanca_wer import normalize
from qwen_hotword.evaluation.egy_corpus_hotwords import expected_ids, order_key, phrase_present
from qwen_hotword.inference.streaming_core import StreamingCandidate
from qwen_hotword.training.edit_distance import sequence_edit_distance

MODES = ("no_rag", "top5", "top7")


def smoke_rows(rows: list[dict[str, Any]], table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cover each source and target presence before filling a deterministic 20-row smoke."""
    ordered = sorted(rows, key=lambda row: order_key(row["id"]))
    selected: dict[str, dict[str, Any]] = {}
    for source in sorted({row["source_group"] for row in rows}):
        for positive in (True, False):
            for row in ordered:
                if (
                    row["source_group"] == source
                    and bool(expected_ids(row["text"], table)) == positive
                ):
                    selected[row["id"]] = row
                    break
    for row in ordered:
        if len(selected) >= min(20, len(rows)):
            break
        selected[row["id"]] = row
    return list(selected.values())


class PrefixReplay:
    """Only release the cache entry for the EXACT currently available prefix."""

    def __init__(self, prefixes: list[dict[str, Any]], active_ids: tuple[str, ...], k: int):
        if k not in (5, 7):
            raise ValueError("expected Top5 or Top7")
        self.prefixes, self.active_ids, self.k = prefixes, active_ids, k
        self.position = 0

    def __call__(
        self, waveform: Any, active_ids: tuple[str, ...]
    ) -> tuple[StreamingCandidate, ...]:
        if active_ids != self.active_ids or self.position >= len(self.prefixes):
            raise ValueError("prefix replay index or active table changed")
        prefix = self.prefixes[self.position]
        if len(waveform) != prefix["end_sample"]:
            raise ValueError("causal prefix boundary mismatch; refusing future context")
        self.position += 1
        # Already selected by the SAME operating policy as the whole-audio baseline.
        matches = prefix["selected_matches"][: self.k]
        if any(m["hotword_id"] not in active_ids for m in matches):
            raise ValueError("cached candidate outside frozen active table")
        return tuple(
            StreamingCandidate(
                **{
                    key: m[key]
                    for key in (
                        "hotword_id",
                        "surface",
                        "score",
                        "edit_ratio",
                        "posterior_confidence",
                    )
                }
            )
            for m in matches
        )

    def finish(self) -> None:
        if self.position != len(self.prefixes):
            raise ValueError("stream did not consume every causal prefix")


def edit_count(ref: list[str], hyp: list[str]) -> int:
    ids = {s: n for n, s in enumerate(sorted(set(ref + hyp)))}
    return sequence_edit_distance(tuple(ids[s] for s in ref), tuple(ids[s] for s in hyp))


def score_row(
    reference: str, prediction: str, table: list[dict[str, Any]], injected: set[str]
) -> dict[str, Any]:
    expected = set(expected_ids(reference, table))
    present_ref = {t["id"] for t in table if phrase_present(reference, t["surface"])}
    present_hyp = {t["id"] for t in table if phrase_present(prediction, t["surface"])}
    ref, hyp = normalize(reference), normalize(prediction)
    correct_prompt = expected & injected
    wrong_prompt = injected - present_ref
    return {
        "expected_ids": sorted(expected),
        "hit_ids": sorted(expected & present_hyp),
        "target_pairs": len(expected),
        "target_hits": len(expected & present_hyp),
        "reference_words": len(ref.split()),
        "word_errors": edit_count(ref.split(), hyp.split()),
        "reference_characters": len(ref.replace(" ", "")),
        "character_errors": edit_count(list(ref.replace(" ", "")), list(hyp.replace(" ", ""))),
        "injected_target_pairs": len(correct_prompt),
        "injected_targets_in_final_text": len(correct_prompt & present_hyp),
        "wrong_injected_pairs": len(wrong_prompt),
        "wrong_injected_in_final_text": len(wrong_prompt & present_hyp),
        "negative_samples": int(not expected),
        "negative_with_unreferenced_table_word": int(
            not expected and bool(present_hyp - present_ref)
        ),
    }


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for cell in cells:
        totals.update({k: v for k, v in cell["metrics"].items() if isinstance(v, int)})
    result: dict[str, Any] = {"samples": len(cells), **dict(totals)}
    for name, num, den in (
        ("target_recall", "target_hits", "target_pairs"),
        ("corpus_WER", "word_errors", "reference_words"),
        ("corpus_CER", "character_errors", "reference_characters"),
        ("correct_prompt_adoption", "injected_targets_in_final_text", "injected_target_pairs"),
        ("wrong_prompt_landing", "wrong_injected_in_final_text", "wrong_injected_pairs"),
        ("negative_false_word_rate", "negative_with_unreferenced_table_word", "negative_samples"),
    ):
        result[name] = totals[num] / totals[den] if totals[den] else None
    return result


def summary(cells: list[dict[str, Any]], expected_sample_ids: set[str]) -> dict[str, Any]:
    grouped = {mode: {c["id"]: c for c in cells if c["mode"] == mode} for mode in MODES}
    if len(cells) != 3 * len(expected_sample_ids) or any(
        set(group) != expected_sample_ids for group in grouped.values()
    ):
        raise ValueError("incomplete or duplicate three-way paired results")
    result = {mode: aggregate(list(group.values())) for mode, group in grouped.items()}
    for mode in ("top5", "top7"):
        gained = lost = 0
        for sid in expected_sample_ids:
            base = grouped["no_rag"][sid]["metrics"]
            enhanced = grouped[mode][sid]["metrics"]
            if base["expected_ids"] != enhanced["expected_ids"]:
                raise ValueError("paired gold target denominator differs")
            gained += len(set(enhanced["hit_ids"]) - set(base["hit_ids"]))
            lost += len(set(base["hit_ids"]) - set(enhanced["hit_ids"]))
        result[mode]["paired_vs_no_rag"] = {
            "target_gains": gained,
            "target_losses": lost,
            "net_target_gain": gained - lost,
        }
    return result
