"""Descriptive diagnostics; folds never change training labels or exact PER."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean, median
from typing import Any

from qwen_hotword.training.edit_distance import sequence_edit_distance, sequence_editops

SV = "\u064e\u064f\u0650"
TAN = "\u064b\u064c\u064d"
# Verbatim heuristic definitions from analysis appendix D6. Counts overlap.
RULES: dict[str, dict[str, Any]] = {
    "ar_msa": {
        "wasla_mid (ʔ a)": (lambda w, d, i: i > 0 and w.startswith("ال"), 2),
        "final_short_vowel": (
            lambda w, d, i: re.search("[" + SV + "]$", d.replace("\u0651", "")) is not None,
            1,
        ),
        "final_tanwin": (
            lambda w, d, i: re.search("([" + TAN + "]|\u064bا|\u064bى)$", d) is not None,
            2,
        ),
        "taa_marbuta_t": (lambda w, d, i: re.search("ة[" + SV + TAN + "]$", d) is not None, 1),
        "final_i_y (fake j)": (lambda w, d, i: d.endswith("\u0650ي") and len(w) > 1, 1),
        "final_u_w (fake w)": (lambda w, d, i: d.endswith("\u064fو") and len(w) > 1, 1),
    },
    "ar_egy": {
        "fatha+y/w (e:/o:)": (
            lambda w, d, i: len(re.findall("\u064e[يو](?![\u064b-\u0652اىيو])(?=.)", d)) > 0,
            2,
        ),
        "final_long_vowel": (
            lambda w, d, i: (
                re.search("(\u0650ي|\u064fو|\u064eا|\u064eى)$", d) is not None and len(w) > 1
            ),
            1,
        ),
        "th/dh/zh letters": (lambda w, d, i: any(c in d for c in "ثذظ"), 1),
        "respelled t/d->th/dh": (
            lambda w, d, i: any(
                x in w and y in d and y not in w
                for x, y in (("ت", "ث"), ("د", "ذ"), ("س", "ث"), ("ض", "ظ"), ("ز", "ذ"), ("ز", "ظ"))
            ),
            1,
        ),
        "final_vowel+h": (
            lambda w, d, i: re.search("[" + SV + "]ه$", d) is not None and w not in ("لله", "الله"),
            1,
        ),
        "clitic_underscore": (lambda w, d, i: "_" in d, 1),
        "ashaan": (lambda w, d, i: w in ("عشان", "علشان"), 2),
    },
}


def exposure(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter(utterances=1, words=len(row["words"]), units=len(row["ids"]))
    seen = set()
    for i, (word, diac) in enumerate(zip(row["words"], row["diacritized_words"], strict=True)):
        for name, (predicate, units) in RULES[row["group"]].items():
            if predicate(word, diac, i):
                counts[f"{name}/words"] += 1
                counts[f"{name}/estimated_units"] += units
                seen.add(name)
                if name not in ("respelled t/d->th/dh", "clitic_underscore"):
                    counts["approx_fragile_units_overlapping"] += units
    for name in seen:
        counts[f"{name}/utterances"] += 1
    return counts


def fold_phones(phones: Sequence[str], mode: str) -> list[str]:
    result = []
    for token in phones:
        if not token.startswith("ar_ipa:"):
            result.append(token)
            continue
        phone = token.removeprefix("ar_ipa:")
        vowel = phone.rstrip("ː") in {"a", "i", "u", "e", "o"}
        if (mode in ("F1", "combined") and vowel) or (mode in ("F2", "combined") and not vowel):
            phone = phone.rstrip("ː")
        if mode in ("F3", "combined") and phone == "ʔ":
            continue
        result.append("ar_ipa:" + phone)
    return result


def encode_pair(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ids = {p: i for i, p in enumerate(sorted(set(reference) | set(hypothesis)))}
    return tuple(ids[p] for p in reference), tuple(ids[p] for p in hypothesis)


def decoded_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        per = sorted(r["phoneme_error_rate"] for r in items)
        denominator = sum(len(r["reference"]) for r in items)
        edits: Counter[tuple[str, str | None, str | None]] = Counter()
        folded = {
            mode: Counter(errors=0, reference_tokens=0) for mode in ("F1", "F2", "F3", "combined")
        }
        for row in items:
            ref, hyp = row["reference"], row["hypothesis"]
            for op, a, b in sequence_editops(*encode_pair(ref, hyp)):
                edits[
                    op, None if op == "insert" else ref[a], None if op == "delete" else hyp[b]
                ] += 1
            for mode, values in folded.items():
                r, h = fold_phones(ref, mode), fold_phones(hyp, mode)
                values["errors"] += sequence_edit_distance(*encode_pair(r, h))
                values["reference_tokens"] += len(r)
        return {
            "samples": len(items),
            "reference_tokens": denominator,
            "exact_per": sum(r["errors"] for r in items) / denominator,
            "mean_utterance_per": mean(per),
            "quantiles": {
                f"p{int(q * 100)}": per[min(len(per) - 1, int(q * len(per)))]
                for q in (0.5, 0.9, 0.99)
            },
            "share_per_gt_0.4": sum(v > 0.4 for v in per) / len(per),
            "share_per_gt_0.7": sum(v > 0.7 for v in per) / len(per),
            "top_errors": [
                {"operation": op, "reference": r, "hypothesis": h, "count": n}
                for (op, r, h), n in edits.most_common(30)
            ],
            "folds": {
                mode: {
                    **v,
                    "per": v["errors"] / v["reference_tokens"] if v["reference_tokens"] else None,
                }
                for mode, v in folded.items()
            },
        }

    groups: dict[str, list[Any]] = defaultdict(list)
    for row in samples:
        groups["all"].append(row)
        for name, value in row["groups"].items():
            groups[f"{name}/{value}"].append(row)
        if row.get("duration_seconds", 0) > 0:
            rate = len(row["reference"]) / row["duration_seconds"]
            bucket = next(
                (f"{lo}-{hi}" for lo, hi in ((0, 9), (9, 12), (12, 15)) if lo <= rate < hi), "15+"
            )
            group = row["groups"].get("source_group", "all")
            groups[f"phones_per_second/{group}/{bucket}"].append(row)
    return {name: summarize(items) for name, items in sorted(groups.items())}


def normalize_arabic(text: str) -> str:
    text = re.sub("[ً-ْٰـ]", "", text)
    text = (
        re.sub("[أإآٱ]", "ا", text)
        .replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def agreement_scores(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = normalize_arabic(reference), normalize_arabic(hypothesis)
    result: dict[str, Any] = {}
    for name, a, b in (
        ("cer", list(ref.replace(" ", "")), list(hyp.replace(" ", ""))),
        ("wer", ref.split(), hyp.split()),
    ):
        if not a:
            raise ValueError("empty normalized reference requires review")
        edits = Counter(op for op, _, _ in sequence_editops(*encode_pair(a, b)))
        result[name] = sum(edits.values()) / len(a)
        result[name + "_counts"] = {"errors": sum(edits.values()), "reference": len(a), **edits}
    return result


def agreement_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in items:
        for key in (
            f"{row['split']}/all",
            f"{row['split']}/group/{row['source_group']}",
            f"{row['split']}/source/{row['source']}",
        ):
            grouped[key].append(row)
    return {
        name: {
            "samples": len(part),
            "median_CER": median(r["cer"] for r in part),
            "mean_utterance_WER": mean(r["wer"] for r in part),
            "corpus_WER": sum(r["wer_counts"]["errors"] for r in part)
            / sum(r["wer_counts"]["reference"] for r in part),
            "share_CER_gt_0.3": sum(r["cer"] > 0.3 for r in part) / len(part),
        }
        for name, part in sorted(grouped.items())
    }
