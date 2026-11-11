"""Versioned automatic Arabic labels from CALIMA analyses and explicit phone rules."""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

from qwen_hotword.training.g2p_prep import extract_word_tokens

CONSONANTS = dict(
    zip(
        "ءأإؤئبتثجحخدذرزسشصضطظعغفقكلمنهو يپڤچژ".replace(" ", ""),
        (
            "ʔ",
            "ʔ",
            "ʔ",
            "ʔ",
            "ʔ",
            "b",
            "t",
            "θ",
            "dʒ",
            "ħ",
            "x",
            "d",
            "ð",
            "r",
            "z",
            "s",
            "ʃ",
            "sˤ",
            "dˤ",
            "tˤ",
            "ðˤ",
            "ʕ",
            "ɣ",
            "f",
            "q",
            "k",
            "l",
            "m",
            "n",
            "h",
            "w",
            "j",
            "p",
            "v",
            "tʃ",
            "ʒ",
        ),
        strict=True,
    )
)
VOWELS = {"َ": ("a",), "ِ": ("i",), "ُ": ("u",), "ً": ("a", "n"), "ٍ": ("i", "n"), "ٌ": ("u", "n")}
KNOWN_MARKS = set(VOWELS) | {"ّ", "ْ", "ٰ"}
SUN_LETTERS = set("تثدذرزسشصضطظلن")
CAPHI = {
    "2": "ʔ",
    "3": "ʕ",
    "7": "ħ",
    "th": "θ",
    "dh": "ð",
    "sh": "ʃ",
    "kh": "x",
    "gh": "ɣ",
    "j": "dʒ",
    "s.": "sˤ",
    "d.": "dˤ",
    "t.": "tˤ",
    "dh.": "ðˤ",
    "aa": "aː",
    "ii": "iː",
    "uu": "uː",
    # CAPHI y is the palatal glide (Arabic ي), not IPA's front rounded vowel y.
    "y": "j",
    **{c: c for c in "abdfhiklmnqrs tuwvz".replace(" ", "")},
}


def fold_gemination(units: list[str]) -> list[str]:
    result: list[str] = []
    for unit in units:
        if result and unit == result[-1] and unit not in {"a", "i", "u"} and "ː" not in unit:
            result[-1] += "ː"
        else:
            result.append(unit)
    return result


def caphi_units(caphi: str) -> list[str]:
    if not caphi or caphi == "*":
        raise ValueError("no_CAPHI")
    symbols = caphi.split("_")
    missing = sorted(set(symbols) - CAPHI.keys())
    if missing:
        raise ValueError("unsupported_CAPHI:" + ",".join(missing))
    return fold_gemination([CAPHI[unit] for unit in symbols])


@lru_cache(maxsize=100000)
def egyptian_units(diacritized: str) -> tuple[str, ...]:
    """Broad word-local candidate. Cairo j/q rules are explicit experimental defaults."""
    clusters: list[tuple[str, set[str]]] = []
    for char in unicodedata.normalize("NFC", diacritized):
        if char == "ـ":
            continue
        if unicodedata.category(char).startswith("M"):
            if not clusters or char not in KNOWN_MARKS:
                raise ValueError("unsupported_diacritic:" + char)
            clusters[-1][1].add(char)
        else:
            clusters.append((char, set()))
    result: list[str] = []
    for index, (base, marks) in enumerate(clusters):
        previous_marks = clusters[index - 1][1] if index else set()
        if base in {"ا", "ى", "آ", "ٱ"}:
            if previous_marks.intersection({"ً", "ٌ", "ٍ"}):
                continue
            if base == "آ":
                result.extend(("ʔ", "aː"))
            elif index == 0:
                result.append("ʔ")
                vowel = next((v for m, v in VOWELS.items() if m in marks), None)
                result.extend(
                    vowel or (("i",) if len(clusters) > 1 and clusters[1][0] == "ل" else ("a",))
                )
            elif result and result[-1] == "a":
                result[-1] = "aː"
            else:
                result.append("aː")
            continue
        if base in {"و", "ي"} and not marks and result and result[-1] == {"و": "u", "ي": "i"}[base]:
            result[-1] += "ː"
            continue
        if base == "ة":
            unit = "t" if marks.intersection(VOWELS) else ""
        else:
            unit = {"ج": "ɡ", "ق": "ʔ"}.get(base, CONSONANTS.get(base, ""))
            if not unit:
                raise ValueError("unsupported_Arabic_letter:" + base)
        sun = index + 1 < len(clusters) and clusters[index + 1][0] in SUN_LETTERS
        article_lam = (
            base == "ل"
            and index > 0
            and clusters[index - 1][0] in {"ا", "ٱ"}
            and sun
            and "ّ" in clusters[index + 1][1]
        )
        if unit and not article_lam:
            result.append(unit + ("ː" if "ّ" in marks else ""))
        vowels = [VOWELS[mark] for mark in marks.intersection(VOWELS)]
        if len(vowels) > 1:
            raise ValueError("conflicting_vowels")
        if vowels:
            result.extend(vowels[0])
        if "ٰ" in marks:
            if result and result[-1] == "a":
                result[-1] = "aː"
            else:
                result.append("aː")
    if not result:
        raise ValueError("empty_pronunciation")
    return tuple(result)


def input_words(text: str) -> list[str]:
    if any(
        c.isdigit()
        or c in "&@#%*"
        or unicodedata.category(c).startswith(("S", "C"))
        and not c.isspace()
        for c in text
    ):
        raise ValueError("number_symbol_or_control_requires_review")
    words = extract_word_tokens(text)
    if not words or any("ARABIC" not in unicodedata.name(c, "") for word in words for c in word):
        raise ValueError("non_Arabic_or_empty_text")
    return words


class CalimaFrontend:
    def __init__(self, assets: Path, group: str, tokens: list[str]):
        from camel_tools.disambig.mle import MLEDisambiguator  # type: ignore[import-not-found]
        from camel_tools.morphology.analyzer import Analyzer  # type: ignore[import-not-found]
        from camel_tools.morphology.database import MorphologyDB  # type: ignore[import-not-found]

        dialect = "egy" if group == "ar_egy" else "msa"
        name = f"calima-{dialect}-r13"
        database = MorphologyDB(str(assets / "morphology_db" / name / "morphology.db"))
        self.model = MLEDisambiguator(
            Analyzer(database, backoff="NONE"),
            str(assets / "disambig_mle" / name / "model.json"),
            top=1,
        )
        self.group = group
        self.ids = {token: index for index, token in enumerate(tokens)}

    def label(self, text: str) -> dict[str, Any]:
        words = input_words(text)
        analyses = self.model.disambiguate(words)
        if len(analyses) != len(words):
            raise ValueError("analysis_token_count_changed")
        units: list[str] = []
        diacritized: list[str] = []
        for word in analyses:
            if not word.analyses:
                raise ValueError("no_analysis:" + word.word)
            analysis = word.analyses[0].analysis
            if analysis.get("source") not in {"lex", "spvar"}:
                raise ValueError("nonlexical_analysis:" + word.word)
            diacritized.append(analysis["diac"])
            if self.group == "ar_egy":
                units.extend(egyptian_units(analysis["diac"]))
            else:
                units.extend(caphi_units(analysis.get("caphi", "")))
        try:
            token_ids = [self.ids["ar_ipa:" + unit] for unit in units]
        except KeyError as error:
            raise ValueError("phone_OOV:" + str(error)) from error
        return {"words": words, "diacritized_words": diacritized, "phones": units, "ids": token_ids}
