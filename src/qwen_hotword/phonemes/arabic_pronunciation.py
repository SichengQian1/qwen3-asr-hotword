"""Auditable Arabic pronunciation rules; the historical CALIMA frontend stays immutable.

Rules are broad pronunciation conventions, not a substitute for listening. In particular,
word-based MLE still cannot resolve all homographs. Each word retains its original analysis.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from qwen_hotword.phonemes.arabic_native import (
    CAPHI,
    CalimaFrontend,
    caphi_units,
    egyptian_units,
    fold_gemination,
    input_words,
)

RULES = frozenset(
    {
        "egy_clitic_boundary",
        "msa_final_long_vowel",
        "egy_monophthong",
        "egy_restore_surface_consonant",
        "egy_interdentals",
        "egy_final_shortening",
        "egy_pronominal_h",
        "linked_article",
        "msa_pause_endings",
    }
)
VOWELS = frozenset({"a", "i", "u", "e", "o", "aː", "iː", "uː", "eː", "oː"})
PAUSES = frozenset(".?!؟؛;،,\n")


def bare(text: str) -> str:
    return "".join(
        c
        for c in unicodedata.normalize("NFC", text)
        if not unicodedata.category(c).startswith("M") and c not in "ـ_"
    )


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != 1 or set(policy.get("rules", {})) != RULES:
        raise ValueError("pronunciation policy must name every supported rule")
    if any(type(value) is not bool for value in policy["rules"].values()):
        raise ValueError("pronunciation rules must be boolean")
    seen = set()
    for entry in policy.get("exceptions", []):
        key = (entry["group"], entry["surface"], entry.get("diac"))
        if key in seen or entry["group"] not in {"ar_egy", "ar_msa"}:
            raise ValueError("duplicate or invalid pronunciation exception")
        seen.add(key)
        if (
            entry["status"] not in {"candidate", "approved"}
            or not entry.get("evidence")
            or not entry.get("phones")
            or not entry.get("diac")
        ):
            raise ValueError("exceptions require scoped diac, phones, evidence and review status")
        if entry["status"] == "approved" and not entry.get("reviewer"):
            raise ValueError("approved exceptions require reviewer attribution")


def fixed_caphi(value: str, enabled: bool) -> tuple[list[str], list[str]]:
    symbols = value.split("_")
    fired = []
    # Only a single FINAL glide. i_y_a and i_y_y must retain the consonant.
    if enabled and len(symbols) >= 2 and symbols[-2:] in (["i", "y"], ["u", "w"]):
        symbols[-2:] = ["ii" if symbols[-2] == "i" else "uu"]
        fired.append("msa_final_long_vowel")
    missing = sorted(set(symbols) - CAPHI.keys())
    if not value or value == "*":
        raise ValueError("no_CAPHI")
    if missing:
        raise ValueError("unsupported_CAPHI:" + ",".join(missing))
    return fold_gemination([CAPHI[s] for s in symbols]), fired


def restore_surface(diac: str, surface: str) -> tuple[str, bool]:
    original, analyzed = bare(surface), bare(diac)
    allowed = {("ث", "ت"), ("ث", "س"), ("ذ", "د"), ("ذ", "ز"), ("ظ", "ض"), ("ظ", "ز")}
    if len(original) != len(analyzed) or any(
        a != b and (a, b) not in allowed for a, b in zip(analyzed, original, strict=False)
    ):
        return diac, False
    index, result = 0, []
    for char in diac:
        if unicodedata.category(char).startswith("M") or char in "ـ_":
            result.append(char)
        else:
            result.append(original[index])
            index += 1
    return "".join(result), original != analyzed


def fixed_egyptian(
    diac: str, surface: str, analysis: dict[str, Any], rules: dict[str, bool]
) -> tuple[list[str], list[str]]:
    fired = []
    if "_" in diac and rules["egy_clitic_boundary"]:
        diac = diac.replace("_", "")
        fired.append("egy_clitic_boundary")
    if rules["egy_restore_surface_consonant"]:
        diac, restored = restore_surface(diac, surface)
        if restored:
            fired.append("egy_restore_surface_consonant")
    # Use a marked long-vowel spelling for the legacy cluster parser, then restore quality.
    # Explicit sukun, shadda or a vowel on the glide protects true diphthongs/consonants.
    slots: list[str] = []
    if rules["egy_monophthong"]:

        def mono(match: re.Match[str]) -> str:
            slots.append("eː" if match[1] == "ي" else "oː")
            return "ِي" if match[1] == "ي" else "ُو"

        # Track positions through independent cluster chunks; placeholders are not Arabic input.
        parts = re.split(r"(َ[يو](?![\u064b-\u0652\u0670]))", diac)
        units: list[str] = []
        if len(parts) > 1:
            # Identify each replaced vowel through its parsed prefix; then parse the full word.
            current = diac
            matches = list(re.finditer(r"َ([يو])(?![\u064b-\u0652\u0670])", current))
            positions: list[tuple[int, str]] = []
            for match in matches:
                prefix = current[: match.start()] + mono(match)
                position = len(egyptian_units(prefix)) - 1
                positions.append((position, slots[-1]))
            current = re.sub(r"َ([يو])(?![\u064b-\u0652\u0670])", mono, current)
            units = list(egyptian_units(current))
            # Earlier substitutions shorten the old prefix output by one unit each.
            for ordinal, (position, vowel) in enumerate(positions):
                units[position - ordinal] = vowel
            fired.append("egy_monophthong")
        else:
            units = list(egyptian_units(diac))
    else:
        units = list(egyptian_units(diac))
    if rules["egy_interdentals"]:
        mapping = {"θ": "s", "ð": "z", "ðˤ": "zˤ"}
        changed = [
            mapping.get(p.removesuffix("ː"), p.removesuffix("ː")) + ("ː" if p.endswith("ː") else "")
            for p in units
        ]
        if changed != units:
            units = changed
            fired.append("egy_interdentals")
    if (
        rules["egy_pronominal_h"]
        and analysis.get("enc0") in {"3ms_dobj", "3ms_poss"}
        and len(units) > 1
        and units[-1] == "h"
        and units[-2] in {"i", "u"}
    ):
        units.pop()
        fired.append("egy_pronominal_h")
    if rules["egy_final_shortening"] and units and units[-1] in {"aː", "iː", "uː"}:
        units[-1] = units[-1][:-1]
        fired.append("egy_final_shortening")
    return units, fired


def word_pronunciation(
    surface: str,
    analysis: dict[str, Any],
    group: str,
    policy: dict[str, Any],
    *,
    linked: bool = False,
    previous_phone: str | None = None,
    pause_after: bool = False,
    include_alternatives: bool = True,
) -> dict[str, Any]:
    if group not in {"ar_egy", "ar_msa"}:
        raise ValueError("unknown Arabic group")
    rules = policy["rules"]
    diac = analysis["diac"]
    try:
        old = (
            list(egyptian_units(diac))
            if group == "ar_egy"
            else caphi_units(analysis.get("caphi", ""))
        )
        old_issue = None
    except ValueError as error:
        old, old_issue = [], str(error)
    if group == "ar_egy":
        units, fired = fixed_egyptian(diac, surface, analysis, rules)
    else:
        units, fired = fixed_caphi(analysis.get("caphi", ""), rules["msa_final_long_vowel"])
    applied_exception = None
    candidates = []
    for entry in policy.get("exceptions", []):
        if entry["group"] == group and entry["surface"] == bare(surface) and entry["diac"] == diac:
            candidates.append(entry)
            if entry["status"] == "approved":
                units = list(entry["phones"])
                applied_exception = entry
                fired.append("approved_exception")
    # This optional rule handles definite-article wasl only, never arbitrary hamza.
    if (
        rules["linked_article"]
        and linked
        and bare(diac).startswith(("ال", "ٱل"))
        and units[:2] in (["ʔ", "a"], ["ʔ", "i"])
    ):
        units.pop(0)
        if previous_phone in VOWELS:
            units.pop(0)
        fired.append("linked_article")
    # Pausal endings are distinct from deleting all final vowels. Protect construct state,
    # attached pronouns and verbs. Accusative tanwin outside ta marbuta becomes long a.
    if (
        rules["msa_pause_endings"]
        and group == "ar_msa"
        and pause_after
        and analysis.get("pos", "").startswith(("noun", "adj"))
        and analysis.get("cas") in {"n", "a", "g"}
        and analysis.get("stt") != "c"
        and analysis.get("enc0") in {"0", "na"}
    ):
        end_ta = bare(diac).endswith("ة")
        if len(units) >= 2 and units[-1] == "n" and units[-2] in {"a", "i", "u"}:
            vowel = units[-2]
            units = units[:-2]
            if vowel == "a" and not end_ta:
                units.append("aː")
        elif units and units[-1] in {"a", "i", "u"}:
            units.pop()
        if end_ta and units and units[-1] == "t":
            units.pop()
        fired.append("msa_pause_endings")
    if not units:
        raise ValueError("empty_pronunciation")
    relevant = {
        k: analysis[k]
        for k in (
            "diac",
            "caphi",
            "lex",
            "pos",
            "source",
            "cas",
            "stt",
            "enc0",
            "prc0",
            "prc1",
            "prc2",
            "prc3",
            "stem",
        )
        if k in analysis
    }
    alternatives = []
    if include_alternatives:
        for rule in ("linked_article", "msa_pause_endings"):
            if rules[rule]:
                continue
            if rule == "linked_article" and not linked:
                continue
            if rule == "msa_pause_endings" and (group != "ar_msa" or not pause_after):
                continue
            try:
                alternative = word_pronunciation(
                    surface,
                    analysis,
                    group,
                    {**policy, "rules": {**rules, rule: True}},
                    linked=linked,
                    previous_phone=previous_phone,
                    pause_after=pause_after,
                    include_alternatives=False,
                )
            except ValueError as error:
                alternatives.append(
                    {"rule": rule, "error": str(error), "status": "invalid_candidate_not_active"}
                )
                continue
            if alternative["phones"] != units:
                alternatives.append(
                    {
                        "rule": rule,
                        "phones": alternative["phones"],
                        "status": "listening_required_not_active",
                    }
                )
    return {
        "surface": surface,
        "analysis": relevant,
        "baseline_phones": old,
        "baseline_issue": old_issue,
        "phones": units,
        "rules": fired,
        "exception": applied_exception,
        "exception_candidates": candidates,
        "linked": linked,
        "pause_after": pause_after,
        "rule_alternatives": alternatives,
    }


class PronunciationFrontend(CalimaFrontend):
    """One implementation for sentence training labels and downstream queries."""

    def __init__(self, assets: Any, group: str, tokens: list[str], policy: dict[str, Any]):
        validate_policy(policy)
        super().__init__(assets, group, tokens)
        self.policy = policy

    def label(self, text: str, *, mode: str = "sentence") -> dict[str, Any]:
        if mode not in {"sentence", "isolated", "linked"}:
            raise ValueError("mode must be sentence, isolated or linked")
        words = input_words(text)
        normalized = unicodedata.normalize("NFC", text).casefold()
        cursor, gaps = 0, []
        for word in words:
            start = normalized.find(word, cursor)
            if start < 0:
                raise ValueError("cannot_preserve_text_boundaries")
            gaps.append(normalized[cursor:start])
            cursor = start + len(word)
        gaps.append(normalized[cursor:])
        analyses = self.model.disambiguate(words)
        if len(analyses) != len(words):
            raise ValueError("analysis_token_count_changed")
        traces: list[dict[str, Any]] = []
        units: list[str] = []
        for index, (surface, analyzed) in enumerate(zip(words, analyses, strict=True)):
            if not analyzed.analyses:
                raise ValueError("no_analysis:" + surface)
            analysis = analyzed.analyses[0].analysis
            if analysis.get("source") not in {"lex", "spvar"}:
                raise ValueError("nonlexical_analysis:" + surface)
            linked = (index > 0 and not set(gaps[index]) & PAUSES) or (
                index == 0 and mode == "linked"
            )
            trace = word_pronunciation(
                surface,
                analysis,
                self.group,
                self.policy,
                linked=linked,
                previous_phone=units[-1] if units else None,
                pause_after=bool(set(gaps[index + 1]) & PAUSES) or index == len(words) - 1,
            )
            trace["phone_start"] = len(units)
            units.extend(trace["phones"])
            trace["phone_end"] = len(units)
            traces.append(trace)
        try:
            ids = [self.ids["ar_ipa:" + unit] for unit in units]
        except KeyError as error:
            raise ValueError("phone_OOV:" + str(error)) from error
        return {
            "words": words,
            "diacritized_words": [t["analysis"]["diac"] for t in traces],
            "phones": units,
            "ids": ids,
            "word_traces": traces,
            "pronunciation_policy": self.policy["name"],
            "mode": mode,
            "context_disambiguation": "word_based_MLE_Top1",
            "human_pronunciation_review_completed": False,
            "unreviewed_exception_words": [
                t["surface"] for t in traces if t["exception_candidates"] and not t["exception"]
            ],
        }
