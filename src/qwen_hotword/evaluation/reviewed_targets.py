"""Import a deletion-only human review without rewriting its original checksum seal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import LANGUAGES, _checked, combine_variants
from qwen_hotword.evaluation.wave_keywords import _json, _key
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.external_keyword_retrieval import KeywordBundle
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.phonemes.coverage import PhonemeVocab, tokenize_ipa_to_vocab
from qwen_hotword.training.spanish_capacity import _verified


def import_review(
    table: Path, vocab: PhonemeVocab, identities: dict[str, Any]
) -> tuple[dict[str, KeywordBundle], dict[str, Any], dict[str, Any]]:
    """The edited table is new input; immutable decisions prove each retained variant."""
    _checked(table, identities)
    _checked(table.parent / "sha256.txt", identities)
    original_report = _json(_verified(table.parent, "report.json", identities))
    decision_path = _verified(table.parent, "all_target_decisions.jsonl", identities)
    decisions = [
        json.loads(line) for line in decision_path.read_text().splitlines() if line.strip()
    ]
    if original_report.get("scope") != "multilingual_declared_targets_review_only":
        raise ValueError("expected original target-review provenance")
    original = {entry["id"]: entry for entry in combine_variants(decisions)}
    reviewed = _json(table)
    if (
        reviewed.get("schema_version") != 1
        or reviewed.get("scope") != original_report["scope"]
        or not isinstance(reviewed.get("entries"), list)
        or not reviewed["entries"]
    ):
        raise ValueError("reviewed table requires nonempty entries in the original review schema")
    seen: set[str] = set()
    variants: dict[str, list[Any]] = {lang: [] for lang in LANGUAGES}
    targets: dict[str, dict[str, list[str]]] = {lang: {} for lang in LANGUAGES}
    for entry in reviewed["entries"]:
        identifier = entry.get("id")
        if identifier in seen or identifier not in original or entry != original[identifier]:
            raise ValueError(
                "review must only delete whole entries; duplicate/new/edited entry: "
                f"{identifier}. Preserve each retained entry's original language/phonemes."
            )
        seen.add(identifier)
        for row in entry["variants"]:
            lang, surface = row["language"], row["surface"]
            tokenized = tokenize_ipa_to_vocab(row["pronunciation"], vocab)
            if (
                lang not in LANGUAGES
                or row["normalized"] != _key(surface)
                or tokenized.oov_units
                or not tokenized.token_ids
                or tokenized.token_ids != row["token_ids"]
                or tokenized.tokens != row["phoneme_tokens"]
            ):
                raise ValueError(f"invalid retained phoneme variant: {identifier}/{lang}")
            hotword_id = f"review_{lang}_{identifier}"
            variants[lang].append(
                HotwordEntry(
                    hotword_id=hotword_id,
                    language={"en": "en", "es": "es", "pt": "pt-BR"}[lang],
                    surface=surface,
                    normalized=row["normalized"],
                    words=normalize_match_words(surface),
                    pronunciation=row["pronunciation"],
                    phoneme_tokens=tuple(row["phoneme_tokens"]),
                    token_ids=tuple(row["token_ids"]),
                    source=str(table),
                    validation_occurrences=1,
                )
            )
            for group in row["source_groups"]:
                targets[lang].setdefault(group, []).append(hotword_id)
    bundles = {}
    for lang, entries in variants.items():
        if not entries:
            raise ValueError(f"review contains no {lang} entries; cannot run that language")
        entries.sort(key=lambda e: e.normalized)
        if len({e.normalized for e in entries}) != len(entries):
            raise ValueError(f"duplicate normalized retained words: {lang}")
        bundles[lang] = KeywordBundle(
            tuple(entries), {e.surface: e.pronunciation for e in entries}, {}
        )
    audit = {
        "reviewed_table_sha256": identities[str(table)]["sha256"],
        "original_generated_table_sha256": _original_table_sha(table),
        "original_unique_surfaces": len(original),
        "reviewed_unique_surfaces": len(seen),
        "manually_deleted_unique_surfaces": len(original) - len(seen),
        "keywords_by_language": {lang: len(b.entries) for lang, b in bundles.items()},
        "filler_entries_added": 0,
        "original_checksum_manifest_rewritten": False,
        "review_policy": "user_deleted_whole_entries; surviving_variants_unchanged",
    }
    return bundles, targets, audit


def _original_table_sha(table: Path) -> str:
    for line in (table.parent / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if name.strip().removeprefix("*").removeprefix("./") == table.name:
            return digest
    raise ValueError("original table identity missing from review provenance")
