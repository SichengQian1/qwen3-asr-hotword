from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from qwen_hotword.phonemes.coverage import (
    load_phoneme_vocab,
    normalization_key,
    tokenize_ipa_to_vocab,
)

ROOT = Path(__file__).resolve().parents[1] / "configs" / "phonemes"
VOCAB = ROOT / "zh_en_ar_ipa_vocab.v1.0.json"
MAPPING = ROOT / "zh_en_ar_ipa_mapping.v1.0.tsv"


def test_published_mapping_covers_all_sources_and_matches_the_tokenizer() -> None:
    spec = json.loads(VOCAB.read_text(encoding="utf-8"))
    vocab = load_phoneme_vocab(VOCAB)
    policy = spec["normalization_policy"]
    assert len(vocab.tokens) == spec["statistics"]["classes"] == 248
    assert vocab.tokens[:2] == ("<blank>", "<unk>")
    assert list(vocab.tokens[2:]) == sorted({normalization_key(p) for p in vocab.tokens[2:]})

    expected = {
        (name, unit)
        for name, source in spec["sources"].items()
        for unit in source["source_units"]
    }
    observed: set[tuple[str, str]] = set()
    units_by_language: dict[str, set[str]] = {lang: set() for lang in ("zh", "en", "ar")}
    with MAPPING.open(encoding="utf-8", newline="") as mapping:
        for row in csv.DictReader(mapping, delimiter="\t"):
            name, unit = row["source"], row["source_unit"]
            assert (name, unit) in expected - observed
            observed.add((name, unit))
            language = row["language"]
            assert language == spec["sources"][name]["language"]
            parsed = tokenize_ipa_to_vocab(unit, vocab)
            if unit in policy["source_only_symbols"]:
                assert row["shared_ipa_unit"] == row["token_id"] == ""
                assert row["disposition"] == policy["source_only_symbols"][unit]
                assert parsed.token_ids == [] and parsed.oov_units == [unit]
                continue
            canonical = normalization_key(policy["exact_aliases"].get(unit, unit))
            assert row["shared_ipa_unit"] == canonical
            assert parsed.oov_units == []
            assert parsed.token_ids == [int(row["token_id"])]
            assert parsed.tokens == [canonical]
            units_by_language[language].add(canonical)

    assert observed == expected
    assert set.union(*units_by_language.values()) == set(vocab.tokens[2:])
    assert {lang: len(units) for lang, units in units_by_language.items()} == {
        "zh": 142,
        "en": 78,
        "ar": 95,
    }
    for language, units in units_by_language.items():
        assert units == set(spec["languages"][language]["inventory"])
    assert len(set.intersection(*units_by_language.values())) == 13


def test_standard_arabic_preserves_consonant_and_vowel_length() -> None:
    spec = json.loads(VOCAB.read_text(encoding="utf-8"))
    ar = set(spec["languages"]["ar"]["inventory"])
    core = spec["sources"]["ar_msa_reference"]
    assert len(set(core["consonants"])) == 28
    assert len(set(core["vowels"])) == 6
    assert {c + "ː" for c in core["consonants"]} <= ar
    assert {"a", "aː", "i", "iː", "u", "uː", "ɪ", "ʊ", "x", "χ", "s", "sˤ"} <= ar


def test_cairene_profile_keeps_dialect_phones_and_their_provenance() -> None:
    spec = json.loads(VOCAB.read_text(encoding="utf-8"))
    profiles = spec["languages"]["ar"]["profiles"]
    egyptian = {
        p
        for source in profiles["egyptian_cairene"]
        for p in spec["sources"][source]["source_units"]
    }
    assert len(egyptian) == 79
    assert {"ɡ", "ʔ", "q", "eː", "oː", "zˤ", "zˤː", "rˤ", "ɾ", "pː", "vː"} <= egyptian
    assert not {"ar_mfa_historical_dictionary", "ar_msa_reference"} & set(
        profiles["egyptian_cairene"]
    )


@pytest.mark.parametrize(
    "sequence",
    [
        "a aː a˥ a˨˩˦",
        "k kʰ kʷ",
        "r rˤ ɾ ɹ ɻ",
        "h ħ ʔ ʕ",
        "b bː",
        "x xː χ χː",
        "ɡ dʒ ɣ",
        "z zˤ zˤː ðˤ",
        "e eː o oː ɔ ɔː",
    ],
)
def test_close_phones_keep_distinct_ids(sequence: str) -> None:
    result = tokenize_ipa_to_vocab(sequence, load_phoneme_vocab(VOCAB))
    assert result.oov_units == []
    assert len(set(result.token_ids)) == len(sequence.split())


def test_explicit_ipa_aliases_and_unknown_units() -> None:
    vocab = load_phoneme_vocab(VOCAB)
    assert tokenize_ipa_to_vocab("g d͡ʒ t͜ʃ", vocab).tokens == ["ɡ", "dʒ", "tʃ"]
    for unit in ("<eps>", "<unk>", "<blank>", "ʷ", "ˈ", "不存在"):
        result = tokenize_ipa_to_vocab(f"m {unit} a", vocab)
        assert result.oov_units == [unit]
        assert result.token_ids == []
