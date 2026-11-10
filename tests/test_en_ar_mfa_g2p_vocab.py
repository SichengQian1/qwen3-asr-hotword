from __future__ import annotations

import csv
import json
from pathlib import Path

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab

PHONEMES = Path(__file__).resolve().parents[1] / "configs" / "phonemes"
VOCAB = PHONEMES / "en_ar_mfa_g2p_vocab.v1.0.json"
PARENT_VOCAB = PHONEMES / "zh_en_ar_mfa_g2p_vocab.v1.1.json"


def read_mapping(filename: str) -> list[dict[str, str]]:
    with (PHONEMES / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_bilingual_vocabulary_keeps_the_verified_units_with_compact_ids() -> None:
    spec = json.loads(VOCAB.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_VOCAB.read_text(encoding="utf-8"))
    vocab = load_phoneme_vocab(VOCAB)
    assert set(spec["models"]) == set(spec["languages"]) == {"en", "ar"}
    english = tuple(spec["languages"]["en"]["inventory"])
    arabic = tuple(spec["languages"]["ar"]["inventory"])
    assert len(english) == 70 and len(arabic) == 42
    assert vocab.tokens == ("<blank>", "<unk>", *english, *arabic)
    assert spec["statistics"]["classes"] == len(vocab.tokens) == 114
    assert spec["statistics"]["output_units_by_language"] == {"en": 70, "ar": 42}
    assert all(unit.startswith("ar_gp:") for unit in arabic)

    for language in ("en", "ar"):
        assert spec["models"][language] == parent["models"][language]
        assert spec["languages"][language] == parent["languages"][language]

    original = [row for row in read_mapping(parent["mapping_file"]) if row["language"] != "zh"]
    compact = read_mapping(spec["mapping_file"])
    for old, new in zip(original, compact, strict=True):
        unit = old["shared_unit"]
        label = str(vocab.token_to_id[unit]) if unit else ""
        assert new == {**old, "token_id": label}
        if unit:
            result = tokenize_ipa_to_vocab(unit, vocab)
            assert result.token_ids == [int(label)] and not result.oov_units


def test_native_mfa_examples_and_invalid_targets() -> None:
    vocab = load_phoneme_vocab(VOCAB)
    assert tokenize_ipa_to_vocab("cʰ æ t", vocab).token_ids == [7, 43, 34]
    assert tokenize_ipa_to_vocab("ar_gp:s ar_gp:a ar_gp:l ar_gp:a ar_gp:m", vocab).token_ids == [
        105,
        81,
        95,
        81,
        97,
    ]
    assert vocab.token_to_id["ar_gp:q"] != vocab.token_to_id["ar_gp:Q"]
    for invalid in ("i˨˩˦", "<unk>", "<epsilon>", "<blank>", "ar_gp:<unk>"):
        result = tokenize_ipa_to_vocab(f"m {invalid}", vocab)
        assert not result.token_ids and result.oov_units == [invalid]
