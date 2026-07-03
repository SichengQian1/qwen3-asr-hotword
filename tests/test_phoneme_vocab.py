from __future__ import annotations

import json
from pathlib import Path

VOCAB_PATH = Path("configs/phonemes/en_ptbr_phoneme_vocab.v0.1.json")
PRECISION_VOCAB_PATH = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")


def test_initial_phoneme_vocab_ids_are_stable() -> None:
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    tokens = vocab["tokens"]

    assert tokens[0] == "<blank>"
    assert tokens[1] == "<unk>"
    assert len(tokens) == 81
    assert len(tokens) == len(set(tokens))

    english_tokens = [token for token in tokens if token.startswith("EN_")]
    portuguese_tokens = [token for token in tokens if token.startswith("PT_")]
    assert len(english_tokens) == 39
    assert len(portuguese_tokens) == 40


def test_initial_phoneme_vocab_contains_representative_hotword_phones() -> None:
    tokens = set(json.loads(VOCAB_PATH.read_text(encoding="utf-8"))["tokens"])

    assert {"EN_K", "EN_UW", "EN_B", "EN_ER", "EN_N"}.issubset(tokens)
    assert {"PT_ɐ̃", "PT_tʃ", "PT_dʒ", "PT_ʃ", "PT_ʒ"}.issubset(tokens)


def test_precision_ipa_vocab_ids_are_stable() -> None:
    vocab = json.loads(PRECISION_VOCAB_PATH.read_text(encoding="utf-8"))
    tokens = vocab["tokens"]

    assert tokens[0] == "<blank>"
    assert tokens[1] == "<unk>"
    assert len(tokens) == vocab["metadata"]["ctc_output_classes"] == 90
    assert len(tokens) == len(set(tokens))
    assert not any(token.startswith(("EN_", "ES_", "PT_")) for token in tokens)


def test_precision_ipa_vocab_is_union_of_source_inventories() -> None:
    vocab = json.loads(PRECISION_VOCAB_PATH.read_text(encoding="utf-8"))
    source_tokens = {
        phone
        for inventory in vocab["source_inventories"].values()
        for phone in inventory["phones"]
    }

    assert set(vocab["tokens"][2:]) == source_tokens
    assert set(vocab["source_inventories"]) == {"en-US", "es-419", "pt-BR"}


def test_precision_ipa_vocab_keeps_near_pronunciation_contrasts() -> None:
    tokens = set(json.loads(PRECISION_VOCAB_PATH.read_text(encoding="utf-8"))["tokens"])

    assert {"i", "ɪ", "u", "ʊ", "e", "ɛ", "o", "ɔ"}.issubset(tokens)
    assert {"r", "ɾ", "t", "t̪", "d", "d̪"}.issubset(tokens)
    assert {"ʃ", "ʒ", "tʃ", "dʒ", "θ", "ð"}.issubset(tokens)
    assert {"ɐ̃", "ẽ", "ĩ", "õ", "ũ", "j̃", "w̃"}.issubset(tokens)
    assert {"β", "ɣ", "x", "ɲ", "ʎ"}.issubset(tokens)
