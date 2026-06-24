from __future__ import annotations

import json
from pathlib import Path

VOCAB_PATH = Path("configs/phonemes/en_ptbr_phoneme_vocab.v0.1.json")


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
