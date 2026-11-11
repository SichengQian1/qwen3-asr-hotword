from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.phonemes.arabic_native import CAPHI, caphi_units, egyptian_units, input_words
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

ROOT = Path(__file__).resolve().parents[1]


def test_native_vocabulary_preserves_english_ids() -> None:
    old = json.loads((ROOT / "configs/phonemes/en_ar_mfa_g2p_vocab.v1.0.json").read_text())
    native = json.loads((ROOT / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json").read_text())
    assert native["tokens"][:72] == old["tokens"][:72]
    assert len(native["tokens"]) == len(set(native["tokens"])) == 167
    assert all(token.startswith("ar_ipa:") for token in native["tokens"][72:])
    loaded = load_phoneme_vocab(ROOT / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json")
    assert loaded.tokenization_policy == "mfa_units_v0.3"
    assert len(loaded.token_to_id) == 167


def test_egyptian_candidate_keeps_short_vowels_length_and_gemination() -> None:
    assert egyptian_units("مِش") == ("m", "i", "ʃ")
    assert egyptian_units("حاجَة") == ("ħ", "aː", "ɡ", "a")
    assert egyptian_units("دِي") == ("d", "iː")
    assert egyptian_units("لِإِنّ") == ("l", "i", "ʔ", "i", "nː")
    assert egyptian_units("سَهْلَة") == ("s", "a", "h", "l", "a")


def test_msa_caphi_mapping_and_unknown_units() -> None:
    assert caphi_units("sh_u_k_r_a_n") == ["ʃ", "u", "k", "r", "a", "n"]
    assert caphi_units("l_a_k_u_m") == ["l", "a", "k", "u", "m"]
    assert caphi_units("m_a_d_d") == ["m", "a", "dː"]
    assert caphi_units("b_a_y_t") == ["b", "a", "j", "t"]
    assert caphi_units("3_a_l_i_y_y") == ["ʕ", "a", "l", "i", "jː"]
    with pytest.raises(ValueError, match="unsupported_CAPHI"):
        caphi_units("sh_unverified_symbol")


def test_every_supported_caphi_phone_and_geminate_fits_native_vocabulary() -> None:
    vocab = json.loads((ROOT / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json").read_text())
    inventory = set(vocab["arabic_inventory"])
    for symbol in CAPHI:
        assert set(caphi_units(symbol)) <= inventory, symbol
        assert set(caphi_units(symbol + "_" + symbol)) <= inventory, symbol


def test_input_rejects_digits_and_mixed_script_without_dropping_them() -> None:
    assert input_words("شكرا لكم.") == ["شكرا", "لكم"]
    for text in ("شكرا 123", "شكرا meeting", "شكرا\u200bلكم", "شكرا%"):
        with pytest.raises(ValueError):
            input_words(text)
