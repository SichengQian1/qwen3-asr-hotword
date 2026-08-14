from pathlib import Path

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab

VOCAB_PATH = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")


def test_modifier_letter_small_w_maps_to_shared_w_phone() -> None:
    vocab = load_phoneme_vocab(VOCAB_PATH)

    result = tokenize_ipa_to_vocab("kʷ tʷ ɟʷ", vocab)

    assert result.normalized_ipa == "kw tw ɟw"
    assert result.tokens == ["k", "w", "t", "w", "ɟ", "w"]
    assert result.oov_units == []
