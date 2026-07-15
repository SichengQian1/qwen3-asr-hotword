from pathlib import Path

from qwen_hotword.training.g2p_prep import (
    digit_fragments,
    extract_word_tokens,
    normalize_training_text,
    prepare_mfa_wordlist,
)


def test_portuguese_text_normalization_preserves_pronunciation_spelling() -> None:
    assert normalize_training_text("  VOCÊ  d’água — NÃO  ") == "você d'água - não"
    assert extract_word_tokens("Você d'água, não? bem-vindo!") == [
        "você",
        "d'água",
        "não",
        "bem-vindo",
    ]


def test_digit_fragments_are_reported_without_silent_expansion() -> None:
    assert digit_fragments("Temos 4 modelos e vídeo em 8K.") == ["4", "8k."]


def test_prepare_mfa_wordlist_writes_unique_words_and_counts(tmp_path: Path) -> None:
    tsv_path = tmp_path / "português.tsv"
    tsv_path.write_text(
        "audio\ttext\n"
        "a.wav\tBom dia, você!\n"
        "b.wav\tBom dia 4 vezes.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "g2p"

    summary = prepare_mfa_wordlist(tsv_path, output_dir)

    assert summary.status == "pass"
    assert summary.records_seen == 2
    assert summary.total_word_tokens == 6
    assert summary.unique_words == 4
    assert summary.fragments_with_digits == 1
    assert (output_dir / "words.txt").read_text(encoding="utf-8").splitlines() == [
        "bom",
        "dia",
        "vezes",
        "você",
    ]
    counts = (output_dir / "word_counts.tsv").read_text(encoding="utf-8")
    assert "bom\t2" in counts
    assert "dia\t2" in counts
