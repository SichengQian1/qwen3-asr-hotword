from __future__ import annotations

import csv
import json
from pathlib import Path

from qwen_hotword.training.mfa_audit import audit_mfa_dictionary


def test_audit_mfa_dictionary_reports_missing_and_oov(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    dictionary = tmp_path / "g2p.dict"
    vocab = tmp_path / "vocab.json"
    counts = tmp_path / "word_counts.tsv"
    output_dir = tmp_path / "audit"
    words.write_text("bom\ncafé\nxpto\n", encoding="utf-8")
    dictionary.write_text("bom\tb o\ncafé\tk a f ɛ\n", encoding="utf-8")
    vocab.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", "b", "o", "k", "a", "f"]}),
        encoding="utf-8",
    )
    with counts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["word", "count"])
        writer.writerow(["bom", 10])
        writer.writerow(["café", 3])
        writer.writerow(["xpto", 2])

    summary = audit_mfa_dictionary(
        words,
        dictionary,
        vocab,
        output_dir,
        word_counts_path=counts,
    )

    assert summary.input_unique_words == 3
    assert summary.dictionary_unique_words == 2
    assert summary.missing_words == 1
    assert summary.dictionary_word_coverage == 2 / 3
    assert summary.corpus_token_coverage == 13 / 15
    assert summary.words_with_oov_phones == 1
    assert summary.corpus_weighted_oov_phone_units == 3
    assert summary.training_labels_ready is False
    assert "xpto\t2" in (output_dir / "missing_words.tsv").read_text(encoding="utf-8")
    assert "ɛ\t1\t3" in (output_dir / "oov_phone_counts.tsv").read_text(encoding="utf-8")


def test_audit_mfa_dictionary_accepts_complete_in_vocab_output(tmp_path: Path) -> None:
    words = tmp_path / "words.txt"
    dictionary = tmp_path / "g2p.dict"
    vocab = tmp_path / "vocab.json"
    words.write_text("bom\n", encoding="utf-8")
    dictionary.write_text("bom b o\n", encoding="utf-8")
    vocab.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", "b", "o"]}),
        encoding="utf-8",
    )

    summary = audit_mfa_dictionary(words, dictionary, vocab, tmp_path / "audit")

    assert summary.dictionary_word_coverage == 1.0
    assert summary.corpus_token_coverage is None
    assert summary.training_labels_ready is True
