from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab

ROOT = Path(__file__).resolve().parents[1] / "configs" / "phonemes"
VOCAB = ROOT / "zh_en_ar_mfa_g2p_vocab.v1.1.json"
MAPPING = ROOT / "zh_en_ar_mfa_g2p_mapping.v1.1.tsv"


def test_mapping_reconstructs_each_official_symbol_table() -> None:
    spec = json.loads(VOCAB.read_text(encoding="utf-8"))
    vocab = load_phoneme_vocab(VOCAB)
    assert vocab.tokens[:2] == ("<blank>", "<unk>")
    assert len(vocab.tokens) == spec["statistics"]["classes"] == 219
    specials = spec["normalization_policy"]["source_only_symbols"]
    with MAPPING.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    inventories = {}
    for lang, count in (("zh", 134), ("en", 70), ("ar", 42)):
        source_rows = [row for row in rows if row["language"] == lang]
        assert len({row["source_unit"] for row in source_rows}) == len(source_rows)
        assert [int(row["fst_symbol_id"]) for row in source_rows] == list(range(len(source_rows)))
        external = sorted(
            (int(row["external_symbol_id"]), row["source_unit"])
            for row in source_rows
            if row["external_symbol_id"] != ""
        )
        source_table = "".join(f"{unit}\t{symbol_id}\n" for symbol_id, unit in external)
        assert (
            hashlib.sha256(source_table.encode()).hexdigest()
            == (spec["models"][lang]["files_sha256"]["phones.sym"])
        )
        embedded_only = [
            row["source_unit"] for row in source_rows if row["external_symbol_id"] == ""
        ]
        assert embedded_only == (["m̩"] if lang == "en" else [])
        assert spec["models"][lang]["embedded_only_output_symbols"] == embedded_only
        if lang == "en":
            embedded = "".join(f"{r['source_unit']}\t{r['fst_symbol_id']}\n" for r in source_rows)
            assert (
                hashlib.sha256(embedded.encode()).hexdigest()
                == (spec["models"][lang]["files_sha256"]["phones.txt"])
            )
        inventory = set()
        for row in source_rows:
            raw, unit, label = row["source_unit"], row["shared_unit"], row["token_id"]
            if raw in specials:
                assert unit == label == ""
                assert row["disposition"] == specials[raw]
                continue
            model_unit = f"ar_gp:{raw}" if lang == "ar" else raw
            result = tokenize_ipa_to_vocab(model_unit, vocab)
            assert result.tokens == [unit] and result.token_ids == [int(label)]
            assert not result.oov_units
            inventory.add(unit)
        assert len(inventory) == spec["models"][lang]["output_units"] == count
        assert spec["models"][lang]["arc_output_units"] == count
        assert inventory == set(spec["languages"][lang]["inventory"])
        inventories[lang] = inventory

    assert len(inventories["zh"] & inventories["en"]) == 29
    assert not inventories["ar"] & (inventories["zh"] | inventories["en"])
    assert set(vocab.tokens[2:]) == set.union(*inventories.values())


def test_arabic_source_labels_are_atomic_case_sensitive_and_namespaced() -> None:
    vocab = load_phoneme_vocab(VOCAB)
    result = tokenize_ipa_to_vocab("ar_gp:a ar_gp:alal ar_gp:q ar_gp:Q", vocab)
    assert result.tokens == ["ar_gp:a", "ar_gp:alal", "ar_gp:q", "ar_gp:Q"]
    assert len(set(result.token_ids)) == 4
    assert vocab.token_to_id["a"] != vocab.token_to_id["ar_gp:a"]


@pytest.mark.parametrize("unit", ["<eps>", "<epsilon>", "<unk>", "<blank>", "ar_gp:<unk>", "alal"])
def test_source_specials_and_unmapped_labels_do_not_enter_ctc_targets(unit: str) -> None:
    result = tokenize_ipa_to_vocab(f"m {unit}", load_phoneme_vocab(VOCAB))
    assert result.oov_units == [unit]
    assert result.token_ids == []
