from __future__ import annotations

import json
from pathlib import Path

from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.hotwords.simulation import (
    build_simulated_hotword_assets,
    load_simulated_cases,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab


def test_build_simulated_hotwords_uses_only_validation_and_is_deterministic(
    tmp_path: Path,
) -> None:
    phrases = [
        "cidade bonita",
        "mercado central",
        "janela aberta",
        "cachorro feliz",
        "telefone antigo",
        "computador moderno",
        "viagem tranquila",
        "montanha verde",
        "cozinha pequena",
        "jardim florido",
        "praia distante",
        "trabalho remoto",
    ]
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"sample-{index:03d}",
                    "split": "validation",
                    "language": "pt-BR",
                    "text": phrase,
                },
                ensure_ascii=False,
            )
            + "\n"
            for index, phrase in enumerate(phrases)
        ),
        encoding="utf-8",
    )
    words = sorted({word for phrase in phrases for word in phrase.split()})
    dictionary = tmp_path / "mfa.dict"
    dictionary.write_text(
        "".join(f"{word}\t{' '.join(word)}\n" for word in words),
        encoding="utf-8",
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", *list("abcdefghijklmnopqrstuvwxyz")]}),
        encoding="utf-8",
    )

    first = build_simulated_hotword_assets(
        manifest,
        dictionary,
        vocab,
        tmp_path / "first",
        hotword_count=6,
        case_count=8,
        active_hotwords_per_case=4,
        positive_ratio=0.5,
        max_phonemes=40,
        seed=17,
    )
    second = build_simulated_hotword_assets(
        manifest,
        dictionary,
        vocab,
        tmp_path / "second",
        hotword_count=6,
        case_count=8,
        active_hotwords_per_case=4,
        positive_ratio=0.5,
        max_phonemes=40,
        seed=17,
    )

    first_table = Path(first.hotword_table_path)
    second_table = Path(second.hotword_table_path)
    assert first_table.read_text(encoding="utf-8") == second_table.read_text(
        encoding="utf-8"
    )
    assert first.test_set_used is False
    assert first.selected_hotwords == 6
    assert first.positive_cases > 0
    assert first.negative_cases > 0
    loaded_vocab = load_phoneme_vocab(vocab)
    hotwords = load_hotword_table(first_table, vocab=loaded_vocab)
    cases = load_simulated_cases(first.cases_path)
    known_ids = {entry.hotword_id for entry in hotwords}
    assert len(hotwords) == 6
    assert all(set(case.active_hotword_ids) <= known_ids for case in cases)
    assert all(
        set(case.expected_hotword_ids) <= set(case.active_hotword_ids) for case in cases
    )
