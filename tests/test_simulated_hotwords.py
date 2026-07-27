from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.hotwords.simulation import (
    HotwordLengthBucket,
    build_simulated_hotword_assets,
    build_stratified_hotword_assets,
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


def test_build_stratified_v2_is_separate_bucketed_and_covers_all_hotwords(
    tmp_path: Path,
) -> None:
    pronunciations = {
        "alpha": "a l f a",
        "bravo": "b r a v o",
        "claro": "c l a r o",
        "praia": "p r a i a",
        "janela": "j a n e l a",
        "cidade": "c i d a d e",
        "mercado": "m e r c a d o",
        "telefone": "t e l e f o n e",
    }
    manifest = tmp_path / "validation.jsonl"
    rows = []
    index = 0
    for word in pronunciations:
        for _ in range(2):
            rows.append(
                {
                    "id": f"sample-{index:03d}",
                    "split": "validation",
                    "language": "pt-BR",
                    "text": f"uma {word}",
                }
            )
            index += 1
    for _ in range(6):
        rows.append(
            {
                "id": f"sample-{index:03d}",
                "split": "validation",
                "language": "pt-BR",
                "text": "eu uma",
            }
        )
        index += 1
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    dictionary = tmp_path / "mfa.dict"
    dictionary.write_text(
        "".join(
            f"{word}\t{pronunciation}\n"
            for word, pronunciation in pronunciations.items()
        ),
        encoding="utf-8",
    )
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", *list("abcdefghijklmnopqrstuvwxyz")]}),
        encoding="utf-8",
    )
    previous = tmp_path / "simulated_hotwords_v1.jsonl"
    previous.write_text(
        json.dumps(
            {
                "hotword_id": "sim_hw_ptbr_0001",
                "language": "pt-BR",
                "surface": "alpha",
                "normalized": "alpha",
                "words": ["alpha"],
                "pronunciation": "a l f a",
                "phoneme_tokens": ["a", "l", "f", "a"],
                "token_ids": [2, 13, 7, 2],
                "source": "test-v1",
                "validation_occurrences": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "v2"
    summary = build_stratified_hotword_assets(
        manifest,
        dictionary,
        vocab_path,
        output,
        length_buckets=(
            HotwordLengthBucket("phonemes_4_5", 4, 5, 3),
            HotwordLengthBucket("phonemes_6_8", 6, 8, 3),
        ),
        exclude_hotword_table_path=previous,
        case_count=12,
        active_hotwords_per_case=5,
        positive_ratio=0.5,
        seed=23,
    )

    vocab = load_phoneme_vocab(vocab_path)
    hotwords = load_hotword_table(summary.hotword_table_path, vocab=vocab)
    cases = load_simulated_cases(summary.cases_path)
    assert summary.asset_version == "simulated-hotwords-v2-stratified"
    assert summary.selected_hotwords == 6
    assert summary.length_bucket_counts == {
        "phonemes_4_5": 3,
        "phonemes_6_8": 3,
    }
    assert summary.covered_hotwords_in_positive_cases == 6
    assert summary.hotwords_with_multiple_validation_occurrences == 6
    assert "alpha" not in {entry.normalized for entry in hotwords}
    assert {entry.hotword_id for entry in hotwords} == {
        hotword_id
        for case in cases
        for hotword_id in case.expected_hotword_ids
    }
    with pytest.raises(FileExistsError, match="new empty directory"):
        build_stratified_hotword_assets(
            manifest,
            dictionary,
            vocab_path,
            output,
            length_buckets=(
                HotwordLengthBucket("phonemes_4_5", 4, 5, 3),
                HotwordLengthBucket("phonemes_6_8", 6, 8, 3),
            ),
            exclude_hotword_table_path=previous,
            case_count=12,
            active_hotwords_per_case=5,
        )
