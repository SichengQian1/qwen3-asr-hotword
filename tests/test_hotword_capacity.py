from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.capacity_assets import build_hotword_capacity_assets
from qwen_hotword.hotwords.capacity_benchmark import benchmark_hotword_capacity
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table, write_hotword_table
from qwen_hotword.hotwords.scoring import (
    DecodedPhoneme,
    HotwordScoringConfig,
    profile_decoded_hotwords,
    score_decoded_hotwords,
    score_hotwords,
)
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab


def _letters(index: int) -> str:
    value = index
    output = []
    for _ in range(4):
        output.append(chr(ord("a") + value % 26))
        value //= 26
    return "".join(reversed(output))


def _tokens(index: int, *, width: int) -> tuple[int, ...]:
    return tuple(((index // (6**position)) % 6) + 1 for position in range(width))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _capacity_inputs(tmp_path: Path) -> dict[str, Path]:
    tokens = ("<blank>", "a", "b", "c", "d", "e", "f")
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"tokens": list(tokens)}), encoding="utf-8")
    base_entries = []
    for index in range(100):
        token_ids = _tokens(index + 2_000, width=5)
        surface = f"base{_letters(index)}"
        base_entries.append(
            HotwordEntry(
                hotword_id=f"h{index:03d}",
                language="pt-BR",
                surface=surface,
                normalized=surface,
                words=(surface,),
                pronunciation=" ".join(tokens[item] for item in token_ids),
                phoneme_tokens=tuple(tokens[item] for item in token_ids),
                token_ids=token_ids,
                source="test_base",
                validation_occurrences=1,
            )
        )
    hotwords_path = tmp_path / "base_hotwords.jsonl"
    write_hotword_table(hotwords_path, base_entries)
    active = [entry.hotword_id for entry in base_entries]
    cases_path = tmp_path / "base_cases.jsonl"
    _write_jsonl(
        cases_path,
        [
            {
                "case_id": "positive",
                "sample_id": "sample-positive",
                "reference_text": base_entries[0].surface,
                "normalized_reference_text": base_entries[0].surface,
                "language": "pt-BR",
                "primary_group": "single_hotword",
                "expected_hotword_ids": [base_entries[0].hotword_id],
                "active_hotword_ids": active,
            },
            {
                "case_id": "negative",
                "sample_id": "sample-negative",
                "reference_text": "frase sem alvo",
                "normalized_reference_text": "frase sem alvo",
                "language": "pt-BR",
                "primary_group": "negative",
                "expected_hotword_ids": [],
                "active_hotword_ids": active,
            },
        ],
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selection_profile": "formal100",
                "test_set_used": False,
                "samples": [
                    {
                        "case_id": "positive",
                        "sample_id": "sample-positive",
                        "expected_hotword_ids": [base_entries[1].hotword_id],
                        "expected_surfaces": [base_entries[1].surface],
                    },
                    {
                        "case_id": "negative",
                        "sample_id": "sample-negative",
                        "expected_hotword_ids": [],
                        "expected_surfaces": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    train_rows: list[dict[str, object]] = []
    dictionary_lines: list[str] = []
    word_index = 0
    for row_index in range(80):
        words = []
        for _ in range(4):
            word = f"palavra{_letters(word_index + 500)}"
            token_ids = _tokens(word_index, width=4)
            dictionary_lines.append(f"{word} {' '.join(tokens[item] for item in token_ids)}\n")
            words.append(word)
            word_index += 1
        train_rows.append(
            {
                "id": f"train-{row_index}",
                "split": "train",
                "text": " ".join(words),
            }
        )
    train_path = tmp_path / "train.jsonl"
    dictionary_path = tmp_path / "dictionary.dict"
    _write_jsonl(train_path, train_rows)
    dictionary_path.write_text("".join(dictionary_lines), encoding="utf-8")
    return {
        "vocab": vocab_path,
        "base_hotwords": hotwords_path,
        "base_cases": cases_path,
        "selection": selection_path,
        "train": train_path,
        "dictionary": dictionary_path,
    }


def test_decoded_scoring_is_equivalent_to_logits_and_exposes_phase_timings() -> None:
    torch = pytest.importorskip("torch")
    vocab = PhonemeVocab(
        tokens=("<blank>", "a", "b", "c", "d"),
        phone_tokens=("a", "b", "c", "d"),
        token_to_id={"<blank>": 0, "a": 1, "b": 2, "c": 3, "d": 4},
    )
    entry = HotwordEntry(
        hotword_id="target",
        language="pt-BR",
        surface="target",
        normalized="target",
        words=("target",),
        pronunciation="a b c d",
        phoneme_tokens=("a", "b", "c", "d"),
        token_ids=(1, 2, 3, 4),
        source="test",
        validation_occurrences=1,
    )
    raw_ids = [0, 1, 1, 0, 2, 0, 3, 0, 4, 0]
    logits = torch.full((len(raw_ids), len(vocab.tokens)), -8.0)
    for row, token_id in enumerate(raw_ids):
        logits[row, token_id] = 8.0
    decoded = (
        DecodedPhoneme(1, 0.99, 1, 3),
        DecodedPhoneme(2, 0.99, 4, 5),
        DecodedPhoneme(3, 0.99, 6, 7),
        DecodedPhoneme(4, 0.99, 8, 9),
    )
    config = HotwordScoringConfig(score_threshold=0.86, top_k=5)
    from_logits = score_hotwords(
        logits,
        input_length=len(raw_ids),
        hotwords=[entry],
        config=config,
    )
    from_replay = score_decoded_hotwords(
        decoded,
        effective_time_steps=len(raw_ids),
        hotwords=[entry],
        config=config,
    )
    profiled = profile_decoded_hotwords(
        decoded,
        effective_time_steps=len(raw_ids),
        hotwords=[entry],
        config=config,
    )

    assert from_replay.ranked_matches[0].hotword_id == from_logits.ranked_matches[0].hotword_id
    assert from_replay.ranked_matches[0].edit_distance == 0
    assert profiled.result == from_replay
    assert profiled.matching_seconds >= 0
    assert profiled.sorting_seconds >= 0
    assert profiled.selection_seconds >= 0


def test_capacity_assets_are_nested_train_only_and_replay_benchmark_is_deterministic(
    tmp_path: Path,
) -> None:
    paths = _capacity_inputs(tmp_path)
    assets = tmp_path / "assets"
    summary = build_hotword_capacity_assets(
        training_manifest_path=paths["train"],
        dictionary_path=paths["dictionary"],
        vocab_path=paths["vocab"],
        base_hotwords_path=paths["base_hotwords"],
        base_cases_path=paths["base_cases"],
        selection_path=paths["selection"],
        output_dir=assets,
        sizes=(100, 101),
        candidate_pool_multiplier=2,
        print_progress=False,
    )

    assert summary["status"] == "pass"
    assert summary["base_cases"] == 2
    assert summary["base_expected_hotwords"] == 1
    vocab = load_phoneme_vocab(paths["vocab"])
    for profile in ("representative", "hard_negative"):
        base_cases = [
            json.loads(line)
            for line in (assets / profile / "size_100" / "cases.jsonl").read_text().splitlines()
        ]
        extended_cases = [
            json.loads(line)
            for line in (assets / profile / "size_101" / "cases.jsonl").read_text().splitlines()
        ]
        for base, extended in zip(base_cases, extended_cases, strict=True):
            assert len(base["active_hotword_ids"]) == 100
            assert len(extended["active_hotword_ids"]) == 101
            assert set(base["active_hotword_ids"]).issubset(extended["active_hotword_ids"])
        assert base_cases[0]["expected_hotword_ids"] == ["h001"]
        entries = load_hotword_table(
            assets / profile / "size_101" / "hotwords.jsonl",
            vocab=vocab,
        )
        assert any(entry.source == "capacity_v1_train_only_real_ngram" for entry in entries)

    target = load_hotword_table(paths["base_hotwords"], vocab=vocab)[1]
    replay = tmp_path / "replay.jsonl"
    _write_jsonl(
        replay,
        [
            {
                "case_id": "positive",
                "sample_id": "sample-positive",
                "primary_group": "single_hotword",
                "expected_hotword_ids": [target.hotword_id],
                "chunk_id": 0,
                "cumulative_audio_sec": 2.0,
                "is_final": True,
                "is_tail_flush": False,
                "effective_time_steps": 8,
                "decoded": [
                    {
                        "token_id": token_id,
                        "confidence": 0.99,
                        "start_step": index,
                        "end_step": index + 1,
                    }
                    for index, token_id in enumerate(target.token_ids)
                ],
                "source_timings": {},
            },
            {
                "case_id": "negative",
                "sample_id": "sample-negative",
                "primary_group": "negative",
                "expected_hotword_ids": [],
                "chunk_id": 0,
                "cumulative_audio_sec": 2.0,
                "is_final": True,
                "is_tail_flush": False,
                "effective_time_steps": 8,
                "decoded": [],
                "source_timings": {},
            },
        ],
    )
    benchmark = tmp_path / "benchmark"
    report = benchmark_hotword_capacity(
        assets_root=assets,
        replay_path=replay,
        vocab_path=paths["vocab"],
        output_dir=benchmark,
        sizes=(100, 101),
        warmup_queries=0,
        continue_after_deadline_failure=True,
        print_progress=False,
    )

    assert report["status"] == "pass"
    quality = json.loads((benchmark / "quality_summary.json").read_text())
    assert quality["representative"]["100"]["raw_recall_at_5"] == 1.0
    assert quality["representative"]["101"]["raw_recall_at_5"] == 1.0
    performance = json.loads((benchmark / "performance_summary.json").read_text())
    assert performance["representative"]["101"]["registry_load"]["entries"] >= 101
    recommendation = json.loads((benchmark / "capacity_recommendation.json").read_text())
    assert recommendation["verified_maximum"] == 101
    assert recommendation["recommended_online_cap"] == 100
    assert (benchmark / "sha256.txt").is_file()


def test_capacity_asset_builder_rejects_sealed_test_manifest(tmp_path: Path) -> None:
    paths = _capacity_inputs(tmp_path)
    paths["train"].write_text(
        json.dumps({"id": "sealed", "split": "test", "text": "frase selada"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed test"):
        build_hotword_capacity_assets(
            training_manifest_path=paths["train"],
            dictionary_path=paths["dictionary"],
            vocab_path=paths["vocab"],
            base_hotwords_path=paths["base_hotwords"],
            base_cases_path=paths["base_cases"],
            output_dir=tmp_path / "assets",
            sizes=(100, 101),
            candidate_pool_multiplier=2,
            print_progress=False,
        )


def test_capacity_asset_builder_rejects_explicit_forced_selection(
    tmp_path: Path,
) -> None:
    paths = _capacity_inputs(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selection["retrieval_mode"] = "forced_topk"
    paths["selection"].write_text(json.dumps(selection), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy formal100 or operating"):
        build_hotword_capacity_assets(
            training_manifest_path=paths["train"],
            dictionary_path=paths["dictionary"],
            vocab_path=paths["vocab"],
            base_hotwords_path=paths["base_hotwords"],
            base_cases_path=paths["base_cases"],
            selection_path=paths["selection"],
            output_dir=tmp_path / "assets",
            sizes=(100, 101),
            candidate_pool_multiplier=2,
            print_progress=False,
        )
