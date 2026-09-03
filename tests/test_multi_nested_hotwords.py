from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.multi_nested import (
    CaseScore,
    HotwordFamily,
    MultiNestedCase,
    _score_row,
    build_multi_nested_assets,
    evaluate_multi_nested_case_scores,
    load_multi_nested_case_scores,
    load_multi_nested_cases,
    load_validation_samples,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.hotwords.scoring import HotwordMatch
from qwen_hotword.phonemes.coverage import load_phoneme_vocab


def _entry(hotword_id: str, surface: str, token_count: int = 5) -> HotwordEntry:
    words = tuple(surface.split())
    return HotwordEntry(
        hotword_id=hotword_id,
        language="pt-BR",
        surface=surface,
        normalized=surface,
        words=words,
        pronunciation=" ".join("a" for _ in range(token_count)),
        phoneme_tokens=tuple("a" for _ in range(token_count)),
        token_ids=tuple(1 for _ in range(token_count)),
        source="test",
        validation_occurrences=1,
    )


def _match(hotword_id: str, score: float = 0.9) -> HotwordMatch:
    return HotwordMatch(
        hotword_id=hotword_id,
        surface=hotword_id,
        language="pt-BR",
        score=score,
        edit_similarity=score,
        edit_distance=0,
        edit_ratio=0.0,
        posterior_confidence=0.95,
        decoded_start=0,
        decoded_end=5,
        start_step=0,
        end_step=8,
    )


def _case(
    case_id: str,
    group: str,
    expected: tuple[str, ...],
    *,
    containment: tuple[str, ...] | None = None,
    longest: tuple[str, ...] | None = None,
    families: tuple[str, ...] = (),
    independent: tuple[str, ...] = (),
) -> MultiNestedCase:
    active = tuple(f"h{index}" for index in range(100))
    all_expected = containment if containment is not None else expected
    return MultiNestedCase(
        case_id=case_id,
        sample_id=f"sample-{case_id}",
        audio_path=f"/audio/{case_id}.wav",
        reference_text="texto natural",
        normalized_reference_text="texto natural",
        language="pt-BR",
        primary_group=group,
        expected_hotword_ids=expected,
        expected_surfaces=expected,
        expected_word_spans={item: (index, index + 1) for index, item in enumerate(expected)},
        containment_expected_ids=all_expected,
        longest_match_expected_ids=longest if longest is not None else all_expected,
        active_hotword_ids=active,
        nested_family_ids=families,
        hard_negative_ids=(),
        independent_expected_ids=independent,
        selection_reason="test",
    )


def _score(
    case: MultiNestedCase,
    ranking: tuple[str, ...],
    operating: tuple[str, ...] | None = None,
) -> CaseScore:
    return CaseScore(
        case_id=case.case_id,
        sample_id=case.sample_id,
        primary_group=case.primary_group,
        ranked_matches=tuple(
            _match(item, 0.9 - index * 0.01) for index, item in enumerate(ranking)
        ),
        operating_matches=tuple(_match(item) for item in (operating or ())),
    )


def test_multi_hotword_metrics_keep_ranking_and_operating_separate() -> None:
    one = _case("one", "single_hotword", ("h1",))
    two = _case("two", "two_independent", ("h1", "h2"), independent=("h1", "h2"))
    three = _case(
        "three",
        "three_independent",
        ("h1", "h2", "h3"),
        independent=("h1", "h2", "h3"),
    )
    negative = _case("negative", "negative", ())
    cases = (one, two, three, negative)
    scores = (
        _score(one, ("h1", "h4", "h5", "h6", "h7"), ("h1",)),
        _score(two, ("h1", "h2", "h4", "h5", "h6"), ("h1",)),
        _score(three, ("h1", "h2", "h3", "h4", "h5"), ("h1", "h2")),
        _score(negative, ("h4", "h5", "h6", "h7", "h8"), ()),
    )
    hotwords = tuple(_entry(f"h{index}", f"palavra{index}") for index in range(100))

    report = evaluate_multi_nested_case_scores(cases, hotwords, (), scores)
    by_group = report["by_primary_group"]
    assert isinstance(by_group, dict)
    three_metrics = by_group["three_independent"]
    assert three_metrics["raw_precision_at_5"] == pytest.approx(0.6)
    assert three_metrics["all_3_hit_at_5"] == 1.0
    assert three_metrics["ranking"][2]["mean_hits_at_k"] == 3.0
    assert report["overall"]["ranking"][2]["micro_recall_at_k"] == 1.0
    assert report["overall"]["operating"]["recall"] == pytest.approx(4 / 6)
    assert report["overall"]["operating"]["negative_case_false_positive_rate"] == 0.0


def test_nested_ground_truth_false_trigger_and_slot_crowding() -> None:
    ordinary = _case(
        "ordinary",
        "three_independent",
        ("h1", "h2", "h3"),
        independent=("h1", "h2", "h3"),
    )
    short_only = _case(
        "short-only",
        "nested_short_only",
        ("h10",),
        families=("family",),
    )
    long_present = _case(
        "long",
        "nested_long_present",
        ("h10", "h11"),
        containment=("h10", "h11"),
        longest=("h11",),
        families=("family",),
    )
    nested_plus = _case(
        "nested-plus",
        "nested_family_plus_two",
        ("h10", "h11", "h1", "h2"),
        containment=("h10", "h11", "h1", "h2"),
        longest=("h11", "h1", "h2"),
        families=("family",),
        independent=("h1", "h2"),
    )
    cases = (ordinary, short_only, long_present, nested_plus)
    scores = (
        _score(ordinary, ("h1", "h2", "h3", "h4", "h5")),
        _score(short_only, ("h10", "h11", "h4", "h5", "h6")),
        _score(long_present, ("h10", "h11", "h4", "h5", "h6"), ("h10", "h11")),
        _score(nested_plus, ("h10", "h11", "h1", "h4", "h5")),
    )
    hotwords = tuple(
        _entry(
            f"h{index}",
            "termo longo" if index == 11 else f"palavra{index}",
            token_count=10 if index == 11 else 5,
        )
        for index in range(100)
    )
    family = HotwordFamily("family", "h10", "h11", "curto", "termo longo")

    report = evaluate_multi_nested_case_scores(cases, hotwords, (family,), scores)
    nested = report["nested"]
    assert nested["short_only_long_ranking_false_trigger_rate_at_5"] == 1.0
    assert nested["short_only_long_operating_false_trigger_rate"] == 0.0
    assert nested["containment"]["ranking"][2]["micro_recall_at_k"] == pytest.approx(5 / 6)
    assert nested["longest_match"]["ranking"][2]["micro_recall_at_k"] == 0.75
    assert nested["longest_match"]["operating"]["precision"] == 1.0
    assert nested["family_duplicate_or_redundant_hits"] >= 3
    assert nested["ordinary_three_independent_recall_at_5"] == 1.0
    assert nested["nested_plus_two_independent_recall_at_5"] == 0.5
    assert nested["slot_crowding_loss"] == 0.5
    assert nested["crowding_attribution_cases"][0]["case_id"] == "nested-plus"


def test_v3_case_score_loader_round_trip(tmp_path: Path) -> None:
    case = _case("score", "single_hotword", ("h1",))
    score = _score(case, ("h1", "h2", "h3", "h4", "h5"), ("h1",))
    row = {
        "case_id": score.case_id,
        "sample_id": score.sample_id,
        "primary_group": score.primary_group,
        "effective_time_steps": 42,
        "decoded_token_count": 7,
        "ranking_top5": [match.to_dict() for match in score.ranked_matches],
        "operating_matches": [match.to_dict() for match in score.operating_matches],
    }
    path = tmp_path / "scores.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_multi_nested_case_scores(path)

    assert loaded[0].case_id == "score"
    assert loaded[0].effective_time_steps == 42
    assert [match.hotword_id for match in loaded[0].ranked_matches] == [
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
    ]


def test_v3_case_score_loader_prefers_complete_ranking_and_checks_top5(
    tmp_path: Path,
) -> None:
    case = _case("full-score", "single_hotword", ("h1",))
    score = _score(case, tuple(f"h{index}" for index in range(10)))
    ranked = [match.to_dict() for match in score.ranked_matches]
    row = {
        "case_id": score.case_id,
        "sample_id": score.sample_id,
        "primary_group": score.primary_group,
        "effective_time_steps": 42,
        "decoded_token_count": 7,
        "ranking_top5": ranked[:5],
        "ranked_matches": ranked,
        "ranked_matches_available": 10,
        "ranked_matches_complete": True,
        "active_hotword_count": 10,
        "operating_matches": [],
    }
    path = tmp_path / "full-scores.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = load_multi_nested_case_scores(path)

    assert len(loaded[0].ranked_matches) == 10
    row["ranking_top5"] = ranked[1:6]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not preserve ranking_top5"):
        load_multi_nested_case_scores(path)


def test_v3_score_row_keeps_legacy_top5_and_full_ranking() -> None:
    case = _case("serialized", "single_hotword", ("h1",))
    score = _score(case, tuple(f"h{index}" for index in range(10)))

    row = _score_row(case, score, {})

    assert len(row["ranking_top5"]) == 5
    assert len(row["ranked_matches"]) == 10
    assert row["ranked_matches_available"] == 10
    assert row["ranked_matches_complete"] is False
    assert row["active_hotword_count"] == 100


def _alpha_name(index: int) -> str:
    letters = []
    value = index
    for _ in range(3):
        letters.append(chr(ord("a") + value % 26))
        value //= 26
    return "termo" + "".join(letters)


def test_v3_asset_build_is_deterministic_has_spans_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    words = [_alpha_name(index) for index in range(130)]
    rows = []
    for index in range(130):
        row_words = [words[(index + offset) % 130] for offset in range(6)]
        rows.append(
            {
                "id": f"sample-{index:03d}",
                "split": "validation",
                "language": "pt-BR",
                "audio_path": f"/audio/{index:03d}.wav",
                "text": " ".join(row_words),
            }
        )
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dictionary = tmp_path / "mfa.dict"
    dictionary.write_text(
        "".join(f"{word}\t{' '.join(word)}\n" for word in words), encoding="utf-8"
    )
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", *list("abcdefghijklmnopqrstuvwxyz")]}),
        encoding="utf-8",
    )
    targets = {
        "single_hotword": 1,
        "two_independent": 1,
        "three_independent": 1,
        "nested_short_only": 1,
        "nested_long_present": 1,
        "nested_family_plus_two": 1,
        "negative": 1,
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = build_multi_nested_assets(
        manifest, dictionary, vocab_path, first, seed=7, group_targets=targets
    )
    build_multi_nested_assets(
        manifest, dictionary, vocab_path, second, seed=7, group_targets=targets
    )

    assert (first / "multi_nested_cases_v3.jsonl").read_text() == (
        second / "multi_nested_cases_v3.jsonl"
    ).read_text()
    assert first_summary["actual_case_counts"] == targets
    cases = load_multi_nested_cases(first / "multi_nested_cases_v3.jsonl")
    assert all(len(case.active_hotword_ids) == 100 for case in cases)
    capacity_rows = [
        json.loads(line)
        for line in (first / "multi_nested_cases_v3.jsonl").read_text().splitlines()
    ]
    capacity_rows[0]["active_hotword_ids"].append("capacity-extra-hotword")
    capacity_path = tmp_path / "capacity_cases.jsonl"
    capacity_path.write_text(
        "".join(json.dumps(row) + "\n" for row in capacity_rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="100 unique active hotwords"):
        load_multi_nested_cases(capacity_path)
    capacity_cases = load_multi_nested_cases(
        capacity_path,
        expected_active_hotwords=None,
    )
    assert len(capacity_cases[0].active_hotword_ids) == 101
    assert all(
        left[1] <= right[0] or right[1] <= left[0]
        for case in cases
        if case.primary_group in {"two_independent", "three_independent"}
        for index, left in enumerate(case.expected_word_spans.values())
        for right in list(case.expected_word_spans.values())[index + 1 :]
    )
    vocab = load_phoneme_vocab(vocab_path)
    load_hotword_table(first / "multi_nested_hotwords_v3.jsonl", vocab=vocab)
    with pytest.raises(FileExistsError, match="new empty directory"):
        build_multi_nested_assets(
            manifest, dictionary, vocab_path, first, seed=7, group_targets=targets
        )


def test_v3_manifest_loader_rejects_train_and_sealed_test(tmp_path: Path) -> None:
    for split in ("train", "test"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "id": "sample",
                    "split": split,
                    "language": "pt-BR",
                    "audio_path": "/audio.wav",
                    "text": "texto válido",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_validation_samples(path)
