from __future__ import annotations

from pathlib import Path

import pytest

from qwen_hotword.hotwords.evaluation import (
    HotwordCaseScore,
    evaluate_hotword_ranking,
    evaluate_hotword_threshold,
)
from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.hotwords.scoring import (
    HotwordMatch,
    HotwordScoringConfig,
    decode_ctc_posterior,
    score_hotwords,
)
from qwen_hotword.phonemes.coverage import PhonemeVocab


def _vocab() -> PhonemeVocab:
    tokens = ("<blank>", "a", "b", "c", "d", "e", "f")
    return PhonemeVocab(
        tokens=tokens,
        phone_tokens=tokens[1:],
        token_to_id={token: index for index, token in enumerate(tokens)},
    )


def _entry(hotword_id: str, token_ids: tuple[int, ...]) -> HotwordEntry:
    vocab = _vocab()
    return HotwordEntry(
        hotword_id=hotword_id,
        language="pt-BR",
        surface=hotword_id,
        normalized=hotword_id,
        words=(hotword_id,),
        pronunciation=" ".join(vocab.tokens[index] for index in token_ids),
        phoneme_tokens=tuple(vocab.tokens[index] for index in token_ids),
        token_ids=token_ids,
        source="test",
        validation_occurrences=1,
    )


def _logits(raw_ids: list[int], classes: int = 7):
    torch = pytest.importorskip("torch")
    logits = torch.full((len(raw_ids), classes), -8.0)
    for row, token_id in enumerate(raw_ids):
        logits[row, token_id] = 8.0
    return logits


def test_decode_and_score_exact_hotword_on_effective_time_axis() -> None:
    logits = _logits([0, 1, 1, 0, 2, 0, 3, 0, 4, 0, 6, 6])
    decoded = decode_ctc_posterior(logits, input_length=10)

    assert [item.token_id for item in decoded] == [1, 2, 3, 4]
    assert decoded[0].start_step == 1
    assert decoded[-1].end_step == 9

    result = score_hotwords(
        logits,
        input_length=10,
        hotwords=[_entry("exact", (1, 2, 3, 4)), _entry("wrong", (2, 3, 5, 6))],
        config=HotwordScoringConfig(score_threshold=0.75, minimum_top1_margin=0.03),
    )

    assert result.effective_time_steps == 10
    assert result.selected_matches[0].hotword_id == "exact"
    assert result.selected_matches[0].edit_distance == 0
    assert result.selected_matches[0].score > 0.99


def test_equal_confusable_scores_are_suppressed() -> None:
    result = score_hotwords(
        _logits([0, 1, 0, 2, 0, 3, 0, 4, 0]),
        input_length=9,
        hotwords=[_entry("left", (1, 2, 3, 5)), _entry("right", (1, 2, 3, 6))],
        config=HotwordScoringConfig(
            score_threshold=0.70,
            maximum_edit_ratio=0.30,
            minimum_top1_margin=0.03,
        ),
    )

    assert result.selected_matches == ()
    assert result.suppressed_reason == "ambiguous_top_matches"


def test_default_multi_hotword_scoring_keeps_close_matches() -> None:
    result = score_hotwords(
        _logits([0, 1, 0, 2, 0, 3, 0, 4, 0]),
        input_length=9,
        hotwords=[_entry("left", (1, 2, 3, 5)), _entry("right", (1, 2, 3, 6))],
        config=HotwordScoringConfig(
            score_threshold=0.70,
            maximum_edit_ratio=0.30,
        ),
    )

    assert [match.hotword_id for match in result.selected_matches] == [
        "left",
        "right",
    ]
    assert result.suppressed_reason is None


def test_long_hotword_against_short_decode_returns_below_threshold() -> None:
    result = score_hotwords(
        _logits([0, 1, 0]),
        input_length=3,
        hotwords=[_entry("long", (1, 2, 3, 4, 5, 6))],
    )

    assert result.selected_matches == ()
    assert result.ranked_matches[0].edit_distance == 5


def _match(hotword_id: str, score: float) -> HotwordMatch:
    return HotwordMatch(
        hotword_id=hotword_id,
        surface=hotword_id,
        language="pt-BR",
        score=score,
        edit_similarity=score,
        edit_distance=0,
        edit_ratio=0.0,
        posterior_confidence=0.9,
        decoded_start=0,
        decoded_end=4,
        start_step=0,
        end_step=8,
    )


def test_threshold_metrics_count_positive_hits_and_negative_false_alarms() -> None:
    cases = [
        HotwordCaseScore(
            case_id="positive",
            sample_id="sample-1",
            case_type="positive_confusable",
            active_hotword_ids=("wanted", "other"),
            expected_hotword_ids=("wanted",),
            effective_time_steps=20,
            decoded_token_count=8,
            ranked_matches=(_match("wanted", 0.90), _match("other", 0.40)),
        ),
        HotwordCaseScore(
            case_id="negative",
            sample_id="sample-2",
            case_type="negative",
            active_hotword_ids=("other",),
            expected_hotword_ids=(),
            effective_time_steps=20,
            decoded_token_count=8,
            ranked_matches=(_match("other", 0.80),),
        ),
    ]

    loose = evaluate_hotword_threshold(
        cases,
        threshold=0.75,
        top_k=1,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.03,
    )
    strict = evaluate_hotword_threshold(
        cases,
        threshold=0.85,
        top_k=1,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.03,
    )

    assert loose.recall == 1.0
    assert loose.precision == 0.5
    assert loose.negative_case_false_positive_rate == 1.0
    assert strict.recall == strict.precision == 1.0
    assert strict.negative_case_false_positive_rate == 0.0


def test_threshold_metrics_keep_close_valid_multi_hotword_matches_without_margin() -> None:
    cases = [
        HotwordCaseScore(
            case_id="multi-positive",
            sample_id="sample-1",
            case_type="positive_confusable",
            active_hotword_ids=("short", "long"),
            expected_hotword_ids=("short", "long"),
            effective_time_steps=20,
            decoded_token_count=8,
            ranked_matches=(_match("short", 0.99), _match("long", 0.98)),
        )
    ]

    retained = evaluate_hotword_threshold(
        cases,
        threshold=0.90,
        top_k=5,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.0,
    )
    suppressed = evaluate_hotword_threshold(
        cases,
        threshold=0.90,
        top_k=5,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.03,
    )

    assert retained.recall == retained.precision == 1.0
    assert retained.positive_case_hit_rate == 1.0
    assert suppressed.recall == suppressed.positive_case_hit_rate == 0.0


def test_ranking_recall_reports_top1_top3_and_top5_without_threshold() -> None:
    cases = [
        HotwordCaseScore(
            case_id=f"case-{index}",
            sample_id=f"sample-{index}",
            case_type="positive_confusable",
            active_hotword_ids=("a", "b", "c", "d", "e"),
            expected_hotword_ids=(expected,),
            effective_time_steps=20,
            decoded_token_count=8,
            ranked_matches=tuple(
                _match(hotword_id, 1.0 - rank * 0.1)
                for rank, hotword_id in enumerate(ranking)
            ),
        )
        for index, (expected, ranking) in enumerate(
            (
                ("a", ("a", "b", "c", "d", "e")),
                ("b", ("a", "b", "c", "d", "e")),
                ("e", ("a", "b", "c", "d", "e")),
            )
        )
    ]

    metrics = evaluate_hotword_ranking(cases, ks=(1, 3, 5))

    assert [item.recall_at_k for item in metrics] == [
        pytest.approx(1 / 3),
        pytest.approx(2 / 3),
        1.0,
    ]
    assert [item.positive_case_hit_rate_at_k for item in metrics] == [
        pytest.approx(1 / 3),
        pytest.approx(2 / 3),
        1.0,
    ]


def test_hotword_loader_rejects_out_of_range_ids_cleanly(tmp_path: Path) -> None:
    table = tmp_path / "hotwords.jsonl"
    table.write_text(
        '{"hotword_id":"bad","language":"pt-BR","surface":"bad",'
        '"normalized":"bad","words":["bad"],"pronunciation":"a",'
        '"phoneme_tokens":["a"],"token_ids":[99],"source":"test",'
        '"validation_occurrences":1}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="out-of-range"):
        load_hotword_table(table, vocab=_vocab())


def test_hotword_loader_accepts_canonically_equivalent_tokens(tmp_path: Path) -> None:
    table = tmp_path / "hotwords.jsonl"
    table.write_text(
        '{"hotword_id":"cedilla","language":"English","surface":"façade",'
        '"normalized":"façade","words":["façade"],"pronunciation":"ç",'
        '"phoneme_tokens":["ç"],"token_ids":[1],"source":"test",'
        '"validation_occurrences":1}\n',
        encoding="utf-8",
    )
    vocab = PhonemeVocab(
        tokens=("<blank>", "ç"),
        phone_tokens=("ç",),
        token_to_id={"<blank>": 0, "ç": 1},
    )

    entries = load_hotword_table(table, vocab=vocab)

    assert entries[0].phoneme_tokens == ("ç",)
