from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.multi_nested import (
    CaseScore,
    MultiNestedCase,
    evaluate_multi_nested_case_scores,
)
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import HotwordMatch
from qwen_hotword.hotwords.v3_operating_calibration import (
    _replay_scores,
    calibrate_v3_operating_points,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _match(hotword_id: str, score: float, posterior: float = 0.9) -> HotwordMatch:
    return HotwordMatch(
        hotword_id=hotword_id,
        surface=hotword_id,
        language="pt-BR",
        score=score,
        edit_similarity=0.9,
        edit_distance=1,
        edit_ratio=0.1,
        posterior_confidence=posterior,
        decoded_start=0,
        decoded_end=4,
        start_step=0,
        end_step=4,
    )


def _case(index: int, group: str, expected: tuple[str, ...]) -> MultiNestedCase:
    return MultiNestedCase(
        case_id=f"case-{index}",
        sample_id=f"sample-{index}",
        audio_path=f"/audio/{index}.wav",
        reference_text="texto",
        normalized_reference_text="texto",
        language="pt-BR",
        primary_group=group,
        expected_hotword_ids=expected,
        expected_surfaces=expected,
        expected_word_spans={item: (0, 1) for item in expected},
        containment_expected_ids=expected,
        longest_match_expected_ids=expected,
        active_hotword_ids=tuple(f"h{item}" for item in range(100)),
        nested_family_ids=(),
        hard_negative_ids=(),
        independent_expected_ids=expected,
        selection_reason="test",
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, full_ranking: bool = False) -> dict[str, Path]:
    vocab = tmp_path / "vocab.json"
    phones = tuple(f"p{index}" for index in range(10))
    _write_json(vocab, {"tokens": ["<blank>", "<unk>", *phones]})
    entries = [
        HotwordEntry(
            hotword_id=f"h{index}",
            language="pt-BR",
            surface=f"termo{index}",
            normalized=f"termo{index}",
            words=(f"termo{index}",),
            pronunciation=f"{phones[index // 10]} {phones[index % 10]}",
            phoneme_tokens=(phones[index // 10], phones[index % 10]),
            token_ids=(index // 10 + 2, index % 10 + 2),
            source="test",
            validation_occurrences=1,
        )
        for index in range(100)
    ]
    hotwords = tmp_path / "hotwords.jsonl"
    hotwords.write_text(
        "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    families = tmp_path / "families.jsonl"
    families.write_text("", encoding="utf-8")
    groups = (
        "single_hotword",
        "two_independent",
        "three_independent",
        "nested_short_only",
        "nested_long_present",
        "nested_family_plus_two",
        "negative",
    )
    cases_value = tuple(
        _case(index, group, () if group == "negative" else ("h1",))
        for index, group in enumerate(groups)
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "".join(json.dumps(case.to_dict(), sort_keys=True) + "\n" for case in cases_value),
        encoding="utf-8",
    )
    scores_value = []
    for case in cases_value:
        positive = case.primary_group != "negative"
        leading_scores = (
            (0.90, 0.85, 0.80, 0.75, 0.70)
            if positive
            else (
                0.84,
                0.80,
                0.76,
                0.72,
                0.68,
            )
        )
        ranked = [
            _match(hotword_id, score)
            for hotword_id, score in zip(
                ("h1", "h2", "h3", "h4", "h0"), leading_scores, strict=True
            )
        ]
        ranked.extend(
            _match(f"h{index}", max(0.0, 0.68 - position * 0.006))
            for position, index in enumerate(range(5, 100), start=1)
        )
        scores_value.append(
            CaseScore(
                case_id=case.case_id,
                sample_id=case.sample_id,
                primary_group=case.primary_group,
                ranked_matches=tuple(ranked if full_ranking else ranked[:5]),
                operating_matches=((_match("h1", 0.90),) if positive else ()),
                effective_time_steps=10,
                decoded_token_count=4,
            )
        )
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": score.case_id,
                    "sample_id": score.sample_id,
                    "primary_group": score.primary_group,
                    "effective_time_steps": score.effective_time_steps,
                    "decoded_token_count": score.decoded_token_count,
                    "ranking_top5": [item.to_dict() for item in score.ranked_matches[:5]],
                    **(
                        {
                            "ranked_matches": [item.to_dict() for item in score.ranked_matches],
                            "ranked_matches_available": len(score.ranked_matches),
                            "ranked_matches_complete": True,
                            "active_hotword_count": 100,
                        }
                        if full_ranking
                        else {}
                    ),
                    "operating_matches": [item.to_dict() for item in score.operating_matches],
                },
                sort_keys=True,
            )
            + "\n"
            for score in scores_value
        ),
        encoding="utf-8",
    )
    report = tmp_path / "candidate.json"
    scores_tuple = tuple(scores_value)
    metrics = evaluate_multi_nested_case_scores(cases_value, entries, (), scores_tuple)
    scoring_config = {
        "top_k": 5,
        "threshold": 0.86,
        "maximum_edit_ratio": 0.35,
        "posterior_weight": 0.25,
        "minimum_posterior_confidence": 0.0,
        "minimum_top1_margin": 0.0,
    }
    if full_ranking:
        scoring_config.update({"saved_ranked_matches": 100, "ranked_matches_complete": True})
    _write_json(
        report,
        {
            "test_set_used": False,
            "checkpoint_sha256": "a" * 64,
            "vocab_sha256": _sha256(vocab),
            "hotword_table_sha256": _sha256(hotwords),
            "families_sha256": _sha256(families),
            "cases_sha256": _sha256(cases),
            "case_scores_sha256": _sha256(scores),
            "scoring_config": scoring_config,
            "metrics": metrics,
        },
    )
    return {
        "vocab": vocab,
        "hotwords": hotwords,
        "families": families,
        "cases": cases,
        "scores": scores,
        "report": report,
    }


def test_v3_calibration_writes_exact_candidates_and_marks_truncated_points(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "calibration"

    summary = calibrate_v3_operating_points(
        vocab_path=paths["vocab"],
        hotword_path=paths["hotwords"],
        families_path=paths["families"],
        cases_path=paths["cases"],
        case_scores_path=paths["scores"],
        candidate_report_path=paths["report"],
        output_dir=output,
        top_ks=(3, 5),
        thresholds=(0.70, 0.75, 0.86),
        minimum_posterior_confidences=(0.0, 0.95),
    )

    assert summary["status"] in {
        "guarded_recall_gain_candidate_available",
        "exact_sweep_complete_no_guarded_recall_gain",
    }
    assert summary["sweep_point_count"] == 12
    assert 0 < summary["exact_point_count"] < 12
    assert summary["recommended_candidates"]
    assert (output / "exact_pareto_frontier.jsonl").is_file()
    assert (output / "sha256.txt").is_file()
    rows = [
        json.loads(line)
        for line in (output / "operating_point_sweep.jsonl").read_text().splitlines()
    ]
    source = next(row for row in rows if row["is_source_operating_point"])
    assert source["replay_exact"] is True
    assert any(not row["replay_exact"] for row in rows)


def test_v3_calibration_rejects_top_k_above_saved_rank_depth(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    with pytest.raises(ValueError, match="rank depth"):
        calibrate_v3_operating_points(
            vocab_path=paths["vocab"],
            hotword_path=paths["hotwords"],
            families_path=paths["families"],
            cases_path=paths["cases"],
            case_scores_path=paths["scores"],
            candidate_report_path=paths["report"],
            output_dir=tmp_path / "output",
            top_ks=(5, 7),
        )


def test_v3_calibration_accepts_top7_with_complete_ranking(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, full_ranking=True)

    summary = calibrate_v3_operating_points(
        vocab_path=paths["vocab"],
        hotword_path=paths["hotwords"],
        families_path=paths["families"],
        cases_path=paths["cases"],
        case_scores_path=paths["scores"],
        candidate_report_path=paths["report"],
        output_dir=tmp_path / "full-output",
        top_ks=(5, 7),
        thresholds=(0.83, 0.86),
        minimum_posterior_confidences=(0.0,),
    )

    assert summary["sweep_point_count"] == 4
    assert summary["exact_point_count"] == 4
    assert summary["non_exact_point_count"] == 0


def test_replay_completeness_requires_filled_cap_or_score_cutoff() -> None:
    score = CaseScore(
        case_id="case",
        sample_id="sample",
        primary_group="single_hotword",
        ranked_matches=(
            _match("h1", 0.90, posterior=0.9),
            _match("h2", 0.85, posterior=0.4),
            _match("h3", 0.80, posterior=0.4),
            _match("h4", 0.75, posterior=0.4),
            _match("h5", 0.70, posterior=0.4),
        ),
        operating_matches=(),
    )

    _, incomplete = _replay_scores(
        (score,),
        top_k=3,
        threshold=0.70,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.5,
    )
    assert incomplete == ("case",)

    replayed, complete = _replay_scores(
        (score,),
        top_k=3,
        threshold=0.86,
        maximum_edit_ratio=0.35,
        minimum_posterior_confidence=0.5,
    )
    assert complete == ()
    assert [item.hotword_id for item in replayed[0].operating_matches] == ["h1"]
