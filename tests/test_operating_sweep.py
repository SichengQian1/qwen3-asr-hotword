from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.operating_sweep import sweep_operating_points


def _match(
    hotword_id: str, *, score: float, edit_ratio: float, posterior: float = 0.9
) -> dict[str, object]:
    return {
        "hotword_id": hotword_id,
        "surface": hotword_id,
        "language": "pt-BR",
        "score": score,
        "edit_similarity": 1.0 - edit_ratio,
        "edit_distance": round(edit_ratio * 10),
        "edit_ratio": edit_ratio,
        "posterior_confidence": posterior,
        "decoded_start": 0,
        "decoded_end": 4,
        "start_step": 0,
        "end_step": 4,
    }


def _benchmark(tmp_path: Path, *, complete: bool = True) -> Path:
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "run_config.json").write_text(
        json.dumps(
            {
                "retrieval_config": {
                    "threshold": 0.86,
                    "top_k": 1,
                    "minimum_phonemes": 4,
                    "maximum_edit_ratio": 0.35,
                    "posterior_weight": 0.25,
                    "minimum_posterior_confidence": 0.0,
                    "minimum_top1_margin": 0.0,
                },
                "test_set_used": False,
            }
        ),
        encoding="utf-8",
    )
    common = {
        "schema_version": 1,
        "profile": "representative",
        "size": 100,
        "window": "full_current",
        "shortlist_size": 2,
        "chunk_id": 0,
        "cumulative_audio_sec": 2.0,
        "is_final": True,
        "is_tail_flush": False,
        "candidate_count": 2,
        "ranked_matches_available": 2,
        "ranked_matches_complete": complete,
        "retrieval_seconds": 0.01,
    }
    rows = [
        {
            **common,
            "case_id": "positive",
            "expected_hotword_ids": ["distractor", "target"],
            "top_matches": [
                _match("distractor", score=0.88, edit_ratio=0.10),
                _match("target", score=0.83, edit_ratio=0.40),
            ],
        },
        {
            **common,
            "case_id": "negative",
            "expected_hotword_ids": [],
            "top_matches": [
                _match("wrong", score=0.87, edit_ratio=0.10),
                _match("weak", score=0.70, edit_ratio=0.20),
            ],
        },
    ]
    (benchmark / "query_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return benchmark


def test_operating_sweep_selects_recall_first_point_and_preserves_baseline(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark(tmp_path)
    output = tmp_path / "sweep"

    summary = sweep_operating_points(
        benchmark_dir=benchmark,
        output_dir=output,
        profile="representative",
        size=100,
        shortlist_size=2,
        top_ks=(1, 2),
        thresholds=(0.0, 0.82, 0.86),
        maximum_edit_ratios=(0.35, 0.40),
        minimum_posterior_confidences=(0.0,),
        minimum_top1_margins=(0.0,),
        target_recall=1.0,
        diagnostic_precision_target=0.85,
    )

    assert summary["status"] == "pass"
    assert summary["sweep_points"] == 12
    assert summary["retrieval_p95_le_deadline"] is True
    recommendation = json.loads(
        (output / "recommended_config.json").read_text(encoding="utf-8")
    )
    assert recommendation["status"] == "target_recall_met"
    assert recommendation["source_baseline"]["final"]["recall"] == 0.5
    assert recommendation["recall_first"]["config"] == {
        "maximum_edit_ratio": 0.4,
        "minimum_posterior_confidence": 0.0,
        "minimum_top1_margin": 0.0,
        "threshold": 0.82,
        "top_k": 2,
    }
    assert recommendation["recall_first"]["final"]["recall"] == 1.0
    assert recommendation["recall_first"]["final"]["precision"] == pytest.approx(2 / 3)
    assert recommendation["strict_recall_and_precision_point_count"] == 0
    assert set(recommendation["source_gates_by_top_k"]) == {"1", "2"}
    assert recommendation["best_by_top_k"]["1"]["status"] == "target_recall_not_met"
    assert recommendation["best_by_top_k"]["1"]["recall_first"]["final"]["recall"] == 0.5
    assert recommendation["best_by_top_k"]["2"]["status"] == "target_recall_met"
    assert (output / "pareto_frontier.jsonl").is_file()
    assert (output / "sha256.txt").is_file()


def test_operating_sweep_rejects_truncated_ranked_matches(tmp_path: Path) -> None:
    benchmark = _benchmark(tmp_path, complete=False)

    with pytest.raises(ValueError, match="complete ranked shortlist"):
        sweep_operating_points(
            benchmark_dir=benchmark,
            output_dir=tmp_path / "sweep",
            profile="representative",
            size=100,
            shortlist_size=2,
            top_ks=(1, 2),
            thresholds=(0.86,),
            maximum_edit_ratios=(0.35,),
            minimum_posterior_confidences=(0.0,),
            minimum_top1_margins=(0.0,),
        )
