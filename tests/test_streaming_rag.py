from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.scoring import (
    HotwordMatch,
    HotwordScoringConfig,
    HotwordScoringResult,
)
from qwen_hotword.inference.streaming_backends import select_streaming_ctc_matches
from qwen_hotword.inference.streaming_core import StreamingSample
from qwen_hotword.inference.streaming_rag import (
    _build_latency_summary,
    _build_summary,
    _collect_shards,
    _file_identity,
    _prepare_output,
    _validate_offline_control,
    _write_shard,
)


def _sample() -> StreamingSample:
    return StreamingSample(
        case_id="case-1",
        sample_id="sample-1",
        reference_text="hello hot word",
        language="English",
        expected_hotword_ids=("hot",),
        expected_surfaces=("hot word",),
        active_hotword_ids=("hot",),
        audio_path="/ignored/audio.wav",
    )


def test_forced_topk_uses_ranked_matches_without_operating_guards() -> None:
    def match(index: int) -> HotwordMatch:
        return HotwordMatch(
            hotword_id=f"h{index}",
            language="pt-BR",
            surface=f"term {index}",
            score=1.0 - index * 0.1,
            edit_similarity=1.0 - index * 0.1,
            edit_distance=index,
            edit_ratio=index * 0.1,
            posterior_confidence=0.9,
            decoded_start=0,
            decoded_end=1,
            start_step=0,
            end_step=1,
        )

    ranked = tuple(match(index) for index in range(6))
    scored = HotwordScoringResult(
        effective_time_steps=10,
        decoded_token_ids=(1,),
        decoded_confidences=(0.9,),
        ranked_matches=ranked,
        selected_matches=(ranked[0],),
        suppressed_reason=None,
    )
    config = HotwordScoringConfig(top_k=5, score_threshold=0.86)
    assert select_streaming_ctc_matches(
        scored, scoring_config=config, retrieval_mode="operating"
    ) == (ranked[0],)
    assert select_streaming_ctc_matches(
        scored, scoring_config=config, retrieval_mode="forced_topk"
    ) == ranked[:5]


def test_sample_shards_are_resumable_and_collect_deterministically(tmp_path: Path) -> None:
    output = tmp_path / "run"
    config = {"schema_version": 1, "chunk_size_sec": 2.0}
    _prepare_output(output, config, resume=False)
    sample = _sample()
    result = {
        "case_id": sample.case_id,
        "sample_id": sample.sample_id,
        "experiment_group": "C",
        "reference_text": sample.reference_text,
        "prediction": "hello",
        "expected_hotword_ids": ["hot"],
        "expected_hotwords": ["hot word"],
        "matched_expected_hotword_ids": [],
        "injected_hotword_ids": [],
        "injected_hotwords": [],
        "boundary_bucket": None,
        "hotword_metrics": [],
        "partial_modification_count": 1,
        "inference_seconds": 0.5,
        "failure_reason": "unknown_requires_review",
    }
    timeline = (
        {
            "case_id": sample.case_id,
            "experiment_group": "C",
            "chunk_id": 0,
        },
    )
    shard_dir = output / "sample_shards"
    shard_dir.mkdir()
    _write_shard(shard_dir, "C", sample.case_id, result, timeline)

    first = _collect_shards(shard_dir, ("C",), (sample,))
    _prepare_output(output, config, resume=True)
    second = _collect_shards(shard_dir, ("C",), (sample,))
    assert first == second
    with pytest.raises(ValueError, match="config differs"):
        _prepare_output(output, {"schema_version": 2}, resume=True)


def test_summary_and_latency_keep_missing_timestamps_explicit() -> None:
    rows = [
        {
            "case_id": "case-1",
            "sample_id": "sample-1",
            "experiment_group": "D",
            "reference_text": "hello hot word",
            "prediction": "hello hot word",
            "expected_hotword_ids": ["hot"],
            "expected_hotwords": ["hot word"],
            "matched_expected_hotword_ids": ["hot"],
            "injected_hotword_ids": ["hot"],
            "injected_hotwords": ["hot word"],
            "injected_candidates": [
                {"hotword_id": "hot", "surface": "hot word"},
                {"hotword_id": "wrong", "surface": "cold word"},
            ],
            "boundary_bucket": None,
            "hotword_metrics": [
                {
                    "ctc_first_detect_latency_sec": None,
                    "first_correct_latency_sec": None,
                    "stabilization_latency_sec": None,
                    "chunks_from_injection_to_first_correct": 0,
                    "mutable_at_first_detect": True,
                }
            ],
            "partial_modification_count": 1,
            "inference_seconds": 0.5,
            "failure_reason": None,
        }
    ]
    summary = _build_summary(rows)
    assert summary["groups"]["D"]["hotword_exact_recall"] == 1.0
    assert summary["groups"]["D"]["final_prompted_hotword_precision"] == 1.0
    assert summary["groups"]["D"]["wrong_injected_write_through_rate"] == 0.0
    timeline = [
        {
            "case_id": "case-1",
            "experiment_group": "D",
            "chunk_id": 0,
            "compute_timing": {
                "step_total_seconds": 0.2,
                "retrieval_seconds": 0.04,
                "qwen_streaming_seconds": 0.1,
                "retrieval_over_50ms": False,
            },
        }
    ]
    latency = _build_latency_summary(rows, timeline)
    assert latency["groups"]["D"]["ctc_first_detect_latency_sec"]["count"] == 0
    assert latency["groups"]["D"]["chunks_from_injection_to_first_correct"]["median"] == 0.0
    compute = latency["groups"]["D"]["compute"]
    assert compute["chunk_metrics"]["retrieval_seconds"]["p99"] == pytest.approx(0.04)
    assert compute["retrieval_over_50ms_rate"] == 0.0


def test_multi_nested_offline_control_requires_exact_top5_and_inputs(tmp_path: Path) -> None:
    offline = tmp_path / "offline"
    offline.mkdir()
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    paths = {
        "offline": offline,
        "model": model,
        "validation": tmp_path / "validation.jsonl",
        "vocab": tmp_path / "vocab.json",
        "hotwords": tmp_path / "hotwords.jsonl",
        "cases": tmp_path / "cases.jsonl",
        "families": tmp_path / "families.jsonl",
        "checkpoint": tmp_path / "ctc.pt",
        "ctc_report": tmp_path / "ctc_report.json",
    }
    for key in ("validation", "vocab", "hotwords", "cases", "families", "checkpoint"):
        paths[key].write_text(key + "\n", encoding="utf-8")
    paths["ctc_report"].write_text(
        json.dumps(
            {
                "checkpoint_sha256": _file_identity(paths["checkpoint"])["sha256"],
                "scoring_config": {
                    "threshold": 0.86,
                    "top_k": 5,
                    "maximum_edit_ratio": 0.35,
                    "posterior_weight": 0.25,
                    "minimum_posterior_confidence": 0.0,
                    "minimum_phonemes": 4,
                    "minimum_top1_margin": 0.0,
                    "time_axis": "temporal_upsample_2x_only",
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        version = importlib.metadata.version("qwen-asr")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed-local"
    report = {
        "status": "pass",
        "test_set_used": False,
        "qwen_asr_version": version,
        "retrieval_config": {
            "threshold": 0.86,
            "top_k": 5,
            "maximum_edit_ratio": 0.35,
            "posterior_weight": 0.25,
            "minimum_posterior_confidence": 0.0,
            "minimum_top1_margin": 0.0,
        },
        "prompt_interface": {"template": "Reference: {hotwords}", "language": "Portuguese"},
        "model": {
            "path": str(model),
            "dtype": "bfloat16",
            "max_new_tokens": 128,
        },
        "selection": {"profile": "formal100", "total_cases": 100},
        "inputs": {
            report_key: _file_identity(paths[path_key])
            for report_key, path_key in {
                "validation_manifest": "validation",
                "vocab": "vocab",
                "hotword_table": "hotwords",
                "cases": "cases",
                "hotword_families": "families",
                "ctc_report": "ctc_report",
            }.items()
        },
    }
    (offline / "multi_nested_prompt_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    control = _validate_offline_control(
        offline_format="multi_nested_v3",
        paths=paths,
        threshold=0.86,
        top_k=5,
        retrieval_mode="operating",
        maximum_edit_ratio=0.35,
        posterior_weight=0.25,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.0,
        prompt_template="Reference: {hotwords}",
        language="Portuguese",
        dtype="bfloat16",
        max_new_tokens=None,
    )
    assert control["status"] == "pass"
    assert control["max_new_tokens"] == 128
    assert control["total_cases"] == 100
    with pytest.raises(ValueError, match="retrieval_config.top_k"):
        _validate_offline_control(
            offline_format="multi_nested_v3",
            paths=paths,
            threshold=0.86,
            top_k=3,
            retrieval_mode="operating",
            maximum_edit_ratio=0.35,
            posterior_weight=0.25,
            minimum_posterior_confidence=0.0,
            minimum_top1_margin=0.0,
            prompt_template="Reference: {hotwords}",
            language="Portuguese",
            dtype="bfloat16",
            max_new_tokens=None,
        )

    paths["hotwords"].write_text("expanded-hotwords\n", encoding="utf-8")
    paths["cases"].write_text("expanded-cases\n", encoding="utf-8")
    selection_only = _validate_offline_control(
        offline_format="multi_nested_v3",
        control_mode="selection_only",
        paths=paths,
        threshold=0.75,
        top_k=7,
        retrieval_mode="operating",
        maximum_edit_ratio=0.35,
        posterior_weight=0.25,
        minimum_posterior_confidence=0.5,
        minimum_top1_margin=0.0,
        prompt_template="Reference: {hotwords}",
        language="Portuguese",
        dtype="bfloat16",
        max_new_tokens=None,
    )
    assert selection_only["control_mode"] == "selection_only"
    assert selection_only["validated_retrieval_config"] is None
    assert selection_only["current_retrieval_config_not_compared"] == {
        "mode": "operating",
        "threshold": 0.75,
        "top_k": 7,
        "maximum_edit_ratio": 0.35,
        "minimum_posterior_confidence": 0.5,
        "minimum_top1_margin": 0.0,
        "posterior_weight": 0.25,
        "minimum_phonemes": 4,
        "guards_applied": True,
        "candidate_source": "operating_matches",
    }


def test_multi_nested_offline_control_accepts_explicit_forced_topk(tmp_path: Path) -> None:
    offline = tmp_path / "offline"
    offline.mkdir()
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    paths = {
        "offline": offline,
        "model": model,
        "validation": tmp_path / "validation.jsonl",
        "vocab": tmp_path / "vocab.json",
        "hotwords": tmp_path / "hotwords.jsonl",
        "cases": tmp_path / "cases.jsonl",
        "families": tmp_path / "families.jsonl",
        "checkpoint": tmp_path / "ctc.pt",
        "ctc_report": tmp_path / "ctc_report.json",
    }
    for key in ("validation", "vocab", "hotwords", "cases", "families", "checkpoint"):
        paths[key].write_text(key + "\n", encoding="utf-8")
    paths["ctc_report"].write_text(
        json.dumps(
            {
                "checkpoint_sha256": _file_identity(paths["checkpoint"])["sha256"],
                "scoring_config": {
                    "threshold": 0.86,
                    "top_k": 5,
                    "maximum_edit_ratio": 0.35,
                    "posterior_weight": 0.25,
                    "minimum_posterior_confidence": 0.0,
                    "minimum_phonemes": 4,
                    "minimum_top1_margin": 0.0,
                    "time_axis": "temporal_upsample_2x_only",
                },
            }
        ),
        encoding="utf-8",
    )
    try:
        version = importlib.metadata.version("qwen-asr")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed-local"
    report = {
        "status": "pass",
        "test_set_used": False,
        "qwen_asr_version": version,
        "retrieval_config": {
            "mode": "forced_topk",
            "threshold": None,
            "top_k": 5,
            "maximum_edit_ratio": None,
            "posterior_weight": 0.25,
            "minimum_posterior_confidence": None,
            "minimum_top1_margin": None,
            "minimum_phonemes": 4,
            "guards_applied": False,
            "candidate_source": "ranked_matches[:5]",
        },
        "prompt_interface": {"template": "Reference: {hotwords}", "language": "Portuguese"},
        "model": {"path": str(model), "dtype": "bfloat16", "max_new_tokens": 128},
        "selection": {"profile": "formal100", "total_cases": 100},
        "inputs": {
            report_key: _file_identity(paths[path_key])
            for report_key, path_key in {
                "validation_manifest": "validation",
                "vocab": "vocab",
                "hotword_table": "hotwords",
                "cases": "cases",
                "hotword_families": "families",
                "ctc_report": "ctc_report",
            }.items()
        },
    }
    (offline / "multi_nested_prompt_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    control = _validate_offline_control(
        offline_format="multi_nested_v3",
        paths=paths,
        threshold=0.86,
        top_k=5,
        retrieval_mode="forced_topk",
        maximum_edit_ratio=0.35,
        posterior_weight=0.25,
        minimum_posterior_confidence=0.0,
        minimum_top1_margin=0.0,
        prompt_template="Reference: {hotwords}",
        language="Portuguese",
        dtype="bfloat16",
        max_new_tokens=None,
    )
    assert control["validated_retrieval_config"]["threshold"] is None
    assert control["validated_retrieval_config"]["candidate_source"] == "ranked_matches[:5]"
