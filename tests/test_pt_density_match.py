from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.training.pt_density_evaluation import check_actual_lengths, verify_selection
from qwen_hotword.training.pt_density_match import (
    build_subset,
    density,
    optimal_pairs,
    read_validation,
)
from qwen_hotword.training.pt_mfa_pilot import sha256_file


def row(i: int, tokens: int, language: str = "pt") -> dict[str, Any]:
    return {
        "id": f"{language}_{i}",
        "audio_path": f"/unread/{language}_{i}.wav",
        "experiment": "full-ctc-v1",
        "split": "validation",
        "text": "unchanged",
        "language": language,
        "source_corpus": "source",
        "release_source": "original_ready",
        "phoneme_token_ids": [1] * tokens,
        "label_length": tokens,
        "estimated_ctc_input_length": 50,
        "effective_ctc_input_length": 100,
        "ctc_time_upsampling_factor": 2,
        "duration_seconds": float(i + 1),
        "balanced_language_bucket": language,
    }


def test_global_match_agrees_with_brute_force_and_is_prediction_independent() -> None:
    pt = [row(i, x) for i, x in enumerate([12, 24, 34, 47, 56, 62])]
    es = [row(i, x, "es") for i, x in enumerate([20, 27, 60])]
    result = optimal_pairs(pt, es, 42)
    cost = sum(abs(density(a) - density(b)) for a, b in result)
    expected = min(
        sum(abs(density(a) - density(b)) for a, b in zip(es, subset, strict=True))
        for subset in itertools.combinations(pt, len(es))
    )
    assert cost == pytest.approx(expected)
    assert len({b["id"] for _, b in result}) == len(es)
    ids = [(a["id"], b["id"]) for a, b in result]
    for r in pt:
        r["per"] = 0.99
        r["prediction"] = "never inspected"
    again = optimal_pairs(pt[::-1], es[::-1], 42)
    assert [(a["id"], b["id"]) for a, b in again] == ids


def fixture(tmp: Path) -> dict[str, Any]:
    pt = [row(i, x) for i, x in enumerate([10, 20, 30, 40, 50, 60])]
    ref = [row(i, x, "es") for i, x in enumerate([20, 30, 40])] + [pt[0]]
    for name, rows in (("pt.jsonl", pt), ("ref.jsonl", ref)):
        (tmp / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {
        "pt_manifest": "pt.jsonl",
        "pt_sha256": sha256_file(tmp / "pt.jsonl"),
        "reference_manifest": "ref.jsonl",
        "reference_sha256": sha256_file(tmp / "ref.jsonl"),
        "output_dir": "selection",
        "seed": 42,
        "max_ks": 0.05,
        "max_p95_pair_gap": 0.025,
        "max_pair_gap": 0.1,
    }


def test_build_preserves_inputs_labels_and_freezes_verified_ids(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    before = (tmp_path / "pt.jsonl").read_bytes()
    report = build_subset(config, tmp_path)
    assert report["status"] == "matched"
    assert report["quality"]["ks"] == 0
    assert report["portuguese_selected"]["samples"] == 3
    assert report["prediction_used"] is False
    root, selected = verify_selection(config, tmp_path)
    pool = {r["id"]: r for r in read_validation(tmp_path / "pt.jsonl", config["pt_sha256"])}
    assert all(pool[r["id"]] == r for r in selected)
    assert before == (tmp_path / "pt.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build_subset(config, tmp_path)
    (root / "selected_ids.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_selection(config, tmp_path)


def test_unmatched_tails_are_reported_and_cannot_be_evaluated(tmp_path: Path) -> None:
    config = fixture(tmp_path)
    path = tmp_path / "ref.jsonl"
    rows = [row(i, 90, "es") for i in range(3)] + [row(7, 10)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    config["reference_sha256"] = sha256_file(path)
    report = build_subset(config, tmp_path)
    assert report["status"] == "insufficient_density_match"
    assert report["spanish_reference"]["samples"] == 3  # Never drop ES tails.
    with pytest.raises(ValueError, match="pass frozen thresholds"):
        verify_selection(config, tmp_path)


@pytest.mark.parametrize("change", ["split", "frames", "length", "duplicate", "sha"])
def test_invalid_source_rejected(tmp_path: Path, change: str) -> None:
    r = row(0, 20)
    if change == "split":
        r["split"] = "test"
    elif change == "frames":
        r["effective_ctc_input_length"] = 50
    elif change == "length":
        r["label_length"] = 99
    path = tmp_path / "bad.jsonl"
    path.write_text((json.dumps(r) + "\n") * (2 if change == "duplicate" else 1))
    with pytest.raises(ValueError):
        read_validation(path, "bad" if change == "sha" else sha256_file(path))


def test_feature_lengths_cannot_silently_change_density(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen_hotword.training import sharded_ctc

    r = row(0, 20)
    sample = SimpleNamespace(
        sample_id=r["id"],
        token_ids=tuple(r["phoneme_token_ids"]),
        hidden_states=SimpleNamespace(shape=(50, 1024)),
    )
    monkeypatch.setattr(sharded_ctc, "load_feature_shard", lambda *a, **k: [sample])
    cache = SimpleNamespace(shards=["mock"], ctc_time_upsampling_factor=2)
    check_actual_lengths(cache, [r], 90)
    sample.hidden_states.shape = (49, 1024)
    with pytest.raises(ValueError, match="Actual feature length"):
        check_actual_lengths(cache, [r], 90)


def test_tied_densities_use_distinct_samples_and_insufficient_pool_fails() -> None:
    pt = [row(i, 30) for i in range(6)]
    es = [row(i, 30, "es") for i in range(3)]
    a = optimal_pairs(pt, es, 8)
    b = optimal_pairs(pt[::-1], es, 8)
    assert a == b
    assert len({r["id"] for _, r in a}) == 3
    with pytest.raises(ValueError):
        optimal_pairs(pt[:2], es, 8)


@pytest.mark.parametrize("actual_match", [True, False])
def test_evaluation_uses_same_subset_three_heads_and_preserves_legacy(
    actual_match: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from qwen_hotword import config as config_module
    from qwen_hotword.modeling import qwen_backbone
    from qwen_hotword.phonemes import coverage
    from qwen_hotword.training import (
        ctc_diagnostics,
        ctc_overfit,
        feature_cache,
        pt_density_evaluation,
        sharded_ctc,
    )

    config = fixture(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (model / name).write_text("{}")
    cache_root = tmp_path / "reference_cache"
    cache_root.mkdir()
    (cache_root / "cache_config.json").write_text(
        json.dumps(
            {
                "model": {
                    "config_sha256": sha256_file(model / "config.json"),
                    "weight_index_sha256": sha256_file(model / "model.safetensors.index.json"),
                    "dtype": "bfloat16",
                }
            }
        )
    )
    (tmp_path / "head.pt").write_bytes(b"synthetic only")
    config.update(
        {
            "checkpoints": dict.fromkeys(
                ["multilingual_baseline", "portuguese_old", "failed_recovery_ablation"], "head.pt"
            ),
            "checkpoint_sha256": {},
            "reference_cache": "reference_cache",
            "vocab": "vocab",
            "model_config": "model_config",
        }
    )
    build_subset(config, tmp_path)
    before = (tmp_path / "ref.jsonl").read_bytes()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)),
    )
    monkeypatch.setattr(
        config_module,
        "load_workzone_config",
        lambda *a, **k: SimpleNamespace(
            model=SimpleNamespace(
                device="cuda:0", local_files_only=True, path=model, dtype="bfloat16"
            )
        ),
    )
    monkeypatch.setattr(
        coverage, "load_phoneme_vocab", lambda *a: SimpleNamespace(tokens=range(90))
    )
    monkeypatch.setattr(pt_density_evaluation, "validate_heads", lambda *a: None)

    def actual_rows(cache: Any, rows: list[dict[str, Any]], classes: int) -> Any:
        scale = 2 if not actual_match and all(r["language"] == "pt" for r in rows) else 1
        return [
            dict(r, effective_ctc_input_length=r["effective_ctc_input_length"] * scale)
            for r in rows
        ], {"status": "consistent"}

    monkeypatch.setattr(pt_density_evaluation, "validated_actual_rows", actual_rows)
    monkeypatch.setattr(
        sharded_ctc,
        "load_disk_feature_cache",
        lambda *a, **k: SimpleNamespace(ctc_time_upsampling_factor=2),
    )
    monkeypatch.setattr(ctc_overfit, "load_experiment_records", lambda *a, **k: ["record"])
    monkeypatch.setattr(qwen_backbone, "load_asr_model", lambda *a: object())
    monkeypatch.setattr(
        feature_cache,
        "cache_feature_split",
        lambda *a, **k: SimpleNamespace(to_dict=lambda: {"sample_count": 3}),
    )
    calls = []

    def diagnose(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["selected_sample_ids"])
        return {
            "head_config": {"time_upsampling_factor": 2},
            "validation": {"sample_count": len(kwargs["selected_sample_ids"])},
            "validation_by_dimension": {},
        }

    monkeypatch.setattr(ctc_diagnostics, "diagnose_ctc_checkpoint", diagnose)
    result = pt_density_evaluation.evaluate_subset(config, tmp_path)
    if not actual_match:
        assert result["status"] == "insufficient_actual_density_match"
        assert calls == []
        assert result["head_evaluation_started"] is False
        assert (Path(result["output_dir"]) / "sha256.txt").is_file()
        return
    assert result["status"] == "completed"
    assert result["actual_density"]["status"] == "matched"
    assert calls[0] == calls[3] == calls[5] == {"pt_1", "pt_2", "pt_3"}
    assert calls[1] == calls[4] == calls[6] == {"pt_0"}
    assert calls[2] == {"es_0", "es_1", "es_2"}
    assert not result["training_started"] and not result["external_asr_used"]
    assert before == (tmp_path / "ref.jsonl").read_bytes()
    output = Path(result["output_dir"])
    assert (output / "sha256.txt").is_file()


def test_cache_audit_distinguishes_all_frame_and_label_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen_hotword.training import sharded_ctc
    from qwen_hotword.training.pt_density_evaluation import audit_cache_lengths

    rows = [row(i, 20, "pt" if i < 2 else "es") for i in range(4)]
    samples = [
        SimpleNamespace(
            sample_id=r["id"],
            token_ids=tuple(r["phoneme_token_ids"]),
            hidden_states=SimpleNamespace(shape=(50, 1024)),
        )
        for r in rows[:3]
    ]
    samples[0].hidden_states.shape = (49, 1024)
    samples[1].token_ids = (2,) * 20  # Same count, different reference; must still detect it.
    samples[2].hidden_states.shape = (51, 1024)
    monkeypatch.setattr(sharded_ctc, "load_feature_shard", lambda *a, **k: samples)
    report = audit_cache_lengths(
        SimpleNamespace(shards=["mock"], ctc_time_upsampling_factor=2), rows, 90
    )
    assert report["mismatch_counts"] == {
        "pt::frame_mismatch": 1,
        "pt::label_mismatch": 1,
        "es::frame_mismatch": 1,
    }
    assert report["frame_delta_counts"] == {-2: 1, 2: 1}
    assert report["examples"][0]["labels_equal"] is True
    assert report["examples"][1]["labels_equal"] is False
    assert report["missing_count"] == 1


def test_actual_density_rows_preserve_labels_and_reject_label_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qwen_hotword.training import sharded_ctc
    from qwen_hotword.training.pt_density_evaluation import validated_actual_rows

    r = row(0, 20)
    before = json.dumps(r, sort_keys=True)
    sample = SimpleNamespace(
        sample_id=r["id"],
        token_ids=tuple(r["phoneme_token_ids"]),
        hidden_states=SimpleNamespace(shape=(49, 1024)),
    )
    monkeypatch.setattr(sharded_ctc, "load_feature_shard", lambda *a, **k: [sample])
    cache = SimpleNamespace(shards=["mock"], ctc_time_upsampling_factor=2)
    actual, audit = validated_actual_rows(cache, [r], 90)
    assert actual[0]["effective_ctc_input_length"] == 98
    assert actual[0]["phoneme_token_ids"] == r["phoneme_token_ids"]
    assert audit["frame_delta_counts"] == {-2: 1}
    assert json.dumps(r, sort_keys=True) == before
    sample.token_ids = (2,) * 20
    with pytest.raises(ValueError, match="label/ID identity mismatch"):
        validated_actual_rows(cache, [r], 90)
    sample.sample_id = "missing"
    with pytest.raises(ValueError, match="label/ID identity mismatch"):
        validated_actual_rows(cache, [r], 90)
