from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.en_egy_dynamic import read
from scripts import run_en_egy_head as paired


def fixture_run(root: Path) -> Path:
    source = root / "baseline"
    source.mkdir()
    vocab = root / "vocab.json"
    write_json(vocab, {"fixture": True})
    head = {
        "status": "completed",
        "head_config": {
            "head_type": "temporal_upsample",
            "time_upsampling_factor": 2,
            "hidden_dimension": 512,
            "kernel_size": 5,
            "dropout": 0.1,
        },
        "train_sampling_policy": "all_samples_once",
        "minimum_epochs": 5,
        "early_stopping_patience": 6,
        "early_stopping_min_delta": 0.0005,
        "early_stopping_metric": "validation_macro_per",
        "checkpoint_selection_metric": "validation_macro_per",
        "train_batch_size": 256,
        "weight_decay": 0.0001,
        "max_gradient_norm": 5.0,
        "scheduler_patience": 2,
        "scheduler_factor": 0.5,
        "minimum_learning_rate": 5e-6,
        "seed": 20260720,
        "epochs_requested": 40,
        "initial_learning_rate": 0.0003,
    }
    write_json(source / "head/report.json", head)
    best = source / "head/ctc_head_best.pt"
    best.write_bytes(b"source head")
    manifests = {}
    for split in ("train", "validation"):
        manifest = source / f"{split}.jsonl"
        manifest.write_text("fixture manifest")
        manifests[split] = {"path": str(manifest), "sha256": sha(manifest)}
        folder = source / "cache" / split
        write_json(
            folder / "cache_config.json",
            {
                "model": {
                    "path": "r15-view",
                    "dtype": "bfloat16",
                    "config_sha256": "cfg",
                    "weight_index_sha256": "index",
                }
            },
        )
        write_json(folder / "cache_index.json", {"fixture": True})
        write_json(folder / "cache_summary.json", {"fixture": True})
    prepared = {"arabic_msa_used": False, "manifests": manifests, "vocab_sha256": sha(vocab)}
    write_json(source / "prepare_report.json", prepared)
    write_json(source / "contract.json", {"configuration": {"vocab": "vocab.json"}})
    write_json(
        source / "training_contract.json",
        {
            "prepare_report_sha256": sha(source / "prepare_report.json"),
            "encoder": {
                "view": "r15-view",
                "index_sha256": "index",
                "files": {
                    "model.safetensors": {"sha256": "r15-weights"},
                    "config.json": {"sha256": "cfg"},
                },
            },
        },
    )
    write_json(
        source / "report.json",
        {
            "status": "completed",
            "head": head,
            "selection": prepared,
            "test_set_used": False,
            "vocab_sha256": sha(vocab),
            "checkpoint_sha256": sha(best),
            "model_weight_sha256": "r15-weights",
        },
    )
    for name in (
        "scripts/arabic_diagnostic_common.py",
        "src/qwen_hotword/training/sharded_ctc.py",
        "src/qwen_hotword/modeling/ctc_head.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture")
    return source


def test_new_egy_comparison_reuses_new_cache_and_only_changes_factor(tmp_path, monkeypatch):
    source = fixture_run(tmp_path)
    output = tmp_path / "up3"
    original, _ = paired.source_contract(tmp_path, source, 2)
    up3, _ = paired.source_contract(tmp_path, source, 3)
    assert up3["training_options"] == {
        **original["training_options"],
        "head_time_upsampling_factor": 3,
    }
    caches = Mock(
        side_effect=lambda *a, **k: SimpleNamespace(
            shards=[SimpleNamespace(sample_ids=("training",))]
        )
    )
    monkeypatch.setattr(paired, "load_disk_feature_cache", caches)
    monkeypatch.setattr(paired, "load_validation_sample_groups", lambda *a, **k: {"val": "ar"})
    monkeypatch.setattr(paired, "load_phoneme_vocab", lambda *a: "vocab")
    monkeypatch.setattr(paired, "wait_for_gpu", Mock())
    invocations = []

    def train(*args, **kwargs):
        invocations.append(kwargs)
        assert kwargs["head_time_upsampling_factor"] == 3
        assert kwargs["epochs"] == 40
        assert kwargs["minimum_learning_rate"] == 5e-6
        assert kwargs["early_stopping_min_delta"] == 0.0005
        head = args[3]
        (head / "training_state_latest.pt").write_bytes(b"state")
        if len(invocations) == 1:
            raise RuntimeError("simulated interruption")
        (head / "ctc_head_best.pt").write_bytes(b"trained")
        write_json(head / "report.json", {"status": "completed"})

    monkeypatch.setattr(paired, "train_sharded_ctc_head", train)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        paired.run(tmp_path, source, output, 5)
    assert not invocations[0]["resume"]
    report = paired.run(tmp_path, source, output, 5)
    assert invocations[1]["resume"]
    assert report["encoder_recomputed"] is False
    assert report["source_run"] == str(source)
    assert report["physical_gpu"] == 5
    assert paired.run(tmp_path, source, output, 5) == report
    assert len(invocations) == 2
    for call in caches.call_args_list:
        assert call.args[0].is_relative_to(source / "cache")
        assert call.kwargs["verify_sha256"] is True
    # Frozen original baseline is unchanged; old x3 directories are never consulted.
    assert (source / "head/ctc_head_best.pt").read_bytes() == b"source head"


@pytest.mark.parametrize("change", ["not_completed", "msa", "wrong_encoder", "changed_manifest"])
def test_source_must_be_completed_new_egy_run(tmp_path, change):
    source = fixture_run(tmp_path)
    report = read(source / "report.json")
    if change == "not_completed":
        report["status"] = "running"
    elif change == "msa":
        report["selection"]["arabic_msa_used"] = True
        write_json(source / "prepare_report.json", report["selection"])
    elif change == "wrong_encoder":
        write_json(source / "cache/train/cache_config.json", {"model": {"path": "base"}})
    else:
        (source / "train.jsonl").write_text("modified")
    write_json(source / "report.json", report)
    with pytest.raises(ValueError):
        paired.source_contract(tmp_path, source, 3)


def test_separate_output_and_unknown_partial_directory_rejected(tmp_path):
    source = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        paired.run(tmp_path, source, source / "up3", 5)
    output = tmp_path / "unknown"
    output.mkdir()
    with pytest.raises(ValueError, match="unsealed"):
        paired.run(tmp_path, source, output, 5)
