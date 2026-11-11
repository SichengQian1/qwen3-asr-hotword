from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training import en_ar_backbone_refresh as refresh
from scripts import run_en_ar_merged as run
from scripts.run_en_ar_training import NATIVE_VOCAB, VOCAB

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    tensors = pytest.importorskip("safetensors.numpy")
    np = pytest.importorskip("numpy")
    model = tmp_path / "checkpoint-5100-merged"
    model.mkdir()
    tensors.save_file(
        {
            "thinker.audio_tower.ln_post.weight": np.ones(4, dtype="float32"),
            "thinker.audio_tower.layers.0.self_attn.q_proj.weight": np.eye(2, dtype="float32"),
            "thinker.audio_tower.proj1.weight": np.eye(2, dtype="float32"),
            "thinker.audio_tower.proj2.weight": np.eye(2, dtype="float32"),
        },
        str(model / "model.safetensors"),
    )
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        write_json(model / name, {"model_type": "qwen3_asr"})
    architecture = {"model_type": "qwen3_asr", "ctc_tap_dimension": 1024}
    (tmp_path / "architecture.yaml").write_text(yaml.safe_dump({"architecture": architecture}))
    (tmp_path / "cache.yaml").write_text((ROOT / "configs/480h_cache.workzone.yaml").read_text())
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("run_en_ar_merged.py", "run_en_ar_training.py"):
        (scripts / name).write_text("# paired-run code fixture\n")
    config = tmp_path / "config.json"
    write_json(
        config,
        {
            "model": str(model),
            "baseline_english": "base-en",
            "baseline_arabic": "base-ar",
            "cache_config": "cache.yaml",
            "architecture_manifest": "architecture.yaml",
            "train_hours_per_language": 0.0002,
            "minimum_gpu_free_gib": 40,
        },
    )
    for language, vocab in (("en", VOCAB), ("ar", NATIVE_VOCAB)):
        directory = tmp_path / f"base-{language}"
        directory.mkdir()
        report: dict[str, Any] = {
            "status": "labels_complete",
            "languages": [language],
            "vocab_sha256": sha(vocab),
            "train_target_hours_per_language": 0.0002,
            "validation_target_hours": {language: 0.0002},
            "rows_by_group": {},
        }
        for split in ("train", "validation"):
            name = f"{language}-{split}"
            row = {
                "id": name,
                "audio_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "audio_path": str(tmp_path / f"missing-{name}.wav"),
                "text": name,
                "phoneme_token_ids": [4, 17] if language == "en" else [74, 72],
                "ctc_minimum_input_length": 2,
                "label_length": 2,
                "duration_seconds": 2.0,
                "ctc_time_upsampling_factor": 2,
                "language": language,
                "split": split,
                "experiment": "full-ctc-v1",
            }
            manifest = directory / f"{split}.jsonl"
            manifest.write_text(json.dumps(row) + "\n")
            report[f"{split}_manifest_sha256"] = sha(manifest)
            report["rows_by_group"][f"{split}/{language}"] = 1
        write_json(directory / "label_report.json", report)
    monkeypatch.setattr(refresh, "inspect_local_config", lambda _config: object())
    monkeypatch.setattr(refresh, "inspection_dict", lambda _value: architecture.copy())
    return config, tmp_path / "new-run"


def prepare(config: Path, output: Path) -> dict[str, Any]:
    return refresh.prepare(
        config.parent, config, output, legacy_vocab=VOCAB, native_vocab=NATIVE_VOCAB
    )


def test_new_encoder_run_preserves_exact_data_and_original_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    before = {str(p): sha(p) for p in (tmp_path / "checkpoint-5100-merged").iterdir()}
    plan = prepare(config, output)
    assert plan["base_hidden_states_reused"] is False
    assert len(plan["manifests"]) == 4
    for entry in plan["manifests"].values():
        assert Path(entry["source"]).read_bytes() == Path(entry["path"]).read_bytes()
    view = Path(plan["model"]["view"])
    assert view.name == "Qwen3-ASR-1.7B"
    assert (view / "model.safetensors").is_symlink()
    index = json.loads((view / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) == {"model.safetensors"}
    assert "thinker.audio_tower.ln_post.weight" in index["weight_map"]
    assert yaml.safe_load((output / "cache.workzone.yaml").read_text())["model"]["path"] == str(
        view
    )
    assert before == {str(p): sha(p) for p in (tmp_path / "checkpoint-5100-merged").iterdir()}
    assert not (output / "english/cache").exists()
    assert not (output / "cache").exists()
    assert refresh.verify_plan(output)["model"] == plan["model"]
    (tmp_path / "checkpoint-5100-merged/model.safetensors").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="model file changed"):
        refresh.verify_plan(output)


def test_pending_baseline_and_cross_split_overlap_stop_before_model_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    ar_report = tmp_path / "base-ar/label_report.json"
    saved = ar_report.read_bytes()
    ar_report.unlink()
    with pytest.raises(ValueError, match="not sealed yet"):
        prepare(config, output)
    assert not output.exists()
    ar_report.write_bytes(saved)
    english = json.loads((tmp_path / "base-en/train.jsonl").read_text())
    ar_manifest = tmp_path / "base-ar/validation.jsonl"
    row = json.loads(ar_manifest.read_text())
    row["audio_sha256"] = english["audio_sha256"]
    ar_manifest.write_text(json.dumps(row) + "\n")
    report = json.loads(ar_report.read_text())
    report["validation_manifest_sha256"] = sha(ar_manifest)
    write_json(ar_report, report)
    with (
        patch.object(refresh, "make_model_view") as model,
        pytest.raises(ValueError, match="overlapping"),
    ):
        prepare(config, output)
    model.assert_not_called()


def test_new_cache_command_uses_new_model_and_original_language_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    plan = prepare(config, output)
    with (
        patch.object(run, "wait_for_gpu"),
        patch.object(run, "verify_cache_backbone"),
        patch.object(subprocess, "run") as command,
    ):
        run.cache_language(output, plan, "en", 4, 40)
    args = command.call_args.args[0]
    assert args[args.index("--config") + 1] == str(output / "cache.workzone.yaml")
    assert args[args.index("--train-manifest") + 1] == str(output / "english/train.jsonl")
    assert args[args.index("--vocab") + 1] == str(VOCAB)
    assert command.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"


def test_resume_rejects_changed_training_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    prepare(config, output)
    (tmp_path / "scripts/run_en_ar_training.py").write_text("# changed training semantics\n")
    with pytest.raises(ValueError, match="missing/changed pinned input"):
        refresh.verify_plan(output)


def test_wrong_tap_architecture_is_rejected_before_feature_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(refresh, "inspection_dict", lambda _value: {"ctc_tap_dimension": 999})
    with pytest.raises(ValueError, match="architecture differs"):
        prepare(config, output)
    assert not (output / "plan.json").exists()
    assert not (output / "cache").exists()


def test_completed_base_cache_cannot_be_reused_for_merged_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    plan = prepare(config, output)
    cache = output / "english/cache"
    write_json(cache / "feature_cache_report.json", {"status": "pass"})
    write_json(cache / "train/cache_config.json", {"model": {"path": "old-base-model"}})
    with (
        patch.object(subprocess, "run") as command,
        pytest.raises(ValueError, match="different model"),
    ):
        run.cache_language(output, plan, "en", 4, 40)
    command.assert_not_called()


@pytest.mark.parametrize("resume", [False, True])
def test_run_recaches_both_languages_then_trains_with_new_english_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    config, output = fixture(tmp_path, monkeypatch)
    plan = prepare(config, output)
    head = output / "head"
    head.mkdir()
    if resume:
        (head / "training_state_latest.pt").write_bytes(b"resume test state")
    events = []

    def cache(
        _output: Path, _plan: dict[str, Any], language: str, gpu: int, minimum: float
    ) -> None:
        events.append(language)

    def train(*args: Any, **kwargs: Any) -> None:
        assert events == ["en", "ar"]
        assert args == (output, 4, output / "english")
        assert kwargs == {"vocab_path": NATIVE_VOCAB, "resume": resume}
        write_json(head / "report.json", {"status": "completed"})
        write_json(
            output / "cache_sources.json", {"reused_english_output": str(output / "english")}
        )

    with (
        patch.object(run, "cache_language", side_effect=cache),
        patch.object(run, "wait_for_gpu"),
        patch.object(run, "train_stage", autospec=True, side_effect=train),
    ):
        result = run.execute(output, plan, 4, 40)
    assert result["status"] == "completed"
    assert result["baseline_hidden_states_reused"] is False
    assert result["english_encoder_recomputed"] and result["arabic_encoder_recomputed"]


def test_gpu_gate_waits_without_loading_a_model() -> None:
    with (
        patch.object(
            subprocess,
            "run",
            side_effect=[SimpleNamespace(stdout="1000"), SimpleNamespace(stdout="100000")],
        ),
        patch.object(time, "sleep") as sleep,
    ):
        run.wait_for_gpu(4, 40)
    sleep.assert_called_once_with(5)


def test_running_baseline_manifest_seal_is_waited_for_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = fixture(tmp_path, monkeypatch)
    path = tmp_path / "base-ar/label_report.json"
    saved = path.read_bytes()
    path.unlink()
    with (
        patch.object(run, "ROOT", tmp_path),
        patch.object(time, "sleep", side_effect=lambda _seconds: path.write_bytes(saved)) as sleep,
        patch.object(subprocess, "run") as gpu,
    ):
        run.wait_for_manifests(config)
    sleep.assert_called_once_with(5)
    gpu.assert_not_called()


def test_failed_baseline_selection_is_not_waited_for_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = fixture(tmp_path, monkeypatch)
    write_json(tmp_path / "base-ar/label_report.json", {"status": "insufficient_labeled_hours"})
    with (
        patch.object(run, "ROOT", tmp_path),
        pytest.raises(ValueError, match="did not complete successfully"),
    ):
        run.wait_for_manifests(config)
