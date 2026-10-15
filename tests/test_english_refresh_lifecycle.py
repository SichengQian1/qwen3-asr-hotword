"""Exercise production preparation/cache/Head contracts with synthetic Encoder features on CPU."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_english_refresh import prepared_inputs as prepared_inputs
from test_english_refresh_fast import legacy_contract
from test_english_refresh_gpu import forbid_audio_io

from qwen_hotword.training import ctc_overfit
from qwen_hotword.training import english_refresh as original
from qwen_hotword.training import english_refresh_fast as fast
from qwen_hotword.training import english_refresh_gpu as gpu
from qwen_hotword.training.spanish_capacity import _sha

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[1]
REAL_RUN = subprocess.run


@pytest.fixture
def lifecycle_inputs(prepared_inputs, monkeypatch):
    root, cfg, calls = prepared_inputs
    # Use the same relative fixed_freeze configuration as the deployed entry.
    cfg["fixed_freeze"] = str(Path(cfg["fixed_freeze"]).relative_to(root))

    def before_byte_scan(*args):
        raise RuntimeError("source labels completed; byte scan not started")

    monkeypatch.setattr(original, "freeze", before_byte_scan)
    with pytest.raises(RuntimeError, match="source labels completed"):
        original.prepare(root, cfg, root / "original", workers=2, jobs=2)
    return root, cfg, calls


def load_cli(name):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location("lifecycle_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_fast_outputs_through_smoke_cache_pilot_formal_and_cache_resume(
    lifecycle_inputs,
    monkeypatch,
):
    root, cfg, mfa_calls = lifecycle_inputs
    output = root / "fast"
    fast.prepare(root, cfg, root / "original", output)
    assert not (output / "frozen/audio_fingerprints.jsonl").exists()
    # Model the deployed completed preparation's exact known source-code identity.
    legacy_contract(output, "5d69e4a99650ec42469c87c274f1975c3a1e18c8d2fd3b6bc0590a5620cb83cf")
    before = {
        p: _sha(p)
        for folder in (output, root / "original")
        for p in folder.rglob("*")
        if p.is_file()
    }
    monkeypatch.chdir(root)
    forbid_audio_io(monkeypatch)
    cache_cli = load_cli("cache_full_training_features.py")
    head_cli = load_cli("train_full_ctc.py")
    monkeypatch.setattr(cache_cli, "load_asr_model", lambda _: SimpleNamespace())
    extracted = []

    def synthetic_encoder(records, wrapper, **kwargs):
        extracted.extend(r.sample_id for r in records)
        return (
            [
                ctc_overfit.CachedSample(
                    r.sample_id,
                    torch.zeros((8, 1024), dtype=torch.bfloat16),
                    r.token_ids,
                )
                for r in records
            ],
            1,
            0.0,
        )

    monkeypatch.setattr(ctc_overfit, "extract_frozen_features", synthetic_encoder)
    calls = []

    def child(command, *args, **kwargs):
        if len(command) < 3 or Path(command[2]).name not in {
            "cache_full_training_features.py",
            "train_full_ctc.py",
        }:
            return REAL_RUN(command, *args, **kwargs)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4" and kwargs["check"] is True
        calls.append(list(command))
        argv = command[2:]
        if Path(argv[0]).name == "train_full_ctc.py":
            index = argv.index("--device") + 1
            assert argv[index] == "cuda:0"
            argv[index] = "cpu"  # Only the hardware boundary changes for this local test.
            module = head_cli
        else:
            assert "--prevalidated-audio-paths" in argv
            module = cache_cli
        with monkeypatch.context() as child_context:
            child_context.setattr(sys, "argv", argv)
            assert module.main() == 0
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gpu.subprocess, "run", child)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert fast.run_stage(root, output, "smoke", "4")["status"] == "smoke_completed"
        plan_path = output / "training/run_plan.json"
        plan_sha = _sha(plan_path)
        plan = json.loads(plan_path.read_text())
        assert plan["protection_provenance"]["original_fingerprints"] == str(
            root / cfg["fixed_freeze"] / "audio_fingerprints.jsonl"
        )
        assert fast.run_stage(root, output, "cache", "4")["status"] == "cache_completed"
        cached = len(extracted)
        tensors = {p: _sha(p) for p in (output / "training/cache").rglob("*.pt")}
        pilot = fast.run_stage(root, output, "pilot", "4")
        assert pilot["status"] == "pilot_completed"
        assert pilot["head_report"]["epochs_completed"] == 5
        assert pilot["head_report"]["resumed_from_epoch"] == 0
        formal = fast.run_stage(root, output, "formal", "4")
        assert formal["status"] == "formal_completed"
        assert formal["head_report"]["resumed_from_epoch"] == 5
        assert 5 < formal["head_report"]["epochs_completed"] <= 30
        assert formal["head_report"]["cache_sha256_verified"] is True
        assert formal["head_report"]["test_set_used"] is False
        assert len(calls) == 5 and len(extracted) == cached
        # Reuse completed feature shards through the real production CLI/checks.
        assert fast.run_stage(root, output, "cache", "4")["status"] == "cache_completed"
        assert len(extracted) == cached
        assert all(_sha(path) == digest for path, digest in tensors.items())
        assert _sha(plan_path) == plan_sha
    finally:
        torch.set_num_threads(previous_threads)
    assert all(_sha(path) == digest for path, digest in before.items())
    assert len(mfa_calls) == 1
    assert not (output / "frozen/audio_fingerprints.jsonl").exists()


@pytest.mark.parametrize("role", ["original_validation", "new_test"])
def test_real_fast_protection_inputs_detect_training_overlap(lifecycle_inputs, monkeypatch, role):
    root, cfg, _ = lifecycle_inputs
    output = root / "fast"
    fast.prepare(root, cfg, root / "original", output)
    training = json.loads((output / "training_config_path_only.json").read_text())
    validation = {
        str(ctc_overfit.canonical_audio_path(r["audio_path"]))
        for r in gpu._rows(Path(training["validation_manifest"]))
    }
    if role == "original_validation":
        overlap = next(iter(validation))
    else:
        overlap = next(gpu._rows(root / "original/source/test_audio_paths.jsonl"))["audio_path"]
    frozen = json.loads((output / "frozen/report.json").read_text())
    monkeypatch.chdir(root)
    forbid_audio_io(monkeypatch)
    with pytest.raises(ValueError, match="overlaps protected"):
        gpu.verify_protected_paths(output / "training", {overlap}, validation, frozen, {})


@pytest.mark.parametrize("problem", ["original_fingerprints", "new_tests", "count", "validation"])
def test_protection_metadata_and_coverage_remain_strict(lifecycle_inputs, monkeypatch, problem):
    root, cfg, _ = lifecycle_inputs
    output = root / "fast"
    fast.prepare(root, cfg, root / "original", output)
    training = json.loads((output / "training_config_path_only.json").read_text())
    validation = {r["audio_path"] for r in gpu._rows(Path(training["validation_manifest"]))}
    frozen = json.loads((output / "frozen/report.json").read_text())
    if problem == "original_fingerprints":
        (root / cfg["fixed_freeze"] / "audio_fingerprints.jsonl").write_text("changed")
    elif problem == "new_tests":
        (root / "original/source/test_audio_paths.jsonl").write_text("changed")
    elif problem == "count":
        frozen["path_identity_audit"]["protected_paths"] += 1
    else:
        validation.add(str(root / "unknown.wav"))
    monkeypatch.chdir(root)
    forbid_audio_io(monkeypatch)
    with pytest.raises(ValueError):
        gpu.verify_protected_paths(output / "training", set(), validation, frozen, {})
