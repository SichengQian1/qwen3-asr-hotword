"""Real prepare/cache/smoke/train lifecycle, with only MFA/Encoder/GPU boundaries replaced."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_english_refresh import prepared_inputs as prepared_inputs
from test_english_refresh_gpu import forbid_audio_io

from qwen_hotword.training import ctc_diagnostics, ctc_overfit, sharded_ctc
from qwen_hotword.training import english_newval as prep
from qwen_hotword.training import english_newval_cache as cache
from qwen_hotword.training import english_newval_run as run
from qwen_hotword.training import english_refresh as original
from qwen_hotword.training import english_refresh_fast as fast
from qwen_hotword.training import english_refresh_gpu as gpu
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha

torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False))


@pytest.fixture
def source_run(prepared_inputs, monkeypatch, request):
    root, cfg, mfa_calls = prepared_inputs
    monkeypatch.chdir(root)
    # A 9s English target accommodates all three tiny country sources (3s per row).
    baseline_path = Path(cfg["baseline_training_config"])
    baseline = json.loads(baseline_path.read_text())
    val_path = Path(baseline["validation_manifest"])
    rows = list(_rows(val_path))
    rows[0]["duration_seconds"] = 9
    val_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    baseline["validation_sha256"] = _sha(val_path)
    write(baseline_path, baseline)
    old_cache_config = root / "old_cache/cache_config.json"
    old_cfg = json.loads(old_cache_config.read_text())
    old_cfg["source_manifest"]["sha256"] = _sha(val_path)
    write(old_cache_config, old_cfg)
    if getattr(request, "param", None) == "test_overlap":
        country = cfg["countries"][0]
        path = Path(country["directory"]) / "train.json"
        source_rows = json.loads(path.read_text())
        source_rows[0]["audios"] = [str(Path(country["old_prefix"]) / "audios/test.wav")]
        write(path, source_rows)

    def stop(*args):
        raise RuntimeError("completed labels; no byte scan")

    monkeypatch.setattr(original, "freeze", stop)
    with pytest.raises(RuntimeError, match="completed labels"):
        original.prepare(root, cfg, root / "original", workers=2, jobs=2)
    source = root / "fast"
    fast.prepare(root, cfg, root / "original", source)
    training = json.loads((source / "training_config_path_only.json").read_text())
    gpu.prepare(training, source / "training")
    extracted = []

    def synthetic(records, wrapper, **kwargs):
        extracted.extend(r.sample_id for r in records)
        return (
            [
                ctc_overfit.CachedSample(
                    r.sample_id,
                    torch.full((8, 1024), len(r.sample_id) / 100, dtype=torch.bfloat16),
                    r.token_ids,
                )
                for r in records
            ],
            317477504,
            0.0,
        )

    for split, manifest in (
        ("train", Path(training["freeze_dir"]) / "full_ctc_train.jsonl"),
        ("validation", Path(training["validation_manifest"])),
    ):
        cache_feature_split(
            cache.records(manifest, split),
            None,
            source / "training/cache" / split,
            split=split,
            source_manifest_path=manifest,
            model_path=root / "model/Qwen3-ASR-1.7B",
            model_dtype="bfloat16",
            vocab_path=training["vocab"],
            extractor=synthetic,
        )
    new = json.loads((REPO / "configs/english_countries_newval.workzone.json").read_text())
    new.update(
        source_run=str(source),
        output=str(root / "newval"),
        source_run_plan_sha256=_sha(source / "training/run_plan.json"),
        source_train_sha256=training["train_sha256"],
        source_validation_sha256=training["validation_sha256"],
    )
    extracted.clear()
    return root, new, extracted, synthetic, mfa_calls


def cli(name):
    spec = importlib.util.spec_from_file_location(name[:-3], REPO / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_unused_pool_cache_smoke_fresh_40_epoch_training_resume_and_old_control(
    source_run,
    monkeypatch,
):
    root, cfg, extracted, synthetic, mfa_calls = source_run
    output = Path(cfg["output"])
    source = Path(cfg["source_run"])
    previous = {p: _sha(p) for p in source.rglob("*") if p.is_file()}
    forbid_audio_io(monkeypatch)
    result = prep.prepare(root, cfg, output)
    assert result["status"] == "new_validation_prepared"
    assert result["validation_target_english_hours"] == 9 / 3600
    assert result["selected_english"]["records"] == 3
    assert result["validation"]["by_language"] == {"en": 3, "es": 1, "pt": 1}
    assert result["audio_byte_audit_performed"] is False
    plan = prep.verify(output)
    train = list(_rows(Path(plan["train_manifest"])))
    english = list(_rows(output / "validation_en.jsonl"))
    assert not {r["id"] for r in train} & {r["id"] for r in english}
    assert not {r["audio_path"] for r in train} & {r["audio_path"] for r in english}
    assert {r["country"] for r in english} == {"0", "1", "2"}
    assert all(r["source_split"] == "train" for r in english)
    assert len(mfa_calls) == 1
    assert output.joinpath("legacy_es_pt_rows.jsonl").read_text() == "".join(
        line
        for line in Path(plan["legacy_validation_manifest"]).read_text().splitlines(True)
        if json.loads(line)["balanced_language_bucket"] in {"es", "pt"}
    )
    loads = []
    monkeypatch.setattr(
        cache, "load_asr_model", lambda model: loads.append(model) or SimpleNamespace()
    )
    monkeypatch.setattr(ctc_overfit, "extract_frozen_features", synthetic)
    head_cli = cli("train_full_ctc.py")
    calls = []
    real_diagnose = ctc_diagnostics.diagnose_ctc_checkpoint

    def cpu_diagnose(*args, **kwargs):
        kwargs["device"] = "cpu"
        return real_diagnose(*args, **kwargs)

    monkeypatch.setattr(ctc_diagnostics, "diagnose_ctc_checkpoint", cpu_diagnose)

    def child(command, **kwargs):
        assert command[2] == "scripts/train_full_ctc.py"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "4"
        assert "--initial-head-checkpoint" not in command
        assert "--skip-cache-sha256-verification" not in command
        calls.append(list(command))
        argv = command[2:]
        argv[argv.index("--device") + 1] = "cpu"
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", argv)
            code = head_cli.main()
            if code:
                raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(run.subprocess, "run", child)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert run.execute(root, cfg, output, "cache", "4")["status"] == "cache_completed"
        assert set(extracted) == {r["id"] for r in english} and len(loads) == 1
        run.execute(root, cfg, output, "smoke", "4")
        assert len(loads) == 1 and len(extracted) == 3
        # Preserve a genuine saved epoch/optimizer state, then interrupt before the report.
        real_history = sharded_ctc._write_metric_history

        def interrupt(path, history):
            real_history(path, history)
            raise InterruptedError("local test interruption after saved epoch")

        with monkeypatch.context() as context:
            context.setattr(sharded_ctc, "_write_metric_history", interrupt)
            with pytest.raises(subprocess.CalledProcessError):
                run.execute(root, cfg, output, "train", "4")
        assert (output / "head/training_state_latest.pt").exists()
        assert not (output / "training_completed.json").exists()
        with pytest.raises(FileExistsError, match="resume"):
            run.execute(root, cfg, output, "train", "4")
        real_save = run.save

        def after_completed_report(path, value):
            if path.name == "training_completed.json":
                raise InterruptedError("completed report before orchestration marker")
            return real_save(path, value)

        with monkeypatch.context() as context:
            context.setattr(run, "save", after_completed_report)
            with pytest.raises(InterruptedError, match="completed report"):
                run.execute(root, cfg, output, "run", "4", resume=True)
        assert (output / "head/report.json").exists()
        assert not (output / "training_completed.json").exists()
        finished = run.execute(root, cfg, output, "run", "4", resume=True)
        report = finished["head_report"]
        assert report["epochs_requested"] == 40
        assert report["resumed_from_epoch"] == 1
        assert report["minimum_learning_rate"] == 5e-6
        assert report["early_stopping_min_delta"] == 0.0005
        assert report["early_stopping_patience"] == 6 and report["minimum_epochs"] == 5
        assert report["cache_sha256_verified"] is True and report["test_set_used"] is False
        assert not (output / "pilot").exists()
        assert [c[c.index("--epochs") + 1] for c in calls] == ["1", "40", "40"]
        assert "--resume" not in calls[1] and "--resume" in calls[2]
        assert Path(finished["return_file"]).exists()
        control = json.loads((output / "old_domain_control.json").read_text())
        assert control["diagnostics"]["validation_by_group"]["en"]["sample_count"] == 1
        assert control["test_set_used"] is False
        assert len(loads) == 1 and len(extracted) == 3
        run.execute(root, cfg, output, "run", "4", resume=True)
        assert len(calls) == 3 and len(loads) == 1
    finally:
        torch.set_num_threads(previous_threads)
    # Repacked ES/PT tensors and labels equal the original cached values exactly.
    old_cache = cache.open_cache(
        Path(plan["legacy_validation_cache"]),
        "validation",
        Path(plan["legacy_validation_manifest"]),
        Path(plan["vocab"]),
    )
    combined = cache.open_cache(
        output / "cache/validation",
        "validation",
        output / "full_ctc_validation.jsonl",
        Path(plan["vocab"]),
    )
    old_values = {
        s.sample_id: s
        for d in old_cache.shards
        for s in sharded_ctc.load_feature_shard(d, num_classes=90)
    }
    new_values = {
        s.sample_id: s
        for d in combined.shards
        for s in sharded_ctc.load_feature_shard(d, num_classes=90)
    }
    for name in ("validation_es", "validation_pt"):
        assert torch.equal(old_values[name].hidden_states, new_values[name].hidden_states)
        assert old_values[name].token_ids == new_values[name].token_ids
    assert "validation_en" not in new_values
    assert all(_sha(path) == digest for path, digest in previous.items())
    assert not (output / "cache/train").exists()


def test_prepare_preserves_sources_refuses_existing_and_pins_manifests(source_run, monkeypatch):
    root, cfg, _, _, _ = source_run
    output = Path(cfg["output"])
    forbid_audio_io(monkeypatch)
    prep.prepare(root, cfg, output)
    with pytest.raises(FileExistsError, match="NEW"):
        prep.prepare(root, cfg, output)
    with pytest.raises(ValueError, match="separate"):
        prep.prepare(root, cfg, Path(cfg["source_run"]) / "nested")
    (output / "validation_en.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        prep.verify(output)


def test_source_identity_and_insufficient_unused_capacity_fail_before_publish(source_run):
    root, cfg, _, _, _ = source_run
    output = Path(cfg["output"])
    with pytest.raises(ValueError, match="identity"):
        prep.prepare(root, {**cfg, "source_train_sha256": "bad"}, output)
    assert not output.exists()
    # A fixed stratifier failure must not leave a reusable prepared directory.
    with pytest.MonkeyPatch.context() as context:

        def fail(*args):
            raise ValueError("insufficient eligible capacity")

        context.setattr(prep, "stratified_select", fail)
        with pytest.raises(ValueError, match="insufficient"):
            prep.prepare(root, cfg, output)
    assert not output.exists()


def test_modified_source_tensor_rejected_before_copy(source_run, monkeypatch):
    root, cfg, _, synthetic, _ = source_run
    output = Path(cfg["output"])
    prep.prepare(root, cfg, output)
    monkeypatch.setattr(cache, "load_asr_model", lambda _: SimpleNamespace())
    monkeypatch.setattr(ctc_overfit, "extract_frozen_features", synthetic)
    tensor = Path(cfg["source_run"]) / "training/cache/validation/shards/shard-000000.pt"
    data = bytearray(tensor.read_bytes())
    data[-20] ^= 1
    tensor.write_bytes(data)
    with pytest.raises(ValueError, match="identity"):
        cache.build_cache(output, prep.verify(output))


@pytest.mark.parametrize("source_run", ["test_overlap"], indirect=True)
def test_qualified_but_test_overlapping_audio_never_enters_new_validation(source_run):
    root, cfg, _, _, _ = source_run
    output = Path(cfg["output"])
    prep.prepare(root, cfg, output)
    source_tests = set(
        r["audio_path"] for r in _rows(root / "original/source/test_audio_paths.jsonl")
    )
    source_qualified = list(_rows(Path(cfg["source_run"]) / "frozen/released_english_pool.jsonl"))
    assert any(r["audio_path"] in source_tests for r in source_qualified)
    assert not source_tests & {r["audio_path"] for r in _rows(output / "validation_en.jsonl")}
    selected = list(_rows(output / "validation_en.jsonl"))
    candidate_ids = {r["id"] for r in source_qualified}
    assert {r["id"] for r in selected} <= candidate_ids


def test_english_cache_resume_avoids_second_encoder_load_and_rejects_cache_tampering(
    source_run,
    monkeypatch,
):
    root, cfg, extracted, synthetic, _ = source_run
    output = Path(cfg["output"])
    prep.prepare(root, cfg, output)
    plan = prep.verify(output)
    loads = []
    monkeypatch.setattr(cache, "load_asr_model", lambda _: loads.append(True) or SimpleNamespace())
    monkeypatch.setattr(ctc_overfit, "extract_frozen_features", synthetic)

    def interrupt(*args):
        raise InterruptedError("English complete before repack")

    with monkeypatch.context() as context:
        context.setattr(cache, "materialize", interrupt)
        with pytest.raises(InterruptedError, match="English complete"):
            cache.build_cache(output, plan)
    assert len(extracted) == 3 and len(loads) == 1
    cache.build_cache(output, plan)
    cache.build_cache(output, plan)
    assert len(extracted) == 3 and len(loads) == 1
    tensor = output / "cache/validation/shards/shard-000000.pt"
    tensor.write_bytes(tensor.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        cache.build_cache(output, plan)


def test_completed_preparation_rejects_different_training_configuration(source_run):
    root, cfg, _, _, _ = source_run
    output = Path(cfg["output"])
    prep.prepare(root, cfg, output)
    with pytest.raises(ValueError, match="configuration changed"):
        run.execute(root, {**cfg, "epochs": 30}, output, "cache", "4")


def test_actual_prepare_cli_and_small_bundle(source_run, monkeypatch):
    import zipfile

    root, cfg, _, _, _ = source_run
    config = root / "new_config.json"
    write(config, cfg)
    module = cli("run_english_newval.py")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_english_newval.py", "prepare", "--root", str(root), "--config", str(config)],
    )
    assert module.main() == 0
    output = Path(cfg["output"])
    with zipfile.ZipFile(output / "prepare_bundle.zip") as archive:
        assert set(archive.namelist()) == {"report.json", "plan.json", "sha256.txt"}
        for line in archive.read("sha256.txt").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert digest == _sha(output / name)
