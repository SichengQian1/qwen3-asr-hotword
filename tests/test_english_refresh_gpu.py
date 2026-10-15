from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_multilingual_480_run import fixture as historical_fixture
from test_multilingual_480_run import smoke_report

from qwen_hotword.training import english_refresh_gpu as gpu
from qwen_hotword.training import multilingual_480_run as historical
from qwen_hotword.training.ctc_generalization import validate_disjoint_records
from qwen_hotword.training.ctc_overfit import extract_frozen_features, load_experiment_records
from qwen_hotword.training.stage_progress import stage_progress


@pytest.fixture
def prepared(tmp_path):
    cfg, output = historical_fixture(tmp_path)
    cfg["audio_isolation_policy"] = gpu.POLICY
    freeze = Path(cfg["freeze_dir"])
    report_path = freeze / "report.json"
    report = json.loads(report_path.read_text())
    report.update(audio_isolation_policy=gpu.POLICY, path_identity_audit={"status": "pass"})
    report_path.write_text(json.dumps(report))
    (freeze / "sha256.txt").write_text(
        "".join(
            f"{gpu._sha(p)}  {p.name}\n" for p in sorted(freeze.iterdir()) if p.name != "sha256.txt"
        )
    )
    return cfg, output


def forbid_audio_io(monkeypatch):
    for method in ("resolve", "stat", "lstat", "is_file", "open", "readlink"):
        original = getattr(Path, method)

        def checked(path, *args, _original=original, **kwargs):
            if path.suffix.lower() in {".wav", ".flac"}:
                raise AssertionError("audio filesystem I/O during metadata preflight")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, checked)


def test_metadata_preflight_matches_historical_plan_without_touching_audio(prepared, monkeypatch):
    cfg, output = prepared
    legacy = output.parent / "legacy"
    expected = historical.prepare(cfg, legacy)
    before = {p: gpu._sha(p) for p in Path(cfg["freeze_dir"]).iterdir() if p.is_file()}
    forbid_audio_io(monkeypatch)
    # This reproduces the old slow path even though no WAV contents were being read.
    with pytest.raises(AssertionError, match="audio filesystem"):
        historical.scan_manifest(
            Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl", "train", per_group=16
        )
    plan = gpu.prepare(cfg, output)
    for key in (
        "config",
        "train",
        "validation",
        "estimated_feature_bytes",
        "recommended_free_bytes",
    ):
        assert plan[key] == expected[key]
    for split in ("train", "validation"):
        name = f"smoke_{split}.jsonl"
        assert (output / name).read_bytes() == (legacy / name).read_bytes()
    for path, digest in before.items():
        assert gpu._sha(path) == digest
    assert plan["test_set_used"] is False and plan["model_loaded"] is False
    with pytest.raises(FileExistsError):
        gpu.prepare(cfg, output)


def test_smoke_sampling_cap_and_full_scan_counts(prepared, monkeypatch, capsys):
    cfg, _ = prepared
    path = Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl"
    rows = list(gpu._rows(path))
    expanded = []
    for row in rows:
        for number in range(18):
            expanded.append(
                {
                    **row,
                    "id": f"{row['id']}-{number}",
                    "audio_path": row["audio_path"].replace(".wav", f"-{number}.wav"),
                }
            )
    path.write_text("".join(json.dumps(row) + "\n" for row in expanded))
    forbid_audio_io(monkeypatch)
    stats, smoke, paths, ids = gpu.scan_manifest(path, "train", per_group=16, expected=108)
    assert len(smoke) == 96 and stats["records"] == len(paths) == len(ids) == 108
    assert set(stats["smoke_groups"].values()) == {16}
    assert "108/108 (100.0%)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "problem", ["count", "duplicate", "relative", "parent", "duration", "factor"]
)
def test_metadata_scan_retains_preflight_rejections(prepared, problem):
    cfg, _ = prepared
    path = Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl"
    rows = list(gpu._rows(path))
    expected = 6
    if problem == "count":
        expected = 7
    elif problem == "duplicate":
        rows[1]["audio_path"] = rows[0]["audio_path"]
    elif problem in {"relative", "parent"}:
        rows[0]["audio_path"] = "relative.wav" if problem == "relative" else "/data/a/../b.wav"
    elif problem == "duration":
        rows[0]["duration_seconds"] = float("nan")
    else:
        rows[0]["ctc_time_upsampling_factor"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        gpu.scan_manifest(path, "train", per_group=16, expected=expected)


def test_interrupted_scan_does_not_publish_partial_training_directory(prepared, monkeypatch):
    cfg, output = prepared

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(gpu, "scan_manifest", interrupted)
    with pytest.raises(KeyboardInterrupt):
        gpu.prepare(cfg, output)
    assert not output.exists()


def test_prevalidated_loader_keeps_labels_and_overlap_checks_without_audio_io(
    tmp_path, monkeypatch
):
    train, val = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    row = {
        "id": "train",
        "audio_path": str(tmp_path / "missing.wav"),
        "experiment": "full-ctc-v1",
        "split": "train",
        "text": "hello",
        "language": "en",
        "phoneme_token_ids": [1, 2],
        "label_length": 2,
        "ctc_minimum_input_length": 2,
        "ctc_time_upsampling_factor": 2,
    }
    train.write_text(json.dumps(row) + "\n")
    val.write_text(json.dumps({**row, "id": "validation", "split": "validation"}) + "\n")
    with pytest.raises(FileNotFoundError, match="audio file"):
        load_experiment_records(train, num_classes=90, expected_experiment="full-ctc-v1")
    forbid_audio_io(monkeypatch)
    progress = []
    records = load_experiment_records(
        train,
        num_classes=90,
        expected_experiment="full-ctc-v1",
        prevalidated_audio_paths=True,
        progress=progress.append,
    )
    validation = load_experiment_records(
        val,
        num_classes=90,
        expected_experiment="full-ctc-v1",
        expected_split="validation",
        prevalidated_audio_paths=True,
    )
    assert progress == [1]
    with pytest.raises(ValueError, match="share audio paths"):
        validate_disjoint_records(records, validation, prevalidated_audio_paths=True)
    train.write_text(json.dumps({**row, "phoneme_token_ids": [0, 2]}) + "\n")
    with pytest.raises(ValueError, match="blank or out-of-range"):
        load_experiment_records(
            train, num_classes=90, expected_experiment="full-ctc-v1", prevalidated_audio_paths=True
        )


def test_missing_audio_still_fails_when_encoder_actually_reads_it(tmp_path, monkeypatch):
    from qwen_hotword.training.ctc_overfit import ExperimentRecord

    def missing(path, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(load=missing))
    record = ExperimentRecord("missing", tmp_path / "missing.wav", "hello", "en", (1, 2), 2, 2)
    tower = SimpleNamespace(eval=lambda: None, parameters=lambda: [])
    wrapper = SimpleNamespace(model=SimpleNamespace(thinker=SimpleNamespace(audio_tower=tower)))
    with pytest.raises(FileNotFoundError):
        extract_frozen_features([record], wrapper, encoder_batch_size=8)


def test_stages_use_same_head_options_and_preserve_existing_plan(prepared, monkeypatch):
    cfg, output = prepared
    # Existing legacy-generated plans are valid; the fix does not rewrite their identity.
    plan = historical.prepare(cfg, output)
    before = gpu._sha(output / "run_plan.json")
    calls = []

    def child(cmd, *, env, check):
        assert env["CUDA_VISIBLE_DEVICES"] == "4" and check
        calls.append(cmd)
        if cmd[2].endswith("train_full_ctc.py"):
            assert cmd == historical.commands(cfg, output, "smoke")[1]
            smoke_report(output)
        else:
            assert "--prevalidated-audio-paths" in cmd
            assert cmd[cmd.index("--expected-train-samples") + 1] == "6"
            if str(output / "cache") in cmd:
                p = output / "cache/feature_cache_report.json"
                p.parent.mkdir()
                p.write_text(
                    json.dumps(
                        {
                            "status": "pass",
                            "train": {"sample_count": 6},
                            "validation": {"sample_count": 6},
                        }
                    )
                )

    monkeypatch.setattr(gpu.subprocess, "run", child)
    monkeypatch.setattr(gpu.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**4))
    assert gpu.run_stage(cfg, output, "smoke", "4")["status"] == "smoke_completed"
    assert gpu.run_stage(cfg, output, "cache", "4")["status"] == "cache_completed"
    assert len(calls) == 3 and gpu._sha(output / "run_plan.json") == before
    assert json.loads((output / "run_plan.json").read_text()) == plan
    cfg["encoder_batch_size"] = 16
    with pytest.raises(ValueError, match="configuration changed"):
        gpu.run_stage(cfg, output, "cache", "4")


def test_cache_still_requires_smoke_disk_space_and_pinned_inputs(prepared, monkeypatch):
    cfg, output = prepared
    gpu.prepare(cfg, output)
    with pytest.raises(FileNotFoundError):
        gpu.run_stage(cfg, output, "cache", "4")
    smoke_report(output)
    monkeypatch.setattr(gpu.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="disk space"):
        gpu.run_stage(cfg, output, "cache", "4")
    (output / "smoke_train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        gpu.run_stage(cfg, output, "smoke", "4")


def test_progress_shows_counts_and_interruption_without_completion_claim(capsys):
    with pytest.raises(KeyboardInterrupt), stage_progress("scan", total=10, every=2) as update:
        update(2)
        raise KeyboardInterrupt
    text = capsys.readouterr().out
    assert "2/10 (20.0%)" in text and "elapsed=" in text
    assert "interrupted or failed" in text and "100.0%" not in text


def test_cache_cli_defers_audio_checks_and_exposes_model_load(prepared, monkeypatch, capsys):
    cfg, output = prepared
    script_path = Path(__file__).resolve().parents[1] / "scripts/cache_full_training_features.py"
    spec = importlib.util.spec_from_file_location("cache_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_asr_model", lambda _: SimpleNamespace())

    def cache(records, wrapper, dest, *, split, **kwargs):
        return SimpleNamespace(to_dict=lambda: {"status": "pass", "sample_count": len(records)})

    monkeypatch.setattr(module, "cache_feature_split", cache)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--config",
            cfg["workzone_config"],
            "--vocab",
            cfg["vocab"],
            "--train-manifest",
            str(Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl"),
            "--validation-manifest",
            cfg["validation_manifest"],
            "--output-dir",
            str(output),
            "--prevalidated-audio-paths",
            "--expected-train-samples",
            "6",
            "--expected-validation-samples",
            "6",
        ],
    )
    # Complete the fixture labels expected by the real production loader.
    for path in (
        Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl",
        Path(cfg["validation_manifest"]),
    ):
        rows = list(gpu._rows(path))
        for row in rows:
            row.update(label_length=2, ctc_minimum_input_length=2)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    forbid_audio_io(monkeypatch)
    assert module.main() == 0
    assert "Load Qwen model onto GPU" in capsys.readouterr().out
    assert json.loads((output / "feature_cache_report.json").read_text())["status"] == "pass"
