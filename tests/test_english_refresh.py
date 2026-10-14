from __future__ import annotations

import json
import shutil
import subprocess
import wave
import zipfile
from pathlib import Path

import pytest
import yaml

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.training import english_refresh as pipeline
from qwen_hotword.training.english_refresh_freeze import unique_candidates
from qwen_hotword.training.english_refresh_source import audio_path, stage_sources, transcript
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha

REPO = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def wav(path, salt, seconds=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes([salt, 0]) * 16000 * seconds)


def setup(root):
    cfg = json.loads((REPO / "configs/english_countries_refresh.workzone.json").read_text())
    countries = []
    for index, name in enumerate(("智利", "哥伦比亚", "墨西哥")):
        directory, old = root / "data" / name, root / "old" / name
        rows = []
        for n in range(4):
            relative = Path("audios/shards_000000000") / f"{index}_{n}.wav"
            wav(directory / relative, 1 + index * 10 + n)
            response = "language English<asr_text>Hello world!"
            rows.append(
                {
                    "audios": [str(old / relative)],
                    "response": response,
                    "language": "English",
                    "messages": [{"role": "assistant", "content": response}],
                }
            )
        write(directory / "train.json", rows)
        wav(directory / "audios/test.wav", 80 + index)
        # Test transcript deliberately lacks the required training format: it must never be read.
        write(
            directory / "test.json",
            [{"audios": [str(old / "audios/test.wav")], "response": "DO NOT USE TEST LABELS"}],
        )
        countries.append({"name": str(index), "directory": str(directory), "old_prefix": str(old)})
    cfg["countries"] = countries
    cfg["target_hours"] = 4 / 3600
    cfg["vocab"] = str(REPO / cfg["vocab"])
    cfg["g2p_model"] = str(root / "assets/g2p.zip")
    (root / "assets").mkdir()
    (root / "assets/g2p.zip").write_bytes(b"mock local G2P model, never loaded")
    cfg["g2p_model_sha256"] = _sha(root / "assets/g2p.zip")
    old = root / "old_freeze"
    old.mkdir()
    fixed, validation, prints = [], [], []
    for index, lang in enumerate(("en", "es", "pt")):
        path = root / f"old_audio/{lang}.wav"
        wav(path, 100 + index, seconds=4)
        row = {
            "id": f"old_{lang}",
            "audio_path": str(path),
            "language": lang,
            "duration_seconds": 4,
            "experiment": "full-ctc-v1",
            "dataset_version": "old",
            "split": "train",
            "balanced_language_bucket": lang,
            "release_source": "original_ready",
            "ctc_time_upsampling_factor": 2,
            "estimated_ctc_input_length": 50,
            "phoneme_token_ids": [1, 2],
            "text": "old",
        }
        fixed.append(row)
        (old / f"full_ctc_train_{lang}.jsonl").write_text(json.dumps(row) + "\n")
        prints.append({"audio_path": str(path), "sha256": _sha(path), "roles": ["train"]})
        val_path = root / f"old_audio/validation_{lang}.wav"
        wav(val_path, 110 + index)
        validation.append(
            {**row, "id": "validation_" + lang, "audio_path": str(val_path), "split": "validation"}
        )
        prints.append(
            {"audio_path": str(val_path), "sha256": _sha(val_path), "roles": ["validation"]}
        )
    (old / "full_ctc_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fixed))
    (old / "audio_fingerprints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prints))
    (root / "validation").mkdir()
    val = root / "validation/current.jsonl"
    val.write_text("".join(json.dumps(r) + "\n" for r in validation))
    write(
        old / "report.json",
        {
            "status": "completed",
            "training_manifest_ready": True,
            "artifacts": {"combined": {"sha256": _sha(old / "full_ctc_train.jsonl")}},
        },
    )
    seal(old)
    baseline = json.loads((REPO / "configs/480h_training.workzone.json").read_text())
    (root / "config").mkdir()
    zone = root / "config/old_zone.yaml"
    zone.write_text(
        yaml.safe_dump(
            {
                "paths": {"cache_dir": "old/cache", "output_dir": "old"},
                "model": {"path": "/fake/Qwen3-ASR-1.7B"},
            }
        )
    )
    baseline.update(
        freeze_dir=str(old),
        train_sha256=_sha(old / "full_ctc_train.jsonl"),
        validation_manifest=str(val),
        validation_sha256=_sha(val),
        validation_count=3,
        workzone_config=str(zone),
        baseline_validation_cache=str(root / "old_cache"),
        vocab=cfg["vocab"],
    )
    write(root / "config/baseline.json", baseline)
    cfg.update(
        baseline_training_config=str(root / "config/baseline.json"),
        fixed_freeze=str(old),
        fixed_languages={
            lang: {"sha256": _sha(old / f"full_ctc_train_{lang}.jsonl"), "records": 1}
            for lang in ("es", "pt")
        },
    )
    return cfg


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    cfg = setup(tmp_path)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "mock-mfa")
    calls = []

    def mfa(command, **kwargs):
        calls.append(command)
        words = Path(command[-3]).read_text().splitlines()
        Path(command[-1]).write_text(
            "".join(
                f"{w}\t{' '.join(['t'] * 30) if w == 'dense' else 't a'}\n"
                for w in words
                if w != "missing"
            )
        )

    monkeypatch.setattr(pipeline.subprocess, "run", mfa)
    return tmp_path, cfg, calls


def test_complete_pipeline_processes_all_before_selection_and_preserves_fixed_assets(
    prepared_inputs,
):
    root, cfg, calls = prepared_inputs
    before = {p: _sha(p) for p in (root / "old_freeze").iterdir() if p.is_file()}
    report = pipeline.prepare(root, cfg, root / "new", workers=2, jobs=2)
    assert report["status"] == "data_prepared"
    assert report["full_english"]["source_records"] == report["full_english"]["ready_records"] == 12
    assert report["selection"]["en"]["records"] < 12
    assert report["selection"]["en"]["hours"] >= cfg["target_hours"]
    frozen = Path(report["freeze_dir"])
    for lang in ("es", "pt"):
        assert (
            _sha(frozen / f"full_ctc_train_{lang}.jsonl") == cfg["fixed_languages"][lang]["sha256"]
        )
    training = json.loads((root / "new/training_config.json").read_text())
    baseline = json.loads((root / "config/baseline.json").read_text())
    assert training["training_options"] == baseline["training_options"]
    assert training["validation_manifest"] == baseline["validation_manifest"]
    assert not report["training_started"] and not report["test_labels_used"]
    assert all(_sha(p) == sha for p, sha in before.items())
    with zipfile.ZipFile(root / "new/return_bundle.zip") as bundle:
        assert "freeze_report.json" in bundle.namelist()
        assert not any(name.endswith(".wav") for name in bundle.namelist())
    with pytest.raises(FileExistsError):
        pipeline.prepare(root, cfg, root / "new", workers=2)
    resumed = pipeline.prepare(root, cfg, root / "new", resume=True, workers=2)
    assert resumed == report and len(calls) == 1
    write(Path(cfg["countries"][0]["directory"]) / "train.json", [])
    with pytest.raises(ValueError, match="inputs/code/configuration differ"):
        pipeline.prepare(root, cfg, root / "new", resume=True, workers=2)


def test_review_rows_heldout_and_fixed_train_duplicates_are_excluded_without_changing_sources(
    prepared_inputs,
):
    root, cfg, _ = prepared_inputs
    country = cfg["countries"][0]
    p = Path(country["directory"]) / "train.json"
    rows = json.loads(p.read_text())
    for n, text in ((0, "missing"), (1, "hello 25")):
        rows[n]["response"] = "language English<asr_text>" + text
        rows[n]["messages"][0]["content"] = rows[n]["response"]
    shutil.copyfile(root / "old_audio/es.wav", audio_path(rows[2], country))
    shutil.copyfile(Path(country["directory"]) / "audios/test.wav", audio_path(rows[3], country))
    write(p, rows)
    report = pipeline.prepare(root, cfg, root / "new", workers=2)
    assert report["full_english"]["source_records"] == 12
    assert report["full_english"]["review_records"] == 2
    assert report["excluded_english"]["fixed_es_pt_or_heldout_audio_overlap"] == 2
    review = list(_rows(root / "new/manifest/needs_review.jsonl"))
    assert {i["reason"] for r in review for i in r["issues"]} >= {
        "dictionary_missing",
        "unresolved_digit",
    }
    assert {"missing", "hello 25"} == {r["text"] for r in review}


def test_unsupported_source_contracts_are_not_silently_skipped(tmp_path):
    cfg = setup(tmp_path)
    p = Path(cfg["countries"][0]["directory"]) / "train.json"
    rows = json.loads(p.read_text())
    rows[0]["messages"][0]["content"] = "different reference"
    write(p, rows)
    with pytest.raises(ValueError, match="row 1.*disagree"):
        stage_sources(cfg, tmp_path / "stage")
    assert (
        transcript({"response": "language English<asr_text>Don't re-write it!"})
        == "Don't re-write it!"
    )
    with pytest.raises(ValueError):
        audio_path({"audios": ["/unmapped/wrong.wav"]}, cfg["countries"][0])


def test_insufficient_capacity_preserves_complete_processed_pool_and_never_repeats(prepared_inputs):
    root, cfg, _ = prepared_inputs
    cfg["target_hours"] = 480
    with pytest.raises(ValueError, match="insufficient eligible capacity"):
        pipeline.prepare(root, cfg, root / "new", workers=2)
    assert (root / "new/manifest/train_ready.jsonl").exists()
    assert not (root / "new/training_config.json").exists()
    assert len(list(_rows(root / "new/manifest/train_ready.jsonl"))) == 12


def test_consistent_duplicates_keep_one_and_conflicting_labels_quarantine_entire_group(tmp_path):
    wav(tmp_path / "a.wav", 1)
    shutil.copyfile(tmp_path / "a.wav", tmp_path / "b.wav")
    base = {
        "id": "a",
        "audio_path": str(tmp_path / "a.wav"),
        "duration_seconds": 3,
        "estimated_ctc_input_length": 38,
        "phoneme_token_ids": [1, 2],
        "ctc_minimum_input_length": 2,
        "source_corpus": "chile",
        "release_source": "original_ready",
        "country": "chile",
        "text": "hello",
    }
    pool = tmp_path / "pool.jsonl"
    second = {**base, "id": "b", "audio_path": str(tmp_path / "b.wav")}
    pool.write_text(json.dumps(base) + "\n" + json.dumps(second) + "\n")
    out = tmp_path / "consistent"
    out.mkdir()
    rows, excluded = unique_candidates(pool, out, set(), set(), 2, 20260923)
    assert len(rows) == 1 and excluded == {"duplicate_consistent_file_bytes": 1}
    second["text"] = "different"
    pool.write_text(json.dumps(base) + "\n" + json.dumps(second) + "\n")
    out = tmp_path / "conflicting"
    out.mkdir()
    rows, excluded = unique_candidates(pool, out, set(), set(), 2, 20260923)
    assert not rows and excluded == {"same_file_bytes_conflicting_labels_or_metadata": 2}


def test_temporal_recovery_uses_the_historical_two_x_policy(prepared_inputs):
    root, cfg, _ = prepared_inputs
    p = Path(cfg["countries"][0]["directory"]) / "train.json"
    rows = json.loads(p.read_text())
    rows[0]["response"] = "language English<asr_text>dense"
    rows[0]["messages"][0]["content"] = rows[0]["response"]
    write(p, rows)
    report = pipeline.prepare(root, cfg, root / "new", workers=2)
    assert report["full_english"]["review_records"] == 1
    released = list(_rows(Path(report["freeze_dir"]) / "released_english_pool.jsonl"))
    recovery = next(r for r in released if r["text"] == "dense")
    assert recovery["release_source"] == "temporal_2x_recovery"
    assert recovery["ctc_minimum_input_length"] > recovery["estimated_ctc_input_length"]
    assert recovery["effective_ctc_target_ratio"] <= 0.90


def test_mfa_interruption_preserves_failed_attempt_and_resume_creates_new_attempt(
    prepared_inputs, monkeypatch
):
    root, cfg, _ = prepared_inputs
    working_mfa = pipeline.subprocess.run
    failed_paths = []

    def failed_mfa(command, **kwargs):
        path = Path(command[-1])
        path.write_text("partial\tt\n")
        failed_paths.append(path)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(pipeline.subprocess, "run", failed_mfa)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.prepare(root, cfg, root / "new", workers=2)
    failed_sha = _sha(failed_paths[0])
    monkeypatch.setattr(pipeline.subprocess, "run", working_mfa)
    report = pipeline.prepare(root, cfg, root / "new", resume=True, workers=2)
    assert report["status"] == "data_prepared" and _sha(failed_paths[0]) == failed_sha
    assert len(list((root / "new").glob("mfa_attempt_*"))) == 2


def test_gpu_stage_delegates_to_old_training_policy_and_blocks_generated_config_drift(
    prepared_inputs, monkeypatch
):
    root, cfg, _ = prepared_inputs
    pipeline.prepare(root, cfg, root / "new", workers=2)
    monkeypatch.chdir(REPO)
    observed = []

    def stage(config, output, action, gpu):
        observed.append((config, output, action, gpu))
        return {"status": "mock_only"}

    monkeypatch.setattr(pipeline.historical, "run_stage", stage)
    assert pipeline.run_stage(root, root / "new", "smoke", "4")["status"] == "mock_only"
    assert observed[0][1:] == (root / "new/training", "smoke", "4")
    assert observed[0][0]["training_options"]["train-sampling-policy"] == "all_samples_once"
    config = root / "new/training_config.json"
    value = json.loads(config.read_text())
    value["encoder_batch_size"] = 99
    write(config, value)
    with pytest.raises(ValueError, match="configuration identity changed"):
        pipeline.run_stage(root, root / "new", "smoke", "4")
