from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from qwen_hotword.training import multilingual_480_run as run

REPO = Path(__file__).resolve().parents[1]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def fixture(tmp_path):
    cfg = json.loads((REPO / "configs/480h_training.workzone.json").read_text())
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    train, val = [], []
    for lang in ("en", "es", "pt"):
        for split in ("train", "validation"):
            for release in ("original_ready", "temporal_2x_recovery"):
                identifier = f"{split}-{lang}-{release}"
                row = {
                    "id": identifier,
                    "audio_path": str(tmp_path / f"{identifier}.wav"),
                    "balanced_language_bucket": lang,
                    "language": lang,
                    "split": split,
                    "experiment": "full-ctc-v1",
                    "release_source": release,
                    "ctc_time_upsampling_factor": 2,
                    "estimated_ctc_input_length": 20,
                    "duration_seconds": 1.5,
                    "phoneme_token_ids": [1, 2],
                    "text": "foo",
                }
                (train if split == "train" else val).append(row)
    for name, rows in (("full_ctc_train.jsonl", train), ("validation.jsonl", val)):
        (freeze / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg.update(
        freeze_dir=str(freeze),
        train_sha256=run._sha(freeze / "full_ctc_train.jsonl"),
        train_counts={k: 2 for k in ("en", "es", "pt")},
        validation_manifest=str(freeze / "validation.jsonl"),
        validation_sha256=run._sha(freeze / "validation.jsonl"),
        validation_count=6,
        vocab=str(REPO / cfg["vocab"]),
    )
    write_json(
        freeze / "report.json",
        {
            "status": "completed",
            "training_manifest_ready": True,
            "audio_identity_audit": {"conflict_groups_by_reason": {}},
            "artifacts": {"combined": {"sha256": cfg["train_sha256"]}},
        },
    )
    (freeze / "audio_fingerprints.jsonl").write_text(
        "".join(
            json.dumps({"audio_path": r["audio_path"], "roles": ["validation"]}) + "\n" for r in val
        )
    )
    (freeze / "sha256.txt").write_text(
        "".join(f"{run._sha(p)}  {p.name}\n" for p in sorted(freeze.iterdir()))
    )
    model = tmp_path / "Qwen3-ASR-1.7B"
    for name in ("config.json", "model.safetensors.index.json"):
        write_json(model / name, {})
    zone = yaml.safe_load((REPO / "configs/480h_cache.workzone.yaml").read_text())
    zone["model"]["path"] = str(model)
    zone_path = tmp_path / "zone.yaml"
    zone_path.write_text(yaml.safe_dump(zone))
    cfg["workzone_config"] = str(zone_path)
    cfg["baseline_validation_cache"] = str(tmp_path / "old_cache")
    write_json(
        tmp_path / "old_cache/cache_config.json",
        {
            "source_manifest": {"sha256": cfg["validation_sha256"]},
            "vocab": {"sha256": cfg["vocab_sha256"]},
            "ctc_time_upsampling_factor": 2,
            "tap_module": "thinker.audio_tower.ln_post",
            "model": {
                "path": str(model),
                "dtype": "bfloat16",
                "config_sha256": run._sha(model / "config.json"),
                "weight_index_sha256": run._sha(model / "model.safetensors.index.json"),
            },
        },
    )
    return cfg, tmp_path / "run"


def test_prepare_pins_data_preserves_inputs_and_keeps_both_releases(tmp_path):
    cfg, out = fixture(tmp_path)
    original = run._sha(Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl")
    plan = run.prepare(cfg, out)
    assert plan["train"]["records"] == plan["validation"]["records"] == 6
    assert len(plan["train"]["smoke_groups"]) == 6
    assert plan["estimated_feature_bytes"] == 12 * 20 * 2048
    assert not plan["test_set_used"] and not plan["model_loaded"]
    assert run._sha(Path(cfg["freeze_dir"]) / "full_ctc_train.jsonl") == original
    with pytest.raises(FileExistsError):
        run.prepare(cfg, out)


@pytest.mark.parametrize("problem", ["model", "train", "validation", "protection", "overlap"])
def test_prepare_rejects_changed_identity_or_unprotected_validation(tmp_path, problem):
    cfg, out = fixture(tmp_path)
    if problem == "model":
        (tmp_path / "Qwen3-ASR-1.7B/config.json").write_text('{"changed":true}')
    elif problem in {"train", "validation"}:
        cfg[f"{problem}_sha256"] = "0" * 64
    else:
        freeze = Path(cfg["freeze_dir"])
        if problem == "protection":
            (freeze / "audio_fingerprints.jsonl").write_text("")
        else:
            val = list(run._rows(Path(cfg["validation_manifest"])))
            val[0]["audio_path"] = next(run._rows(freeze / "full_ctc_train.jsonl"))["audio_path"]
            Path(cfg["validation_manifest"]).write_text("".join(json.dumps(r) + "\n" for r in val))
            cfg["validation_sha256"] = run._sha(Path(cfg["validation_manifest"]))
            p = tmp_path / "old_cache/cache_config.json"
            cache = json.loads(p.read_text())
            cache["source_manifest"]["sha256"] = cfg["validation_sha256"]
            write_json(p, cache)
        (freeze / "sha256.txt").write_text(
            "".join(
                f"{run._sha(p)}  {p.name}\n"
                for p in sorted(freeze.iterdir())
                if p.name != "sha256.txt"
            )
        )
    with pytest.raises(ValueError):
        run.prepare(cfg, out)
    assert not out.exists()


def test_commands_keep_full_data_and_separate_smoke_from_formal(tmp_path):
    cfg, out = fixture(tmp_path)
    smoke = run.commands(cfg, out, "smoke")
    assert len(smoke) == 2 and "smoke_head" in " ".join(smoke[1])
    pilot = run.commands(cfg, out, "pilot")[0]
    formal = run.commands(cfg, out, "formal")[0]
    assert pilot[pilot.index("--epochs") + 1] == "5"
    assert formal[formal.index("--epochs") + 1] == "30" and formal[-1] == "--resume"
    assert pilot[pilot.index("--train-sampling-policy") + 1] == "all_samples_once"
    assert "--skip-cache-sha256-verification" not in pilot
    assert len(run.commands(cfg, out, "cache")) == 1
    with pytest.raises(ValueError, match="separate step"):
        run.run_stage(cfg, out, "formal", "3")


def smoke_report(out):
    write_json(
        out / "smoke_head/report.json",
        {
            "status": "completed",
            "epochs_completed": 1,
            "final_validation_loss": 1.0,
            "final_validation_phoneme_error_rate": 0.6,
        },
    )


def test_stage_requires_smoke_and_keeps_single_gpu_resume_policy(tmp_path, monkeypatch):
    cfg, out = fixture(tmp_path)
    run.prepare(cfg, out)
    with pytest.raises(FileNotFoundError):
        run.run_stage(cfg, out, "cache", "3")
    (out / "smoke_head").mkdir()
    (out / "smoke_head/training_state_latest.pt").write_bytes(b"stub")
    calls = []

    def child(cmd, *, env, check):
        calls.append(cmd)
        assert env["CUDA_VISIBLE_DEVICES"] == "3" and check
        if cmd[2].endswith("train_full_ctc.py"):
            assert "--resume" in cmd
            smoke_report(out)
        elif str(out / "cache") in cmd:
            write_json(
                out / "cache/feature_cache_report.json",
                {
                    "status": "pass",
                    "train": {"sample_count": 6},
                    "validation": {"sample_count": 6},
                },
            )

    monkeypatch.setattr(run.subprocess, "run", child)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=1024**4))
    assert run.run_stage(cfg, out, "smoke", "3")["status"] == "smoke_completed"
    assert len(calls) == 2
    assert run.run_stage(cfg, out, "cache", "3")["status"] == "cache_completed"
    assert len(calls) == 3
    assert all("--epochs" not in cmd for cmd in calls if cmd[2].endswith("features.py"))
    cfg["encoder_batch_size"] = 16
    with pytest.raises(ValueError, match="configuration changed"):
        run.run_stage(cfg, out, "cache", "3")


def test_cache_rejects_low_disk_or_changed_smoke_manifest(tmp_path, monkeypatch):
    cfg, out = fixture(tmp_path)
    run.prepare(cfg, out)
    smoke_report(out)
    monkeypatch.setattr(run.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="disk space"):
        run.run_stage(cfg, out, "cache", "3")
    (out / "smoke_train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        run.run_stage(cfg, out, "smoke", "3")


def test_inputs_changed_during_child_run_block_completion(tmp_path, monkeypatch):
    cfg, out = fixture(tmp_path)
    run.prepare(cfg, out)

    def child(cmd, **kwargs):
        if cmd[2].endswith("train_full_ctc.py"):
            smoke_report(out)
            Path(cfg["validation_manifest"]).write_text("changed during execution")

    monkeypatch.setattr(run.subprocess, "run", child)
    with pytest.raises(ValueError, match="SHA256"):
        run.run_stage(cfg, out, "smoke", "3")
