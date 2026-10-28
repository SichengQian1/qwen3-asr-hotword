from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_hotword.training.noise_recovery import (
    HISTORICAL_CODE,
    RECOVERY,
    audit_all,
    audited_report,
    encoder_frames,
    frame_bounds,
    processor_probe,
    recovered_views,
    verify_source,
)
from qwen_hotword.training.sharded_ctc import load_feature_shard
from qwen_hotword.training.spanish_capacity import _sha

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from qwen_hotword.modeling.ctc_head import build_ctc_head, ctc_head_config
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab
    from qwen_hotword.training.ctc_overfit import (
        CachedSample,
        EpochMetrics,
        ExperimentRecord,
        save_ctc_head_checkpoint,
    )
    from qwen_hotword.training.feature_cache import cache_feature_split

    torch.set_num_threads(1)
    run = tmp_path / "run"
    run.mkdir()
    vocab = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    baseline = build_ctc_head(
        head_type="temporal_upsample",
        input_dimension=1024,
        num_classes=90,
        hidden_dimension=4,
        kernel_size=3,
        dropout=0.1,
    )
    head_path = tmp_path / "baseline.pt"
    save_ctc_head_checkpoint(
        head_path, baseline, load_phoneme_vocab(vocab), EpochMetrics(27, 1, 1, 1, 1), 11
    )
    cfg = {
        "seed": 123,
        "initial_head_sha256": _sha(head_path),
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 3e-5},
    }
    plan = {
        "configuration": cfg,
        "parts": [],
        "input_sha256": {},
        "encoder_batch_size": 2,
        "vocab": str(vocab),
        "head_config": ctc_head_config(baseline),
        "initial_head": str(head_path),
    }
    for split in ("train", "validation"):
        base_rows, noise_rows = [], []
        for i in range(2):
            path = tmp_path / f"{split}-{i}.wav"
            sf.write(path, np.sin(np.arange(108944) * 0.03) * 0.2, 16000)
            row = {
                "id": f"{split}-{i}",
                "audio_path": str(path),
                "text": "a",
                "language": "en",
                "balanced_language_bucket": "en",
                "phoneme_token_ids": [2, 3, 2],
                "label_length": 3,
                "ctc_minimum_input_length": 3,
                "ctc_time_upsampling_factor": 2,
                "experiment": "full-ctc-v1",
                "split": split,
                "selected_30": i == 0,
            }
            base_rows.append(row)
            noise_rows.append(
                {**row, "id": ("noise10db::" if split == "validation" else "") + row["id"]}
            )
        base_manifest = tmp_path / f"{split}.jsonl"
        folder = run / "noise" / split / "part-000000"
        folder.mkdir(parents=True)
        noise_manifest = folder / "manifest.jsonl"
        for manifest, rows, frames, cache_path in (
            (base_manifest, base_rows, 89, tmp_path / "cache" / split),
            (noise_manifest, noise_rows, 88, folder / "features"),
        ):
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            records = [
                ExperimentRecord(r["id"], Path(r["audio_path"]), "a", "en", (2, 3, 2), 3, 2)
                for r in rows
            ]

            def extract(
                batch: list, wrapper: object, frames: int = frames, **kwargs: object
            ) -> tuple:
                return (
                    [
                        CachedSample(r.sample_id, torch.ones(frames, 1024), r.token_ids)
                        for r in batch
                    ],
                    317477504,
                    0.01,
                )

            cache_feature_split(
                records,
                None,
                cache_path,
                split=split,
                source_manifest_path=manifest,
                model_path=model,
                model_dtype="bfloat16",
                vocab_path=vocab,
                samples_per_shard=2,
                encoder_batch_size=2,
                extractor=extract,
            )
        plan[f"{split}_cache"] = str(tmp_path / "cache" / split)
        plan[f"{split}_manifest"] = str(base_manifest)
        (folder / "mix_audit.json").write_text(
            json.dumps(
                [
                    {
                        "id": r["id"],
                        "source_samples": 108944,
                        "snr_db": 10.0,
                        "original_audio_sha256": _sha(Path(r["audio_path"])),
                    }
                    for r in noise_rows
                ]
            )
        )
        plan["parts"].append(
            {
                "split": split,
                "base_shard": 0,
                "records": 2,
                "manifest": noise_manifest.relative_to(run).as_posix(),
                "sha256": _sha(noise_manifest),
            }
        )
    (run / "noise_inventory.json").write_text("{}")
    plan["noise_inventory_sha256"] = _sha(run / "noise_inventory.json")
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "prepared.json").write_text(json.dumps({"plan_sha256": _sha(run / "plan.json")}))
    marker = {
        "status": "noise_cache_completed",
        "parts": 2,
        "mixing": {},
        "sha256": {str(p): _sha(p) for p in (run / "noise").rglob("*") if p.is_file()},
    }
    (run / "cache_completed.json").write_text(json.dumps(marker))
    return run, plan, cfg


def snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(p): _sha(p) for root in paths for p in root.rglob("*") if p.is_file()}


def test_real_processor_and_qwen_lengths_reproduce_reported_boundary() -> None:
    pytest.importorskip("transformers")
    probe = processor_probe()
    assert probe["mel_frames_alone_with_longer"] == [680, 681]
    assert probe["qwen_frames_alone_with_longer"] == [88, 89]
    assert frame_bounds(108944) == (88, 89)
    assert frame_bounds(16000) == (13, 13)
    assert encoder_frames(200) == 26


def test_full_audit_preserves_inputs_and_strict_default(tmp_path: Path) -> None:
    run, plan, _ = fixture(tmp_path)
    before = snapshot([run, tmp_path / "cache"])
    report = audit_all(run, plan)
    assert report["parts_checked"] == 2
    assert report["counts"]["train_records"] == 2
    assert report["counts"]["train_selected_30"] == 1
    assert len(report["frame_pairs"]) == 4
    assert snapshot([run, tmp_path / "cache"]) == before
    views30 = recovered_views(run, plan, 30, report)
    views50 = recovered_views(run, plan, 50, report)
    a, b = views30[0].shards[0], views50[0].shards[0]
    assert len(a.replacement_frame_pairs) == 1 and len(b.replacement_frame_pairs) == 2
    assert [s.hidden_states.shape[0] for s in load_feature_shard(a, num_classes=90)] == [88, 89]
    assert [s.hidden_states.shape[0] for s in load_feature_shard(b, num_classes=90)] == [88, 88]
    with pytest.raises(ValueError, match="changes labels or Encoder"):
        load_feature_shard(replace(a, replacement_frame_pairs=()), num_classes=90)
    with pytest.raises(ValueError, match="audited pair"):
        load_feature_shard(
            replace(a, replacement_frame_pairs=(("train-0", 88, 89),)), num_classes=90
        )
    with pytest.raises(ValueError, match="invalid audited"):
        load_feature_shard(
            replace(a, replacement_frame_pairs=(("train-0", 89, 87),)), num_classes=90
        )
    # Explicit frame permission must never permit a changed target label.
    torch = pytest.importorskip("torch")
    payload = torch.load(a.replacement.feature_path, weights_only=True)
    payload["token_ids"][0] = 4
    wrong_labels = tmp_path / "wrong-labels.pt"
    torch.save(payload, wrong_labels)
    invalid = replace(a, replacement=replace(a.replacement, feature_path=wrong_labels))
    with pytest.raises(ValueError, match="changes labels"):
        load_feature_shard(invalid, num_classes=90)


@pytest.mark.parametrize("failure", ["labels", "frames", "samples", "audio", "ctc", "batch"])
def test_audit_rejects_unexplained_changes(tmp_path: Path, failure: str) -> None:
    torch = pytest.importorskip("torch")
    run, plan, _ = fixture(tmp_path)
    folder = run / "noise/train/part-000000"
    path = folder / "features/shards/shard-000000.pt"
    payload = torch.load(path, weights_only=True)
    if failure == "labels":
        payload["token_ids"][0] = 4
    elif failure in {"frames", "ctc"}:
        payload["hidden_offsets"][1] = 2 if failure == "frames" else 1
    elif failure == "batch":
        plan["encoder_batch_size"] = 1
        audits = json.loads((folder / "mix_audit.json").read_text())
        audits[0]["source_samples"] = 108800
        (folder / "mix_audit.json").write_text(json.dumps(audits))
    elif failure == "samples":
        audits = json.loads((folder / "mix_audit.json").read_text())
        # Same frame bounds, but different actual waveform length.
        audits[0]["source_samples"] = 108943
        (folder / "mix_audit.json").write_text(json.dumps(audits))
    else:
        with Path(_rows_for_test(folder / "manifest.jsonl")[0]["audio_path"]).open("ab") as audio:
            audio.write(b"changed")
    torch.save(payload, path)
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["feature_bytes"] = path.stat().st_size
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        audit_all(run, plan)
    assert not (run / RECOVERY).exists()


def _rows_for_test(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_only_exact_historical_code_upgrades_are_accepted(tmp_path: Path) -> None:
    run, plan, cfg = fixture(tmp_path)
    code = ROOT / "src/qwen_hotword/training/sharded_ctc.py"
    plan["input_sha256"][str(code)] = HISTORICAL_CODE[code.name]
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "prepared.json").write_text(json.dumps({"plan_sha256": _sha(run / "plan.json")}))
    source = verify_source(ROOT, run, cfg)
    assert len(source["identity"]["additive_code_compatibility"]) == 1
    plan["input_sha256"][str(code)] = "unrelated-historical-hash"
    (run / "plan.json").write_text(json.dumps(plan))
    (run / "prepared.json").write_text(json.dumps({"plan_sha256": _sha(run / "plan.json")}))
    with pytest.raises(ValueError, match="unapproved input/code drift"):
        verify_source(ROOT, run, cfg)


def test_recovery_reuses_audit_and_resumes_head_without_reextracting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen_hotword.training import noise_recovery as recovery
    from qwen_hotword.training.noise_run import train

    run, plan, cfg = fixture(tmp_path)
    source = verify_source(ROOT, run, cfg)
    report = audited_report(run, source)
    monkeypatch.setattr(recovery, "audit_all", lambda *a: pytest.fail("re-audit"))
    assert audited_report(run, source) == report
    views = recovered_views(run, plan, 30, report)
    strict_views = (
        replace(
            views[0],
            shards=tuple(replace(s, replacement_frame_pairs=()) for s in views[0].shards),
        ),
        views[1],
        views[2],
    )
    assert strict_views[0].fingerprint == views[0].fingerprint
    # A crash immediately after epoch-0 validation must remain resumable under
    # the unchanged data fingerprint, including the saved empty optimizer.
    from qwen_hotword.training import sharded_ctc

    original_train = sharded_ctc._train_cache_epoch
    monkeypatch.setattr(
        sharded_ctc,
        "_train_cache_epoch",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interrupt")),
    )
    with pytest.raises(RuntimeError, match="interrupt"):
        train(run, plan, 30, resume=True, device="cpu", audited_views=strict_views)
    monkeypatch.setattr(sharded_ctc, "_train_cache_epoch", original_train)
    result = train(run, plan, 30, resume=True, device="cpu", audited_views=views)
    assert result["epochs_completed"] == 1 and result["resumed_from_epoch"] == 0
    assert (run / "head_30/completed.json").exists()
    assert train(run, plan, 30, resume=True, device="cpu") == result
    archive = recovery.recovery_bundle(run)
    with zipfile.ZipFile(archive) as z:
        assert f"{RECOVERY}/audit.json" in z.namelist()
    source["identity"]["cache_marker_sha256"] = "changed"
    with pytest.raises(ValueError, match="saved recovery identity changed"):
        audited_report(run, source)


def test_all_orchestration_reuses_features_and_completed_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen_hotword.training import noise_cache, noise_evaluation, noise_recovery

    run, plan, cfg = fixture(tmp_path)
    immutable = snapshot([run / "noise", tmp_path / "cache"])
    plan_sha = _sha(run / "plan.json")
    actual_train = noise_recovery.train

    def train_cpu(*args: object, **kwargs: object) -> dict:
        return actual_train(*args, **kwargs, device="cpu")

    def evaluate(*args: object, **kwargs: object) -> dict:
        assert all((run / f"head_{ratio}/completed.json").exists() for ratio in (30, 50))
        return {"status": "completed", "mock_encoder": True}

    monkeypatch.setattr(noise_recovery, "train", train_cpu)
    monkeypatch.setattr(noise_evaluation, "evaluate", evaluate)
    monkeypatch.setattr(noise_cache, "build_noise_cache", lambda *a: pytest.fail("reextract"))
    result = noise_recovery.recover(ROOT, run, cfg, "all")
    assert result["status"] == "completed" and result["features_recomputed"] is False
    assert _sha(run / "plan.json") == plan_sha
    assert snapshot([run / "noise", tmp_path / "cache"]) == immutable
    heads = snapshot([run / "head_30", run / "head_50"])
    assert noise_recovery.recover(ROOT, run, cfg, "all")["status"] == "completed"
    assert snapshot([run / "head_30", run / "head_50"]) == heads
    with zipfile.ZipFile(result["return_bundle"]) as archive:
        for line in archive.read("sha256.txt").decode().splitlines():
            digest, name = line.split(None, 1)
            import hashlib

            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
