from __future__ import annotations

import hashlib
from unittest.mock import Mock

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training import en_egy_dynamic as pipeline
from scripts import run_en_egy_dynamic as entry


def candidate(bucket="open_source", **changes):
    return {
        "group": bucket,
        "role": "train_pool",
        "bucket": bucket,
        "text_relation": "arabic_spelling_normalization_equivalent",
        "issues": ["arabic_spelling_normalization_equivalent"],
        "declared_dialect": "EGY",
        "assistant_text": "في البيت",
        "response_text": "في البيت",
        "text": "في البيت",
        "audio_path": "/audio/clip.wav",
        "duration_seconds": 5.0,
        "source": "source.json",
        "row": 1,
        "ids": [72, 73],
        **changes,
    }


def test_admission_respects_selected_subset_and_unknown_dialect():
    assert pipeline.admission(candidate()) is None
    assert pipeline.admission(candidate(text_relation="exact")) == "text_relation_not_selected"
    assert pipeline.admission(candidate("gaoshan_train", text_relation="exact")) is None
    assert (
        pipeline.admission(candidate("coarse_train", text_relation="punctuation_or_spacing_only"))
        is None
    )
    assert (
        pipeline.admission(candidate(declared_dialect="", issues=["conflicting_dialect_field"]))
        is None
    )
    assert pipeline.admission(
        candidate(declared_dialect="MSA", issues=["conflicting_dialect_field"])
    )
    assert pipeline.admission(candidate(text_relation="content_difference"))
    assert pipeline.admission(candidate("ar_msa"))


@pytest.mark.parametrize("embedded", ["swift-MGB-3-validation.json", "test.json", "foo_dev.jsonl"])
def test_embedded_heldout_is_protected_before_admission(embedded):
    assert pipeline.admission(candidate(embedded_source=embedded)) == "protected_test_or_validation"
    assert pipeline.admission(candidate(role="original_test")) == "protected_test_or_validation"


def test_dynamic_hours_have_no_480_or_minimum_gate():
    ratios = dict(zip(pipeline.BUCKETS, (0.8, 0.1, 0.1), strict=True))
    assert (
        pipeline.capacity(700, dict(zip(pipeline.BUCKETS, (400, 100, 100), strict=True)), ratios)
        == 500
    )
    assert (
        pipeline.capacity(300, dict(zip(pipeline.BUCKETS, (400, 100, 100), strict=True)), ratios)
        == 300
    )
    assert (
        pipeline.capacity(100, dict(zip(pipeline.BUCKETS, (0.8, 0.1, 0.1), strict=True)), ratios)
        == 1
    )
    with pytest.raises(ValueError, match="no usable"):
        pipeline.capacity(100, dict(zip(pipeline.BUCKETS, (10, 0, 10), strict=True)), ratios)


def test_complete_recordings_deterministic_rounding():
    rows = [{"audio_sha256": str(i), "duration_seconds": float(i + 1)} for i in range(100)]
    a = pipeline.select_seconds(rows, 325.2, 9)
    b = pipeline.select_seconds(list(reversed(rows)), 325.2, 9)
    assert a == b
    assert abs(sum(r["duration_seconds"] for r in a) - 325.2) <= 100
    assert len({r["audio_sha256"] for r in a}) == len(a)
    assert pipeline.select_seconds(rows, 0, 9) == []


def test_recognized_recordings_share_split_key_and_unknown_is_disclosed():
    key, basis = pipeline.group_key(candidate(audio_path="/a/abc__digest__seg001.wav"))
    assert key == pipeline.group_key(candidate(audio_path="/b/abc__other__seg999.wav"))[0]
    assert basis == "recognized_recording"
    key = pipeline.group_key(candidate(audio_path="/a/" + "a" * 32 + "_001.wav"))[0]
    assert key == pipeline.group_key(candidate(audio_path="/b/" + "a" * 32 + "_002.wav"))[0]
    assert pipeline.group_key(candidate())[1] == "file_only_unknown_recording"


def test_seals_and_atomic_files_refuse_different_content(tmp_path):
    p = tmp_path / "sealed.json"
    pipeline.seal(p, {"x": 1})
    pipeline.seal(p, {"x": 1})
    with pytest.raises(ValueError, match="changed"):
        pipeline.seal(p, {"x": 2})
    p = tmp_path / "rows.jsonl"
    pipeline.save_rows(p, [{"x": 1}])
    with pytest.raises(ValueError):
        pipeline.save_rows(p, [{"x": 2}])
    assert list(pipeline.iter_jsonl(p)) == [{"x": 1}]


def test_label_worker_quarantines_incomplete_and_candidate_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    cfg = {
        "vocab": "vocab.json",
        "vocab_sha256": "identity",
        "arabic_frontend_config": "frontend.json",
    }
    write_json(tmp_path / "config.json", cfg)
    write_json(tmp_path / "frontend.json", {})
    write_json(tmp_path / "vocab.json", {"tokens": list(range(167))})
    front = Mock()
    front.label.side_effect = [
        {"ids": [72, 73], "unreviewed_exception_words": []},
        {"ids": [72], "unreviewed_exception_words": ["ده"]},
        ValueError("no_analysis:word"),
        {"ids": [72] * 1000},
    ]
    monkeypatch.setattr(entry, "FRONTEND", front)
    rows = [candidate(row=i) for i in range(4)]
    job = (str(tmp_path / "labels"), str(tmp_path / "config.json"), "unused", 0, rows)
    result = entry.label_job(job)
    assert result["ready_rows"] == 1
    assert result["reasons"] == {
        "unreviewed_pronunciation_exception": 1,
        "no_analysis": 1,
        "estimated_CTC_length_infeasible": 1,
    }
    front.label.reset_mock()
    assert entry.label_job(job) == result
    front.label.assert_not_called()
    (tmp_path / "labels" / result["labels"]).write_text("tampered")
    with pytest.raises(ValueError):
        entry.label_job(job)


@pytest.mark.parametrize("cap", [None, 0.01])
def test_freeze_auto_shrinks_protects_duplicates_and_reuses_cache_order(tmp_path, monkeypatch, cap):
    out = tmp_path / "new"
    out.mkdir()
    cfg = {
        "source_run": "old",
        "seed": 7,
        "test_percent": 10,
        "validation_percent": 10,
        "arabic_validation_hours": 0.01,
        "vocab_sha256": "x",
        "ratios": dict(zip(pipeline.BUCKETS, (0.8, 0.1, 0.1), strict=True)),
    }
    if cap is not None:
        cfg["maximum_hours_per_language"] = cap
    ar = []
    for b in pipeline.BUCKETS:
        for i in range(100):
            ar.append(candidate(b, row=i, audio_path=f"/{b}/clip{i}.wav"))
    # Same bytes, different labels: neither row may be trained.
    ar.extend(
        [
            candidate("open_source", row=100, audio_path="/conflict.wav"),
            candidate("coarse_train", row=101, audio_path="/conflict.wav", ids=[74]),
        ]
    )
    # Duplicate across buckets: retain open-source attribution only.
    ar.extend(
        [
            candidate("open_source", row=102, audio_path="/duplicate.wav"),
            candidate("gaoshan_train", row=103, audio_path="/duplicate.wav"),
        ]
    )
    piece = out / "labels/parts/0/labels.jsonl"
    pipeline.save_rows(piece, ar)
    write_json(
        out / "labels/report.json",
        {"pieces": [{"labels": "parts/0/labels.jsonl", "labels_sha256": sha(piece)}]},
    )
    english = [candidate("en_egy", row=i, audio_path=f"/en/{i}.wav", ids=[2, 3]) for i in range(40)]
    english += [candidate("en_us", row=i, audio_path=f"/us/{i}.wav", ids=[2, 3]) for i in range(40)]
    protected = [ar[1], english[0]]
    pipeline.save_rows(out / "english_ready.jsonl", english)
    pipeline.save_rows(out / "protected.jsonl", protected)
    val = candidate("en_validation", audio_path="/en_val.wav", ids=[2, 3])
    pipeline.save_rows(out / "english_validation.jsonl", [val])

    def fingerprint(rows, *_):
        for row in rows:
            row["audio_sha256"] = hashlib.sha256(row["audio_path"].encode()).hexdigest()
            if row["audio_path"] in {ar[2]["audio_path"], english[1]["audio_path"]}:
                row["audio_review_issue"] = "audio_duration_mismatch"

    monkeypatch.setattr(pipeline, "fingerprint", fingerprint)
    old_records = [
        {"id": "en-ar-v1-" + hashlib.sha256(r["audio_path"].encode()).hexdigest()}
        for r in english[::-1]
    ]
    pipeline.save_rows(tmp_path / "old/train.jsonl", old_records)
    write_json(
        tmp_path / "old/plan.json",
        {"manifests": {"en/train": {"path": str(tmp_path / "old/train.jsonl")}}},
    )
    result = pipeline.freeze(tmp_path, cfg, out, 2)
    train = list(pipeline.iter_jsonl(out / "train.jsonl"))
    validation = list(pipeline.iter_jsonl(out / "validation.jsonl"))
    test = list(pipeline.iter_jsonl(out / "test.jsonl"))
    assert result["target_hours_per_language"] < 1
    assert result["arabic_msa_used"] is False
    assert set(r["language"] for r in validation) == {"en", "ar"}
    assert all(
        r["audio_path"]
        not in {"/conflict.wav", protected[0]["audio_path"], protected[1]["audio_path"]}
        for r in train
    )
    assert all(
        r["audio_path"] not in {ar[2]["audio_path"], english[1]["audio_path"]}
        for r in train + validation + test
    )
    assert len({r["id"] for r in train + validation + test}) == len(train + validation + test)
    en_order = [r["id"] for r in train if r["language"] == "en"]
    expected = [r["id"] for r in old_records if r["id"] in set(en_order)]
    assert en_order == expected
    en_hours = sum(r["duration_seconds"] for r in train if r["language"] == "en")
    ar_hours = sum(r["duration_seconds"] for r in train if r["language"] == "ar")
    assert abs(en_hours - ar_hours) <= 20
    if cap is not None:
        assert en_hours <= cap * 3600 and ar_hours <= cap * 3600
    fingerprint_spy = Mock(side_effect=AssertionError("must skip completed freeze"))
    monkeypatch.setattr(pipeline, "fingerprint", fingerprint_spy)
    assert pipeline.freeze(tmp_path, cfg, out, 2) == result
    fingerprint_spy.assert_not_called()
    (out / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        pipeline.freeze(tmp_path, cfg, out, 2)


def test_real_audit_normalized_warning_is_admitted_but_content_difference_is_not(tmp_path):
    import wave

    from qwen_hotword.training.egyptian_source_audit import inspect

    audio = tmp_path / "sample.wav"
    with wave.open(str(audio), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(16000)
        dst.writeframes(b"\0\0" * 16000)
    # The fixture supplies metadata because local soundfile is optional/unavailable.
    from types import SimpleNamespace
    from unittest.mock import patch

    with patch(
        "qwen_hotword.training.egyptian_source_audit.read_audio_metadata",
        return_value=SimpleNamespace(frames=16000, sample_rate=16000, duration_seconds=1.0),
    ):
        audited = inspect(
            {
                "language": "Arabic",
                "dialect": "EGY",
                "audios": [str(audio)],
                "response": "هذه مدرسه",
                "messages": [{"role": "assistant", "content": "هذه مدرسة"}],
            },
            {},
        )
    assert audited["issues"] == ["arabic_spelling_normalization_equivalent"]
    assert pipeline.admission({**candidate(), **audited}) is None
    assert pipeline.admission(
        {**candidate(), **audited, "issues": ["audio_header_unreadable_or_changed"]}
    )


def test_durable_audio_fingerprints_resume_without_reread(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from qwen_hotword.training import verified_audio_hashes

    metadata = Mock(return_value=SimpleNamespace(duration_seconds=5.0))
    monkeypatch.setattr(verified_audio_hashes, "read_audio_metadata", metadata)
    path = tmp_path / "audio.wav"
    path.write_bytes(b"first")
    row = candidate(audio_path=str(path))
    pipeline.fingerprint([row], tmp_path, 2)
    first = row["audio_sha256"]
    pipeline.fingerprint([row], tmp_path, 2)
    assert metadata.call_count == 1
    path.write_bytes(b"second content")
    pipeline.fingerprint([row], tmp_path, 2)
    assert row["audio_sha256"] != first
    assert metadata.call_count == 2


def test_cache_completion_automatically_starts_head_and_completed_run_is_idempotent(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    from qwen_hotword import config
    from qwen_hotword.phonemes import coverage
    from qwen_hotword.training import (
        arabic_refresh_train,
        en_egy_dynamic_train,
        feature_cache,
        sharded_ctc,
    )
    from scripts import arabic_diagnostic_common, run_en_ar_merged

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=Mock()))
    )
    out = tmp_path / "run"
    out.mkdir()
    manifests = {}
    for split in ("train", "validation"):
        records = [
            pipeline.manifest(
                {
                    **candidate(b, ids=ids, audio_path=f"/{split}/{b}.wav"),
                    "audio_sha256": f"{split}-{b}",
                    "recording_key": f"{split}-{b}",
                    "grouping_basis": "file_only",
                },
                split,
            )
            for b, ids in (("en_egy", [2, 3]), ("open_source", [72, 73]))
        ]
        path = out / f"{split}.jsonl"
        pipeline.save_rows(path, records)
        manifests[split] = {"path": str(path), "sha256": sha(path)}
    write_json(out / "prepare_report.json", {"manifests": manifests})
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"mock model identity only")
    write_json(model / "model.safetensors.index.json", {})
    weight = {"path": str(model / "model.safetensors"), "sha256": sha(model / "model.safetensors")}
    cfg = {"vocab": "vocab.json", "minimum_gpu_free_gib": 40, "head_time_upsampling_factor": 2}
    write_json(tmp_path / cfg["vocab"], {"tokens": list(range(167))})
    cfg["vocab_sha256"] = sha(tmp_path / cfg["vocab"])
    source = {
        "head": {},
        "plan": {
            "model": {
                "files": {"model.safetensors": weight},
                "view": str(model),
                "index_sha256": sha(model / "model.safetensors.index.json"),
            },
            "manifests": {"en/train": manifests["train"]},
            "cache_config": {"path": "mock"},
        },
    }
    monkeypatch.setattr(
        config,
        "load_workzone_config",
        lambda *a, **k: SimpleNamespace(model=SimpleNamespace(path=model, dtype="bfloat16")),
    )
    monkeypatch.setattr(coverage, "load_phoneme_vocab", lambda *a: object())
    monkeypatch.setattr(arabic_diagnostic_common, "load_cache", lambda *a: object())
    monkeypatch.setattr(arabic_diagnostic_common, "training_options", lambda *a: {"epochs": 40})
    monkeypatch.setattr(run_en_ar_merged, "wait_for_gpu", Mock())
    monkeypatch.setattr(
        en_egy_dynamic_train.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="")
    )
    monkeypatch.setattr(
        arabic_refresh_train,
        "ReuseOrExtract",
        lambda *a: SimpleNamespace(wrapper=None, loaded={}, reused=2, extracted=2),
    )
    events = []

    def cache(records, wrapper, destination, **kwargs):
        assert kwargs["encoder_batch_size"] == 8 and kwargs["samples_per_shard"] == 512
        assert {r.language for r in records} == {"en", "ar"}
        events.append("cache/" + kwargs["split"])

    monkeypatch.setattr(feature_cache, "cache_feature_split", cache)
    monkeypatch.setattr(sharded_ctc, "load_disk_feature_cache", lambda *a, **k: object())

    def train(*args, **kwargs):
        assert events == ["cache/train", "cache/validation"]
        assert kwargs["epochs"] == 40 and kwargs["resume"] is False
        assert set(kwargs["validation_sample_groups"].values()) == {"en", "ar"}
        assert "initial_head_checkpoint" not in kwargs
        events.append("train")
        head = out / "head"
        head.mkdir(exist_ok=True)
        write_json(head / "report.json", {"status": "completed"})
        (head / "ctc_head_best.pt").write_bytes(b"mock")

    monkeypatch.setattr(sharded_ctc, "train_sharded_ctc_head", train)
    result = en_egy_dynamic_train.run(tmp_path, cfg, out, source, 4)
    assert events == ["cache/train", "cache/validation", "train"]
    assert result["status"] == "completed"
    assert en_egy_dynamic_train.run(tmp_path, cfg, out, source, 4) == result
    assert len(events) == 3
