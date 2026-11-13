from __future__ import annotations

import copy
import hashlib
import io
import json
import sqlite3
import tarfile
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training import zh_en_data as data
from qwen_hotword.training import zh_en_labels as labels
from qwen_hotword.training import zh_en_run as runner
from qwen_hotword.training.zh_en_common import completed, finish, read, rows, save_rows

ROOT = Path(__file__).resolve().parents[1]


def wav_bytes(value: int = 1) -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(value.to_bytes(2, "little", signed=True) * 16000)
    return stream.getvalue()


def add_member(archive, name, content):
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def test_tar_direct_access_duplicate_pairs_and_unicode(tmp_path):
    path = tmp_path / "shard.tar"
    audio = wav_bytes()
    with tarfile.open(path, "w") as archive:
        add_member(archive, "nested/中文_001.wav", audio)
        add_member(archive, "nested/中文_001.txt", "重庆天气".encode())
        add_member(archive, "duplicate.wav", audio)
        add_member(archive, "duplicate.wav", audio)
        add_member(archive, "duplicate.txt", b"text")
        add_member(archive, "missing.wav", audio)
    result = list(data.chinese_rows(path, 0))
    good = next(r for r in result if not r["issues"])
    assert good["text"] == "重庆天气"
    assert data.audio_bytes(good) == audio
    assert sum("missing_or_duplicate_tar_pair" in r["issues"] for r in result) == 2
    assert not (tmp_path / "nested").exists()
    good["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source audio changed"):
        data.audio_bytes(good)


def test_english_path_fallback_transcript_and_missing_audio(tmp_path):
    actual = tmp_path / "mounted" / "speaker" / "a.wav"
    actual.parent.mkdir(parents=True)
    actual.write_bytes(wav_bytes())
    response = "language English<asr_text>Hello world."
    source = tmp_path / "us.json"
    write_json(
        source,
        [
            dict(
                audios=["/old/speaker/a.wav"],
                response=response,
                messages=[{"role": "assistant", "content": response}],
                language="English",
            )
        ],
    )
    result = list(data.english_rows(source, "us", {"/old": str(tmp_path / "mounted")}))
    assert result[0]["audio"]["path"] == str(actual)
    assert result[0]["text"] == "Hello world."
    assert result[0]["issues"] == []
    actual.unlink()
    failed = data.inspect_audio(result[0])
    assert failed["issues"][0].startswith("audio:FileNotFoundError:")


def test_pcm_deduplication_and_family_union_isolation(tmp_path):
    records = []
    for i, (family, pcm, text) in enumerate(
        [
            ("family-A", "same", "hello"),
            ("family-B", "same", "hello"),
            ("family-A", "unique-a", "world"),
            ("family-B", "unique-b", "other"),
            ("family-C", "conflict", "one"),
            ("family-D", "conflict", "two"),
        ]
    ):
        records.append(
            dict(id=str(i), family=family, pcm_sha256=pcm, text=text, issues=[], group="us")
        )
    source = tmp_path / "source.jsonl"
    save_rows(source, records)
    parts = data.split_inventory(tmp_path, [source], 20261113)
    got = list(rows(parts[0]))
    assert got[1]["issues"] == ["duplicate_pcm"]
    assert len({r["split_group"] for r in got[:4]}) == 1
    assert len({r["split"] for r in got[:4]}) == 1
    assert all(r["issues"] == ["duplicate_pcm_conflicting_text"] for r in got[4:])
    assert read(tmp_path / "split_report.json")["speaker_disjoint_verified"] is False


def synthetic_records():
    records = []
    for group in ("zh", "us", "uk"):
        for split, count in (("train", 10), ("validation", 2), ("test", 1)):
            for i in range(count):
                records.append(
                    dict(
                        id=f"{group}_{split}_{i}",
                        group=group,
                        split=split,
                        seconds=1.0,
                        text="fixture",
                        issues=[],
                        phoneme_token_ids=[2, 3],
                        ctc_minimum_input_length=2,
                        ctc_time_upsampling_factor=2,
                        audio={"path": "/fixture.wav"},
                        samples_16k=16000,
                    )
                )
    return records


def fixture_config():
    cfg = read(ROOT / "configs/zh_en_noise70.workzone.json")
    cfg["hours"] = {s: {g: 1 for g in ("zh", "us", "uk")} for s in ("train", "validation")}
    cfg["minimum_hours"] = {s: {g: 0 for g in ("zh", "us", "uk")} for s in ("train", "validation")}
    cfg["part_records"] = 3
    return cfg


def test_manifests_noise_ratio_paired_validation_sealed_test_and_resume(tmp_path):
    source = tmp_path / "labels.jsonl"
    values = synthetic_records()
    values.append({**values[0], "id": "bad", "issues": ["unsupported_symbol_or_character:☃"]})
    save_rows(source, values)
    cfg = fixture_config()
    result = data.manifests(tmp_path, [source], cfg)
    for group in ("zh", "us", "uk"):
        assert result["groups"]["train/" + group]["records"] == 10
        assert result["groups"]["train/" + group]["noise_records"] == 7
    ready = [r for p in result["parts"] for r in rows(Path(p["path"]))]
    assert all(r["split"] != "test" for r in ready)
    for group in ("zh", "us", "uk"):
        train = [r for r in ready if r["group"] == group and r["split"] == "train"]
        assert len(train) == 10 and sum(r["noisy"] for r in train) == 7
        clean = {
            r["base_id"]: r for r in ready if r["group"] == group and r["condition"] == "original"
        }
        noisy = {
            r["base_id"]: r for r in ready if r["group"] == group and r["condition"] == "noise10db"
        }
        assert clean.keys() == noisy.keys()
        assert len(clean) == 2
        assert all(clean[k]["phoneme_token_ids"] == noisy[k]["phoneme_token_ids"] for k in clean)
    assert result["failure_reasons"]["zh/unsupported_symbol_or_character"] == 1
    assert len(list(rows(tmp_path / "prepared/sealed_test.jsonl"))) == 3
    assert data.manifests(tmp_path, [source], cfg) == result
    changed = Path(result["parts"][0]["path"])
    changed.write_text("{}\n")
    with pytest.raises(ValueError):
        data.manifests(tmp_path, [source], cfg)


def test_shortfall_fails_with_saved_counts_not_test_backfill(tmp_path):
    source = tmp_path / "labels.jsonl"
    save_rows(source, synthetic_records())
    cfg = fixture_config()
    cfg["minimum_hours"]["train"]["zh"] = 1
    with pytest.raises(ValueError, match="insufficient usable train/zh"):
        data.manifests(tmp_path, [source], cfg)
    report = read(tmp_path / "label_report.json")
    assert report["groups"]["train/zh"]["records"] == 10
    assert not (tmp_path / "prepared/completed.json").exists()


@pytest.mark.parametrize("text", ["call 123", "hello ☃ world", "NASA mission", "A&B"])
def test_unknown_pronunciation_never_becomes_partial_success(text):
    with pytest.raises(ValueError):
        labels.frontend(text, "us")


def test_context_pinyin_and_mixed_script():
    pytest.importorskip("pypinyin")
    pytest.importorskip("jieba")
    result = labels.frontend("银行行长来到重庆。西安天气，打开c。", "zh")
    assert "".join(r["surface"] for r in result) == "银行行长来到重庆西安天气打开c"
    assert any(r["word"] == "chóngqìng" for r in result)
    assert any(r["word"] == "xī'ān" for r in result)
    assert result[-1] == {"surface": "c", "word": "c", "language": "us"}
    assert labels.frontend("King’s road", "uk")[0]["word"] == "king's"


def test_label_generation_rejects_missing_full_word_and_preserves_repeats(tmp_path):
    db_path = tmp_path / "pronunciations.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE words (lang TEXT, word TEXT, payload TEXT)")
        for word, value in [
            ("good", {"status": "ok", "candidates": [{"token_ids": [2, 2, 3], "usable": True}]}),
            ("bad", {"status": "unknown_graphemes", "candidates": []}),
        ]:
            db.execute("INSERT INTO words VALUES (?,?,?)", ("us", word, json.dumps(value)))
    source = tmp_path / "source.jsonl"
    base = dict(
        id="1",
        group="us",
        split="train",
        issues=[],
        samples_16k=16000,
        seconds=1.0,
        frontend=[{"language": "us", "word": "good"}],
    )
    bad = copy.deepcopy(base)
    bad.update(id="2")
    bad["frontend"].append({"language": "us", "word": "bad"})
    test = copy.deepcopy(base)
    test.update(id="3", split="test", frontend=[{"language": "us", "word": "unqueried"}])
    save_rows(source, [base, bad, test])
    write_json(tmp_path / "g2p_report.json", {"identity": "fixture"})
    path = labels.label_part(tmp_path, source, 0, db_path, 2)
    good, bad, sealed = list(rows(path))
    assert good["phoneme_token_ids"] == [2, 2, 3]
    assert good["ctc_minimum_input_length"] == 4
    assert bad["issues"] and "phoneme_token_ids" not in bad
    assert "phoneme_token_ids" not in sealed
    assert labels.label_part(tmp_path, source, 0, db_path, 2) == path


def test_training_is_random_and_resumes_only_its_own_state(tmp_path, monkeypatch):
    from qwen_hotword.training import ctc_diagnostics, sharded_ctc

    cfg = fixture_config()
    foundation = read(ROOT / cfg["foundation_config"])
    cache = SimpleNamespace(fingerprint="fixture")
    groups = {
        "zh1": "zh/original",
        "zh2": "zh/noise10db",
        "us1": "en/original",
        "uk1": "en/original",
        "us2": "en/noise10db",
    }
    monkeypatch.setattr(runner, "cache_views", lambda *a: (cache, cache, groups, groups))
    calls = []

    def trainer(*args, **kwargs):
        calls.append(kwargs)
        folder = args[3]
        folder.mkdir(exist_ok=True)
        (folder / "ctc_head_best.pt").write_bytes(b"fake checkpoint, not real training")
        report = dict(status="completed", num_classes=183, test_set_used=False, best_epoch=2)
        write_json(folder / "report.json", report)
        return SimpleNamespace(to_dict=lambda: report)

    monkeypatch.setattr(sharded_ctc, "train_sharded_ctc_head", trainer)
    monkeypatch.setattr(
        ctc_diagnostics,
        "diagnose_ctc_checkpoint",
        lambda *a, **k: {"validation_by_group": {"zh/original": {"phoneme_error_rate": 0.2}}},
    )
    result = runner.train(ROOT, tmp_path, cfg, foundation)
    assert calls[0]["initial_head_checkpoint"] is None
    assert calls[0]["head_type"] == "temporal_upsample"
    assert calls[0]["resume"] is False
    assert calls[0]["validation_sample_groups"] == groups
    assert result["num_classes"] == 183
    runner.train(ROOT, tmp_path, cfg, foundation)
    assert len(calls) == 1
    (tmp_path / "head/ctc_head_best.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="Head identity changed"):
        runner.train(ROOT, tmp_path, cfg, foundation)


def test_atomic_completion_does_not_accept_mutated_identity(tmp_path):
    artifact = tmp_path / "output.json"
    write_json(artifact, {"value": "one"})
    finish(tmp_path, "identity-one", [artifact], {"ok": True})
    assert completed(tmp_path, "identity-one") == {"ok": True}
    with pytest.raises(ValueError, match="changed stage identity"):
        completed(tmp_path, "identity-two")
    artifact.write_text("{}")
    with pytest.raises(ValueError):
        completed(tmp_path, "identity-one")


def test_mixed_extractor_uses_distinct_noise_pools_without_mutating_source(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    speech = tmp_path / "speech.wav"
    speech.write_bytes(wav_bytes(3000))
    noises = {}
    for split, value in [("train", 100), ("validation", 500)]:
        path = tmp_path / (split + ".wav")
        path.write_bytes(wav_bytes(value))
        noises[split] = [dict(path=str(path), sha256=sha(path))]
    before = speech.read_bytes()
    row = dict(
        id="x",
        base_id="x",
        split="train",
        noisy=True,
        audio={"path": str(speech)},
        samples_16k=16000,
        audio_sha256=hashlib.sha256(before).hexdigest(),
    )
    extractor = runner.MixedExtractor([row], noises, 1)
    mixed = extractor.load(SimpleNamespace(sample_id="x"))
    assert len(mixed) == 16000 and np.isfinite(mixed).all()
    assert extractor.audit[0]["noise_path"] == noises["train"][0]["path"]
    assert speech.read_bytes() == before
    row = {**row, "id": "v", "split": "validation"}
    val = runner.MixedExtractor([row], noises, 1)
    val.load(SimpleNamespace(sample_id="v"))
    assert val.audit[0]["noise_path"] == noises["validation"][0]["path"]
