from __future__ import annotations

import hashlib
import json
import sqlite3
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, FeatureShardDescriptor
from scripts import run_en_ar_training as run


def wav(path: Path, *, heldout: bool, seed: int) -> str:
    for n in range(seed, seed + 1000):
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(n.to_bytes(2, "little") * 16000)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (int(digest[:12], 16) % 100 < 3) == heldout:
            return digest
    raise AssertionError("test waveform could not be assigned")


def test_freeze_gates_audio_and_reuses_english_g2p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run,
        "VALIDATION_HOURS",
        {"en_egy": 0.0002, "en_us_donor": 0, "ar_egy": 0.0002, "ar_msa": 0.0002},
    )
    source_rows, arabic = [], []
    for number, (group, heldout) in enumerate(
        [
            ("en_egy", True),
            ("en_egy", False),
            ("en_us_donor", False),
            ("ar_egy", True),
            ("ar_egy", False),
            ("ar_msa", True),
        ],
        1,
    ):
        path = tmp_path / f"audio-{number}.wav"
        wav(path, heldout=heldout, seed=number * 1000)
        native_text = "Hi." if group.startswith("en") else "مش"
        item = {
            "group": group,
            "row": number,
            "source": f"{group}.json",
            "text": native_text,
            "audio_path": str(path),
            "duration_seconds": 1.0,
            "issues": [],
        }
        source_rows.append(item)
        if group.startswith("ar"):
            arabic.append({**item, "romanized": "mish", "ids": [97, 91, 72]})
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    ar = tmp_path / "arabic.jsonl"
    ar.write_text("".join(json.dumps(row) + "\n" for row in arabic), encoding="utf-8")
    en = tmp_path / "english.json"
    en.write_text(
        json.dumps(
            {
                "status": "english_g2p_completed",
                "vocab_sha256": run.VOCAB_SHA,
                "donor_sha256": run.DONOR_SHA,
                "inventory_sha256": sha(inventory),
                "words": {
                    "hi": {
                        "status": "ok",
                        "candidates": [{"usable": True, "oov_units": [], "token_ids": [20, 60]}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    inputs = dict.fromkeys(run.AUDITS, inventory)
    output = tmp_path / "release"
    output.mkdir()
    report = run.freeze(output, inputs, en, ar, 0.0002)
    assert report["status"] == "labels_complete"
    training = list(iter_jsonl(output / "train.jsonl"))
    validation = list(iter_jsonl(output / "validation.jsonl"))
    assert {row["language"] for row in training} == {"en", "ar"}
    assert {row["source_group"] for row in validation} == {"en_egy", "ar_egy", "ar_msa"}
    assert {row["audio_sha256"] for row in training}.isdisjoint(
        row["audio_sha256"] for row in validation
    )
    assert {tuple(row["phoneme_token_ids"]) for row in training} == {(20, 60), (97, 91, 72)}

    en_output = tmp_path / "english_freeze"
    en_output.mkdir()
    en_report = run.freeze(en_output, inputs, en, None, 0.0002)
    assert en_report["status"] == "labels_complete"
    assert en_report["languages"] == ["en"]
    write_json(en_output / "label_report.json", en_report)
    protected = run.frozen_english_audio(en_output, 0.0002)
    assert len(protected) == 2
    ar_output = tmp_path / "arabic_freeze"
    ar_output.mkdir()
    ar_report = run.freeze(ar_output, inputs, None, ar, 0.0002, protected_audio=protected)
    assert ar_report["status"] == "labels_complete"
    assert ar_report["languages"] == ["ar"]
    assert all(
        row["audio_sha256"] not in protected
        for split in ("train", "validation")
        for row in iter_jsonl(ar_output / f"{split}.jsonl")
    )

    native_labels = tmp_path / "native_labels"
    native_labels.mkdir()
    native_manifest = native_labels / "labels.jsonl"
    native_rows = [
        {**row, "words": ["مش"], "ids": [99, 89, 154], "vocab_sha256": run.NATIVE_VOCAB_SHA}
        for row in arabic
    ]
    native_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in native_rows), encoding="utf-8"
    )
    write_json(
        native_labels / "report.json",
        {
            "status": "automatic_labels_completed",
            "vocab_sha256": run.NATIVE_VOCAB_SHA,
            "num_classes": 167,
            "labels_sha256": sha(native_manifest),
            "identity": {"inventories": {"ar_egy": sha(inventory), "ar_msa": sha(inventory)}},
            "by_group": {"ar_egy": {"ready_rows": 2}, "ar_msa": {"ready_rows": 1}},
        },
    )
    native_output = tmp_path / "native_release"
    native_output.mkdir()
    with sqlite3.connect(":memory:") as connection:
        run.index_native_labels(native_labels, connection, inputs)
        native_report = run.freeze(
            native_output,
            inputs,
            None,
            native_manifest,
            0.0002,
            protected_audio=protected,
            native_index=connection,
            vocab_path=run.NATIVE_VOCAB,
        )
    assert native_report["status"] == "labels_complete"
    assert native_report["vocab_sha256"] == run.NATIVE_VOCAB_SHA
    assert list(iter_jsonl(native_output / "train.jsonl"))[0]["phoneme_token_ids"] == [99, 89, 154]

    ar.write_text("", encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    assert run.freeze(blocked, inputs, en, ar, 0.0002)["status"] == "insufficient_labeled_hours"


def test_arabic_input_requires_ascii_and_audit_identity(tmp_path: Path) -> None:
    path = tmp_path / "arabic.jsonl"
    row = {"group": "ar_egy", "row": 12941, "source": "egy.json", "text": "مش"}
    path.write_text(json.dumps({**row, "romanized": "mish"}) + "\n", encoding="utf-8")
    assert list(run.romanizations(path))[0]["text"] == "مش"
    path.write_text(json.dumps({**row, "romanized": "مش"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Arabic"):
        list(run.romanizations(path))


def test_combine_references_existing_feature_shards(tmp_path: Path) -> None:
    shard = FeatureShardDescriptor(
        shard_index=0,
        feature_path=tmp_path / "en.pt",
        metadata_path=tmp_path / "en.json",
        record_count=1,
        total_frames=5,
        total_target_tokens=2,
        feature_bytes=10,
        feature_sha256="a" * 64,
        sample_ids=("en-id",),
        ctc_time_upsampling_factor=2,
    )
    en = DiskFeatureCache(
        split="train",
        root=tmp_path / "en",
        sample_count=1,
        shard_count=1,
        feature_bytes=10,
        encoder_frozen_parameters=123,
        fingerprint="a" * 64,
        sha256_verified=True,
        ctc_time_upsampling_factor=2,
        shards=(shard,),
    )
    ar = replace(
        en,
        root=tmp_path / "ar",
        fingerprint="b" * 64,
        shards=(replace(shard, feature_path=tmp_path / "ar.pt", sample_ids=("ar-id",)),),
    )
    combined = run.combine_caches([en, ar], tmp_path / "joined")
    assert combined.sample_count == 2
    assert combined.shards == (*en.shards, *ar.shards)
    assert not combined.root.exists()
    with pytest.raises(ValueError, match="duplicate audio"):
        run.combine_caches([en, en], tmp_path / "joined")
