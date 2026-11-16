from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.en_egy_dynamic import read
from scripts import run_en_egy_up3 as up3


def setup_run(root, monkeypatch):
    source = root / "x2"
    source.mkdir()
    vocab = root / "vocab.json"
    write_json(vocab, {"fixture": True})
    manifests = {}
    for split in ("train", "validation"):
        path = source / f"{split}.jsonl"
        path.write_text(json.dumps({"id": split, "phoneme_token_ids": [2, 3]}) + "\n")
        manifests[split] = {"path": str(path), "sha256": sha(path), "samples": 1}
    model = {
        "view": "r15",
        "index_sha256": "index",
        "files": {
            "config.json": {"sha256": "cfg"},
            "model.safetensors": {"sha256": "weights"},
        },
    }
    base = {"identity": {}, "plan": {"model": model}, "head": {}}
    monkeypatch.setattr(up3, "completed_run", lambda _: base)
    write_json(
        source / "contract.json",
        {
            "source_identity": {},
            "configuration": {
                "source_run": "old-r15",
                "vocab": "vocab.json",
                "vocab_sha256": sha(vocab),
                "maximum_hours_per_language": 480,
                "head_time_upsampling_factor": 2,
            },
        },
    )
    write_json(
        source / "prepare_report.json",
        {
            "status": "prepared",
            "arabic_msa_used": False,
            "vocab_sha256": sha(vocab),
            "manifests": manifests,
        },
    )
    for name in (
        "en_egy_dynamic_train.py",
        "arabic_refresh_train.py",
        "feature_cache.py",
        "sharded_ctc.py",
        "ctc_overfit.py",
    ):
        path = root / "src/qwen_hotword/training" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture")
    path = root / "scripts/arabic_diagnostic_common.py"
    path.parent.mkdir()
    path.write_text("# fixture")
    return source, base, vocab


def published_shard(source, vocab, split="train"):
    folder = source / "cache" / split
    write_json(
        folder / "cache_config.json",
        {
            "model": {
                "path": "r15",
                "dtype": "bfloat16",
                "config_sha256": "cfg",
                "weight_index_sha256": "index",
            },
            "vocab": {"sha256": sha(vocab)},
            "source_manifest": {"sha256": sha(source / f"{split}.jsonl")},
        },
    )
    path = folder / "shards/shard-000000.pt"
    path.parent.mkdir()
    path.write_bytes(b"already complete feature")
    write_json(
        path.with_suffix(".json"),
        {
            "split": split,
            "feature_bytes": path.stat().st_size,
            "feature_sha256": sha(path),
            "feature_path": str(path),
        },
    )
    # An in-flight pt file without atomically published metadata is never imported.
    (path.parent / "shard-000001.pt").write_bytes(b"partial")
    return path


def test_independent_up3_with_no_x2_report_or_cache(tmp_path, monkeypatch):
    from qwen_hotword.training import en_egy_dynamic_train

    source, base, _ = setup_run(tmp_path, monkeypatch)
    output = tmp_path / "up3"
    train = Mock(return_value={"status": "completed"})
    monkeypatch.setattr(en_egy_dynamic_train, "run", train)
    # Running x2 may hold its own lock for hours; x3 must never acquire that lock.
    with up3.exclusive_job(source):
        result = up3.run(tmp_path, source, output, 4)
    assert result["status"] == "completed"
    assert not (source / "report.json").exists()
    cfg = train.call_args.args[1]
    assert cfg["head_time_upsampling_factor"] == 3
    assert cfg["maximum_hours_per_language"] == 480
    assert train.call_args.args[3] is base and train.call_args.args[4] == 4
    for split in ("train", "validation"):
        assert (source / f"{split}.jsonl").read_bytes() == (output / f"{split}.jsonl").read_bytes()
    # A subsequently completed/changing x2 Head must not invalidate x3's source contract.
    write_json(source / "head/report.json", {"status": "running", "epoch": 1})
    up3.run(tmp_path, source, output, 4)
    assert read(output / "standalone_contract.json")["wait_for_x2_completion"] is False
    assert not (output / "test.jsonl").exists()


def test_snapshot_completed_pairs_and_resume_without_copying_active_shard(tmp_path, monkeypatch):
    source, base, vocab = setup_run(tmp_path, monkeypatch)
    output = tmp_path / "up3"
    up3.prepare(tmp_path, source, output)
    original = published_shard(source, vocab)
    original_metadata = original.with_suffix(".json").read_bytes()
    up3.snapshot_shards(source, output, base["plan"]["model"], vocab)
    linked = output / "cache/train/shards/shard-000000.pt"
    assert linked.stat().st_ino == original.stat().st_ino
    assert original.with_suffix(".json").read_bytes() == original_metadata
    assert read(linked.with_suffix(".json"))["feature_path"] == str(linked)
    assert not linked.with_name("shard-000001.pt").exists()
    up3.snapshot_shards(source, output, base["plan"]["model"], vocab)
    # Removing/replacing our link cannot delete or replace the x2 feature inode.
    linked.unlink()
    assert original.read_bytes() == b"already complete feature"


@pytest.mark.parametrize("change", ["audio_manifest", "weights", "shard"])
def test_invalid_partial_cache_never_reused(tmp_path, monkeypatch, change):
    source, base, vocab = setup_run(tmp_path, monkeypatch)
    output = tmp_path / "up3"
    up3.prepare(tmp_path, source, output)
    original = published_shard(source, vocab)
    if change == "audio_manifest":
        (source / "train.jsonl").write_text("changed")
    elif change == "weights":
        path = source / "cache/train/cache_config.json"
        config = read(path)
        config["model"]["weight_index_sha256"] = "other"
        write_json(path, config)
    else:
        original.write_bytes(b"different feature")
    with pytest.raises(ValueError):
        up3.snapshot_shards(source, output, base["plan"]["model"], vocab)


def test_resume_rejects_changed_frozen_labels(tmp_path, monkeypatch):
    source, _, _ = setup_run(tmp_path, monkeypatch)
    output = tmp_path / "up3"
    up3.prepare(tmp_path, source, output)
    (output / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError):
        up3.prepare(tmp_path, source, output)
