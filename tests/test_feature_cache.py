from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.training.ctc_overfit import CachedSample, ExperimentRecord
from qwen_hotword.training.feature_cache import (
    cache_feature_split,
    exclusive_feature_cache_run,
)
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)


def _records(tmp_path: Path, count: int) -> list[ExperimentRecord]:
    records = []
    for index in range(count):
        audio_path = tmp_path / f"audio-{index}.wav"
        _write_wav(audio_path)
        records.append(
            ExperimentRecord(
                sample_id=f"sample-{index}",
                audio_path=audio_path,
                text="bom dia",
                language="pt-BR",
                token_ids=(2, 3, 4),
                ctc_minimum_input_length=3,
            )
        )
    return records


def _identity_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "manifest.jsonl"
    vocab = tmp_path / "vocab.json"
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    manifest.write_text("manifest\n", encoding="utf-8")
    vocab.write_text("{}\n", encoding="utf-8")
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return manifest, vocab, model


def _fake_extractor(
    records: list[ExperimentRecord],
    _wrapper: object,
    **_kwargs: object,
) -> tuple[list[CachedSample], int, float]:
    torch = pytest.importorskip("torch")
    samples = [
        CachedSample(
            sample_id=record.sample_id,
            hidden_states=torch.full((5 + index, 1024), float(index)),
            token_ids=record.token_ids,
        )
        for index, record in enumerate(records)
    ]
    return samples, 317_477_504, 1.25


def test_feature_cache_is_sharded_validated_and_resumable(tmp_path: Path) -> None:
    records = _records(tmp_path, 5)
    manifest, vocab, model = _identity_inputs(tmp_path)
    output = tmp_path / "cache"
    wrapper = SimpleNamespace()

    first = cache_feature_split(
        records,
        wrapper,
        output,
        split="train",
        source_manifest_path=manifest,
        model_path=model,
        model_dtype="bfloat16",
        vocab_path=vocab,
        encoder_batch_size=2,
        samples_per_shard=2,
        extractor=_fake_extractor,
    )
    second = cache_feature_split(
        records,
        wrapper,
        output,
        split="train",
        source_manifest_path=manifest,
        model_path=model,
        model_dtype="bfloat16",
        vocab_path=vocab,
        encoder_batch_size=1,
        samples_per_shard=2,
        extractor=_fake_extractor,
    )

    assert first.status == "pass"
    assert first.shard_count == 3
    assert first.generated_shards == 3
    assert first.resumed_shards == 0
    assert first.total_frames == 27
    assert second.generated_shards == 0
    assert second.resumed_shards == 3
    index = json.loads((output / "cache_index.json").read_text(encoding="utf-8"))
    assert index["status"] == "pass"
    assert index["completed_samples"] == 5
    assert len(list((output / "shards").glob("*.pt"))) == 3


def test_feature_cache_rejects_corrupted_completed_shard(tmp_path: Path) -> None:
    records = _records(tmp_path, 3)
    manifest, vocab, model = _identity_inputs(tmp_path)
    output = tmp_path / "cache"
    kwargs = {
        "split": "validation",
        "source_manifest_path": manifest,
        "model_path": model,
        "model_dtype": "bfloat16",
        "vocab_path": vocab,
        "samples_per_shard": 2,
        "extractor": _fake_extractor,
    }
    cache_feature_split(records, SimpleNamespace(), output, **kwargs)
    shard = output / "shards/shard-000000.pt"
    shard.write_bytes(shard.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="size mismatch"):
        cache_feature_split(records, SimpleNamespace(), output, **kwargs)


def test_feature_cache_rejects_test_split(tmp_path: Path) -> None:
    records = _records(tmp_path, 1)
    manifest, vocab, model = _identity_inputs(tmp_path)
    with pytest.raises(ValueError, match="only train or validation"):
        cache_feature_split(
            records,
            SimpleNamespace(),
            tmp_path / "cache",
            split="test",
            source_manifest_path=manifest,
            model_path=model,
            model_dtype="bfloat16",
            vocab_path=vocab,
            extractor=_fake_extractor,
        )


def test_feature_cache_honors_temporal_upsampling_manifest_contract(
    tmp_path: Path,
) -> None:
    records = _records(tmp_path, 1)
    temporal_record = ExperimentRecord(
        sample_id=records[0].sample_id,
        audio_path=records[0].audio_path,
        text=records[0].text,
        language=records[0].language,
        token_ids=(2, 3, 4, 5),
        ctc_minimum_input_length=8,
        ctc_time_upsampling_factor=2,
    )
    manifest, vocab, model = _identity_inputs(tmp_path)

    summary = cache_feature_split(
        [temporal_record],
        SimpleNamespace(),
        tmp_path / "temporal-cache",
        split="train",
        source_manifest_path=manifest,
        model_path=model,
        model_dtype="bfloat16",
        vocab_path=vocab,
        extractor=_fake_extractor,
    )

    assert summary.status == "pass"
    config = json.loads((tmp_path / "temporal-cache/cache_config.json").read_text(encoding="utf-8"))
    assert config["ctc_time_upsampling_factor"] == 2
    disk_cache = load_disk_feature_cache(
        tmp_path / "temporal-cache",
        expected_split="train",
        source_manifest_path=manifest,
        vocab_path=vocab,
    )
    assert disk_cache.ctc_time_upsampling_factor == 2
    assert disk_cache.shards[0].ctc_time_upsampling_factor == 2


def test_feature_cache_lock_rejects_duplicate_process(tmp_path: Path) -> None:
    output = tmp_path / "cache"
    with (
        exclusive_feature_cache_run(output),
        pytest.raises(RuntimeError, match="another feature-cache process"),
        exclusive_feature_cache_run(output),
    ):
        raise AssertionError("duplicate lock must not be acquired")
