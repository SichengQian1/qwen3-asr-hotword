from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_diagnostics import (
    diagnose_ctc_checkpoint,
    load_validation_sample_groups,
)
from qwen_hotword.training.ctc_overfit import CachedSample, EpochMetrics, ExperimentRecord
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.sharded_ctc import (
    _early_stopping_value,
    exclusive_training_run,
    load_disk_feature_cache,
    load_feature_shard,
    train_sharded_ctc_head,
)


def _records(tmp_path: Path, split: str, count: int) -> list[ExperimentRecord]:
    records: list[ExperimentRecord] = []
    token_sequences = ((1,), (2,), (1, 2), (2, 1))
    for index in range(count):
        audio_path = tmp_path / f"{split}-{index}.wav"
        audio_path.write_bytes(b"unused by fake extractor")
        token_ids = token_sequences[index % len(token_sequences)]
        records.append(
            ExperimentRecord(
                sample_id=f"{split}-{index}",
                audio_path=audio_path,
                text="synthetic",
                language="pt-BR",
                token_ids=token_ids,
                ctc_minimum_input_length=len(token_ids),
            )
        )
    return records


def _identity_inputs(tmp_path: Path) -> tuple[Path, Path]:
    vocab = tmp_path / "vocab.json"
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    vocab.write_text(
        json.dumps({"tokens": ["<blank>", "a", "b"]}) + "\n",
        encoding="utf-8",
    )
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    return vocab, model


def _fake_extractor(
    records: list[ExperimentRecord],
    _wrapper: object,
    **_kwargs: object,
) -> tuple[list[CachedSample], int, float]:
    torch = pytest.importorskip("torch")

    samples = []
    for index, record in enumerate(records):
        hidden = torch.zeros((6, 1024), dtype=torch.float32)
        hidden[:, index % 4] = 1.0
        samples.append(
            CachedSample(
                sample_id=record.sample_id,
                hidden_states=hidden,
                token_ids=record.token_ids,
            )
        )
    return samples, 317_477_504, 0.01


def _build_cache_pair(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    vocab, model = _identity_inputs(tmp_path)
    train_manifest = tmp_path / "train.jsonl"
    validation_manifest = tmp_path / "validation.jsonl"
    train_manifest.write_text("train identity\n", encoding="utf-8")
    validation_manifest.write_text("validation identity\n", encoding="utf-8")
    train_cache = tmp_path / "train-cache"
    validation_cache = tmp_path / "validation-cache"
    cache_feature_split(
        _records(tmp_path, "train", 4),
        SimpleNamespace(),
        train_cache,
        split="train",
        source_manifest_path=train_manifest,
        model_path=model,
        model_dtype="bfloat16",
        vocab_path=vocab,
        samples_per_shard=2,
        extractor=_fake_extractor,
    )
    cache_feature_split(
        _records(tmp_path, "validation", 2),
        SimpleNamespace(),
        validation_cache,
        split="validation",
        source_manifest_path=validation_manifest,
        model_path=model,
        model_dtype="bfloat16",
        vocab_path=vocab,
        samples_per_shard=2,
        extractor=_fake_extractor,
    )
    return train_cache, validation_cache, train_manifest, validation_manifest, vocab


def test_disk_cache_loader_validates_and_reconstructs_samples(tmp_path: Path) -> None:
    train_dir, _, train_manifest, _, vocab_path = _build_cache_pair(tmp_path)
    cache = load_disk_feature_cache(
        train_dir,
        expected_split="train",
        source_manifest_path=train_manifest,
        vocab_path=vocab_path,
    )
    samples = load_feature_shard(cache.shards[0], num_classes=3)

    assert cache.sample_count == 4
    assert cache.shard_count == 2
    assert cache.sha256_verified is True
    assert [sample.sample_id for sample in samples] == ["train-0", "train-1"]
    assert samples[0].hidden_states.shape == (6, 1024)


def test_disk_cache_loader_rejects_corrupted_shard(tmp_path: Path) -> None:
    train_dir, _, train_manifest, _, vocab_path = _build_cache_pair(tmp_path)
    shard = train_dir / "shards/shard-000000.pt"
    shard.write_bytes(shard.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="file size is invalid"):
        load_disk_feature_cache(
            train_dir,
            expected_split="train",
            source_manifest_path=train_manifest,
            vocab_path=vocab_path,
        )


@pytest.mark.parametrize("head_type", ["linear", "temporal_upsample"])
def test_sharded_ctc_training_saves_and_resumes(
    tmp_path: Path,
    head_type: str,
) -> None:
    torch = pytest.importorskip("torch")
    train_dir, validation_dir, train_manifest, validation_manifest, vocab_path = (
        _build_cache_pair(tmp_path)
    )
    train_cache = load_disk_feature_cache(
        train_dir,
        expected_split="train",
        source_manifest_path=train_manifest,
        vocab_path=vocab_path,
    )
    validation_cache = load_disk_feature_cache(
        validation_dir,
        expected_split="validation",
        source_manifest_path=validation_manifest,
        vocab_path=vocab_path,
    )
    vocab = load_phoneme_vocab(vocab_path)
    output = tmp_path / "run"
    common = {
        "vocab_path": vocab_path,
        "device": torch.device("cpu"),
        "minimum_epochs": 1,
        "early_stopping_patience": 5,
        "train_batch_size": 2,
        "learning_rate": 0.01,
        "log_every_shards": 1,
        "head_type": head_type,
        "head_hidden_dimension": 8,
        "head_kernel_size": 3,
        "head_dropout": 0.0,
        "head_time_upsampling_factor": 2,
    }

    first = train_sharded_ctc_head(
        train_cache,
        validation_cache,
        vocab,
        output,
        epochs=1,
        **common,
    )
    resumed = train_sharded_ctc_head(
        train_cache,
        validation_cache,
        vocab,
        output,
        epochs=2,
        resume=True,
        **common,
    )

    assert first.epochs_completed == 1
    assert resumed.resumed_from_epoch == 1
    assert resumed.epochs_completed == 2
    assert resumed.test_set_used is False
    assert resumed.early_stopping_metric == "validation_loss"
    assert resumed.head_config["head_type"] == head_type
    assert (output / "ctc_head_best.pt").is_file()
    assert (output / "ctc_head_latest.pt").is_file()
    assert (output / "training_state_latest.pt").is_file()
    assert len((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    diagnostics = diagnose_ctc_checkpoint(
        output / "ctc_head_best.pt",
        validation_cache,
        vocab,
        device=torch.device("cpu"),
        batch_size=2,
        sample_groups={"validation-0": "en", "validation-1": "es"},
    )
    assert diagnostics["head_config"]["head_type"] == head_type
    expected_factor = 2 if head_type == "temporal_upsample" else 1
    assert diagnostics["validation"]["input_frames"] == 12 * expected_factor
    assert set(diagnostics["validation_by_group"]) == {"en", "es"}
    macro_per = diagnostics["validation_macro_phoneme_error_rate"]
    assert isinstance(macro_per, float)

    group_manifest = tmp_path / "validation-groups.jsonl"
    group_manifest.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"validation-{index}",
                    "split": "validation",
                    "balanced_language_bucket": group,
                }
            )
            + "\n"
            for index, group in enumerate(("en", "es"))
        ),
        encoding="utf-8",
    )
    assert load_validation_sample_groups(
        group_manifest,
        validation_cache,
        group_column="balanced_language_bucket",
        expected_groups=("en", "es"),
    ) == {"validation-0": "en", "validation-1": "es"}


def test_training_lock_rejects_duplicate_process(tmp_path: Path) -> None:
    output = tmp_path / "run"
    with (
        exclusive_training_run(output),
        pytest.raises(RuntimeError, match="another CTC training process"),
        exclusive_training_run(output),
    ):
        raise AssertionError("duplicate lock must not be acquired")


def test_early_stopping_metric_is_explicit() -> None:
    metrics = EpochMetrics(
        epoch=1,
        loss=0.75,
        phoneme_error_rate=0.25,
        phoneme_errors=25,
        reference_phonemes=100,
    )

    assert _early_stopping_value(metrics, "validation_loss") == 0.75
    assert _early_stopping_value(metrics, "validation_per") == 0.25
