from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    EpochMetrics,
    collapse_ctc_ids,
    collate_cached_samples,
    save_ctc_head_checkpoint,
)

SHARDED_CTC_SCHEMA_VERSION = 1
EarlyStoppingMetric = Literal["validation_loss", "validation_per"]


@dataclass(frozen=True)
class FeatureShardDescriptor:
    shard_index: int
    feature_path: Path
    metadata_path: Path
    record_count: int
    total_frames: int
    total_target_tokens: int
    feature_bytes: int
    feature_sha256: str
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class DiskFeatureCache:
    split: str
    root: Path
    sample_count: int
    shard_count: int
    feature_bytes: int
    encoder_frozen_parameters: int
    fingerprint: str
    sha256_verified: bool
    shards: tuple[FeatureShardDescriptor, ...]


@dataclass(frozen=True)
class ShardedEpochMetrics:
    epoch: int
    learning_rate: float
    train: EpochMetrics
    validation: EpochMetrics
    epoch_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "epoch_seconds": self.epoch_seconds,
        }


@dataclass(frozen=True)
class ShardedCtcReport:
    train_cache_dir: str
    validation_cache_dir: str
    vocab_path: str
    train_sample_count: int
    validation_sample_count: int
    train_shard_count: int
    validation_shard_count: int
    train_feature_bytes: int
    validation_feature_bytes: int
    num_classes: int
    blank_id: int
    head_config: dict[str, object]
    encoder_frozen_parameters: int
    ctc_trainable_parameters: int
    train_batch_size: int
    initial_learning_rate: float
    final_learning_rate: float
    epochs_requested: int
    epochs_completed: int
    resumed_from_epoch: int
    minimum_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_metric: str
    initial_validation_loss: float
    initial_validation_phoneme_error_rate: float
    best_epoch: int
    best_train_loss: float | None
    best_train_phoneme_error_rate: float | None
    best_validation_loss: float
    best_validation_phoneme_error_rate: float
    final_train_loss: float
    final_train_phoneme_error_rate: float
    final_validation_loss: float
    final_validation_phoneme_error_rate: float
    training_seconds: float
    best_checkpoint_path: str
    latest_checkpoint_path: str
    training_state_path: str
    metrics_path: str
    early_stopped: bool
    cache_sha256_verified: bool
    selection_metric: str
    test_set_used: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@contextmanager
def exclusive_training_run(output_dir: str | Path) -> Iterator[None]:
    import fcntl

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".ctc_training.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"another CTC training process owns {output_dir}: {owner}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_disk_feature_cache(
    cache_dir: str | Path,
    *,
    expected_split: str,
    source_manifest_path: str | Path,
    vocab_path: str | Path,
    verify_sha256: bool = True,
) -> DiskFeatureCache:
    if expected_split not in {"train", "validation"}:
        raise ValueError("disk feature cache accepts only train or validation")
    root = Path(cache_dir).expanduser()
    index = _read_object(root / "cache_index.json")
    config = _read_object(root / "cache_config.json")
    summary = _read_object(root / "cache_summary.json")
    if index.get("status") != "pass" or summary.get("status") != "pass":
        raise ValueError(f"feature cache is not complete: {root}")
    if index.get("split") != expected_split or config.get("split") != expected_split:
        raise ValueError(f"feature cache split mismatch: {root}")
    if config.get("experiment") != "full-ctc-v1":
        raise ValueError(f"feature cache is not full-ctc-v1 data: {root}")
    if config.get("hidden_size") != 1024 or config.get("feature_dtype") != "torch.bfloat16":
        raise ValueError(f"feature cache tensor contract is invalid: {root}")
    _validate_identity(
        config.get("source_manifest"),
        Path(source_manifest_path).expanduser(),
        label="source manifest",
    )
    _validate_identity(
        config.get("vocab"),
        Path(vocab_path).expanduser(),
        label="vocabulary",
    )

    sample_count = _required_int(index, "sample_count")
    shard_count = _required_int(index, "shard_count")
    if sample_count != config.get("sample_count") or sample_count != summary.get("sample_count"):
        raise ValueError(f"feature cache sample counts disagree: {root}")
    if shard_count != index.get("completed_shards") or shard_count != summary.get("shard_count"):
        raise ValueError(f"feature cache shard counts disagree: {root}")
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise ValueError(f"feature cache index has an invalid shard list: {root}")

    descriptors: list[FeatureShardDescriptor] = []
    expected_record_start = 0
    all_sample_ids: list[str] = []
    for position, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, dict) or raw_shard.get("shard_index") != position:
            raise ValueError(f"feature cache shard order is invalid: {root}")
        feature_path = root / "shards" / f"shard-{position:06d}.pt"
        metadata_path = root / "shards" / f"shard-{position:06d}.json"
        metadata = _read_object(metadata_path)
        record_count = _required_int(metadata, "record_count")
        if metadata.get("split") != expected_split or metadata.get("shard_index") != position:
            raise ValueError(f"feature shard metadata identity mismatch: {metadata_path}")
        if metadata.get("record_start") != expected_record_start:
            raise ValueError(f"feature shard record offsets are not contiguous: {metadata_path}")
        expected_record_start += record_count
        if metadata.get("record_end") != expected_record_start:
            raise ValueError(f"feature shard record end is invalid: {metadata_path}")
        sample_ids = metadata.get("sample_ids")
        if (
            not isinstance(sample_ids, list)
            or len(sample_ids) != record_count
            or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        ):
            raise ValueError(f"feature shard sample IDs are invalid: {metadata_path}")
        feature_bytes = _required_int(metadata, "feature_bytes")
        feature_sha256 = _required_string(metadata, "feature_sha256")
        if not feature_path.is_file() or feature_path.stat().st_size != feature_bytes:
            raise ValueError(f"feature shard file size is invalid: {feature_path}")
        for key in ("record_count", "total_frames", "total_target_tokens", "feature_bytes"):
            if raw_shard.get(key) != metadata.get(key):
                raise ValueError(f"feature shard index mismatch for {key}: {metadata_path}")
        if raw_shard.get("feature_sha256") != feature_sha256:
            raise ValueError(f"feature shard index SHA256 mismatch: {metadata_path}")
        if verify_sha256 and _sha256_file(feature_path) != feature_sha256:
            raise ValueError(f"feature shard SHA256 verification failed: {feature_path}")
        all_sample_ids.extend(sample_ids)
        descriptors.append(
            FeatureShardDescriptor(
                shard_index=position,
                feature_path=feature_path,
                metadata_path=metadata_path,
                record_count=record_count,
                total_frames=_required_int(metadata, "total_frames"),
                total_target_tokens=_required_int(metadata, "total_target_tokens"),
                feature_bytes=feature_bytes,
                feature_sha256=feature_sha256,
                sample_ids=tuple(sample_ids),
            )
        )
        if verify_sha256 and (
            position == 0 or (position + 1) % 50 == 0 or position + 1 == shard_count
        ):
            print(
                f"verified {expected_split} feature shards: {position + 1}/{shard_count}",
                flush=True,
            )
    if expected_record_start != sample_count or len(set(all_sample_ids)) != sample_count:
        raise ValueError(f"feature cache sample coverage is invalid: {root}")
    sample_id_sha256 = _strings_sha256(all_sample_ids)
    if config.get("sample_id_sha256") != sample_id_sha256:
        raise ValueError(f"feature cache sample ID fingerprint is invalid: {root}")

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "config": config,
                "shards": [descriptor.feature_sha256 for descriptor in descriptors],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    encoder_frozen_parameters = _required_int(summary, "encoder_frozen_parameters")
    return DiskFeatureCache(
        split=expected_split,
        root=root,
        sample_count=sample_count,
        shard_count=shard_count,
        feature_bytes=sum(descriptor.feature_bytes for descriptor in descriptors),
        encoder_frozen_parameters=encoder_frozen_parameters,
        fingerprint=fingerprint,
        sha256_verified=verify_sha256,
        shards=tuple(descriptors),
    )


def load_feature_shard(
    descriptor: FeatureShardDescriptor,
    *,
    num_classes: int,
    blank_id: int = 0,
) -> list[CachedSample]:
    import torch

    payload = torch.load(descriptor.feature_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "hidden_states",
        "hidden_offsets",
        "token_ids",
        "token_offsets",
    }:
        raise ValueError(f"feature shard tensor payload is invalid: {descriptor.feature_path}")
    hidden = payload["hidden_states"]
    hidden_offsets = payload["hidden_offsets"]
    tokens = payload["token_ids"]
    token_offsets = payload["token_offsets"]
    if hidden.ndim != 2 or list(hidden.shape[1:]) != [1024] or hidden.dtype != torch.bfloat16:
        raise ValueError(f"feature shard hidden tensor is invalid: {descriptor.feature_path}")
    if (
        hidden_offsets.ndim != 1
        or token_offsets.ndim != 1
        or hidden_offsets.dtype != torch.int64
        or token_offsets.dtype != torch.int64
    ):
        raise ValueError(f"feature shard offsets are invalid: {descriptor.feature_path}")
    if tokens.ndim != 1 or tokens.dtype != torch.int64:
        raise ValueError(f"feature shard labels are invalid: {descriptor.feature_path}")
    hidden_values = hidden_offsets.tolist()
    token_values = token_offsets.tolist()
    expected_offset_count = descriptor.record_count + 1
    if len(hidden_values) != expected_offset_count or len(token_values) != expected_offset_count:
        raise ValueError(f"feature shard offset counts are invalid: {descriptor.feature_path}")
    if (
        hidden_values[0] != 0
        or token_values[0] != 0
        or hidden_values[-1] != hidden.shape[0]
        or token_values[-1] != tokens.shape[0]
        or any(
            left > right
            for left, right in zip(hidden_values, hidden_values[1:], strict=False)
        )
        or any(
            left > right
            for left, right in zip(token_values, token_values[1:], strict=False)
        )
    ):
        raise ValueError(f"feature shard offset values are invalid: {descriptor.feature_path}")
    if (
        hidden.shape[0] != descriptor.total_frames
        or tokens.shape[0] != descriptor.total_target_tokens
    ):
        raise ValueError(f"feature shard totals are invalid: {descriptor.feature_path}")

    samples: list[CachedSample] = []
    for index, sample_id in enumerate(descriptor.sample_ids):
        input_start, input_end = hidden_values[index : index + 2]
        target_start, target_end = token_values[index : index + 2]
        token_ids = tuple(int(value) for value in tokens[target_start:target_end].tolist())
        if not token_ids or any(
            token_id == blank_id or token_id < 0 or token_id >= num_classes
            for token_id in token_ids
        ):
            raise ValueError(f"cached labels are invalid for sample {sample_id}")
        minimum_input_length = len(token_ids) + sum(
            left == right for left, right in zip(token_ids, token_ids[1:], strict=False)
        )
        if input_end - input_start < minimum_input_length:
            raise ValueError(f"cached CTC lengths are infeasible for sample {sample_id}")
        samples.append(
            CachedSample(
                sample_id=sample_id,
                hidden_states=hidden[input_start:input_end],
                token_ids=token_ids,
            )
        )
    return samples


def train_sharded_ctc_head(
    train_cache: DiskFeatureCache,
    validation_cache: DiskFeatureCache,
    vocab: PhonemeVocab,
    output_dir: str | Path,
    *,
    vocab_path: str | Path,
    device: Any,
    epochs: int = 30,
    minimum_epochs: int = 5,
    early_stopping_patience: int = 6,
    early_stopping_min_delta: float = 0.001,
    early_stopping_metric: EarlyStoppingMetric = "validation_loss",
    train_batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_gradient_norm: float = 5.0,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    minimum_learning_rate: float = 1e-5,
    seed: int = 20_260_720,
    log_every_shards: int = 25,
    resume: bool = False,
    head_type: str = "linear",
    head_hidden_dimension: int = 512,
    head_kernel_size: int = 5,
    head_dropout: float = 0.1,
    head_time_upsampling_factor: int = 2,
) -> ShardedCtcReport:
    import torch

    from qwen_hotword.modeling.ctc_head import build_ctc_head, ctc_head_config

    _validate_training_arguments(
        train_cache,
        validation_cache,
        vocab,
        epochs=epochs,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        max_gradient_norm=max_gradient_norm,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        minimum_learning_rate=minimum_learning_rate,
        log_every_shards=log_every_shards,
    )
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    blank_id = 0
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = build_ctc_head(
        head_type=head_type,
        input_dimension=1024,
        num_classes=len(vocab.tokens),
        hidden_dimension=head_hidden_dimension,
        kernel_size=head_kernel_size,
        dropout=head_dropout,
        time_upsampling_factor=head_time_upsampling_factor,
    ).to(device=device, dtype=torch.float32)
    head_configuration = ctc_head_config(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=minimum_learning_rate,
    )
    best_checkpoint_path = destination / "ctc_head_best.pt"
    latest_checkpoint_path = destination / "ctc_head_latest.pt"
    training_state_path = destination / "training_state_latest.pt"
    metrics_path = destination / "metrics.jsonl"
    fingerprint_hyperparameters: dict[str, object] = {
        "train_batch_size": train_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_gradient_norm": max_gradient_norm,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "minimum_learning_rate": minimum_learning_rate,
        "minimum_epochs": minimum_epochs,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_metric": early_stopping_metric,
        "seed": seed,
    }
    # Preserve the fingerprint of existing linear-Head runs so their training
    # state remains resumable after this feature was added.
    if head_configuration["head_type"] != "linear":
        fingerprint_hyperparameters["head_config"] = head_configuration
    run_fingerprint = _training_fingerprint(
        train_cache,
        validation_cache,
        vocab,
        **fingerprint_hyperparameters,
    )

    history: list[ShardedEpochMetrics]
    resumed_from_epoch = 0
    training_seconds = 0.0
    early_stopped = False
    if resume and training_state_path.is_file():
        state = torch.load(training_state_path, map_location=device, weights_only=True)
        if not isinstance(state, dict) or state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("training state does not match the requested cache or hyperparameters")
        saved_head_config = state.get("head_config")
        if saved_head_config is not None and saved_head_config != head_configuration:
            raise ValueError("training state CTC Head structure does not match the requested Head")
        if saved_head_config is None and head_configuration["head_type"] != "linear":
            raise ValueError("training state has no CTC Head structure metadata")
        completed_epoch = _required_int(state, "completed_epoch")
        if completed_epoch > epochs:
            raise ValueError("requested epochs are fewer than the resumed completed epoch")
        head.load_state_dict(state["head_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        raw_history = state.get("history")
        if not isinstance(raw_history, list):
            raise ValueError("training state history must be a list")
        history = [_sharded_metric_from_dict(value) for value in raw_history]
        if len(history) != completed_epoch:
            raise ValueError("training state history is incomplete")
        initial_validation = _epoch_metric_from_dict(state["initial_validation"])
        best_epoch = _required_int(state, "best_epoch")
        best_validation = _epoch_metric_from_dict(state["best_validation"])
        raw_best_train = state.get("best_train")
        best_train = (
            _epoch_metric_from_dict(raw_best_train)
            if isinstance(raw_best_train, dict)
            else None
        )
        patience_reference_value = float(state["patience_reference_value"])
        stale_epochs = _required_int(state, "stale_epochs")
        training_seconds = float(state["training_seconds"])
        resumed_from_epoch = completed_epoch
        start_epoch = completed_epoch + 1
        if not best_checkpoint_path.is_file():
            raise FileNotFoundError("best CTC checkpoint is missing from the resumed run")
        _write_metric_history(metrics_path, history)
        print(f"resumed full CTC training from epoch {completed_epoch}", flush=True)
    else:
        if not resume and any(
            path.exists()
            for path in (training_state_path, metrics_path, best_checkpoint_path)
        ):
            raise ValueError("training output already exists; use --resume or a new output dir")
        print("evaluating fresh random CTC head on validation cache", flush=True)
        initial_validation = _evaluate_cache(
            head,
            validation_cache,
            device=device,
            batch_size=train_batch_size,
            blank_id=blank_id,
            epoch=0,
        )
        best_epoch = 0
        best_train = None
        best_validation = initial_validation
        patience_reference_value = _early_stopping_value(
            initial_validation,
            early_stopping_metric,
        )
        stale_epochs = 0
        history = []
        start_epoch = 1
        save_ctc_head_checkpoint(best_checkpoint_path, head, vocab, best_validation, seed)
        _save_training_state(
            training_state_path,
            run_fingerprint=run_fingerprint,
            completed_epoch=0,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            initial_validation=initial_validation,
            best_epoch=best_epoch,
            best_train=best_train,
            best_validation=best_validation,
            patience_reference_value=patience_reference_value,
            stale_epochs=stale_epochs,
            training_seconds=training_seconds,
            history=history,
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.monotonic()
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _train_cache_epoch(
            head,
            optimizer,
            train_cache,
            device=device,
            batch_size=train_batch_size,
            blank_id=blank_id,
            max_gradient_norm=max_gradient_norm,
            epoch=epoch,
            seed=seed,
            log_every_shards=log_every_shards,
        )
        validation_metrics = _evaluate_cache(
            head,
            validation_cache,
            device=device,
            batch_size=train_batch_size,
            blank_id=blank_id,
            epoch=epoch,
        )
        scheduler.step(validation_metrics.loss)
        epoch_seconds = time.monotonic() - epoch_started
        training_seconds += epoch_seconds
        current = ShardedEpochMetrics(
            epoch=epoch,
            learning_rate=current_learning_rate,
            train=train_metrics,
            validation=validation_metrics,
            epoch_seconds=epoch_seconds,
        )
        history.append(current)

        if (
            validation_metrics.phoneme_error_rate,
            validation_metrics.loss,
        ) < (best_validation.phoneme_error_rate, best_validation.loss):
            best_epoch = epoch
            best_train = train_metrics
            best_validation = validation_metrics
            save_ctc_head_checkpoint(best_checkpoint_path, head, vocab, best_validation, seed)

        early_stopping_value = _early_stopping_value(
            validation_metrics,
            early_stopping_metric,
        )
        if early_stopping_value < patience_reference_value - early_stopping_min_delta:
            patience_reference_value = early_stopping_value
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_ctc_head_checkpoint(latest_checkpoint_path, head, vocab, validation_metrics, seed)
        _save_training_state(
            training_state_path,
            run_fingerprint=run_fingerprint,
            completed_epoch=epoch,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            initial_validation=initial_validation,
            best_epoch=best_epoch,
            best_train=best_train,
            best_validation=best_validation,
            patience_reference_value=patience_reference_value,
            stale_epochs=stale_epochs,
            training_seconds=training_seconds,
            history=history,
        )
        _write_metric_history(metrics_path, history)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.loss:.6f} "
            f"train_PER={train_metrics.phoneme_error_rate:.4f} "
            f"val_loss={validation_metrics.loss:.6f} "
            f"val_PER={validation_metrics.phoneme_error_rate:.4f} "
            f"best_val_PER={best_validation.phoneme_error_rate:.4f} "
            f"lr={float(optimizer.param_groups[0]['lr']):.2e} "
            f"early_stop={early_stopping_metric} stale={stale_epochs} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if epoch >= minimum_epochs and stale_epochs >= early_stopping_patience:
            early_stopped = True
            print(
                f"early stopping at epoch {epoch}; best validation epoch={best_epoch}",
                flush=True,
            )
            break

    if not history:
        raise RuntimeError("CTC training completed without an epoch")
    final = history[-1]
    report = ShardedCtcReport(
        train_cache_dir=str(train_cache.root),
        validation_cache_dir=str(validation_cache.root),
        vocab_path=str(Path(vocab_path)),
        train_sample_count=train_cache.sample_count,
        validation_sample_count=validation_cache.sample_count,
        train_shard_count=train_cache.shard_count,
        validation_shard_count=validation_cache.shard_count,
        train_feature_bytes=train_cache.feature_bytes,
        validation_feature_bytes=validation_cache.feature_bytes,
        num_classes=len(vocab.tokens),
        blank_id=blank_id,
        head_config=head_configuration,
        encoder_frozen_parameters=train_cache.encoder_frozen_parameters,
        ctc_trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
        train_batch_size=train_batch_size,
        initial_learning_rate=learning_rate,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        epochs_requested=epochs,
        epochs_completed=final.epoch,
        resumed_from_epoch=resumed_from_epoch,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
        initial_validation_loss=initial_validation.loss,
        initial_validation_phoneme_error_rate=initial_validation.phoneme_error_rate,
        best_epoch=best_epoch,
        best_train_loss=best_train.loss if best_train else None,
        best_train_phoneme_error_rate=best_train.phoneme_error_rate if best_train else None,
        best_validation_loss=best_validation.loss,
        best_validation_phoneme_error_rate=best_validation.phoneme_error_rate,
        final_train_loss=final.train.loss,
        final_train_phoneme_error_rate=final.train.phoneme_error_rate,
        final_validation_loss=final.validation.loss,
        final_validation_phoneme_error_rate=final.validation.phoneme_error_rate,
        training_seconds=training_seconds,
        best_checkpoint_path=str(best_checkpoint_path),
        latest_checkpoint_path=str(latest_checkpoint_path),
        training_state_path=str(training_state_path),
        metrics_path=str(metrics_path),
        early_stopped=early_stopped,
        cache_sha256_verified=train_cache.sha256_verified
        and validation_cache.sha256_verified,
        selection_metric="validation_phoneme_error_rate_then_validation_loss",
        test_set_used=False,
        status="completed",
    )
    _write_json(destination / "report.json", report.to_dict())
    return report


def _train_cache_epoch(
    head: Any,
    optimizer: Any,
    cache: DiskFeatureCache,
    *,
    device: Any,
    batch_size: int,
    blank_id: int,
    max_gradient_norm: float,
    epoch: int,
    seed: int,
    log_every_shards: int,
) -> EpochMetrics:
    import torch

    from qwen_hotword.modeling.ctc_head import compute_ctc

    generator = random.Random(seed + epoch)
    descriptors = list(cache.shards)
    generator.shuffle(descriptors)
    head.train()
    loss_sum = 0.0
    sample_count = 0
    total_errors = 0
    total_reference = 0
    for shard_position, descriptor in enumerate(descriptors, start=1):
        samples = load_feature_shard(descriptor, num_classes=head.num_classes)
        indices = list(range(len(samples)))
        generator.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = [samples[index] for index in indices[start : start + batch_size]]
            hidden_states, input_lengths, targets, target_lengths = collate_cached_samples(
                batch,
                device=device,
                blank_id=blank_id,
            )
            optimizer.zero_grad(set_to_none=True)
            computation = compute_ctc(
                head,
                hidden_states,
                input_lengths,
                targets,
                target_lengths,
                blank_id=blank_id,
            )
            computation.loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_gradient_norm)
            optimizer.step()
            errors, references = _batch_error_counts(
                computation.logits,
                computation.input_lengths,
                batch,
                blank_id=blank_id,
            )
            loss_sum += float(computation.loss.item()) * len(batch)
            sample_count += len(batch)
            total_errors += errors
            total_reference += references
        if (
            shard_position == 1
            or shard_position % log_every_shards == 0
            or shard_position == len(descriptors)
        ):
            print(
                f"epoch={epoch:03d} trained_shards={shard_position}/{len(descriptors)} "
                f"samples={sample_count}/{cache.sample_count}",
                flush=True,
            )
        del samples
    return _finalize_metrics(
        epoch,
        loss_sum=loss_sum,
        sample_count=sample_count,
        total_errors=total_errors,
        total_reference=total_reference,
    )


def _evaluate_cache(
    head: Any,
    cache: DiskFeatureCache,
    *,
    device: Any,
    batch_size: int,
    blank_id: int,
    epoch: int,
) -> EpochMetrics:
    import torch

    from qwen_hotword.modeling.ctc_head import compute_ctc

    head.eval()
    loss_sum = 0.0
    sample_count = 0
    total_errors = 0
    total_reference = 0
    with torch.no_grad():
        for descriptor in cache.shards:
            samples = load_feature_shard(descriptor, num_classes=head.num_classes)
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                hidden_states, input_lengths, targets, target_lengths = (
                    collate_cached_samples(batch, device=device, blank_id=blank_id)
                )
                computation = compute_ctc(
                    head,
                    hidden_states,
                    input_lengths,
                    targets,
                    target_lengths,
                    blank_id=blank_id,
                )
                errors, references = _batch_error_counts(
                    computation.logits,
                    computation.input_lengths,
                    batch,
                    blank_id=blank_id,
                )
                loss_sum += float(computation.loss.item()) * len(batch)
                sample_count += len(batch)
                total_errors += errors
                total_reference += references
            del samples
    return _finalize_metrics(
        epoch,
        loss_sum=loss_sum,
        sample_count=sample_count,
        total_errors=total_errors,
        total_reference=total_reference,
    )


def _batch_error_counts(
    logits: Any,
    input_lengths: Any,
    samples: list[CachedSample],
    *,
    blank_id: int,
) -> tuple[int, int]:
    from qwen_hotword.training.edit_distance import sequence_edit_distance

    predictions = logits.argmax(dim=-1).detach().cpu()
    errors = 0
    references = 0
    for row, sample in enumerate(samples):
        input_length = int(input_lengths[row].item())
        hypothesis = tuple(
            collapse_ctc_ids(
                predictions[row, :input_length].tolist(),
                blank_id=blank_id,
            )
        )
        errors += sequence_edit_distance(sample.token_ids, hypothesis)
        references += len(sample.token_ids)
    return errors, references


def _finalize_metrics(
    epoch: int,
    *,
    loss_sum: float,
    sample_count: int,
    total_errors: int,
    total_reference: int,
) -> EpochMetrics:
    if sample_count <= 0 or total_reference <= 0:
        raise ValueError("metric accumulator received no samples or reference labels")
    return EpochMetrics(
        epoch=epoch,
        loss=loss_sum / sample_count,
        phoneme_error_rate=total_errors / total_reference,
        phoneme_errors=total_errors,
        reference_phonemes=total_reference,
    )


def _validate_training_arguments(
    train_cache: DiskFeatureCache,
    validation_cache: DiskFeatureCache,
    vocab: PhonemeVocab,
    *,
    epochs: int,
    minimum_epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    early_stopping_metric: EarlyStoppingMetric,
    train_batch_size: int,
    learning_rate: float,
    max_gradient_norm: float,
    scheduler_patience: int,
    scheduler_factor: float,
    minimum_learning_rate: float,
    log_every_shards: int,
) -> None:
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("formal training requires train and validation caches")
    if train_cache.fingerprint == validation_cache.fingerprint:
        raise ValueError("train and validation caches must be distinct")
    if train_cache.encoder_frozen_parameters != validation_cache.encoder_frozen_parameters:
        raise ValueError("encoder parameter counts differ between feature caches")
    if not vocab.tokens or vocab.tokens[0] != "<blank>":
        raise ValueError("CTC vocabulary must place <blank> at token ID 0")
    if epochs <= 0 or minimum_epochs <= 0 or minimum_epochs > epochs:
        raise ValueError("minimum_epochs must be positive and no greater than epochs")
    if early_stopping_patience <= 0 or scheduler_patience < 0:
        raise ValueError("early stopping patience must be positive")
    if early_stopping_min_delta < 0:
        raise ValueError("early stopping min delta cannot be negative")
    if early_stopping_metric not in {"validation_loss", "validation_per"}:
        raise ValueError("early stopping metric must be validation_loss or validation_per")
    if train_batch_size <= 0 or log_every_shards <= 0:
        raise ValueError("batch size and shard log interval must be positive")
    if learning_rate <= 0 or minimum_learning_rate <= 0 or max_gradient_norm <= 0:
        raise ValueError("learning rates and gradient norm must be positive")
    if not 0 < scheduler_factor < 1:
        raise ValueError("scheduler factor must be between zero and one")


def _training_fingerprint(
    train_cache: DiskFeatureCache,
    validation_cache: DiskFeatureCache,
    vocab: PhonemeVocab,
    **hyperparameters: object,
) -> str:
    value = {
        "schema_version": SHARDED_CTC_SCHEMA_VERSION,
        "train_cache": train_cache.fingerprint,
        "validation_cache": validation_cache.fingerprint,
        "vocab_tokens": list(vocab.tokens),
        "hyperparameters": hyperparameters,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _early_stopping_value(
    metrics: EpochMetrics,
    metric: EarlyStoppingMetric,
) -> float:
    if metric == "validation_loss":
        return metrics.loss
    if metric == "validation_per":
        return metrics.phoneme_error_rate
    raise ValueError("early stopping metric must be validation_loss or validation_per")


def _save_training_state(
    path: Path,
    *,
    run_fingerprint: str,
    completed_epoch: int,
    head: Any,
    optimizer: Any,
    scheduler: Any,
    initial_validation: EpochMetrics,
    best_epoch: int,
    best_train: EpochMetrics | None,
    best_validation: EpochMetrics,
    patience_reference_value: float,
    stale_epochs: int,
    training_seconds: float,
    history: list[ShardedEpochMetrics],
) -> None:
    import torch

    from qwen_hotword.modeling.ctc_head import ctc_head_config

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(
        {
            "schema_version": SHARDED_CTC_SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "completed_epoch": completed_epoch,
            "head_config": ctc_head_config(head),
            "head_state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "initial_validation": initial_validation.to_dict(),
            "best_epoch": best_epoch,
            "best_train": best_train.to_dict() if best_train else None,
            "best_validation": best_validation.to_dict(),
            "patience_reference_value": patience_reference_value,
            "stale_epochs": stale_epochs,
            "training_seconds": training_seconds,
            "history": [metric.to_dict() for metric in history],
        },
        temporary,
    )
    temporary.replace(path)


def _sharded_metric_from_dict(value: object) -> ShardedEpochMetrics:
    if not isinstance(value, dict):
        raise ValueError("training history entry must be an object")
    return ShardedEpochMetrics(
        epoch=_required_int(value, "epoch"),
        learning_rate=float(value["learning_rate"]),
        train=_epoch_metric_from_dict(value["train"]),
        validation=_epoch_metric_from_dict(value["validation"]),
        epoch_seconds=float(value["epoch_seconds"]),
    )


def _epoch_metric_from_dict(value: object) -> EpochMetrics:
    if not isinstance(value, dict):
        raise ValueError("epoch metrics must be an object")
    return EpochMetrics(
        epoch=_required_int(value, "epoch"),
        loss=float(value["loss"]),
        phoneme_error_rate=float(value["phoneme_error_rate"]),
        phoneme_errors=_required_int(value, "phoneme_errors"),
        reference_phonemes=_required_int(value, "reference_phonemes"),
    )


def _write_metric_history(path: Path, history: list[ShardedEpochMetrics]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for metric in history:
            handle.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")
    temporary.replace(path)


def _validate_identity(raw: object, path: Path, *, label: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"feature cache has no {label} identity")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if raw.get("size") != path.stat().st_size or raw.get("sha256") != _sha256_file(path):
        raise ValueError(f"feature cache {label} identity does not match: {path}")


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _required_int(value: dict[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ValueError(f"invalid non-negative integer field: {key}")
    return result


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"invalid string field: {key}")
    return result


def _strings_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
