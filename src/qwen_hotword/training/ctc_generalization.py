from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    EpochMetrics,
    ExperimentRecord,
    collate_cached_samples,
    evaluate_cached_samples,
    save_ctc_head_checkpoint,
)


@dataclass(frozen=True)
class GeneralizationEpochMetrics:
    epoch: int
    learning_rate: float
    train: EpochMetrics
    validation: EpochMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class GeneralizationReport:
    train_manifest_path: str
    validation_manifest_path: str
    vocab_path: str
    train_sample_count: int
    validation_sample_count: int
    num_classes: int
    blank_id: int
    encoder_frozen_parameters: int
    ctc_trainable_parameters: int
    encoder_batch_size: int
    train_batch_size: int
    initial_learning_rate: float
    final_learning_rate: float
    epochs_requested: int
    epochs_completed: int
    minimum_epochs: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    best_epoch: int
    best_train_loss: float
    best_train_phoneme_error_rate: float
    best_validation_loss: float
    best_validation_phoneme_error_rate: float
    final_train_loss: float
    final_train_phoneme_error_rate: float
    final_validation_loss: float
    final_validation_phoneme_error_rate: float
    train_feature_extraction_seconds: float
    validation_feature_extraction_seconds: float
    training_seconds: float
    best_checkpoint_path: str
    latest_checkpoint_path: str
    metrics_path: str
    early_stopped: bool
    selection_metric: str
    test_set_used: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_disjoint_records(
    train_records: list[ExperimentRecord],
    validation_records: list[ExperimentRecord],
) -> None:
    train_ids = {record.sample_id for record in train_records}
    validation_ids = {record.sample_id for record in validation_records}
    duplicate_ids = train_ids & validation_ids
    if duplicate_ids:
        raise ValueError(
            "train and validation manifests share sample IDs: "
            f"{sorted(duplicate_ids)[:5]}"
        )

    train_audio = {record.audio_path.resolve() for record in train_records}
    validation_audio = {record.audio_path.resolve() for record in validation_records}
    duplicate_audio = train_audio & validation_audio
    if duplicate_audio:
        raise ValueError(
            "train and validation manifests share audio paths: "
            f"{sorted(str(path) for path in duplicate_audio)[:5]}"
        )


def validation_rank(metrics: GeneralizationEpochMetrics) -> tuple[float, float]:
    return (
        metrics.validation.phoneme_error_rate,
        metrics.validation.loss,
    )


def train_cached_ctc_with_validation(
    train_samples: list[CachedSample],
    validation_samples: list[CachedSample],
    vocab: PhonemeVocab,
    output_dir: str | Path,
    *,
    train_manifest_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    device: Any,
    encoder_frozen_parameters: int,
    train_feature_extraction_seconds: float,
    validation_feature_extraction_seconds: float,
    epochs: int = 30,
    minimum_epochs: int = 5,
    early_stopping_patience: int = 6,
    early_stopping_min_delta: float = 0.001,
    train_batch_size: int = 64,
    encoder_batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_gradient_norm: float = 5.0,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    minimum_learning_rate: float = 1e-5,
    seed: int = 20_260_717,
    log_every: int = 1,
) -> GeneralizationReport:
    import torch

    from qwen_hotword.modeling.ctc_head import LinearCtcHead, compute_ctc

    if not train_samples or not validation_samples:
        raise ValueError("train and validation caches must both contain samples")
    if epochs <= 0 or minimum_epochs <= 0 or minimum_epochs > epochs:
        raise ValueError("minimum_epochs must be positive and no greater than epochs")
    if early_stopping_patience <= 0 or scheduler_patience < 0:
        raise ValueError("early stopping patience must be positive")
    if train_batch_size <= 0 or encoder_batch_size <= 0 or log_every <= 0:
        raise ValueError("batch sizes and log interval must be positive")
    if learning_rate <= 0 or minimum_learning_rate <= 0 or max_gradient_norm <= 0:
        raise ValueError("learning rates and gradient norm must be positive")
    if not 0 < scheduler_factor < 1:
        raise ValueError("scheduler_factor must be between zero and one")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta cannot be negative")

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    blank_id = 0
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = LinearCtcHead(1024, len(vocab.tokens)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=minimum_learning_rate,
    )
    metrics_path = destination / "metrics.jsonl"
    best_checkpoint_path = destination / "ctc_head_best.pt"
    latest_checkpoint_path = destination / "ctc_head_latest.pt"

    initial = _evaluate_epoch(
        head,
        train_samples,
        validation_samples,
        device=device,
        batch_size=train_batch_size,
        blank_id=blank_id,
        epoch=0,
        learning_rate=learning_rate,
    )
    history = [initial]
    _write_metrics(metrics_path, history)
    best = initial
    patience_reference_per = initial.validation.phoneme_error_rate
    stale_epochs = 0
    save_ctc_head_checkpoint(
        best_checkpoint_path,
        head,
        vocab,
        best.validation,
        seed,
    )

    generator = random.Random(seed)
    started = time.monotonic()
    early_stopped = False
    for epoch in range(1, epochs + 1):
        indices = list(range(len(train_samples)))
        generator.shuffle(indices)
        head.train()
        for start in range(0, len(indices), train_batch_size):
            batch = [
                train_samples[index]
                for index in indices[start : start + train_batch_size]
            ]
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

        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        metrics = _evaluate_epoch(
            head,
            train_samples,
            validation_samples,
            device=device,
            batch_size=train_batch_size,
            blank_id=blank_id,
            epoch=epoch,
            learning_rate=current_learning_rate,
        )
        history.append(metrics)
        _append_metric(metrics_path, metrics)
        scheduler.step(metrics.validation.loss)

        if validation_rank(metrics) < validation_rank(best):
            best = metrics
            save_ctc_head_checkpoint(
                best_checkpoint_path,
                head,
                vocab,
                best.validation,
                seed,
            )

        if (
            metrics.validation.phoneme_error_rate
            < patience_reference_per - early_stopping_min_delta
        ):
            patience_reference_per = metrics.validation.phoneme_error_rate
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            next_learning_rate = float(optimizer.param_groups[0]["lr"])
            print(
                f"epoch={epoch:03d} "
                f"train_loss={metrics.train.loss:.6f} "
                f"train_PER={metrics.train.phoneme_error_rate:.4f} "
                f"val_loss={metrics.validation.loss:.6f} "
                f"val_PER={metrics.validation.phoneme_error_rate:.4f} "
                f"best_val_PER={best.validation.phoneme_error_rate:.4f} "
                f"lr={next_learning_rate:.2e} stale={stale_epochs}",
                flush=True,
            )

        if epoch >= minimum_epochs and stale_epochs >= early_stopping_patience:
            early_stopped = True
            print(
                f"early stopping at epoch {epoch}; best validation epoch={best.epoch}",
                flush=True,
            )
            break

    training_seconds = time.monotonic() - started
    final = history[-1]
    save_ctc_head_checkpoint(
        latest_checkpoint_path,
        head,
        vocab,
        final.validation,
        seed,
    )
    report = GeneralizationReport(
        train_manifest_path=str(Path(train_manifest_path)),
        validation_manifest_path=str(Path(validation_manifest_path)),
        vocab_path=str(Path(vocab_path)),
        train_sample_count=len(train_samples),
        validation_sample_count=len(validation_samples),
        num_classes=len(vocab.tokens),
        blank_id=blank_id,
        encoder_frozen_parameters=encoder_frozen_parameters,
        ctc_trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
        encoder_batch_size=encoder_batch_size,
        train_batch_size=train_batch_size,
        initial_learning_rate=learning_rate,
        final_learning_rate=float(optimizer.param_groups[0]["lr"]),
        epochs_requested=epochs,
        epochs_completed=final.epoch,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        best_epoch=best.epoch,
        best_train_loss=best.train.loss,
        best_train_phoneme_error_rate=best.train.phoneme_error_rate,
        best_validation_loss=best.validation.loss,
        best_validation_phoneme_error_rate=best.validation.phoneme_error_rate,
        final_train_loss=final.train.loss,
        final_train_phoneme_error_rate=final.train.phoneme_error_rate,
        final_validation_loss=final.validation.loss,
        final_validation_phoneme_error_rate=final.validation.phoneme_error_rate,
        train_feature_extraction_seconds=train_feature_extraction_seconds,
        validation_feature_extraction_seconds=validation_feature_extraction_seconds,
        training_seconds=training_seconds,
        best_checkpoint_path=str(best_checkpoint_path),
        latest_checkpoint_path=str(latest_checkpoint_path),
        metrics_path=str(metrics_path),
        early_stopped=early_stopped,
        selection_metric="validation_phoneme_error_rate_then_validation_loss",
        test_set_used=False,
        status="completed",
    )
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_epoch(
    head: Any,
    train_samples: list[CachedSample],
    validation_samples: list[CachedSample],
    *,
    device: Any,
    batch_size: int,
    blank_id: int,
    epoch: int,
    learning_rate: float,
) -> GeneralizationEpochMetrics:
    train_metrics = evaluate_cached_samples(
        head,
        train_samples,
        device=device,
        batch_size=batch_size,
        blank_id=blank_id,
        epoch=epoch,
    )
    validation_metrics = evaluate_cached_samples(
        head,
        validation_samples,
        device=device,
        batch_size=batch_size,
        blank_id=blank_id,
        epoch=epoch,
    )
    return GeneralizationEpochMetrics(
        epoch=epoch,
        learning_rate=learning_rate,
        train=train_metrics,
        validation=validation_metrics,
    )


def _write_metrics(path: Path, metrics: list[GeneralizationEpochMetrics]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")


def _append_metric(path: Path, metric: GeneralizationEpochMetrics) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")
