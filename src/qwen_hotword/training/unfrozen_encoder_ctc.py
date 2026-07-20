from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_generalization import validate_disjoint_records
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    EpochMetrics,
    ExperimentRecord,
    build_audio_prompt,
    collapse_ctc_ids,
    load_experiment_records,
)
from qwen_hotword.training.sharded_ctc import EarlyStoppingMetric

SelectionMetric = Literal["validation_phoneme_error_rate_then_validation_loss"]


@dataclass(frozen=True)
class EncoderUnfreezePlan:
    trainable_parameter_names: tuple[str, ...]
    trainable_module_names: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int
    unfreeze_all_encoder: bool
    unfreeze_last_encoder_layers: int
    train_ln_post: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UnfrozenEpochMetrics:
    epoch: int
    head_learning_rate: float
    encoder_learning_rate: float | None
    train: EpochMetrics
    validation: EpochMetrics
    epoch_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "head_learning_rate": self.head_learning_rate,
            "encoder_learning_rate": self.encoder_learning_rate,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "epoch_seconds": self.epoch_seconds,
        }


@dataclass(frozen=True)
class UnfrozenCtcReport:
    train_manifest_path: str
    validation_manifest_path: str
    vocab_path: str
    train_sample_count: int
    validation_sample_count: int
    num_classes: int
    blank_id: int
    ctc_trainable_parameters: int
    encoder_trainable_parameters: int
    encoder_frozen_parameters: int
    unfreeze_plan: dict[str, object]
    train_batch_size: int
    validation_batch_size: int
    gradient_accumulation_steps: int
    effective_train_batch_size: int
    initial_head_learning_rate: float
    initial_encoder_learning_rate: float | None
    final_head_learning_rate: float
    final_encoder_learning_rate: float | None
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
    selection_metric: SelectionMetric
    test_set_used: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OnTheFlyBatch:
    hidden_states: Any
    input_lengths: Any
    targets: Any
    target_lengths: Any
    samples: tuple[CachedSample, ...]


def configure_audio_tower_unfreeze(
    audio_tower: Any,
    *,
    unfreeze_last_encoder_layers: int = 1,
    train_ln_post: bool = True,
    unfreeze_all_encoder: bool = False,
) -> EncoderUnfreezePlan:
    """Freeze an audio tower, then selectively reopen encoder parameters."""
    if unfreeze_last_encoder_layers < 0:
        raise ValueError("unfreeze_last_encoder_layers cannot be negative")
    if unfreeze_all_encoder and unfreeze_last_encoder_layers:
        raise ValueError("use either unfreeze_all_encoder or unfreeze_last_encoder_layers")

    audio_tower.eval()
    for parameter in audio_tower.parameters():
        parameter.requires_grad_(False)

    trainable_modules: list[str] = []
    if unfreeze_all_encoder:
        audio_tower.train()
        for parameter in audio_tower.parameters():
            parameter.requires_grad_(True)
        trainable_modules.append("audio_tower")
    else:
        if unfreeze_last_encoder_layers:
            layers = getattr(audio_tower, "layers", None)
            if (
                layers is None
                or not hasattr(layers, "__len__")
                or not hasattr(layers, "__getitem__")
            ):
                raise ValueError("audio_tower does not expose a layers sequence")
            layer_count = len(layers)
            if unfreeze_last_encoder_layers > layer_count:
                raise ValueError(
                    "unfreeze_last_encoder_layers exceeds available audio encoder layers"
                )
            first_layer = layer_count - unfreeze_last_encoder_layers
            for layer_index in range(first_layer, layer_count):
                _mark_module_trainable(layers[layer_index])
                trainable_modules.append(f"layers.{layer_index}")
        if train_ln_post:
            ln_post = getattr(audio_tower, "ln_post", None)
            if ln_post is None:
                raise ValueError("audio_tower does not expose ln_post")
            _mark_module_trainable(ln_post)
            trainable_modules.append("ln_post")

    trainable_parameter_names = tuple(
        name for name, parameter in audio_tower.named_parameters() if parameter.requires_grad
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in audio_tower.parameters() if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel() for parameter in audio_tower.parameters() if not parameter.requires_grad
    )
    return EncoderUnfreezePlan(
        trainable_parameter_names=trainable_parameter_names,
        trainable_module_names=tuple(trainable_modules),
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        unfreeze_all_encoder=unfreeze_all_encoder,
        unfreeze_last_encoder_layers=unfreeze_last_encoder_layers,
        train_ln_post=train_ln_post,
    )


def train_unfrozen_encoder_ctc(
    wrapper: Any,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    vocab: PhonemeVocab,
    output_dir: str | Path,
    *,
    vocab_path: str | Path,
    device: Any,
    epochs: int = 5,
    minimum_epochs: int = 2,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.001,
    early_stopping_metric: EarlyStoppingMetric = "validation_loss",
    train_batch_size: int = 1,
    validation_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    head_learning_rate: float = 1e-3,
    encoder_learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    max_gradient_norm: float = 1.0,
    scheduler_patience: int = 1,
    scheduler_factor: float = 0.5,
    minimum_head_learning_rate: float = 1e-5,
    minimum_encoder_learning_rate: float = 1e-7,
    seed: int = 20_260_720,
    log_every_batches: int = 200,
    resume: bool = False,
    unfreeze_last_encoder_layers: int = 1,
    train_ln_post: bool = True,
    unfreeze_all_encoder: bool = False,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> UnfrozenCtcReport:
    import torch

    from qwen_hotword.modeling.ctc_head import LinearCtcHead

    _validate_arguments(
        vocab,
        epochs=epochs,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        head_learning_rate=head_learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        max_gradient_norm=max_gradient_norm,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        minimum_head_learning_rate=minimum_head_learning_rate,
        minimum_encoder_learning_rate=minimum_encoder_learning_rate,
        log_every_batches=log_every_batches,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
    )
    train_records = load_experiment_records(
        train_manifest,
        num_classes=len(vocab.tokens),
        blank_id=0,
        expected_experiment="full-ctc-v1",
        expected_split="train",
    )
    validation_records = load_experiment_records(
        validation_manifest,
        num_classes=len(vocab.tokens),
        blank_id=0,
        expected_experiment="full-ctc-v1",
        expected_split="validation",
    )
    validate_disjoint_records(train_records, validation_records)
    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_validation_samples is not None:
        validation_records = validation_records[:max_validation_samples]

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    best_checkpoint_path = destination / "ctc_encoder_best.pt"
    latest_checkpoint_path = destination / "ctc_encoder_latest.pt"
    training_state_path = destination / "training_state_latest.pt"
    blank_id = 0

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = wrapper.model
    model.eval()
    audio_tower = model.thinker.audio_tower
    unfreeze_plan = configure_audio_tower_unfreeze(
        audio_tower,
        unfreeze_last_encoder_layers=unfreeze_last_encoder_layers,
        train_ln_post=train_ln_post,
        unfreeze_all_encoder=unfreeze_all_encoder,
    )
    _set_trainable_modules_train(audio_tower, unfreeze_plan)

    head = LinearCtcHead(1024, len(vocab.tokens)).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(
            head,
            audio_tower,
            head_learning_rate=head_learning_rate,
            encoder_learning_rate=encoder_learning_rate,
            weight_decay=weight_decay,
        )
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=early_stopping_min_delta,
        threshold_mode="abs",
        min_lr=[
            minimum_head_learning_rate,
            *([minimum_encoder_learning_rate] if unfreeze_plan.trainable_parameters else []),
        ],
    )
    run_fingerprint = _training_fingerprint(
        train_manifest_path=train_manifest,
        validation_manifest_path=validation_manifest,
        vocab=vocab,
        train_sample_count=len(train_records),
        validation_sample_count=len(validation_records),
        unfreeze_plan=unfreeze_plan,
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        head_learning_rate=head_learning_rate,
        encoder_learning_rate=encoder_learning_rate,
        weight_decay=weight_decay,
        max_gradient_norm=max_gradient_norm,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        minimum_head_learning_rate=minimum_head_learning_rate,
        minimum_encoder_learning_rate=minimum_encoder_learning_rate,
        minimum_epochs=minimum_epochs,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_metric=early_stopping_metric,
        seed=seed,
    )

    history: list[UnfrozenEpochMetrics]
    resumed_from_epoch = 0
    training_seconds = 0.0
    early_stopped = False
    if resume and training_state_path.is_file():
        state = torch.load(training_state_path, map_location=device, weights_only=True)
        if not isinstance(state, dict) or state.get("run_fingerprint") != run_fingerprint:
            raise ValueError("training state does not match the requested run")
        completed_epoch = _required_int(state, "completed_epoch")
        if completed_epoch > epochs:
            raise ValueError("requested epochs are fewer than the resumed completed epoch")
        head.load_state_dict(state["head_state_dict"], strict=True)
        _load_audio_tower_trainable_state(audio_tower, state["audio_tower_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        history = [_epoch_history_from_dict(value) for value in _required_list(state, "history")]
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
            raise FileNotFoundError("best unfrozen checkpoint is missing from the resumed run")
        _write_metric_history(metrics_path, history)
        print(f"resumed unfrozen encoder CTC training from epoch {completed_epoch}", flush=True)
    else:
        if not resume and any(
            path.exists()
            for path in (training_state_path, metrics_path, best_checkpoint_path)
        ):
            raise ValueError("training output already exists; use --resume or a new output dir")
        print("evaluating fresh unfrozen-control CTC head on validation manifest", flush=True)
        initial_validation = _evaluate_records(
            wrapper,
            head,
            validation_records,
            device=device,
            batch_size=validation_batch_size,
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
        _save_checkpoint(
            best_checkpoint_path,
            head=head,
            audio_tower=audio_tower,
            vocab=vocab,
            metrics=best_validation,
            unfreeze_plan=unfreeze_plan,
            seed=seed,
        )
        _save_training_state(
            training_state_path,
            run_fingerprint=run_fingerprint,
            completed_epoch=0,
            head=head,
            audio_tower=audio_tower,
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
        train_metrics = _train_records_epoch(
            wrapper,
            head,
            optimizer,
            train_records,
            device=device,
            batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            blank_id=blank_id,
            max_gradient_norm=max_gradient_norm,
            epoch=epoch,
            seed=seed,
            log_every_batches=log_every_batches,
            unfreeze_plan=unfreeze_plan,
        )
        validation_metrics = _evaluate_records(
            wrapper,
            head,
            validation_records,
            device=device,
            batch_size=validation_batch_size,
            blank_id=blank_id,
            epoch=epoch,
        )
        scheduler.step(validation_metrics.loss)
        epoch_seconds = time.monotonic() - epoch_started
        training_seconds += epoch_seconds
        current = UnfrozenEpochMetrics(
            epoch=epoch,
            head_learning_rate=float(optimizer.param_groups[0]["lr"]),
            encoder_learning_rate=_encoder_group_lr(optimizer),
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
            _save_checkpoint(
                best_checkpoint_path,
                head=head,
                audio_tower=audio_tower,
                vocab=vocab,
                metrics=best_validation,
                unfreeze_plan=unfreeze_plan,
                seed=seed,
            )

        early_stopping_value = _early_stopping_value(
            validation_metrics,
            early_stopping_metric,
        )
        if early_stopping_value < patience_reference_value - early_stopping_min_delta:
            patience_reference_value = early_stopping_value
            stale_epochs = 0
        else:
            stale_epochs += 1

        _save_checkpoint(
            latest_checkpoint_path,
            head=head,
            audio_tower=audio_tower,
            vocab=vocab,
            metrics=validation_metrics,
            unfreeze_plan=unfreeze_plan,
            seed=seed,
        )
        _save_training_state(
            training_state_path,
            run_fingerprint=run_fingerprint,
            completed_epoch=epoch,
            head=head,
            audio_tower=audio_tower,
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
            f"head_lr={float(optimizer.param_groups[0]['lr']):.2e} "
            f"encoder_lr={_encoder_group_lr(optimizer)} "
            f"stale={stale_epochs} seconds={epoch_seconds:.1f}",
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
        raise RuntimeError("unfrozen encoder CTC training completed without an epoch")
    final = history[-1]
    report = UnfrozenCtcReport(
        train_manifest_path=str(Path(train_manifest)),
        validation_manifest_path=str(Path(validation_manifest)),
        vocab_path=str(Path(vocab_path)),
        train_sample_count=len(train_records),
        validation_sample_count=len(validation_records),
        num_classes=len(vocab.tokens),
        blank_id=blank_id,
        ctc_trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
        encoder_trainable_parameters=unfreeze_plan.trainable_parameters,
        encoder_frozen_parameters=unfreeze_plan.frozen_parameters,
        unfreeze_plan=unfreeze_plan.to_dict(),
        train_batch_size=train_batch_size,
        validation_batch_size=validation_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_train_batch_size=train_batch_size * gradient_accumulation_steps,
        initial_head_learning_rate=head_learning_rate,
        initial_encoder_learning_rate=encoder_learning_rate
        if unfreeze_plan.trainable_parameters
        else None,
        final_head_learning_rate=float(optimizer.param_groups[0]["lr"]),
        final_encoder_learning_rate=_encoder_group_lr(optimizer),
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
        selection_metric="validation_phoneme_error_rate_then_validation_loss",
        test_set_used=False,
        status="completed",
    )
    _write_json(destination / "report.json", report.to_dict())
    return report


def _mark_module_trainable(module: Any) -> None:
    module.train()
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def _set_trainable_modules_train(audio_tower: Any, plan: EncoderUnfreezePlan) -> None:
    audio_tower.eval()
    if plan.unfreeze_all_encoder:
        audio_tower.train()
        return
    layers = getattr(audio_tower, "layers", None)
    for module_name in plan.trainable_module_names:
        if module_name.startswith("layers."):
            if layers is None:
                raise ValueError("audio_tower does not expose layers")
            layers[int(module_name.split(".", maxsplit=1)[1])].train()
        elif module_name == "ln_post":
            audio_tower.ln_post.train()


def _optimizer_groups(
    head: Any,
    audio_tower: Any,
    *,
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = [
        {
            "params": list(head.parameters()),
            "lr": head_learning_rate,
            "weight_decay": weight_decay,
            "name": "ctc_head",
        }
    ]
    encoder_parameters = [
        parameter for parameter in audio_tower.parameters() if parameter.requires_grad
    ]
    if encoder_parameters:
        groups.append(
            {
                "params": encoder_parameters,
                "lr": encoder_learning_rate,
                "weight_decay": weight_decay,
                "name": "audio_encoder",
            }
        )
    return groups


def _train_records_epoch(
    wrapper: Any,
    head: Any,
    optimizer: Any,
    records: list[ExperimentRecord],
    *,
    device: Any,
    batch_size: int,
    gradient_accumulation_steps: int,
    blank_id: int,
    max_gradient_norm: float,
    epoch: int,
    seed: int,
    log_every_batches: int,
    unfreeze_plan: EncoderUnfreezePlan,
) -> EpochMetrics:
    import torch

    from qwen_hotword.modeling.ctc_head import compute_ctc

    generator = random.Random(seed + epoch)
    indices = list(range(len(records)))
    generator.shuffle(indices)
    head.train()
    _set_trainable_modules_train(wrapper.model.thinker.audio_tower, unfreeze_plan)

    optimizer.zero_grad(set_to_none=True)
    pending = 0
    loss_sum = 0.0
    sample_count = 0
    total_errors = 0
    total_reference = 0
    total_batches = (len(indices) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(indices), batch_size), start=1):
        batch_records = [records[index] for index in indices[start : start + batch_size]]
        batch = _extract_batch(
            wrapper,
            batch_records,
            device=device,
            blank_id=blank_id,
            no_grad=False,
        )
        computation = compute_ctc(
            head,
            batch.hidden_states.float(),
            batch.input_lengths,
            batch.targets,
            batch.target_lengths,
            blank_id=blank_id,
        )
        scaled_loss = computation.loss / gradient_accumulation_steps
        scaled_loss.backward()
        pending += 1
        if pending == gradient_accumulation_steps or batch_number == total_batches:
            parameters = list(head.parameters()) + [
                parameter
                for parameter in wrapper.model.thinker.audio_tower.parameters()
                if parameter.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0

        errors, references = _batch_error_counts(
            computation.logits,
            batch.input_lengths,
            batch.samples,
            blank_id=blank_id,
        )
        loss_sum += float(computation.loss.item()) * len(batch_records)
        sample_count += len(batch_records)
        total_errors += errors
        total_reference += references
        if (
            batch_number == 1
            or batch_number % log_every_batches == 0
            or batch_number == total_batches
        ):
            print(
                f"epoch={epoch:03d} trained_batches={batch_number}/{total_batches} "
                f"samples={sample_count}/{len(records)}",
                flush=True,
            )
        del batch, computation
    return _finalize_metrics(
        epoch,
        loss_sum=loss_sum,
        sample_count=sample_count,
        total_errors=total_errors,
        total_reference=total_reference,
    )


def _evaluate_records(
    wrapper: Any,
    head: Any,
    records: list[ExperimentRecord],
    *,
    device: Any,
    batch_size: int,
    blank_id: int,
    epoch: int,
) -> EpochMetrics:
    import torch

    from qwen_hotword.modeling.ctc_head import compute_ctc

    wrapper.model.thinker.audio_tower.eval()
    head.eval()
    loss_sum = 0.0
    sample_count = 0
    total_errors = 0
    total_reference = 0
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch = _extract_batch(
                wrapper,
                batch_records,
                device=device,
                blank_id=blank_id,
                no_grad=True,
            )
            computation = compute_ctc(
                head,
                batch.hidden_states.float(),
                batch.input_lengths,
                batch.targets,
                batch.target_lengths,
                blank_id=blank_id,
            )
            errors, references = _batch_error_counts(
                computation.logits,
                batch.input_lengths,
                batch.samples,
                blank_id=blank_id,
            )
            loss_sum += float(computation.loss.item()) * len(batch_records)
            sample_count += len(batch_records)
            total_errors += errors
            total_reference += references
            del batch, computation
    return _finalize_metrics(
        epoch,
        loss_sum=loss_sum,
        sample_count=sample_count,
        total_errors=total_errors,
        total_reference=total_reference,
    )


def _extract_batch(
    wrapper: Any,
    records: list[ExperimentRecord],
    *,
    device: Any,
    blank_id: int,
    no_grad: bool,
) -> OnTheFlyBatch:
    import torch

    from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post

    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("librosa is required to load training audio") from error

    waveforms = [
        librosa.load(str(record.audio_path), sr=16_000, mono=True)[0] for record in records
    ]
    prompts = [build_audio_prompt(wrapper.processor, record.language) for record in records]
    processor_batch = wrapper.processor(
        text=prompts,
        audio=waveforms,
        return_tensors="pt",
        padding=True,
    )
    input_features = processor_batch["input_features"].to(
        device=wrapper.model.device,
        dtype=wrapper.model.dtype,
    )
    feature_attention_mask = processor_batch["feature_attention_mask"].to(
        device=wrapper.model.device
    )
    encoder_batch = extract_padded_ln_post(
        wrapper.model.thinker.audio_tower,
        input_features,
        feature_attention_mask,
        no_grad=no_grad,
    )
    for row, record in enumerate(records):
        input_length = int(encoder_batch.input_lengths[row].item())
        if input_length < record.ctc_minimum_input_length:
            raise ValueError(
                f"actual encoder length is infeasible for {record.sample_id}: "
                f"actual={input_length}, minimum={record.ctc_minimum_input_length}"
            )

    target_lengths = torch.tensor(
        [len(record.token_ids) for record in records],
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (len(records), int(target_lengths.max().item())),
        fill_value=blank_id,
        dtype=torch.long,
        device=device,
    )
    samples: list[CachedSample] = []
    for row, record in enumerate(records):
        targets[row, : len(record.token_ids)] = torch.tensor(
            record.token_ids,
            dtype=torch.long,
            device=device,
        )
        input_length = int(encoder_batch.input_lengths[row].item())
        samples.append(
            CachedSample(
                sample_id=record.sample_id,
                hidden_states=encoder_batch.hidden_states[row, :input_length],
                token_ids=record.token_ids,
            )
        )
    return OnTheFlyBatch(
        hidden_states=encoder_batch.hidden_states.to(device),
        input_lengths=encoder_batch.input_lengths.to(device),
        targets=targets,
        target_lengths=target_lengths,
        samples=tuple(samples),
    )


def _batch_error_counts(
    logits: Any,
    input_lengths: Any,
    samples: tuple[CachedSample, ...],
    *,
    blank_id: int,
) -> tuple[int, int]:
    from rapidfuzz.distance import Levenshtein

    predictions = logits.argmax(dim=-1).detach().cpu()
    errors = 0
    references = 0
    for row, sample in enumerate(samples):
        input_length = int(input_lengths[row].item())
        hypothesis = collapse_ctc_ids(
            predictions[row, :input_length].tolist(),
            blank_id=blank_id,
        )
        errors += int(Levenshtein.distance(sample.token_ids, hypothesis))
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


def _early_stopping_value(
    metrics: EpochMetrics,
    metric: EarlyStoppingMetric,
) -> float:
    if metric == "validation_loss":
        return metrics.loss
    if metric == "validation_per":
        return metrics.phoneme_error_rate
    raise ValueError("early stopping metric must be validation_loss or validation_per")


def _save_checkpoint(
    path: Path,
    *,
    head: Any,
    audio_tower: Any,
    vocab: PhonemeVocab,
    metrics: EpochMetrics,
    unfreeze_plan: EncoderUnfreezePlan,
    seed: int,
) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": "unfrozen_encoder_ctc",
            "head_type": "LinearCtcHead",
            "input_dimension": 1024,
            "num_classes": len(vocab.tokens),
            "blank_id": 0,
            "vocab_tokens": list(vocab.tokens),
            "epoch_metrics": metrics.to_dict(),
            "seed": seed,
            "unfreeze_plan": unfreeze_plan.to_dict(),
            "head_state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
            "audio_tower_state_dict": _audio_tower_trainable_state(audio_tower),
        },
        temporary,
    )
    temporary.replace(path)


def _save_training_state(
    path: Path,
    *,
    run_fingerprint: str,
    completed_epoch: int,
    head: Any,
    audio_tower: Any,
    optimizer: Any,
    scheduler: Any,
    initial_validation: EpochMetrics,
    best_epoch: int,
    best_train: EpochMetrics | None,
    best_validation: EpochMetrics,
    patience_reference_value: float,
    stale_epochs: int,
    training_seconds: float,
    history: list[UnfrozenEpochMetrics],
) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "run_fingerprint": run_fingerprint,
            "completed_epoch": completed_epoch,
            "head_state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
            "audio_tower_state_dict": _audio_tower_trainable_state(audio_tower),
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


def _audio_tower_trainable_state(audio_tower: Any) -> dict[str, Any]:
    return {
        key: parameter.detach().cpu()
        for key, parameter in audio_tower.named_parameters()
        if parameter.requires_grad
    }


def _load_audio_tower_trainable_state(audio_tower: Any, state_dict: object) -> None:
    if not isinstance(state_dict, dict):
        raise ValueError("audio_tower_state_dict must be a mapping")
    result = audio_tower.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected audio tower checkpoint keys: {result.unexpected_keys}")
    missing_trainable = [
        name
        for name, parameter in audio_tower.named_parameters()
        if parameter.requires_grad and name not in state_dict
    ]
    if missing_trainable:
        raise ValueError(f"missing trainable audio tower checkpoint keys: {missing_trainable[:5]}")


def _training_fingerprint(
    *,
    train_manifest_path: str | Path,
    validation_manifest_path: str | Path,
    vocab: PhonemeVocab,
    train_sample_count: int,
    validation_sample_count: int,
    unfreeze_plan: EncoderUnfreezePlan,
    **hyperparameters: object,
) -> str:
    value = {
        "schema_version": 1,
        "train_manifest_path": str(Path(train_manifest_path).expanduser()),
        "validation_manifest_path": str(Path(validation_manifest_path).expanduser()),
        "train_sample_count": train_sample_count,
        "validation_sample_count": validation_sample_count,
        "vocab_tokens": list(vocab.tokens),
        "unfreeze_plan": unfreeze_plan.to_dict(),
        "hyperparameters": hyperparameters,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_arguments(
    vocab: PhonemeVocab,
    *,
    epochs: int,
    minimum_epochs: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    early_stopping_metric: EarlyStoppingMetric,
    train_batch_size: int,
    validation_batch_size: int,
    gradient_accumulation_steps: int,
    head_learning_rate: float,
    encoder_learning_rate: float,
    max_gradient_norm: float,
    scheduler_patience: int,
    scheduler_factor: float,
    minimum_head_learning_rate: float,
    minimum_encoder_learning_rate: float,
    log_every_batches: int,
    max_train_samples: int | None,
    max_validation_samples: int | None,
) -> None:
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
    if train_batch_size <= 0 or validation_batch_size <= 0:
        raise ValueError("train and validation batch sizes must be positive")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if (
        head_learning_rate <= 0
        or encoder_learning_rate <= 0
        or minimum_head_learning_rate <= 0
        or minimum_encoder_learning_rate <= 0
        or max_gradient_norm <= 0
    ):
        raise ValueError("learning rates and gradient norm must be positive")
    if not 0 < scheduler_factor < 1:
        raise ValueError("scheduler factor must be between zero and one")
    if log_every_batches <= 0:
        raise ValueError("log_every_batches must be positive")
    if max_train_samples is not None and max_train_samples <= 0:
        raise ValueError("max_train_samples must be positive when provided")
    if max_validation_samples is not None and max_validation_samples <= 0:
        raise ValueError("max_validation_samples must be positive when provided")


def _encoder_group_lr(optimizer: Any) -> float | None:
    if len(optimizer.param_groups) < 2:
        return None
    return float(optimizer.param_groups[1]["lr"])


def _write_metric_history(path: Path, metrics: list[UnfrozenEpochMetrics]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _epoch_history_from_dict(value: object) -> UnfrozenEpochMetrics:
    if not isinstance(value, dict):
        raise ValueError("training history entry must be an object")
    return UnfrozenEpochMetrics(
        epoch=_required_int(value, "epoch"),
        head_learning_rate=float(value["head_learning_rate"]),
        encoder_learning_rate=(
            float(value["encoder_learning_rate"])
            if value.get("encoder_learning_rate") is not None
            else None
        ),
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


def _required_int(value: dict[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int):
        raise ValueError(f"{key} must be an integer")
    return item


def _required_list(value: dict[str, object], key: str) -> list[object]:
    item = value.get(key)
    if not isinstance(item, list):
        raise ValueError(f"{key} must be a list")
    return item
