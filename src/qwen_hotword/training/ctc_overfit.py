from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwen_hotword.phonemes.coverage import PhonemeVocab

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ExperimentRecord:
    sample_id: str
    audio_path: Path
    text: str
    language: str
    token_ids: tuple[int, ...]
    ctc_minimum_input_length: int


@dataclass(frozen=True)
class CachedSample:
    sample_id: str
    hidden_states: torch.Tensor
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    phoneme_error_rate: float
    phoneme_errors: int
    reference_phonemes: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OverfitReport:
    manifest_path: str
    vocab_path: str
    sample_count: int
    num_classes: int
    blank_id: int
    encoder_frozen_parameters: int
    ctc_trainable_parameters: int
    encoder_batch_size: int
    train_batch_size: int
    learning_rate: float
    epochs_requested: int
    epochs_completed: int
    target_phoneme_error_rate: float
    initial_loss: float
    initial_phoneme_error_rate: float
    best_epoch: int
    best_loss: float
    best_phoneme_error_rate: float
    final_loss: float
    final_phoneme_error_rate: float
    feature_extraction_seconds: float
    training_seconds: float
    best_checkpoint_path: str
    latest_checkpoint_path: str
    metrics_path: str
    overfit_success: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_experiment_records(
    manifest_path: str | Path,
    *,
    num_classes: int,
    blank_id: int = 0,
) -> list[ExperimentRecord]:
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"experiment manifest does not exist: {path}")
    if num_classes <= 1:
        raise ValueError("num_classes must include blank and at least one label")

    records: list[ExperimentRecord] = []
    seen_ids: set[str] = set()
    seen_audio: set[Path] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise ValueError(f"manifest row {line_number} must be a JSON object")
            if raw.get("experiment") != "A" or raw.get("split") != "train":
                raise ValueError(f"manifest row {line_number} is not Experiment A train data")

            sample_id = _required_string(raw, "id", line_number)
            if sample_id in seen_ids:
                raise ValueError(f"duplicate sample id at row {line_number}: {sample_id}")
            seen_ids.add(sample_id)

            audio_path = Path(_required_string(raw, "audio_path", line_number)).expanduser()
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"audio file from manifest row {line_number} does not exist: {audio_path}"
                )
            if audio_path in seen_audio:
                raise ValueError(f"duplicate audio path at row {line_number}: {audio_path}")
            seen_audio.add(audio_path)

            raw_token_ids = raw.get("phoneme_token_ids")
            if not isinstance(raw_token_ids, list) or not raw_token_ids:
                raise ValueError(f"row {line_number} has no phoneme_token_ids")
            if any(not isinstance(token_id, int) for token_id in raw_token_ids):
                raise ValueError(f"row {line_number} contains a non-integer phoneme token ID")
            token_ids = tuple(raw_token_ids)
            invalid_ids = [
                token_id
                for token_id in token_ids
                if token_id == blank_id or token_id < 0 or token_id >= num_classes
            ]
            if invalid_ids:
                raise ValueError(
                    f"row {line_number} has blank or out-of-range target IDs: {invalid_ids[:5]}"
                )
            if raw.get("label_length") != len(token_ids):
                raise ValueError(f"row {line_number} label_length does not match token IDs")

            minimum_length = raw.get("ctc_minimum_input_length")
            if not isinstance(minimum_length, int) or minimum_length < len(token_ids):
                raise ValueError(f"row {line_number} has invalid CTC minimum input length")
            records.append(
                ExperimentRecord(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    text=_required_string(raw, "text", line_number),
                    language=_required_string(raw, "language", line_number),
                    token_ids=token_ids,
                    ctc_minimum_input_length=minimum_length,
                )
            )
    if not records:
        raise ValueError(f"experiment manifest contains no records: {path}")
    return records


def collapse_ctc_ids(token_ids: list[int], *, blank_id: int = 0) -> list[int]:
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        if token_id != previous and token_id != blank_id:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def edit_distance(reference: tuple[int, ...] | list[int], hypothesis: list[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = int(reference_token != hypothesis_token)
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def freeze_module(module: Any) -> int:
    module.eval()
    parameter_count = 0
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter_count += parameter.numel()
    return parameter_count


def build_audio_prompt(processor: Any, language: str) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    language_name = "Portuguese" if language.lower().startswith("pt") else language
    return str(prompt) + f"language {language_name}<asr_text>"


def extract_frozen_features(
    records: list[ExperimentRecord],
    wrapper: Any,
    *,
    encoder_batch_size: int,
) -> tuple[list[CachedSample], int, float]:
    import torch

    from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post

    if encoder_batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    try:
        import librosa  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("librosa is required to load Experiment A audio") from error

    audio_tower = wrapper.model.thinker.audio_tower
    frozen_parameters = freeze_module(audio_tower)
    cached: list[CachedSample] = []
    started = time.monotonic()
    for start in range(0, len(records), encoder_batch_size):
        batch_records = records[start : start + encoder_batch_size]
        waveforms = [
            librosa.load(str(record.audio_path), sr=16_000, mono=True)[0]
            for record in batch_records
        ]
        prompts = [
            build_audio_prompt(wrapper.processor, record.language) for record in batch_records
        ]
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
            audio_tower,
            input_features,
            feature_attention_mask,
            no_grad=True,
        )
        for row, record in enumerate(batch_records):
            input_length = int(encoder_batch.input_lengths[row].item())
            if input_length < record.ctc_minimum_input_length:
                raise ValueError(
                    f"actual encoder length is infeasible for {record.sample_id}: "
                    f"actual={input_length}, minimum={record.ctc_minimum_input_length}"
                )
            cached.append(
                CachedSample(
                    sample_id=record.sample_id,
                    hidden_states=encoder_batch.hidden_states[row, :input_length]
                    .detach()
                    .to(device="cpu", dtype=torch.bfloat16),
                    token_ids=record.token_ids,
                )
            )
        print(f"cached encoder features: {len(cached)}/{len(records)}", flush=True)
    return cached, frozen_parameters, time.monotonic() - started


def train_cached_ctc_head(
    samples: list[CachedSample],
    vocab: PhonemeVocab,
    output_dir: str | Path,
    *,
    manifest_path: str | Path,
    vocab_path: str | Path,
    device: Any,
    encoder_frozen_parameters: int,
    feature_extraction_seconds: float,
    epochs: int = 200,
    train_batch_size: int = 16,
    encoder_batch_size: int = 8,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    max_gradient_norm: float = 5.0,
    target_phoneme_error_rate: float = 0.05,
    seed: int = 20_260_716,
    log_every: int = 5,
) -> OverfitReport:
    import torch

    from qwen_hotword.modeling.ctc_head import LinearCtcHead, compute_ctc

    if not samples:
        raise ValueError("cannot train without cached samples")
    if epochs <= 0 or train_batch_size <= 0 or log_every <= 0:
        raise ValueError("epochs, train_batch_size, and log_every must be positive")
    if learning_rate <= 0 or max_gradient_norm <= 0:
        raise ValueError("learning rate and gradient norm must be positive")
    if not 0 <= target_phoneme_error_rate <= 1:
        raise ValueError("target_phoneme_error_rate must be between zero and one")

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
    metrics_path = destination / "metrics.jsonl"
    best_checkpoint_path = destination / "ctc_head_best.pt"
    latest_checkpoint_path = destination / "ctc_head_latest.pt"

    initial = evaluate_cached_samples(
        head,
        samples,
        device=device,
        batch_size=train_batch_size,
        blank_id=blank_id,
        epoch=0,
    )
    history = [initial]
    _write_metrics(metrics_path, history)
    best = initial
    _save_checkpoint(best_checkpoint_path, head, vocab, best, seed)

    generator = random.Random(seed)
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        indices = list(range(len(samples)))
        generator.shuffle(indices)
        head.train()
        for start in range(0, len(indices), train_batch_size):
            batch = [samples[index] for index in indices[start : start + train_batch_size]]
            hidden_states, input_lengths, targets, target_lengths = _collate_cached(
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

        metrics = evaluate_cached_samples(
            head,
            samples,
            device=device,
            batch_size=train_batch_size,
            blank_id=blank_id,
            epoch=epoch,
        )
        history.append(metrics)
        _append_metric(metrics_path, metrics)
        if (metrics.phoneme_error_rate, metrics.loss) < (
            best.phoneme_error_rate,
            best.loss,
        ):
            best = metrics
            _save_checkpoint(best_checkpoint_path, head, vocab, best, seed)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} loss={metrics.loss:.6f} "
                f"PER={metrics.phoneme_error_rate:.4f} "
                f"best_PER={best.phoneme_error_rate:.4f}",
                flush=True,
            )
        if best.phoneme_error_rate <= target_phoneme_error_rate:
            print(
                f"target PER reached at epoch {epoch}: {best.phoneme_error_rate:.4f}",
                flush=True,
            )
            break

    training_seconds = time.monotonic() - started
    final = history[-1]
    _save_checkpoint(latest_checkpoint_path, head, vocab, final, seed)
    report = OverfitReport(
        manifest_path=str(Path(manifest_path)),
        vocab_path=str(Path(vocab_path)),
        sample_count=len(samples),
        num_classes=len(vocab.tokens),
        blank_id=blank_id,
        encoder_frozen_parameters=encoder_frozen_parameters,
        ctc_trainable_parameters=sum(parameter.numel() for parameter in head.parameters()),
        encoder_batch_size=encoder_batch_size,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        epochs_requested=epochs,
        epochs_completed=final.epoch,
        target_phoneme_error_rate=target_phoneme_error_rate,
        initial_loss=initial.loss,
        initial_phoneme_error_rate=initial.phoneme_error_rate,
        best_epoch=best.epoch,
        best_loss=best.loss,
        best_phoneme_error_rate=best.phoneme_error_rate,
        final_loss=final.loss,
        final_phoneme_error_rate=final.phoneme_error_rate,
        feature_extraction_seconds=feature_extraction_seconds,
        training_seconds=training_seconds,
        best_checkpoint_path=str(best_checkpoint_path),
        latest_checkpoint_path=str(latest_checkpoint_path),
        metrics_path=str(metrics_path),
        overfit_success=best.phoneme_error_rate <= target_phoneme_error_rate,
        status="completed",
    )
    (destination / "report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def evaluate_cached_samples(
    head: Any,
    samples: list[CachedSample],
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
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            hidden_states, input_lengths, targets, target_lengths = _collate_cached(
                batch,
                device=device,
                blank_id=blank_id,
            )
            computation = compute_ctc(
                head,
                hidden_states,
                input_lengths,
                targets,
                target_lengths,
                blank_id=blank_id,
            )
            loss_sum += float(computation.loss.item()) * len(batch)
            sample_count += len(batch)
            predictions = computation.logits.argmax(dim=-1).cpu()
            for row, sample in enumerate(batch):
                input_length = int(input_lengths[row].item())
                hypothesis = collapse_ctc_ids(
                    predictions[row, :input_length].tolist(),
                    blank_id=blank_id,
                )
                total_errors += edit_distance(sample.token_ids, hypothesis)
                total_reference += len(sample.token_ids)
    return EpochMetrics(
        epoch=epoch,
        loss=loss_sum / sample_count,
        phoneme_error_rate=total_errors / total_reference,
        phoneme_errors=total_errors,
        reference_phonemes=total_reference,
    )


def _collate_cached(
    samples: list[CachedSample],
    *,
    device: Any,
    blank_id: int,
) -> tuple[Any, Any, Any, Any]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    hidden_states = pad_sequence(
        [sample.hidden_states for sample in samples],
        batch_first=True,
        padding_value=0.0,
    ).to(device=device, dtype=torch.float32)
    input_lengths = torch.tensor(
        [sample.hidden_states.shape[0] for sample in samples],
        dtype=torch.long,
        device=device,
    )
    target_lengths = torch.tensor(
        [len(sample.token_ids) for sample in samples],
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (len(samples), int(target_lengths.max().item())),
        fill_value=blank_id,
        dtype=torch.long,
        device=device,
    )
    for row, sample in enumerate(samples):
        targets[row, : len(sample.token_ids)] = torch.tensor(
            sample.token_ids,
            dtype=torch.long,
            device=device,
        )
    return hidden_states, input_lengths, targets, target_lengths


def _required_string(raw: dict[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest row {line_number} has invalid {key}")
    return str(value)


def _save_checkpoint(
    path: Path,
    head: Any,
    vocab: PhonemeVocab,
    metrics: EpochMetrics,
    seed: int,
) -> None:
    import torch

    torch.save(
        {
            "schema_version": 1,
            "head_type": "LinearCtcHead",
            "input_dimension": 1024,
            "num_classes": len(vocab.tokens),
            "blank_id": 0,
            "vocab_tokens": list(vocab.tokens),
            "epoch_metrics": metrics.to_dict(),
            "seed": seed,
            "state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
        },
        path,
    )


def _write_metrics(path: Path, metrics: list[EpochMetrics]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for metric in metrics:
            handle.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")


def _append_metric(path: Path, metric: EpochMetrics) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric.to_dict(), sort_keys=True) + "\n")
