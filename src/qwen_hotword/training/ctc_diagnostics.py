from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import CachedSample, collapse_ctc_ids
from qwen_hotword.training.edit_distance import sequence_editops
from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    load_feature_shard,
)


@dataclass
class ErrorAccumulator:
    sample_count: int = 0
    reference_tokens: int = 0
    hypothesis_tokens: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    input_frames: int = 0
    blank_frames: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            {
                "errors": self.errors,
                "phoneme_error_rate": self.errors / self.reference_tokens,
                "hypothesis_reference_length_ratio": (
                    self.hypothesis_tokens / self.reference_tokens
                ),
                "blank_frame_ratio": self.blank_frames / self.input_frames,
            }
        )
        return result


@dataclass
class DetailedErrorAccumulator:
    totals: ErrorAccumulator = field(default_factory=ErrorAccumulator)
    deleted_tokens: Counter[int] = field(default_factory=Counter)
    inserted_tokens: Counter[int] = field(default_factory=Counter)
    substituted_tokens: Counter[tuple[int, int]] = field(default_factory=Counter)


def diagnose_ctc_checkpoint(
    checkpoint_path: str | Path,
    cache: DiskFeatureCache,
    vocab: PhonemeVocab,
    *,
    device: Any,
    batch_size: int = 256,
) -> dict[str, object]:
    import torch

    from qwen_hotword.modeling.ctc_head import LinearCtcHead, compute_ctc

    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    checkpoint = Path(checkpoint_path).expanduser()
    payload = _load_checkpoint(checkpoint, vocab)
    head = LinearCtcHead(1024, len(vocab.tokens)).to(
        device=device,
        dtype=torch.float32,
    )
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()

    detailed = DetailedErrorAccumulator()
    buckets = {
        "minimum_ratio_le_0_50": ErrorAccumulator(),
        "minimum_ratio_gt_0_50_le_0_75": ErrorAccumulator(),
        "minimum_ratio_gt_0_75_le_0_90": ErrorAccumulator(),
        "minimum_ratio_gt_0_90": ErrorAccumulator(),
    }
    loss_sum = 0.0
    with torch.no_grad():
        for descriptor in cache.shards:
            samples = load_feature_shard(
                descriptor,
                num_classes=len(vocab.tokens),
            )
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                hidden_states, input_lengths, targets, target_lengths = (
                    _collate(batch, device=device)
                )
                computation = compute_ctc(
                    head,
                    hidden_states,
                    input_lengths,
                    targets,
                    target_lengths,
                    blank_id=0,
                )
                loss_sum += float(computation.loss.item()) * len(batch)
                predictions = computation.logits.argmax(dim=-1).detach().cpu()
                for row, sample in enumerate(batch):
                    input_length = int(input_lengths[row].item())
                    raw_prediction = predictions[row, :input_length].tolist()
                    hypothesis = tuple(collapse_ctc_ids(raw_prediction, blank_id=0))
                    bucket = buckets[
                        ctc_pressure_bucket(
                            sample.token_ids,
                            input_length=input_length,
                        )
                    ]
                    _accumulate_errors(
                        detailed,
                        bucket,
                        reference=sample.token_ids,
                        hypothesis=hypothesis,
                        raw_prediction=raw_prediction,
                        blank_id=0,
                    )
            del samples

    total = detailed.totals
    if total.sample_count != cache.sample_count:
        raise RuntimeError("diagnostic did not consume the complete validation cache")
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_epoch_metrics": payload.get("epoch_metrics"),
        "validation_loss": loss_sum / total.sample_count,
        "validation": total.to_dict(),
        "ctc_pressure_buckets": {
            name: accumulator.to_dict()
            for name, accumulator in buckets.items()
            if accumulator.sample_count
        },
        "top_deletions": _top_token_counts(detailed.deleted_tokens, vocab),
        "top_insertions": _top_token_counts(detailed.inserted_tokens, vocab),
        "top_substitutions": _top_substitution_counts(
            detailed.substituted_tokens,
            vocab,
        ),
    }


def ctc_pressure_bucket(token_ids: tuple[int, ...], *, input_length: int) -> str:
    if input_length <= 0 or not token_ids:
        raise ValueError("CTC pressure requires positive input and target lengths")
    minimum_length = len(token_ids) + sum(
        left == right
        for left, right in zip(token_ids, token_ids[1:], strict=False)
    )
    ratio = minimum_length / input_length
    if ratio <= 0.50:
        return "minimum_ratio_le_0_50"
    if ratio <= 0.75:
        return "minimum_ratio_gt_0_50_le_0_75"
    if ratio <= 0.90:
        return "minimum_ratio_gt_0_75_le_0_90"
    return "minimum_ratio_gt_0_90"


def _accumulate_errors(
    detailed: DetailedErrorAccumulator,
    bucket: ErrorAccumulator,
    *,
    reference: tuple[int, ...],
    hypothesis: tuple[int, ...],
    raw_prediction: list[int],
    blank_id: int,
) -> None:
    substitutions = 0
    deletions = 0
    insertions = 0
    for tag, source_position, destination_position in sequence_editops(
        reference,
        hypothesis,
    ):
        if tag == "replace":
            substitutions += 1
            detailed.substituted_tokens[
                (reference[source_position], hypothesis[destination_position])
            ] += 1
        elif tag == "delete":
            deletions += 1
            detailed.deleted_tokens[reference[source_position]] += 1
        elif tag == "insert":
            insertions += 1
            detailed.inserted_tokens[hypothesis[destination_position]] += 1
        else:
            raise ValueError(f"unsupported Levenshtein operation: {tag}")

    for accumulator in (detailed.totals, bucket):
        accumulator.sample_count += 1
        accumulator.reference_tokens += len(reference)
        accumulator.hypothesis_tokens += len(hypothesis)
        accumulator.substitutions += substitutions
        accumulator.deletions += deletions
        accumulator.insertions += insertions
        accumulator.input_frames += len(raw_prediction)
        accumulator.blank_frames += raw_prediction.count(blank_id)


def _load_checkpoint(path: Path, vocab: PhonemeVocab) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"CTC Head checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"CTC Head checkpoint is invalid: {path}")
    if payload.get("input_dimension") != 1024:
        raise ValueError("CTC Head checkpoint input dimension is not 1024")
    if payload.get("num_classes") != len(vocab.tokens):
        raise ValueError("CTC Head checkpoint class count differs from the vocabulary")
    if payload.get("blank_id") != 0 or payload.get("vocab_tokens") != list(vocab.tokens):
        raise ValueError("CTC Head checkpoint vocabulary identity does not match")
    return payload


def _collate(samples: list[CachedSample], *, device: Any) -> tuple[Any, Any, Any, Any]:
    from qwen_hotword.training.ctc_overfit import collate_cached_samples

    return collate_cached_samples(samples, device=device, blank_id=0)


def _top_token_counts(
    counts: Counter[int],
    vocab: PhonemeVocab,
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    return [
        {"token_id": token_id, "token": vocab.tokens[token_id], "count": count}
        for token_id, count in counts.most_common(limit)
    ]


def _top_substitution_counts(
    counts: Counter[tuple[int, int]],
    vocab: PhonemeVocab,
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    return [
        {
            "reference_id": reference_id,
            "reference_token": vocab.tokens[reference_id],
            "hypothesis_id": hypothesis_id,
            "hypothesis_token": vocab.tokens[hypothesis_id],
            "count": count,
        }
        for (reference_id, hypothesis_id), count in counts.most_common(limit)
    ]
