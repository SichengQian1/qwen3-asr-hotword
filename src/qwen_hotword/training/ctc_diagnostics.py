from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import CachedSample, collapse_ctc_ids
from qwen_hotword.training.edit_distance import sequence_editops
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, load_feature_shard


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
        if self.reference_tokens <= 0 or self.input_frames <= 0:
            raise ValueError("cannot summarize empty CTC diagnostic metrics")
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
    sample_groups: Mapping[str, str] | None = None,
) -> dict[str, object]:
    import torch

    from qwen_hotword.modeling.ctc_head import (
        build_ctc_head_from_checkpoint,
        compute_ctc,
        ctc_head_config,
    )

    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    checkpoint = Path(checkpoint_path).expanduser()
    payload = _load_checkpoint(checkpoint, vocab)
    head = build_ctc_head_from_checkpoint(payload)
    if head.input_dimension != 1024 or head.num_classes != len(vocab.tokens):
        raise ValueError(
            "CTC Head structure metadata does not match the feature/vocabulary contract"
        )
    head = head.to(device=device, dtype=torch.float32)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()

    normalized_groups = _validate_sample_groups(cache, sample_groups)
    group_accumulators = {
        group: ErrorAccumulator() for group in sorted(set(normalized_groups.values()))
    }
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
            samples = load_feature_shard(descriptor, num_classes=len(vocab.tokens))
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                hidden_states, input_lengths, targets, target_lengths = _collate(
                    batch,
                    device=device,
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
                    input_length = int(computation.input_lengths[row].item())
                    raw_prediction = predictions[row, :input_length].tolist()
                    hypothesis = tuple(collapse_ctc_ids(raw_prediction, blank_id=0))
                    bucket = buckets[
                        ctc_pressure_bucket(sample.token_ids, input_length=input_length)
                    ]
                    _accumulate_errors(
                        detailed,
                        bucket,
                        reference=sample.token_ids,
                        hypothesis=hypothesis,
                        raw_prediction=raw_prediction,
                        blank_id=0,
                        group_accumulator=(
                            group_accumulators[normalized_groups[sample.sample_id]]
                            if normalized_groups
                            else None
                        ),
                    )
            del samples

    total = detailed.totals
    if total.sample_count != cache.sample_count:
        raise RuntimeError("diagnostic did not consume the complete validation cache")
    result: dict[str, object] = {
        "checkpoint_path": str(checkpoint),
        "head_config": ctc_head_config(head),
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
        "top_substitutions": _top_substitution_counts(detailed.substituted_tokens, vocab),
    }
    if group_accumulators:
        validation_by_group = {
            group: accumulator.to_dict()
            for group, accumulator in group_accumulators.items()
        }
        result["validation_by_group"] = validation_by_group
        result["validation_macro_phoneme_error_rate"] = sum(
            accumulator.errors / accumulator.reference_tokens
            for accumulator in group_accumulators.values()
        ) / len(validation_by_group)
    return result


def load_validation_sample_groups(
    manifest_path: str | Path,
    cache: DiskFeatureCache,
    *,
    group_column: str,
    expected_groups: Sequence[str] = (),
) -> dict[str, str]:
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"validation manifest does not exist: {path}")
    if not group_column:
        raise ValueError("validation group column cannot be empty")

    groups: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid validation JSON at {path}:{line_number}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"validation row must be an object at {path}:{line_number}")
            sample_id = row.get("id")
            group = row.get(group_column)
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"validation row has no sample ID at {path}:{line_number}")
            if not isinstance(group, str) or not group:
                raise ValueError(
                    f"validation row has no {group_column!r} at {path}:{line_number}"
                )
            if row.get("split") != "validation":
                raise ValueError(
                    f"non-validation row found in validation manifest at {path}:{line_number}"
                )
            if sample_id in groups:
                raise ValueError(f"duplicate validation sample ID: {sample_id}")
            groups[sample_id] = group

    _validate_sample_groups(cache, groups)
    actual_groups = set(groups.values())
    if expected_groups and actual_groups != set(expected_groups):
        raise ValueError(
            "validation groups differ from the expected set: "
            f"actual={sorted(actual_groups)}, expected={sorted(set(expected_groups))}"
        )
    return groups


def ctc_pressure_bucket(token_ids: tuple[int, ...], *, input_length: int) -> str:
    if input_length <= 0 or not token_ids:
        raise ValueError("CTC pressure requires positive input and target lengths")
    minimum_length = len(token_ids) + sum(
        left == right for left, right in zip(token_ids, token_ids[1:], strict=False)
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
    group_accumulator: ErrorAccumulator | None = None,
) -> None:
    substitutions = 0
    deletions = 0
    insertions = 0
    for tag, source_position, destination_position in sequence_editops(reference, hypothesis):
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

    accumulators = [detailed.totals, bucket]
    if group_accumulator is not None:
        accumulators.append(group_accumulator)
    for accumulator in accumulators:
        accumulator.sample_count += 1
        accumulator.reference_tokens += len(reference)
        accumulator.hypothesis_tokens += len(hypothesis)
        accumulator.substitutions += substitutions
        accumulator.deletions += deletions
        accumulator.insertions += insertions
        accumulator.input_frames += len(raw_prediction)
        accumulator.blank_frames += raw_prediction.count(blank_id)


def _validate_sample_groups(
    cache: DiskFeatureCache,
    sample_groups: Mapping[str, str] | None,
) -> dict[str, str]:
    if sample_groups is None:
        return {}
    normalized = dict(sample_groups)
    if any(not isinstance(group, str) or not group for group in normalized.values()):
        raise ValueError("validation sample groups must be non-empty strings")
    cache_ids = {
        sample_id for descriptor in cache.shards for sample_id in descriptor.sample_ids
    }
    group_ids = set(normalized)
    if cache_ids != group_ids:
        missing = len(cache_ids - group_ids)
        extra = len(group_ids - cache_ids)
        raise ValueError(
            "validation sample groups do not exactly cover the cache: "
            f"missing={missing}, extra={extra}"
        )
    return normalized


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
