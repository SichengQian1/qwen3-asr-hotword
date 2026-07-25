from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    ExperimentRecord,
    collapse_ctc_ids,
    collate_cached_samples,
)
from qwen_hotword.training.edit_distance import sequence_editops


@dataclass
class SealedTestTotals:
    sample_count: int = 0
    reference_phonemes: int = 0
    hypothesis_phonemes: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    effective_input_frames: int = 0
    blank_frames: int = 0
    loss_sum: float = 0.0
    deleted_tokens: Counter[int] = field(default_factory=Counter)
    inserted_tokens: Counter[int] = field(default_factory=Counter)
    substituted_tokens: Counter[tuple[int, int]] = field(default_factory=Counter)

    @property
    def phoneme_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def metrics(self) -> dict[str, object]:
        if (
            self.sample_count <= 0
            or self.reference_phonemes <= 0
            or self.effective_input_frames <= 0
        ):
            raise ValueError("cannot summarize an empty sealed-test evaluation")
        return {
            "sample_count": self.sample_count,
            "loss": self.loss_sum / self.sample_count,
            "phoneme_error_rate": self.phoneme_errors / self.reference_phonemes,
            "phoneme_errors": self.phoneme_errors,
            "reference_phonemes": self.reference_phonemes,
            "hypothesis_phonemes": self.hypothesis_phonemes,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "hypothesis_reference_length_ratio": (
                self.hypothesis_phonemes / self.reference_phonemes
            ),
            "effective_input_frames": self.effective_input_frames,
            "blank_frames": self.blank_frames,
            "blank_frame_ratio": self.blank_frames / self.effective_input_frames,
        }


def evaluate_sealed_ctc_test(
    records: list[ExperimentRecord],
    wrapper: Any,
    checkpoint_path: str | Path,
    vocab: PhonemeVocab,
    output_path: str | Path,
    *,
    test_manifest_path: str | Path,
    vocab_path: str | Path,
    model_path: str | Path,
    device: Any,
    encoder_batch_size: int = 8,
    evaluation_batch_size: int = 256,
    records_per_chunk: int = 512,
    extractor: Any | None = None,
) -> dict[str, object]:
    import torch

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
        compute_ctc,
        ctc_head_config,
    )
    from qwen_hotword.training.ctc_overfit import extract_frozen_features

    if not records:
        raise ValueError("sealed test manifest contains no records")
    if encoder_batch_size <= 0 or evaluation_batch_size <= 0 or records_per_chunk <= 0:
        raise ValueError("sealed-test batch and chunk sizes must be positive")
    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.name != "ctc_head_best.pt":
        raise ValueError("sealed test must use the fixed ctc_head_best.pt checkpoint")
    destination = Path(output_path).expanduser()
    if destination.exists():
        raise FileExistsError(
            f"sealed-test report already exists and will not be overwritten: {destination}"
        )

    payload = _load_checkpoint(checkpoint, vocab)
    head = build_ctc_head_from_checkpoint(payload)
    if not isinstance(head, TemporalUpsampleCtcHead) or head.time_upsampling_factor != 2:
        raise ValueError(
            "sealed test requires the fixed temporal_upsample Head with a 2x time axis"
        )
    head = head.to(device=device, dtype=torch.float32)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()

    extract = extractor or extract_frozen_features
    totals = SealedTestTotals()
    extraction_seconds = 0.0
    frozen_parameters = 0
    started = time.monotonic()
    with torch.no_grad():
        for chunk_start in range(0, len(records), records_per_chunk):
            chunk = records[chunk_start : chunk_start + records_per_chunk]
            samples, chunk_frozen_parameters, chunk_seconds = extract(
                chunk,
                wrapper,
                encoder_batch_size=encoder_batch_size,
                progress_every_batches=max(1, len(chunk) // encoder_batch_size),
                progress_label="sealed test encoder features",
            )
            if len(samples) != len(chunk):
                raise RuntimeError("sealed-test extractor returned an incomplete chunk")
            if frozen_parameters not in {0, chunk_frozen_parameters}:
                raise RuntimeError("frozen Encoder parameter count changed during test")
            frozen_parameters = chunk_frozen_parameters
            extraction_seconds += chunk_seconds
            _evaluate_samples(
                head,
                samples,
                totals,
                device=device,
                batch_size=evaluation_batch_size,
                compute_ctc=compute_ctc,
            )
            print(
                f"sealed_test samples={totals.sample_count}/{len(records)}",
                flush=True,
            )
            del samples

    if totals.sample_count != len(records):
        raise RuntimeError("sealed-test evaluation did not consume every manifest record")
    manifest = Path(test_manifest_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    model = Path(model_path).expanduser()
    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "one_time_frozen_encoder_ctc_sealed_test_evaluation",
        "one_time_evaluation": True,
        "test_set_used": True,
        "test_manifest": _file_identity(manifest),
        "vocab": _file_identity(vocab_file),
        "model_path": str(model.resolve()),
        "model_config_sha256": _sha256_file(model / "config.json"),
        "model_weight_index_sha256": _sha256_file(
            model / "model.safetensors.index.json"
        ),
        "checkpoint": {
            **_file_identity(checkpoint),
            "epoch_metrics": payload.get("epoch_metrics"),
        },
        "head_config": ctc_head_config(head),
        "decode_policy": {
            "algorithm": "greedy_argmax_ctc_collapse",
            "blank_id": 0,
            "effective_time_axis": "temporal_upsample_2x",
        },
        "metrics": totals.metrics(),
        "top_deletions": _top_token_counts(totals.deleted_tokens, vocab),
        "top_insertions": _top_token_counts(totals.inserted_tokens, vocab),
        "top_substitutions": _top_substitution_counts(
            totals.substituted_tokens,
            vocab,
        ),
        "encoder_frozen_parameters": frozen_parameters,
        "encoder_batch_size": encoder_batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "records_per_chunk": records_per_chunk,
        "feature_cache_written": False,
        "feature_extraction_seconds": extraction_seconds,
        "evaluation_wall_seconds": time.monotonic() - started,
        "checkpoint_selection_or_tuning_permitted": False,
        "status": "pass",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report


def _evaluate_samples(
    head: Any,
    samples: list[CachedSample],
    totals: SealedTestTotals,
    *,
    device: Any,
    batch_size: int,
    compute_ctc: Any,
) -> None:
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        hidden_states, input_lengths, targets, target_lengths = collate_cached_samples(
            batch,
            device=device,
            blank_id=0,
        )
        computation = compute_ctc(
            head,
            hidden_states,
            input_lengths,
            targets,
            target_lengths,
            blank_id=0,
        )
        predictions = computation.logits.argmax(dim=-1).detach().cpu()
        totals.loss_sum += float(computation.loss.item()) * len(batch)
        for row, sample in enumerate(batch):
            input_length = int(computation.input_lengths[row].item())
            raw_prediction = predictions[row, :input_length].tolist()
            hypothesis = tuple(collapse_ctc_ids(raw_prediction, blank_id=0))
            _accumulate_sample(
                totals,
                reference=sample.token_ids,
                hypothesis=hypothesis,
                raw_prediction=raw_prediction,
            )


def _accumulate_sample(
    totals: SealedTestTotals,
    *,
    reference: tuple[int, ...],
    hypothesis: tuple[int, ...],
    raw_prediction: list[int],
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
            totals.substituted_tokens[
                (reference[source_position], hypothesis[destination_position])
            ] += 1
        elif tag == "delete":
            deletions += 1
            totals.deleted_tokens[reference[source_position]] += 1
        elif tag == "insert":
            insertions += 1
            totals.inserted_tokens[hypothesis[destination_position]] += 1
        else:
            raise ValueError(f"unsupported Levenshtein operation: {tag}")
    totals.sample_count += 1
    totals.reference_phonemes += len(reference)
    totals.hypothesis_phonemes += len(hypothesis)
    totals.substitutions += substitutions
    totals.deletions += deletions
    totals.insertions += insertions
    totals.effective_input_frames += len(raw_prediction)
    totals.blank_frames += raw_prediction.count(0)


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


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"sealed-test identity input does not exist: {path}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
