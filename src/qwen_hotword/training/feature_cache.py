from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.training.ctc_overfit import CachedSample, ExperimentRecord

FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_HIDDEN_SIZE = 1024
FEATURE_DTYPE = "torch.bfloat16"

FeatureExtractor = Callable[..., tuple[list[CachedSample], int, float]]


@dataclass(frozen=True)
class FeatureCacheSummary:
    split: str
    output_dir: str
    source_manifest_path: str
    sample_count: int
    shard_count: int
    generated_shards: int
    resumed_shards: int
    total_frames: int
    total_target_tokens: int
    feature_bytes: int
    extraction_seconds: float
    encoder_frozen_parameters: int
    encoder_batch_size: int
    samples_per_shard: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@contextmanager
def exclusive_feature_cache_run(output_dir: str | Path) -> Iterator[None]:
    import fcntl

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".feature_cache.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown process"
            raise RuntimeError(
                f"another feature-cache process owns {output_dir}: {owner}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def cache_feature_split(
    records: list[ExperimentRecord],
    wrapper: Any,
    output_dir: str | Path,
    *,
    split: str,
    source_manifest_path: str | Path,
    model_path: str | Path,
    model_dtype: str,
    vocab_path: str | Path,
    encoder_batch_size: int = 8,
    samples_per_shard: int = 512,
    extractor: FeatureExtractor | None = None,
) -> FeatureCacheSummary:
    if split not in {"train", "validation"}:
        raise ValueError("feature caching accepts only train or validation data")
    if not records:
        raise ValueError(f"cannot cache an empty {split} split")
    if encoder_batch_size <= 0 or samples_per_shard <= 0:
        raise ValueError("encoder batch size and samples per shard must be positive")

    from qwen_hotword.training.ctc_overfit import extract_frozen_features

    extract = extractor or extract_frozen_features
    destination = Path(output_dir).expanduser()
    shards_dir = destination / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(source_manifest_path).expanduser()
    model = Path(model_path).expanduser()
    vocab = Path(vocab_path).expanduser()
    config = _cache_config(
        records,
        split=split,
        source_manifest=source_manifest,
        model_path=model,
        model_dtype=model_dtype,
        vocab_path=vocab,
        samples_per_shard=samples_per_shard,
    )
    _ensure_cache_config(destination / "cache_config.json", config)

    shard_count = math.ceil(len(records) / samples_per_shard)
    generated_shards = 0
    resumed_shards = 0
    extraction_seconds = 0.0
    encoder_frozen_parameters = 0
    shard_metadata: list[dict[str, Any]] = []
    started = time.monotonic()

    for shard_index in range(shard_count):
        record_start = shard_index * samples_per_shard
        shard_records = records[record_start : record_start + samples_per_shard]
        feature_path, metadata_path = _shard_paths(shards_dir, shard_index)
        if feature_path.exists() and metadata_path.exists():
            metadata = validate_feature_shard(
                feature_path,
                metadata_path,
                shard_records,
                split=split,
                shard_index=shard_index,
                record_start=record_start,
            )
            resumed_shards += 1
            print(
                f"resumed {split} feature shard={shard_index:06d} "
                f"samples={len(shard_records)}",
                flush=True,
            )
        else:
            feature_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            cached, frozen_parameters, shard_seconds = extract(
                shard_records,
                wrapper,
                encoder_batch_size=encoder_batch_size,
                progress_every_batches=max(1, math.ceil(len(shard_records) / encoder_batch_size)),
                progress_label=f"{split} shard {shard_index:06d}",
            )
            if encoder_frozen_parameters not in {0, frozen_parameters}:
                raise RuntimeError("audio encoder parameter count changed during extraction")
            encoder_frozen_parameters = frozen_parameters
            metadata = write_feature_shard(
                feature_path,
                metadata_path,
                cached,
                shard_records,
                split=split,
                shard_index=shard_index,
                record_start=record_start,
                extraction_seconds=shard_seconds,
                encoder_frozen_parameters=frozen_parameters,
            )
            extraction_seconds += shard_seconds
            generated_shards += 1
            del cached
            print(
                f"completed {split} feature shard={shard_index:06d} "
                f"samples={len(shard_records)} frames={metadata['total_frames']}",
                flush=True,
            )
        encoder_frozen_parameters = max(
            encoder_frozen_parameters,
            int(metadata.get("encoder_frozen_parameters", 0)),
        )
        shard_metadata.append(metadata)
        _write_index(
            destination / "cache_index.json",
            split=split,
            sample_count=len(records),
            shard_count=shard_count,
            shards=shard_metadata,
            status="building" if len(shard_metadata) < shard_count else "pass",
        )

    summary = FeatureCacheSummary(
        split=split,
        output_dir=str(destination),
        source_manifest_path=str(source_manifest),
        sample_count=len(records),
        shard_count=shard_count,
        generated_shards=generated_shards,
        resumed_shards=resumed_shards,
        total_frames=sum(int(metadata["total_frames"]) for metadata in shard_metadata),
        total_target_tokens=sum(
            int(metadata["total_target_tokens"]) for metadata in shard_metadata
        ),
        feature_bytes=sum(int(metadata["feature_bytes"]) for metadata in shard_metadata),
        extraction_seconds=extraction_seconds,
        encoder_frozen_parameters=encoder_frozen_parameters,
        encoder_batch_size=encoder_batch_size,
        samples_per_shard=samples_per_shard,
        status="pass",
    )
    report = summary.to_dict()
    report["wall_seconds"] = time.monotonic() - started
    _write_json(destination / "cache_summary.json", report)
    return summary


def write_feature_shard(
    feature_path: Path,
    metadata_path: Path,
    cached: list[CachedSample],
    records: list[ExperimentRecord],
    *,
    split: str,
    shard_index: int,
    record_start: int,
    extraction_seconds: float,
    encoder_frozen_parameters: int,
) -> dict[str, Any]:
    import torch

    if len(cached) != len(records) or not records:
        raise ValueError("cached feature count must match the non-empty record shard")
    hidden_parts = []
    hidden_offsets = [0]
    token_parts = []
    token_offsets = [0]
    for sample, record in zip(cached, records, strict=True):
        if sample.sample_id != record.sample_id or sample.token_ids != record.token_ids:
            raise ValueError("cached sample order or labels do not match the source records")
        hidden = sample.hidden_states.detach().to(device="cpu", dtype=torch.bfloat16)
        if hidden.ndim != 2 or hidden.shape[1] != FEATURE_HIDDEN_SIZE:
            raise ValueError(
                f"sample {sample.sample_id} has invalid hidden shape: {list(hidden.shape)}"
            )
        if hidden.shape[0] < record.ctc_minimum_input_length:
            raise ValueError(f"sample {sample.sample_id} is not physically CTC-feasible")
        hidden_parts.append(hidden.contiguous())
        hidden_offsets.append(hidden_offsets[-1] + int(hidden.shape[0]))
        tokens = torch.tensor(record.token_ids, dtype=torch.int64)
        token_parts.append(tokens)
        token_offsets.append(token_offsets[-1] + len(record.token_ids))

    payload = {
        "hidden_states": torch.cat(hidden_parts, dim=0),
        "hidden_offsets": torch.tensor(hidden_offsets, dtype=torch.int64),
        "token_ids": torch.cat(token_parts, dim=0),
        "token_offsets": torch.tensor(token_offsets, dtype=torch.int64),
    }
    temporary_feature = feature_path.with_suffix(feature_path.suffix + ".tmp")
    temporary_feature.unlink(missing_ok=True)
    torch.save(payload, temporary_feature)
    temporary_feature.replace(feature_path)
    input_lengths = [
        hidden_offsets[index + 1] - hidden_offsets[index]
        for index in range(len(records))
    ]
    metadata: dict[str, Any] = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "split": split,
        "shard_index": shard_index,
        "record_start": record_start,
        "record_end": record_start + len(records),
        "record_count": len(records),
        "sample_ids": [record.sample_id for record in records],
        "hidden_size": FEATURE_HIDDEN_SIZE,
        "feature_dtype": FEATURE_DTYPE,
        "total_frames": hidden_offsets[-1],
        "total_target_tokens": token_offsets[-1],
        "minimum_input_length": min(input_lengths),
        "maximum_input_length": max(input_lengths),
        "feature_path": str(feature_path),
        "feature_bytes": feature_path.stat().st_size,
        "feature_sha256": _sha256_file(feature_path),
        "extraction_seconds": extraction_seconds,
        "encoder_frozen_parameters": encoder_frozen_parameters,
    }
    _write_json(metadata_path, metadata)
    return validate_feature_shard(
        feature_path,
        metadata_path,
        records,
        split=split,
        shard_index=shard_index,
        record_start=record_start,
    )


def validate_feature_shard(
    feature_path: Path,
    metadata_path: Path,
    records: list[ExperimentRecord],
    *,
    split: str,
    shard_index: int,
    record_start: int,
) -> dict[str, Any]:
    import torch

    if not feature_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"incomplete feature shard: {feature_path}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"feature shard metadata must be an object: {metadata_path}")
    expected = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "split": split,
        "shard_index": shard_index,
        "record_start": record_start,
        "record_end": record_start + len(records),
        "record_count": len(records),
        "sample_ids": [record.sample_id for record in records],
        "hidden_size": FEATURE_HIDDEN_SIZE,
        "feature_dtype": FEATURE_DTYPE,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"feature shard metadata mismatch for {key}: {metadata_path}")
    if raw.get("feature_bytes") != feature_path.stat().st_size:
        raise ValueError(f"feature shard size mismatch: {feature_path}")
    if raw.get("feature_sha256") != _sha256_file(feature_path):
        raise ValueError(f"feature shard SHA256 mismatch: {feature_path}")

    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "hidden_states",
        "hidden_offsets",
        "token_ids",
        "token_offsets",
    }:
        raise ValueError(f"feature shard has an invalid tensor payload: {feature_path}")
    hidden = payload["hidden_states"]
    hidden_offsets = payload["hidden_offsets"]
    tokens = payload["token_ids"]
    token_offsets = payload["token_offsets"]
    if hidden.ndim != 2 or list(hidden.shape[1:]) != [FEATURE_HIDDEN_SIZE]:
        raise ValueError(f"feature shard hidden tensor has invalid shape: {feature_path}")
    if hidden.dtype != torch.bfloat16 or tokens.dtype != torch.int64:
        raise ValueError(f"feature shard tensor dtypes are invalid: {feature_path}")
    if hidden_offsets.dtype != torch.int64 or token_offsets.dtype != torch.int64:
        raise ValueError(f"feature shard offset dtypes are invalid: {feature_path}")
    if hidden_offsets.ndim != 1 or token_offsets.ndim != 1:
        raise ValueError(f"feature shard offsets must be one-dimensional: {feature_path}")
    if len(hidden_offsets) != len(records) + 1 or len(token_offsets) != len(records) + 1:
        raise ValueError(f"feature shard offset counts are invalid: {feature_path}")
    hidden_values = hidden_offsets.tolist()
    token_values = token_offsets.tolist()
    if hidden_values[0] != 0 or hidden_values[-1] != hidden.shape[0]:
        raise ValueError(f"feature shard hidden offsets are invalid: {feature_path}")
    if token_values[0] != 0 or token_values[-1] != tokens.shape[0]:
        raise ValueError(f"feature shard token offsets are invalid: {feature_path}")
    for index, record in enumerate(records):
        input_length = hidden_values[index + 1] - hidden_values[index]
        if input_length < record.ctc_minimum_input_length:
            raise ValueError(f"cached sample {record.sample_id} is not CTC-feasible")
        actual_tokens = tuple(tokens[token_values[index] : token_values[index + 1]].tolist())
        if actual_tokens != record.token_ids:
            raise ValueError(f"cached labels do not match sample {record.sample_id}")
    if raw.get("total_frames") != hidden.shape[0]:
        raise ValueError(f"feature shard frame count mismatch: {feature_path}")
    if raw.get("total_target_tokens") != tokens.shape[0]:
        raise ValueError(f"feature shard target count mismatch: {feature_path}")
    return raw


def _cache_config(
    records: list[ExperimentRecord],
    *,
    split: str,
    source_manifest: Path,
    model_path: Path,
    model_dtype: str,
    vocab_path: Path,
    samples_per_shard: int,
) -> dict[str, object]:
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "experiment": "full-ctc-v1",
        "split": split,
        "sample_count": len(records),
        "sample_id_sha256": _strings_sha256([record.sample_id for record in records]),
        "source_manifest": _file_identity(source_manifest),
        "model": {
            "path": str(model_path.resolve()),
            "dtype": model_dtype,
            "config_sha256": _sha256_file(model_path / "config.json"),
            "weight_index_sha256": _sha256_file(
                model_path / "model.safetensors.index.json"
            ),
        },
        "vocab": _file_identity(vocab_path),
        "tap_module": "thinker.audio_tower.ln_post",
        "hidden_size": FEATURE_HIDDEN_SIZE,
        "feature_dtype": FEATURE_DTYPE,
        "samples_per_shard": samples_per_shard,
    }


def _ensure_cache_config(path: Path, expected: dict[str, object]) -> None:
    if not path.exists():
        _write_json(path, expected)
        return
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError(
            f"feature cache configuration does not match the requested run: {path}; "
            "use a new output directory"
        )


def _write_index(
    path: Path,
    *,
    split: str,
    sample_count: int,
    shard_count: int,
    shards: list[dict[str, Any]],
    status: str,
) -> None:
    _write_json(
        path,
        {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "split": split,
            "sample_count": sample_count,
            "shard_count": shard_count,
            "completed_shards": len(shards),
            "completed_samples": sum(int(shard["record_count"]) for shard in shards),
            "status": status,
            "shards": [
                {
                    key: shard[key]
                    for key in (
                        "shard_index",
                        "record_start",
                        "record_end",
                        "record_count",
                        "total_frames",
                        "total_target_tokens",
                        "feature_path",
                        "feature_bytes",
                        "feature_sha256",
                    )
                }
                for shard in shards
            ],
        },
    )


def _shard_paths(shards_dir: Path, shard_index: int) -> tuple[Path, Path]:
    stem = f"shard-{shard_index:06d}"
    return shards_dir / f"{stem}.pt", shards_dir / f"{stem}.json"


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"identity input does not exist: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": _sha256_file(path),
    }


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
