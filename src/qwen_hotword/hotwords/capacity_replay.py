from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from qwen_hotword.config import EXPECTED_MODEL_NAME, ModelConfig
from qwen_hotword.hotwords.capacity_assets import CapacityCase, load_capacity_base_cases
from qwen_hotword.hotwords.scoring import DecodedPhoneme, decode_ctc_posterior
from qwen_hotword.inference.streaming_core import schedule_stream_chunks
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.training.ctc_overfit import build_audio_prompt, freeze_module
from qwen_hotword.training.sharded_ctc import (
    load_disk_feature_cache,
    load_feature_shard,
)


def build_offline_capacity_replay(
    *,
    validation_cache_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    checkpoint_path: str | Path,
    cases_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 64,
    verify_cache_sha256: bool = True,
    print_progress: bool = True,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = _validated_paths(
        validation_manifest=validation_manifest_path,
        vocab=vocab_path,
        checkpoint=checkpoint_path,
        cases=cases_path,
        output=output_dir,
    )
    cache_root = Path(validation_cache_path).expanduser()
    if not cache_root.is_dir():
        raise FileNotFoundError(f"validation cache does not exist: {cache_root}")
    destination = paths["output"]
    _prepare_output(destination)

    import torch
    from torch.nn.utils.rnn import pad_sequence

    vocab = load_phoneme_vocab(paths["vocab"])
    head = _load_temporal_head(paths["checkpoint"], vocab, device=device)
    cases = load_capacity_base_cases(paths["cases"])
    case_by_sample = _case_by_sample(cases)
    manifest_rows = _load_validation_rows(paths["validation_manifest"])
    cache = load_disk_feature_cache(
        cache_root,
        expected_split="validation",
        source_manifest_path=paths["validation_manifest"],
        vocab_path=paths["vocab"],
        verify_sha256=verify_cache_sha256,
    )
    cached_ids = {sample_id for shard in cache.shards for sample_id in shard.sample_ids}
    missing = set(case_by_sample) - cached_ids
    if missing:
        raise ValueError(f"{len(missing)} capacity cases are missing from validation cache")

    rows: list[dict[str, object]] = []
    processed = 0
    started = time.monotonic()
    with torch.no_grad():
        for shard_index, descriptor in enumerate(cache.shards, start=1):
            wanted = set(descriptor.sample_ids) & set(case_by_sample)
            if not wanted:
                continue
            samples = [
                sample
                for sample in load_feature_shard(descriptor, num_classes=len(vocab.tokens))
                if sample.sample_id in wanted
            ]
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                hidden = pad_sequence(
                    [sample.hidden_states for sample in batch],
                    batch_first=True,
                    padding_value=0.0,
                ).to(device=device, dtype=torch.float32)
                lengths = torch.tensor(
                    [sample.hidden_states.shape[0] for sample in batch],
                    dtype=torch.long,
                    device=device,
                )
                _synchronize(device)
                head_started = time.perf_counter()
                logits = head(hidden, input_lengths=lengths)
                effective = head.output_lengths(lengths)
                _synchronize(device)
                head_seconds = time.perf_counter() - head_started
                for row_index, sample in enumerate(batch):
                    decode_started = time.perf_counter()
                    decoded = decode_ctc_posterior(
                        logits[row_index],
                        input_length=int(effective[row_index].item()),
                        blank_id=0,
                    )
                    decode_seconds = time.perf_counter() - decode_started
                    case = case_by_sample[sample.sample_id]
                    manifest = manifest_rows[sample.sample_id]
                    rows.append(
                        _replay_row(
                            case,
                            chunk_id=0,
                            cumulative_audio_sec=cast(float, manifest["duration_seconds"]),
                            is_final=True,
                            is_tail_flush=False,
                            effective_time_steps=int(effective[row_index].item()),
                            decoded=decoded,
                            source_timings={
                                "ctc_head_batch_seconds": head_seconds,
                                "ctc_head_batch_size": len(batch),
                                "ctc_decode_seconds": decode_seconds,
                            },
                        )
                    )
                processed += len(batch)
                if print_progress:
                    print(
                        f"capacity offline replay={processed}/{len(cases)} "
                        f"shard={shard_index}/{len(cache.shards)}",
                        flush=True,
                    )
    if len(rows) != len(cases):
        raise RuntimeError(f"offline replay wrote {len(rows)} rows for {len(cases)} cases")
    rows.sort(key=lambda row: str(row["case_id"]))
    return _finalize_replay(
        destination,
        rows,
        mode="offline_validation_feature_cache",
        inputs={
            "validation_cache_index": cache.root / "cache_index.json",
            "validation_manifest": paths["validation_manifest"],
            "vocab": paths["vocab"],
            "checkpoint": paths["checkpoint"],
            "cases": paths["cases"],
        },
        elapsed_seconds=time.monotonic() - started,
        extra_config={
            "device": device,
            "batch_size": batch_size,
            "verify_cache_sha256": verify_cache_sha256,
        },
    )


def build_streaming_capacity_replay(
    *,
    model_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    checkpoint_path: str | Path,
    cases_path: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    language: str = "Portuguese",
    chunk_size_sec: float = 2.0,
    max_samples: int = 0,
    print_progress: bool = True,
) -> dict[str, object]:
    if chunk_size_sec != 2.0:
        raise ValueError("capacity v1 streaming replay is sealed to 2.0 second chunks")
    if max_samples < 0:
        raise ValueError("max_samples must not be negative")
    paths = _validated_paths(
        validation_manifest=validation_manifest_path,
        vocab=vocab_path,
        checkpoint=checkpoint_path,
        cases=cases_path,
        output=output_dir,
    )
    model = Path(model_path).expanduser()
    if model.name != EXPECTED_MODEL_NAME or not model.is_dir():
        raise ValueError(f"model path must be an existing {EXPECTED_MODEL_NAME}: {model}")
    destination = paths["output"]
    _prepare_output(destination)

    import torch

    from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post

    vocab = load_phoneme_vocab(paths["vocab"])
    head = _load_temporal_head(paths["checkpoint"], vocab, device=device)
    wrapper = load_asr_model(
        ModelConfig(
            path=model,
            expected_name=EXPECTED_MODEL_NAME,
            dtype=dtype,
            device=device,
            local_files_only=True,
        )
    )
    freeze_module(wrapper.model.thinker.audio_tower)
    cases = load_capacity_base_cases(paths["cases"])
    manifest_rows = _load_validation_rows(paths["validation_manifest"])
    if max_samples:
        cases = cases[:max_samples]

    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for case_index, case in enumerate(cases, start=1):
        manifest = manifest_rows.get(case.sample_id)
        if manifest is None:
            raise ValueError(f"case {case.case_id} is absent from validation manifest")
        waveform = _load_waveform(str(manifest["audio_path"]))
        chunks = schedule_stream_chunks(
            len(waveform), sample_rate=16_000, chunk_size_sec=chunk_size_sec
        )
        if not chunks:
            raise ValueError(f"case {case.case_id} contains empty audio")
        for chunk in chunks:
            cumulative = waveform[: chunk.end_sample]
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
            processor_started = time.perf_counter()
            prompt = build_audio_prompt(wrapper.processor, language)
            processor_batch = wrapper.processor(
                text=[prompt], audio=[cumulative], return_tensors="pt", padding=True
            )
            processor_seconds = time.perf_counter() - processor_started
            input_features = processor_batch["input_features"].to(
                device=wrapper.model.device,
                dtype=wrapper.model.dtype,
            )
            feature_attention_mask = processor_batch["feature_attention_mask"].to(
                device=wrapper.model.device
            )
            _synchronize(device)
            encoder_started = time.perf_counter()
            encoder = extract_padded_ln_post(
                wrapper.model.thinker.audio_tower,
                input_features,
                feature_attention_mask,
                no_grad=True,
            )
            _synchronize(device)
            encoder_seconds = time.perf_counter() - encoder_started
            hidden = encoder.hidden_states.to(device=device, dtype=torch.float32)
            lengths = encoder.input_lengths.to(device=device)
            _synchronize(device)
            head_started = time.perf_counter()
            with torch.no_grad():
                logits = head(hidden, input_lengths=lengths)
                effective = head.output_lengths(lengths)
            _synchronize(device)
            head_seconds = time.perf_counter() - head_started
            decode_started = time.perf_counter()
            decoded = decode_ctc_posterior(
                logits[0], input_length=int(effective[0].item()), blank_id=0
            )
            decode_seconds = time.perf_counter() - decode_started
            gpu_memory = _gpu_memory_snapshot(device)
            rows.append(
                _replay_row(
                    case,
                    chunk_id=chunk.chunk_id,
                    cumulative_audio_sec=chunk.end_sec,
                    is_final=chunk == chunks[-1],
                    is_tail_flush=chunk.is_tail_flush,
                    effective_time_steps=int(effective[0].item()),
                    decoded=decoded,
                    source_timings={
                        "processor_seconds": processor_seconds,
                        "encoder_seconds": encoder_seconds,
                        "ctc_head_seconds": head_seconds,
                        "ctc_decode_seconds": decode_seconds,
                        **gpu_memory,
                    },
                )
            )
        if print_progress:
            print(
                f"capacity streaming replay={case_index}/{len(cases)} "
                f"case={case.case_id} chunks={len(chunks)}",
                flush=True,
            )
    return _finalize_replay(
        destination,
        rows,
        mode="streaming_cumulative_audio_2s",
        inputs={
            "model_config": model / "config.json",
            "validation_manifest": paths["validation_manifest"],
            "vocab": paths["vocab"],
            "checkpoint": paths["checkpoint"],
            "cases": paths["cases"],
        },
        elapsed_seconds=time.monotonic() - started,
        extra_config={
            "model_path": str(model),
            "device": device,
            "dtype": dtype,
            "language": language,
            "chunk_size_sec": chunk_size_sec,
            "ctc_input_strategy": "cumulative_audio",
            "max_samples": max_samples,
        },
    )


def load_capacity_replay(path: str | Path) -> tuple[dict[str, Any], ...]:
    replay = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with replay.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_object(line, replay, line_number)
            case_id = _required_string(raw, "case_id", line_number)
            chunk_id = raw.get("chunk_id")
            if not isinstance(chunk_id, int) or isinstance(chunk_id, bool) or chunk_id < 0:
                raise ValueError(f"replay row {line_number} has invalid chunk_id")
            key = (case_id, chunk_id)
            if key in seen:
                raise ValueError(f"duplicate replay case/chunk: {key}")
            effective = raw.get("effective_time_steps")
            if not isinstance(effective, int) or effective <= 0:
                raise ValueError(f"replay row {line_number} has invalid effective_time_steps")
            raw["decoded"] = [
                asdict_decoded(item, effective_time_steps=effective, line_number=line_number)
                for item in _object_list(raw, "decoded", line_number)
            ]
            rows.append(raw)
            seen.add(key)
    if not rows:
        raise ValueError("capacity replay is empty")
    return tuple(rows)


def replay_decoded_phonemes(row: Mapping[str, Any]) -> tuple[DecodedPhoneme, ...]:
    decoded = row.get("decoded")
    if not isinstance(decoded, list):
        raise ValueError("replay row has invalid decoded phonemes")
    return tuple(
        DecodedPhoneme(
            token_id=int(item["token_id"]),
            confidence=float(item["confidence"]),
            start_step=int(item["start_step"]),
            end_step=int(item["end_step"]),
        )
        for item in decoded
    )


def asdict_decoded(
    raw: Mapping[str, Any], *, effective_time_steps: int, line_number: int
) -> dict[str, object]:
    token_id = raw.get("token_id")
    confidence = raw.get("confidence")
    start = raw.get("start_step")
    end = raw.get("end_step")
    if (
        not isinstance(token_id, int)
        or isinstance(token_id, bool)
        or token_id <= 0
        or not isinstance(confidence, int | float)
        or isinstance(confidence, bool)
        or not 0.0 <= float(confidence) <= 1.0
        or not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or not 0 <= start < end <= effective_time_steps
    ):
        raise ValueError(f"replay row {line_number} has invalid decoded phoneme")
    return {
        "token_id": token_id,
        "confidence": float(confidence),
        "start_step": start,
        "end_step": end,
    }


def _load_temporal_head(path: Path, vocab: PhonemeVocab, *, device: str) -> Any:
    import torch

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("CTC checkpoint must be a mapping")
    if tuple(payload.get("vocab_tokens", ())) != tuple(vocab.tokens):
        raise ValueError("CTC checkpoint vocabulary differs from capacity vocabulary")
    head = build_ctc_head_from_checkpoint(payload)
    if not isinstance(head, TemporalUpsampleCtcHead) or head.time_upsampling_factor != 2:
        raise ValueError("capacity replay requires the sealed Temporal 2x CTC Head")
    head.load_state_dict(payload["state_dict"], strict=True)
    head = head.to(device=device, dtype=torch.float32)
    freeze_module(head)
    return head


def _replay_row(
    case: CapacityCase,
    *,
    chunk_id: int,
    cumulative_audio_sec: float,
    is_final: bool,
    is_tail_flush: bool,
    effective_time_steps: int,
    decoded: Sequence[DecodedPhoneme],
    source_timings: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": case.case_id,
        "sample_id": case.sample_id,
        "primary_group": case.primary_group,
        "expected_hotword_ids": list(case.expected_hotword_ids),
        "chunk_id": chunk_id,
        "cumulative_audio_sec": cumulative_audio_sec,
        "is_final": is_final,
        "is_tail_flush": is_tail_flush,
        "effective_time_steps": effective_time_steps,
        "decoded": [item.to_dict() for item in decoded],
        "source_timings": dict(source_timings),
    }


def _finalize_replay(
    destination: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    mode: str,
    inputs: Mapping[str, Path],
    elapsed_seconds: float,
    extra_config: Mapping[str, object],
) -> dict[str, object]:
    replay_path = destination / "ctc_replay.jsonl"
    config_path = destination / "run_config.json"
    summary_path = destination / "summary.json"
    _write_jsonl(replay_path, rows)
    config = {
        "schema_version": 1,
        "purpose": "portuguese_hotword_capacity_ctc_replay",
        "mode": mode,
        "inputs": {key: _file_identity(path) for key, path in inputs.items()},
        **dict(extra_config),
        "test_set_used": False,
    }
    summary = {
        "schema_version": 1,
        "status": "pass",
        "mode": mode,
        "replay_rows": len(rows),
        "cases": len({str(row["case_id"]) for row in rows}),
        "final_rows": sum(bool(row["is_final"]) for row in rows),
        "tail_flush_rows": sum(bool(row["is_tail_flush"]) for row in rows),
        "elapsed_seconds": elapsed_seconds,
        "replay_path": str(replay_path),
        "replay_sha256": _sha256_file(replay_path),
        "test_set_used": False,
    }
    _write_json(config_path, config)
    _write_json(summary_path, summary)
    _write_sha256_manifest(destination)
    return summary


def _validated_paths(**values: str | Path) -> dict[str, Path]:
    paths = {key: Path(value).expanduser() for key, value in values.items()}
    for key, path in paths.items():
        if key != "output" and not path.is_file():
            raise FileNotFoundError(f"capacity replay input does not exist: {path}")
    return paths


def _prepare_output(destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"capacity replay output must be a new empty directory: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)


def _case_by_sample(cases: Sequence[CapacityCase]) -> dict[str, CapacityCase]:
    values = {case.sample_id: case for case in cases}
    if len(values) != len(cases):
        raise ValueError("capacity cases have duplicate sample IDs")
    return values


def _load_validation_rows(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_object(line, path, line_number)
            split = _required_string(raw, "split", line_number)
            if split == "test":
                raise ValueError("sealed test data is forbidden in capacity replay")
            if split != "validation":
                raise ValueError(f"capacity replay manifest row {line_number} is not validation")
            sample_id = _required_string(raw, "id", line_number)
            if sample_id in rows:
                raise ValueError(f"capacity replay manifest has duplicate sample ID: {sample_id}")
            duration = raw.get("duration_seconds")
            if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
                raise ValueError(f"validation row {line_number} has invalid duration_seconds")
            rows[sample_id] = {
                "audio_path": _required_string(raw, "audio_path", line_number),
                "duration_seconds": float(duration),
            }
    return rows


def _load_waveform(path: str) -> Any:
    try:
        import librosa
    except ImportError as error:
        raise RuntimeError("librosa is required for streaming capacity replay") from error
    waveform, _ = librosa.load(path, sr=16_000, mono=True)
    return waveform


def _synchronize(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize(device)


def _gpu_memory_snapshot(device: str) -> dict[str, int | None]:
    if not device.startswith("cuda"):
        return {
            "gpu_allocated_bytes": None,
            "gpu_reserved_bytes": None,
            "gpu_peak_allocated_bytes": None,
        }
    import torch

    return {
        "gpu_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "gpu_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _object_list(raw: Mapping[str, Any], key: str, line_number: int) -> list[Mapping[str, Any]]:
    value = raw.get(key)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"replay row {line_number} has invalid {key}")
    return value


def _load_object(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"JSON row {line_number} must be an object")
    return raw


def _required_string(raw: Mapping[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {line_number} has invalid {key}")
    return value.strip()


def _file_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_sha256_manifest(destination: Path) -> None:
    paths = sorted(
        path for path in destination.rglob("*") if path.is_file() and path.name != "sha256.txt"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(destination)}" for path in paths]
    (destination / "sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
