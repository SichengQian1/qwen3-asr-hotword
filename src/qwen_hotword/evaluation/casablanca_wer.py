"""Transcribe local Casablanca shards with Qwen3-ASR and score corpus WER."""

from __future__ import annotations

import io
import json
import os
import unicodedata
from collections import Counter
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.backend_comparison import (
    block_legacy_qwen_vllm_registration,
    read,
    sha,
    write_once,
)
from qwen_hotword.training.stage_progress import stage_progress

DATA_COLUMNS = ("seg_id", "audio", "transcription")
ARABIC_VOWEL_MARKS = frozenset(chr(code) for code in range(0x064B, 0x0653)) | {"ٰ", "ـ"}
NORMALIZATION = (
    "NFC and casefold; remove Arabic short-vowel marks, shadda, sukun, dagger alef and "
    "tatweel; map punctuation to spaces except apostrophes inside words; split on whitespace"
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'")
    text = "".join(char for char in text if char not in ARABIC_VOWEL_MARKS)
    chars = []
    for index, char in enumerate(text):
        if (
            char == "'"
            and 0 < index < len(text) - 1
            and text[index - 1].isalnum()
            and text[index + 1].isalnum()
        ):
            chars.append(char)
        else:
            chars.append(" " if unicodedata.category(char).startswith("P") else char)
    return " ".join("".join(chars).split())


def inspect_checkpoint(model_path: Path, base_model: Path | None = None) -> dict[str, Any]:
    optimizer_locator = model_path.name == "optimizer.pt"
    directory = (model_path.parent if optimizer_locator else model_path).resolve()
    config_path = directory / "adapter_config.json"
    if not config_path.is_file():
        if base_model is not None:
            raise ValueError("--base-model is only used with a PEFT LoRA checkpoint")
        return {
            "format": "full_hf",
            "optimizer_path_used_as_locator": optimizer_locator,
            **inspect_full_model(directory),
        }

    config = read(config_path)
    if config.get("peft_type") != "LORA":
        raise ValueError(f"{config_path}: expected peft_type=LORA, got {config.get('peft_type')!r}")
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        weight = directory / name
        if weight.is_file() and weight.stat().st_size > 0:
            break
        if weight.exists():
            raise ValueError(f"invalid or empty adapter weight: {weight}")
    else:
        raise ValueError(f"{directory}: missing adapter_model.safetensors or adapter_model.bin")

    recorded_base = config.get("base_model_name_or_path")
    base_source = "--base-model" if base_model is not None else "adapter_config.json"
    if base_model is None:
        base_model = Path(recorded_base) if isinstance(recorded_base, str) else Path()
        if not base_model.is_absolute() or not base_model.is_dir():
            raise ValueError(
                f"{config_path}: base_model_name_or_path={recorded_base!r} "
                "is not an accessible absolute local directory. "
                "Pass --base-model with the exact local Qwen3-ASR checkpoint used to train LoRA."
            )
    return {
        "format": "peft_lora",
        "directory": str(directory),
        "optimizer_path_used_as_locator": optimizer_locator,
        "adapter_config_sha256": sha(config_path),
        "adapter_weight": {"file": weight.name, "bytes": weight.stat().st_size},
        "recorded_base": recorded_base,
        "recorded_base_revision": config.get("revision"),
        "base_source": base_source,
        "base_model": inspect_full_model(base_model.resolve()),
    }


def inspect_full_model(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValueError(
            f"checkpoint directory is unavailable: {directory} (check /data_103 mount)"
        )
    if (directory / "adapter_config.json").exists():
        raise ValueError(f"{directory}: the base-model path must be a full HF checkpoint")

    formats = ("model.safetensors", "pytorch_model.bin")
    weights = []
    index = None
    for name in formats:
        if (directory / name).is_file():
            weights = [directory / name]
            break
        index_path = directory / f"{name}.index.json"
        if index_path.is_file():
            index = index_path
            weight_map = read(index).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid or empty weight_map: {index}")
            suffix = Path(name).suffix
            if any(
                not isinstance(shard, str)
                or Path(shard).name != shard
                or not shard.endswith(suffix)
                for shard in weight_map.values()
            ):
                raise ValueError(f"invalid local weight shard in {index}")
            weights = [directory / shard for shard in sorted(set(weight_map.values()))]
            break
    if not weights:
        raise ValueError(
            f"{directory}: no Hugging Face model weights. optimizer.pt is optimizer state. "
            "Supply the complete Qwen3-ASR checkpoint or export the step-4900 model."
        )

    for weight in weights:
        if not weight.is_file() or weight.stat().st_size == 0:
            raise ValueError(f"missing or empty model weight: {weight}")
    metadata = ["config.json", "preprocessor_config.json", "tokenizer_config.json"]
    if (directory / "tokenizer.json").is_file():
        metadata.append("tokenizer.json")
    else:
        metadata.extend(("vocab.json", "merges.txt"))
    for name in metadata:
        if not (directory / name).is_file():
            raise ValueError(f"missing HF model/processor file: {directory / name}")
    if read(directory / "config.json").get("model_type") != "qwen3_asr":
        raise ValueError(f"expected a Qwen3-ASR model_type in {directory / 'config.json'}")
    return {
        "directory": str(directory),
        "config_sha256": sha(directory / "config.json"),
        "weight_index_sha256": sha(index) if index else None,
        "weights": [{"file": p.name, "bytes": p.stat().st_size} for p in weights],
    }


def inspect_shards(paths: list[Path]) -> list[dict[str, Any]]:
    arrow = import_module("pyarrow")
    parquet = import_module("pyarrow.parquet")
    resolved = [path.resolve() for path in paths]
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("supply at least one distinct Parquet path")
    shards = []
    for path in resolved:
        with parquet.ParquetFile(path) as source:
            schema = source.schema_arrow
            missing = set(DATA_COLUMNS) - set(schema.names)
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            audio_type = schema.field("audio").type
            if not arrow.types.is_struct(audio_type) or all(
                audio_type.get_field_index(name) == -1 for name in ("bytes", "path")
            ):
                raise ValueError(f"{path}: audio must be a struct with bytes or path")
            text_type = schema.field("transcription").type
            if not (arrow.types.is_string(text_type) or arrow.types.is_large_string(text_type)):
                raise ValueError(f"{path}: transcription must be a string")
            rows = source.metadata.num_rows
        if rows <= 0:
            raise ValueError(f"{path}: no validation rows")
        shards.append({"path": str(path), "rows": rows, "bytes": path.stat().st_size})
    return shards


def load_qwen(checkpoint: str, gpu: int, batch_size: int, max_new_tokens: int) -> Any:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"CUDA GPU {gpu} is unavailable as the sole visible device")
    with block_legacy_qwen_vllm_registration():
        from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


def attach_lora(model: Any, directory: str) -> int:
    """Keep Qwen's original generation methods and validate the injected weights."""
    peft = import_module("peft")
    config = peft.PeftConfig.from_pretrained(directory, local_files_only=True)
    config.inference_mode = True
    saved = peft.utils.load_peft_weights(directory, device="cpu", local_files_only=True)
    # A PEFT wrapper adds this prefix to saved tensors. The Qwen model itself has no prefix.
    weights = {key.removeprefix("base_model.model."): value for key, value in saved.items()}
    if not weights or len(weights) != len(saved):
        raise ValueError(f"{directory}: empty adapter or duplicate keys after removing PEFT prefix")
    peft.inject_adapter_in_model(config, model)
    expected = peft.get_peft_model_state_dict(model, save_embedding_layers=False)
    missing = sorted(set(expected) - weights.keys())
    result = peft.set_peft_model_state_dict(model, weights)
    if not expected or missing or result.unexpected_keys:
        raise ValueError(
            f"{directory}: adapter tensors do not match the base model "
            f"(expected={len(expected)}, missing={missing[:5]}, "
            f"unexpected={result.unexpected_keys[:5]})"
        )
    model.eval()
    return len(weights)


def decode_audio(audio: Any, shard: Path, location: str) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    if not isinstance(audio, dict):
        raise ValueError(f"{location}: expected a Casablanca audio struct")
    if isinstance(audio.get("bytes"), bytes) and audio["bytes"]:
        source: io.BytesIO | Path = io.BytesIO(audio["bytes"])
    elif isinstance(audio.get("path"), str) and audio["path"]:
        source = shard.parent / audio["path"]
    else:
        raise ValueError(f"{location}: missing audio bytes and path")
    try:
        samples, rate = sf.read(source, dtype="float32", always_2d=True)
    except (OSError, ValueError, RuntimeError) as error:
        raise ValueError(f"{location}: audio decode failed: {error}") from error
    if rate <= 0 or not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"{location}: invalid audio samples or sample rate")
    return samples.mean(axis=1), int(rate)


def evaluate(
    model: Any,
    shards: list[dict[str, Any]],
    output: Path,
    batch_size: int,
    language: str | None,
) -> dict[str, Any]:
    parquet = import_module("pyarrow.parquet")
    jiwer = import_module("jiwer")
    metrics: dict[str, Counter[str]] = {"normalized": Counter(), "raw": Counter()}
    expected = sum(shard["rows"] for shard in shards)
    completed = 0
    with (
        (output / "predictions.jsonl").open("x", encoding="utf-8") as results,
        stage_progress("Transcribe and score validation", total=expected, every=batch_size) as step,
    ):
        for shard in shards:
            path = Path(shard["path"])
            row_index = 0
            with parquet.ParquetFile(path) as source:
                for batch in source.iter_batches(batch_size=batch_size, columns=list(DATA_COLUMNS)):
                    rows = batch.to_pylist()
                    audios = [
                        decode_audio(row["audio"], path, f"{path}:{row_index + offset + 1}")
                        for offset, row in enumerate(rows)
                    ]
                    predictions = model.transcribe(
                        audio=audios, language=language, return_time_stamps=False
                    )
                    if len(predictions) != len(rows):
                        raise RuntimeError(f"{path}:{row_index + 1}: prediction count mismatch")
                    for row, prediction in zip(rows, predictions, strict=True):
                        row_index += 1
                        reference = row["transcription"]
                        if not isinstance(reference, str) or not reference.strip():
                            raise ValueError(f"{path}:{row_index}: missing reference transcription")
                        pairs = {
                            "normalized": (normalize(reference), normalize(prediction.text)),
                            "raw": (reference, prediction.text),
                        }
                        scores = {}
                        for key, (ref, hyp) in pairs.items():
                            ref_words = ref.split()
                            edits = jiwer.process_words(" ".join(ref_words), " ".join(hyp.split()))
                            score = {
                                "reference_words": len(ref_words),
                                "substitutions": edits.substitutions,
                                "deletions": edits.deletions,
                                "insertions": edits.insertions,
                            }
                            scores[key] = score
                            metrics[key].update(score)
                        record = {
                            "shard": path.name,
                            "row": row_index,
                            "seg_id": row["seg_id"],
                            "reference": reference,
                            "prediction": prediction.text,
                            "language": prediction.language,
                            "edits": scores,
                        }
                        results.write(json.dumps(record, ensure_ascii=False) + "\n")
                    results.flush()
                    completed += len(rows)
                    step(completed)
            if row_index != shard["rows"]:
                raise RuntimeError(f"{path}: scored {row_index} rows, expected {shard['rows']}")

    summary: dict[str, Any] = {}
    for key, counts in metrics.items():
        if counts["reference_words"] == 0:
            raise ValueError(f"no reference words for {key} WER")
        wer = (counts["substitutions"] + counts["deletions"] + counts["insertions"]) / counts[
            "reference_words"
        ]
        summary[key] = {**counts, "wer": wer, "wer_percent": 100 * wer}
    return {"expected_samples": expected, "scored_samples": completed, "wer": summary}


def run(
    checkpoint: dict[str, Any],
    shards: list[dict[str, Any]],
    output: Path,
    gpu: int,
    batch_size: int,
    max_new_tokens: int,
    language: str | None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {output}", flush=True)
    packages = ["qwen-asr", "transformers", "torch", "pyarrow", "soundfile", "jiwer"]
    is_lora = checkpoint["format"] == "peft_lora"
    if is_lora:
        packages.append("peft")
    versions = {name: version(name) for name in packages}
    model_path = checkpoint["base_model"]["directory"] if is_lora else checkpoint["directory"]
    with stage_progress("Load local Qwen3-ASR checkpoint"):
        model = load_qwen(model_path, gpu, batch_size, max_new_tokens)
    if is_lora:
        with stage_progress("Attach and verify checkpoint LoRA tensors"):
            loaded_tensors = attach_lora(model.model, checkpoint["directory"])
        checkpoint = {**checkpoint, "adapter_tensors_loaded": loaded_tensors}
    result = {
        "status": "completed",
        "dataset": "Casablanca/Morocco/validation",
        "shards": shards,
        "checkpoint": checkpoint,
        "inference": {
            "backend": "official_qwen_asr_transformers",
            "physical_gpu": gpu,
            "dtype": "bfloat16",
            "language": language or "auto",
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "versions": versions,
        },
        "normalization": NORMALIZATION,
        "output_dir": str(output),
        **evaluate(model, shards, output, batch_size, language),
    }
    write_once(output / "report.json", result)
    return result
