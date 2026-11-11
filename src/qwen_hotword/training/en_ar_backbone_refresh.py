"""Freeze an English/Arabic paired run on a new, fully merged Qwen backbone."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml

from qwen_hotword.config import ModelConfig
from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
from qwen_hotword.modeling.qwen_backbone import inspect_local_config, inspection_dict
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.experiment_a import ctc_minimum_input_length
from qwen_hotword.training.stage_progress import stage_progress

HF_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.jinja",
    "model.safetensors",
)


def make_model_view(
    source: Path, destination: Path, architecture: dict[str, Any]
) -> dict[str, Any]:
    """Read the original export; write only links and a real one-shard index in the new run."""
    from safetensors import safe_open

    required = (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors",
    )
    if not source.is_dir() or (source / "adapter_config.json").exists():
        raise ValueError("a complete merged HF model directory is required")
    if any(not (source / name).is_file() for name in required):
        raise ValueError(f"merged model is missing required HF files: {source}")
    if not (source / "tokenizer.json").is_file() and not all(
        (source / name).is_file() for name in ("vocab.json", "merges.txt")
    ):
        raise ValueError("merged model tokenizer is incomplete")
    if json.loads((source / "config.json").read_text())["model_type"] != "qwen3_asr":
        raise ValueError("merged model must use qwen3_asr architecture")
    destination.mkdir(parents=True, exist_ok=False)
    identities: dict[str, dict[str, Any]] = {}
    files = [name for name in HF_FILES if (source / name).is_file()]
    with stage_progress(
        "Seal merged HF model files (including full weight SHA256)", total=len(files), every=1
    ) as tick:
        for index, name in enumerate(files, 1):
            path = (source / name).resolve(strict=True)
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"model file changed while hashing: {path}")
            identities[name] = {
                "path": str(path),
                "sha256": digest,
                "bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
            }
            (destination / name).symlink_to(path)
            tick(index)
    with safe_open(str(source / "model.safetensors"), framework="np") as handle:
        keys = list(handle.keys())
    if not keys or not any(key.startswith("thinker.audio_tower.ln_post.") for key in keys):
        raise ValueError("merged weights do not expose the expected ln_post tap")
    if any("lora_" in key for key in keys):
        raise ValueError("unmerged LoRA tensors remain in the supplied checkpoint")
    with (source / "model.safetensors").open("rb") as stream:
        header_bytes = int.from_bytes(stream.read(8), "little")
    write_json(
        destination / "model.safetensors.index.json",
        {
            "metadata": {"total_size": identities["model.safetensors"]["bytes"] - header_bytes - 8},
            "weight_map": {key: "model.safetensors" for key in keys},
        },
    )
    with stage_progress("Inspect merged model architecture with empty weights (CPU only)"):
        inspection = inspection_dict(
            inspect_local_config(
                ModelConfig(
                    path=destination,
                    expected_name="Qwen3-ASR-1.7B",
                    dtype="bfloat16",
                    device="cuda:0",
                    local_files_only=True,
                )
            )
        )
    mismatches = {
        key: {"actual": inspection.get(key), "expected": value}
        for key, value in architecture.items()
        if inspection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"merged architecture differs from Qwen3-ASR-1.7B: {mismatches}")
    return {
        "source": str(source),
        "view": str(destination),
        "files": identities,
        "index_sha256": sha(destination / "model.safetensors.index.json"),
        "inspection": inspection,
    }


def copy_manifest(
    source: Path,
    target: Path,
    expected_sha: str,
    *,
    language: str,
    split: str,
    seen_audio: set[str],
    seen_ids: set[str],
    expected_count: int,
) -> dict[str, Any]:
    """Copy bytes exactly; validate labels and cross-language/split isolation without WAV I/O."""
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = frames = 0
    hours = 0.0
    with (
        source.open("rb") as src,
        target.open("xb") as dst,
        stage_progress(
            f"Freeze identical {language}/{split} manifest (no audio reads)", total=expected_count
        ) as tick,
    ):
        for line in src:
            if not line.strip():
                raise ValueError("blank manifest lines are not supported")
            row = json.loads(line)
            ids = row["phoneme_token_ids"]
            lower, upper = (2, 72) if language == "en" else (72, 167)
            audio_sha, sample_id = row["audio_sha256"], row["id"]
            if (
                row["language"] != language
                or row["split"] != split
                or row["experiment"] != "full-ctc-v1"
                or row["ctc_time_upsampling_factor"] != 2
                or not ids
                or any(type(token) is not int or not lower <= token < upper for token in ids)
                or row["label_length"] != len(ids)
                or row["ctc_minimum_input_length"] != ctc_minimum_input_length(ids)
                or not isinstance(audio_sha, str)
                or len(audio_sha) != 64
                or any(char not in "0123456789abcdef" for char in audio_sha)
                or audio_sha in seen_audio
                or sample_id in seen_ids
                or not Path(row["audio_path"]).is_absolute()
            ):
                raise ValueError(
                    f"invalid/overlapping frozen {language}/{split} sample: {sample_id}"
                )
            seconds = float(row["duration_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid frozen audio duration")
            seen_audio.add(audio_sha)
            seen_ids.add(sample_id)
            count += 1
            hours += seconds / 3600
            frames += qwen3_asr_audio_output_length(round(seconds * 16000) // 160)
            digest.update(line)
            dst.write(line)
            tick(count)
    if count != expected_count or digest.hexdigest() != expected_sha:
        raise ValueError(f"frozen manifest SHA256/count differs from sealed baseline: {source}")
    return {
        "source": str(source),
        "path": str(target),
        "sha256": expected_sha,
        "samples": count,
        "hours": hours,
        "estimated_encoder_frames": frames,
    }


def prepare(
    root: Path, config_path: Path, output: Path, *, legacy_vocab: Path, native_vocab: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    legacy, native = (json.loads(path.read_text()) for path in (legacy_vocab, native_vocab))
    if len(native["tokens"]) != 167 or legacy["tokens"][:72] != native["tokens"][:72]:
        raise ValueError("English token IDs changed in the native vocabulary")
    reports = {}
    # Require both manifests to be sealed before touching any GPU or rebuilding selection.
    for language, field, vocab in (
        ("en", "baseline_english", legacy_vocab),
        ("ar", "baseline_arabic", native_vocab),
    ):
        directory = root / config[field]
        path = directory / "label_report.json"
        if not path.is_file():
            raise ValueError(f"baseline {language} manifests are not sealed yet; wait for {path}")
        report = json.loads(path.read_text())
        if (
            report.get("status") != "labels_complete"
            or report.get("languages") != [language]
            or report.get("vocab_sha256") != sha(vocab)
            or report.get("train_target_hours_per_language") != config["train_hours_per_language"]
        ):
            raise ValueError(f"incompatible baseline frozen labels: {directory}")
        reports[language] = (directory, path, report)
    output.mkdir(parents=True, exist_ok=False)
    seen_audio: set[str] = set()
    seen_ids: set[str] = set()
    manifests = {}
    report_identities = {}
    for language, (directory, path, report) in reports.items():
        destination = output / "english" if language == "en" else output
        report_identities[language] = {"path": str(path), "sha256": sha(path)}
        for split in ("train", "validation"):
            manifests[f"{language}/{split}"] = copy_manifest(
                directory / f"{split}.jsonl",
                destination / f"{split}.jsonl",
                report[f"{split}_manifest_sha256"],
                language=language,
                split=split,
                seen_audio=seen_audio,
                seen_ids=seen_ids,
                expected_count=sum(
                    count
                    for key, count in report["rows_by_group"].items()
                    if key.startswith(split + "/")
                ),
            )
        if manifests[f"{language}/train"]["hours"] + 1e-9 < config["train_hours_per_language"]:
            raise ValueError(
                f"actual frozen {language} training hours are below the requested target"
            )
        if manifests[f"{language}/validation"]["hours"] + 1e-9 < sum(
            report["validation_target_hours"].values()
        ):
            raise ValueError(f"actual frozen {language} validation hours are insufficient")
    estimated = sum(item["estimated_encoder_frames"] for item in manifests.values()) * 1024 * 2
    minimum_free = math.ceil(estimated * 1.15) + 2 * 1024**3
    free = shutil.disk_usage(output).free
    if free < minimum_free:
        raise ValueError(
            f"insufficient new cache space: need {minimum_free / 1024**3:.1f} GiB, "
            f"free {free / 1024**3:.1f} GiB"
        )
    expected = yaml.safe_load((root / config["architecture_manifest"]).read_text())["architecture"]
    model = make_model_view(Path(config["model"]), output / "model/Qwen3-ASR-1.7B", expected)
    cache_config = yaml.safe_load((root / config["cache_config"]).read_text())
    cache_config["model"]["path"] = model["view"]
    cache_config["model"]["local_files_only"] = True
    cache_config["paths"]["output_dir"] = str(output)
    cache_config["paths"]["cache_dir"] = str(output / "cache")
    cache_path = output / "cache.workzone.yaml"
    cache_path.write_text(yaml.safe_dump(cache_config, sort_keys=False), encoding="utf-8")
    plan = {
        "status": "prepared",
        "config": {"path": str(config_path), "sha256": sha(config_path)},
        "model": model,
        "source_reports": report_identities,
        "manifests": manifests,
        "cache_config": {"path": str(cache_path), "sha256": sha(cache_path)},
        "vocabularies": {
            "en": {"path": str(legacy_vocab), "sha256": sha(legacy_vocab)},
            "ar": {"path": str(native_vocab), "sha256": sha(native_vocab)},
        },
        "estimated_feature_bytes": estimated,
        "minimum_cache_free_gib": minimum_free / 1024**3,
        "base_hidden_states_reused": False,
        "test_set_used": False,
        "code": [
            {"path": str(path), "sha256": sha(path)}
            for path in (
                Path(__file__).resolve(),
                root / "scripts/run_en_ar_merged.py",
                root / "scripts/run_en_ar_training.py",
            )
        ],
    }
    write_json(output / "plan.json", plan)
    return plan


def verify_plan(output: Path, *, hash_weights: bool = True) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    with stage_progress("Verify new-model run identity (no audio reads)"):
        for item in [
            plan["config"],
            plan["cache_config"],
            *plan["source_reports"].values(),
            *plan["vocabularies"].values(),
            *plan["code"],
        ]:
            pin(Path(item["path"]), item["sha256"])
        for item in plan["manifests"].values():
            pin(Path(item["source"]), item["sha256"])
            pin(Path(item["path"]), item["sha256"])
        view = Path(plan["model"]["view"])
        pin(view / "model.safetensors.index.json", plan["model"]["index_sha256"])
        for name, item in plan["model"]["files"].items():
            path = Path(item["path"])
            stat = path.stat()
            if (view / name).resolve(strict=True) != path:
                raise ValueError("new model link target changed")
            if (stat.st_size, stat.st_mtime_ns) != (item["bytes"], item["mtime_ns"]):
                raise ValueError(f"new model file changed: {path}")
            if hash_weights or name != "model.safetensors":
                pin(path, item["sha256"])
    return plan
