"""Read completed bilingual runs without enforcing their historical orchestration code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, load_disk_feature_cache
from qwen_hotword.training.stage_progress import stage_progress
from scripts.run_en_ar_training import combine_caches


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def protect_output(output: Path, directories: list[Path]) -> None:
    target = output.resolve()
    for directory in directories:
        original = directory.resolve()
        if target == original or original in target.parents:
            raise ValueError(f"output would modify a protected source directory: {original}")


def protected_run_directories(run: dict[str, Any]) -> list[Path]:
    return [Path(run["source"])] + [
        Path(item[key]).parent
        for item in run["plan"]["manifests"].values()
        for key in ("path", "source")
        if key in item
    ]


def rows(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def completed_run(source: Path) -> dict[str, Any]:
    source = source.resolve()
    plan = read(source / "plan.json")
    report = read(source / "report.json")
    head = read(source / "head/report.json")
    caches = read(source / "cache_sources.json")
    if report.get("status") != "completed" or head.get("status") != "completed":
        raise ValueError("source training must be completed")
    if report["head"] != head or report.get("test_set_used") is not False:
        raise ValueError("source training reports disagree or used test data")
    if report["merged_weight_sha256"] != caches["merged_weight_sha256"]:
        raise ValueError("source model identities disagree")
    if report["merged_weight_sha256"] != plan["model"]["files"]["model.safetensors"]["sha256"]:
        raise ValueError("source model differs from frozen plan")
    with stage_progress("Verify completed run manifests and vocabularies (no WAV reads)"):
        for key, item in plan["manifests"].items():
            if key not in {"en/train", "en/validation", "ar/train", "ar/validation"}:
                raise ValueError("unexpected source split")
            pin(Path(item["path"]), item["sha256"])
            if report["paired_manifest_sha256"][key] != item["sha256"]:
                raise ValueError("source manifest reports disagree")
        for item in plan["vocabularies"].values():
            pin(Path(item["path"]), item["sha256"])
    ar_vocab = load_phoneme_vocab(plan["vocabularies"]["ar"]["path"])
    en_vocab = load_phoneme_vocab(plan["vocabularies"]["en"]["path"])
    if ar_vocab.tokens[:72] != en_vocab.tokens[:72]:
        raise ValueError("English token IDs changed")
    if caches["target_vocab_sha256"] != plan["vocabularies"]["ar"]["sha256"]:
        raise ValueError("target vocabulary identity disagrees")
    return {
        "source": str(source),
        "plan": plan,
        "head": head,
        "caches": caches,
        "identity": {
            name: {"path": str(source / name), "sha256": sha(source / name)}
            for name in ("plan.json", "report.json", "head/report.json", "cache_sources.json")
        },
    }


def manifest_rows(run: dict[str, Any], split: str, languages: tuple[str, ...]) -> list[Any]:
    result = []
    for language in languages:
        item = run["plan"]["manifests"][f"{language}/{split}"]
        part = list(rows(Path(item["path"])))
        if len(part) != item["samples"]:
            raise ValueError("manifest count differs from plan")
        if any(r["split"] != split or r["language"] != language for r in part):
            raise ValueError("manifest split/language mismatch")
        result.extend(part)
    if len({r["id"] for r in result}) != len(result):
        raise ValueError("duplicate manifest ID")
    return result


def load_cache(run: dict[str, Any], split: str, output: Path) -> DiskFeatureCache:
    plan, sources = run["plan"], run["caches"]
    parts = []
    with stage_progress(f"Verify {split} cached feature SHA256 (no Encoder execution)"):
        for language in ("en", "ar"):
            manifest = Path(plan["manifests"][f"{language}/{split}"]["path"])
            cache = manifest.parent / "cache" / split
            config = read(cache / "cache_config.json")
            identity = {
                k: config[k] for k in ("model", "tap_module", "hidden_size", "feature_dtype")
            }
            if identity != sources["encoder_identity"]:
                raise ValueError("cached Encoder identity differs from source run")
            expected_model = {
                "path": plan["model"]["view"],
                "dtype": "bfloat16",
                "config_sha256": plan["model"]["files"]["config.json"]["sha256"],
                "weight_index_sha256": plan["model"]["index_sha256"],
            }
            if config["model"] != expected_model:
                raise ValueError("cache does not belong to the planned Encoder")
            parts.append(
                load_disk_feature_cache(
                    cache,
                    expected_split=split,
                    source_manifest_path=manifest,
                    vocab_path=plan["vocabularies"][language]["path"],
                    verify_sha256=True,
                )
            )
    combined = combine_caches(parts, output)
    if combined.fingerprint != sources[f"{split}_fingerprint"]:
        raise ValueError("combined feature fingerprint changed")
    return combined


def recheck(run: dict[str, Any]) -> None:
    for item in run["identity"].values():
        pin(Path(item["path"]), item["sha256"])
    for item in run["plan"]["manifests"].values():
        pin(Path(item["path"]), item["sha256"])


def training_options(head: dict[str, Any], factor: int) -> dict[str, Any]:
    if factor < 2:
        raise ValueError("upsampling factor must be >= 2 (frozen label feasibility floor)")
    if head["train_sampling_policy"] != "all_samples_once":
        raise ValueError("only the source all_samples_once policy is supported")
    config = head["head_config"]
    if config["head_type"] != "temporal_upsample":
        raise ValueError("source Head must use temporal_upsample")
    names = (
        "minimum_epochs",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_metric",
        "checkpoint_selection_metric",
        "train_batch_size",
        "weight_decay",
        "max_gradient_norm",
        "scheduler_patience",
        "scheduler_factor",
        "minimum_learning_rate",
        "seed",
    )
    return {
        **{name: head[name] for name in names},
        "epochs": head["epochs_requested"],
        "learning_rate": head["initial_learning_rate"],
        "head_type": config["head_type"],
        "head_hidden_dimension": config["hidden_dimension"],
        "head_kernel_size": config["kernel_size"],
        "head_dropout": config["dropout"],
        "head_time_upsampling_factor": factor,
    }
