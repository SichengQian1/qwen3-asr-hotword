"""H200-only extraction and fixed-Head evaluation of a frozen density subset."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from qwen_hotword.training.pt_density_match import read_validation, write_checksums
from qwen_hotword.training.pt_mfa_pilot import sha256_file, write_json_new


def verify_selection(config: dict[str, Any], repo: Path) -> tuple[Path, list[dict[str, Any]]]:
    selection = repo / config["output_dir"]
    report = json.loads((selection / "selection_report.json").read_text())
    if report["status"] != "matched" or report["config"] != config:
        raise ValueError("Selection must pass frozen thresholds and use the identical config")
    if report["code_sha256"] != sha256_file(Path(__file__).with_name("pt_density_match.py")):
        raise ValueError("Selection code identity changed; use a new selection output")
    names = set()
    for line in (selection / "sha256.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name in names:
            raise ValueError("Duplicate checksum entry")
        names.add(name)
        if (
            name
            not in {
                "full_ctc_validation.jsonl",
                "selected_ids.txt",
                "pairs.json",
                "selection_report.json",
            }
            or sha256_file(selection / name) != digest
        ):
            raise ValueError("Selection artifact SHA mismatch")
    if names != {
        "full_ctc_validation.jsonl",
        "selected_ids.txt",
        "pairs.json",
        "selection_report.json",
    }:
        raise ValueError("Incomplete checksum list")
    rows = read_validation(
        selection / "full_ctc_validation.jsonl", report["selected_manifest_sha256"]
    )
    if [r["id"] for r in rows] != (selection / "selected_ids.txt").read_text().splitlines():
        raise ValueError("Selected ID list differs from manifest")
    return selection, rows


def check_actual_lengths(cache: Any, rows: list[dict[str, Any]], classes: int) -> None:
    from qwen_hotword.training.sharded_ctc import load_feature_shard

    expected = {r["id"]: r for r in rows}
    seen = set()
    for shard in cache.shards:
        for sample in load_feature_shard(shard, num_classes=classes):
            if sample.sample_id not in expected:
                continue
            row = expected[sample.sample_id]
            if (
                list(sample.token_ids) != row["phoneme_token_ids"]
                or int(sample.hidden_states.shape[0]) * 2 != row["effective_ctc_input_length"]
            ):
                raise ValueError(f"Actual feature length/labels differ: {sample.sample_id}")
            seen.add(sample.sample_id)
    if seen != set(expected):
        raise ValueError("Cache does not contain every required evaluation sample")


def evaluate_subset(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    import torch

    from qwen_hotword.config import load_workzone_config
    from qwen_hotword.modeling.qwen_backbone import load_asr_model
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab
    from qwen_hotword.training.ctc_diagnostics import diagnose_ctc_checkpoint
    from qwen_hotword.training.ctc_overfit import load_experiment_records
    from qwen_hotword.training.feature_cache import cache_feature_split
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

    if not torch.cuda.is_available():
        raise RuntimeError("Run on H200 with one GPU exposed; no local/CPU model fallback")
    selection, rows = verify_selection(config, repo)
    full = read_validation(repo / config["reference_manifest"], config["reference_sha256"])
    es = [r for r in full if r.get("balanced_language_bucket") == "es"]
    legacy = [r for r in full if r.get("balanced_language_bucket") == "pt"]
    checkpoints = {name: repo / path for name, path in config["checkpoints"].items()}
    identities = {}
    for name, path in checkpoints.items():
        digest = sha256_file(path)
        if config["checkpoint_sha256"].get(name, digest) != digest:
            raise ValueError(f"Checkpoint SHA mismatch: {name}")
        identities[name] = {"path": str(path), "sha256": digest}
    model_config = load_workzone_config(repo / config["model_config"], require_existing_model=True)
    if model_config.model.device != "cuda:0" or not model_config.model.local_files_only:
        raise ValueError("Require local Qwen model on logical cuda:0")
    reference_cache_root = repo / config["reference_cache"]
    cache_config = json.loads((reference_cache_root / "cache_config.json").read_text())
    for key, filename in (
        ("config_sha256", "config.json"),
        ("weight_index_sha256", "model.safetensors.index.json"),
    ):
        if cache_config["model"][key] != sha256_file(model_config.model.path / filename):
            raise ValueError("Qwen model identity differs from existing reference feature cache")
    if cache_config["model"]["dtype"] != model_config.model.dtype:
        raise ValueError("Encoder dtype differs from reference cache")
    vocab_path = repo / config["vocab"]
    vocab = load_phoneme_vocab(vocab_path)
    validate_heads(checkpoints, vocab)
    reference_cache = load_disk_feature_cache(
        reference_cache_root,
        expected_split="validation",
        source_manifest_path=repo / config["reference_manifest"],
        vocab_path=vocab_path,
        verify_sha256=True,
    )
    if reference_cache.ctc_time_upsampling_factor != 2:
        raise ValueError("Reference cache requires temporal 2x")
    check_actual_lengths(reference_cache, es + legacy, len(vocab.tokens))
    records = load_experiment_records(
        selection / "full_ctc_validation.jsonl",
        num_classes=len(vocab.tokens),
        expected_experiment="full-ctc-v1",
        expected_split="validation",
    )
    output = Path(tempfile.mkdtemp(prefix="evaluation_", dir=selection))
    print(f"Evaluation output: {output}", flush=True)
    write_json_new(
        output / "inputs.json",
        {
            "config": config,
            "checkpoints": identities,
            "selection_sha256": sha256_file(selection / "selection_report.json"),
            "code_sha256": sha256_file(Path(__file__)),
        },
    )
    print(
        "Caching only selected PT validation features; frozen Qwen encoder, no training", flush=True
    )
    wrapper = load_asr_model(model_config.model)
    summary = cache_feature_split(
        records,
        wrapper,
        output / "validation_cache",
        split="validation",
        source_manifest_path=selection / "full_ctc_validation.jsonl",
        model_path=model_config.model.path,
        model_dtype=model_config.model.dtype,
        vocab_path=vocab_path,
        encoder_batch_size=8,
        samples_per_shard=512,
    )
    del wrapper
    torch.cuda.empty_cache()
    cache = load_disk_feature_cache(
        output / "validation_cache",
        expected_split="validation",
        source_manifest_path=selection / "full_ctc_validation.jsonl",
        vocab_path=vocab_path,
        verify_sha256=True,
    )
    check_actual_lengths(cache, rows, len(vocab.tokens))
    reports = {}
    for name, checkpoint in checkpoints.items():
        print(f"Evaluating {name} on matched PT and legacy PT", flush=True)
        views: dict[str, Any] = {}
        for view, active_cache, active_rows in (
            ("matched_pt", cache, rows),
            ("legacy_pt", reference_cache, legacy),
            *([("reference_es", reference_cache, es)] if name == "multilingual_baseline" else []),
        ):
            result = diagnose_ctc_checkpoint(
                checkpoint,
                active_cache,
                vocab,
                device="cuda:0",
                batch_size=256,
                selected_sample_ids={r["id"] for r in active_rows},
                sample_groupings={
                    key: {r["id"]: r[key] for r in active_rows}
                    for key in ("source_corpus", "release_source")
                },
            )
            head_config = result["head_config"]
            if not isinstance(head_config, dict) or head_config.get("time_upsampling_factor") != 2:
                raise ValueError(f"Head must have temporal 2x: {name}")
            write_json_new(output / f"{name}_{view}.json", result)
            views[view] = {k: result[k] for k in ("validation", "validation_by_dimension")}
        reports[name] = views
    report = {
        "status": "completed",
        "checkpoints": identities,
        "results": reports,
        "selection_report_sha256": sha256_file(selection / "selection_report.json"),
        "new_cache": summary.to_dict(),
        "training_started": False,
        "external_asr_used": False,
        "test_used": False,
        "interpretation": "Density-controlled subset comparison, not label certification",
        "output_dir": str(output),
    }
    write_json_new(output / "report.json", report)
    write_checksums(output, sorted(p.name for p in output.glob("*.json")))
    return report


def validate_heads(checkpoints: dict[str, Path], vocab: Any) -> None:
    from qwen_hotword.modeling.ctc_head import build_ctc_head_from_checkpoint
    from qwen_hotword.training.ctc_diagnostics import _load_checkpoint

    for name, path in checkpoints.items():
        payload = _load_checkpoint(path, vocab)
        head = build_ctc_head_from_checkpoint(payload)
        head.load_state_dict(payload["state_dict"], strict=True)
        if head.time_upsampling_factor != 2:
            raise ValueError(f"Head must have temporal 2x before extraction: {name}")
