"""Freeze new English validation from the qualified pool minus actual training paths."""

from __future__ import annotations

import json
import os
import uuid
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.training.balanced_multilingual import _write_interleaved_manifest
from qwen_hotword.training.english_refresh_compat import verify_preparation_inputs
from qwen_hotword.training.english_refresh_fast import POLICY, path_candidates
from qwen_hotword.training.english_refresh_gpu import scan_manifest, verify_protected_paths
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache
from qwen_hotword.training.spanish_480_plan import _rows, metrics, stratified_select
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress

VERSION = "english-unused-qualified-validation-v1"


def save(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        # Publish a complete new file atomically; never replace an existing user artifact.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pin(path: Path, identities: dict[str, str], expected: str | None = None) -> None:
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"input identity mismatch: {path}")
    identities[str(path)] = actual


def verify(output: Path) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    checks: dict[str, Any] = {}
    _verified(output, "plan.json", checks)
    with stage_progress("Verify sealed new-validation inputs and manifests (no audio reads)"):
        for name, digest in plan["input_sha256"].items():
            pin(Path(name), {}, digest)
        for name, digest in plan["manifest_sha256"].items():
            pin(output / name, {}, digest)
    return plan


def cache_metadata(
    folder: Path, split: str, manifest: Path, vocab: Path, identities: dict[str, str]
) -> dict[str, Any]:
    # Tensor SHA is verified when reading copied shards and by the actual Head trainer.
    cache = load_disk_feature_cache(
        folder,
        expected_split=split,
        source_manifest_path=manifest,
        vocab_path=vocab,
        verify_sha256=False,
    )
    for name in ("cache_index.json", "cache_config.json", "cache_summary.json"):
        pin(folder / name, identities)
    config: dict[str, Any] = json.loads((folder / "cache_config.json").read_text())
    if cache.ctc_time_upsampling_factor != 2:
        raise ValueError("source cache must retain temporal 2x contract")
    return config


def prepare(root: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    with exclusive_feature_cache_run(output.parent / f".{output.name}.preparation"):
        return _prepare(root, cfg, output)


def _prepare(root: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    source = (root / cfg["source_run"]).resolve()
    if output.exists():
        raise FileExistsError("new-validation prepare requires a NEW directory")
    if cfg["epochs"] != 40 or cfg["training_overrides"] != {
        "minimum-learning-rate": 5e-6,
        "early-stopping-min-delta": 0.0005,
    }:
        raise ValueError("expected user-approved 40-epoch/5e-6/0.0005 configuration")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("new output must be separate from previous run")
    identities: dict[str, str] = {}
    with stage_progress("Verify completed source preparation and cache metadata"):
        verify_preparation_inputs(source)
        checked: dict[str, Any] = {}
        source_report = json.loads(_verified(source, "report.json", checked).read_text())
        contract = json.loads(_verified(source, "preparation_contract.json", checked).read_text())
        if any(output.is_relative_to(Path(name).parent) for name in contract["input_sha256"]):
            raise ValueError("new output must be outside sealed input directories")
        training_path = _verified(source, "training_config_path_only.json", checked)
        training = json.loads(training_path.read_text())
        if (
            source_report["status"] != "data_prepared"
            or training["audio_isolation_policy"] != POLICY
        ):
            raise ValueError("completed path-only preparation required")
        pin(training_path, identities, source_report["training_config_sha256"])
        for name, data in checked.items():
            identities[name] = data["sha256"]
        pin(source / "training/run_plan.json", identities, cfg["source_run_plan_sha256"])
        old_plan = json.loads((source / "training/run_plan.json").read_text())
        if old_plan["config"] != training:
            raise ValueError("source run plan/configuration mismatch")
        for name, digest in old_plan["input_sha256"].items():
            pin(Path(name), identities, digest)
        train = Path(training["freeze_dir"]) / "full_ctc_train.jsonl"
        legacy = Path(training["validation_manifest"])
        vocab = Path(training["vocab"])
        pin(train, identities, cfg["source_train_sha256"])
        pin(legacy, identities, cfg["source_validation_sha256"])
        pin(vocab, identities, training["vocab_sha256"])
        released = _verified(source, "frozen/released_english_pool.jsonl", checked)
        pin(released, identities, checked[str(released)]["sha256"])
        frozen = json.loads(_verified(source, "frozen/report.json", checked).read_text())
        pin(
            source / "frozen/report.json",
            identities,
            checked[str(source / "frozen/report.json")]["sha256"],
        )
        train_cache = source / "training/cache/train"
        old_cache = source / "training/cache/validation"
        a = cache_metadata(train_cache, "train", train, vocab, identities)
        b = cache_metadata(old_cache, "validation", legacy, vocab, identities)
        for key in ("model", "tap_module", "hidden_size", "feature_dtype", "vocab"):
            if a[key] != b[key]:
                raise ValueError(f"source train/validation cache mismatch: {key}")
        for filename, key in (
            ("config.json", "config_sha256"),
            ("model.safetensors.index.json", "weight_index_sha256"),
        ):
            pin(Path(a["model"]["path"]) / filename, identities, a["model"][key])
        if a["sample_count"] != sum(training["train_counts"].values()):
            raise ValueError("source train cache count mismatch")
        if b["sample_count"] != training["validation_count"]:
            raise ValueError("source validation cache count mismatch")

    train_stats, smoke_train, train_paths, train_ids = scan_manifest(
        train, "train", per_group=16, expected=sum(training["train_counts"].values())
    )
    _, _, old_paths, old_ids = scan_manifest(
        legacy, "validation", per_group=8, expected=training["validation_count"]
    )
    with stage_progress("Check old and new heldout protection paths"):
        protection = verify_protected_paths(source / "training", train_paths, old_paths, frozen, {})
        fingerprints = Path(protection["original_fingerprints"])
        tests = Path(protection["new_test_audio_paths"])
        for path in (fingerprints, tests):
            pin(path, identities, contract["input_sha256"][str(path)])
        forbidden = train_paths | old_paths
        forbidden.update(
            r["audio_path"] for r in _rows(fingerprints) if set(r["roles"]) & {"validation", "test"}
        )
        forbidden.update(r["audio_path"] for r in _rows(tests))
    with stage_progress("Select unused qualified English pool (metadata only)"):
        capacity_path = _verified(source, "frozen/capacity_report.json", checked)
        capacity = json.loads(capacity_path.read_text())
        pin(capacity_path, identities, checked[str(capacity_path)]["sha256"])
        pool, excluded = path_candidates(
            released,
            forbidden,
            contract["configuration"]["selection_seed"],
            capacity["released_records"],
        )
        expected_sources = {
            f"new_english_{country['name']}" for country in contract["configuration"]["countries"]
        }
        if {r["source"] for r in pool} != expected_sources:
            raise ValueError("unused qualified pool must have capacity in all three countries")
        forbidden_ids = train_ids | old_ids
        if any(r["id"] in forbidden_ids for r in pool):
            raise ValueError("unused English pool has a train/legacy-validation ID overlap")
        target_seconds = sum(
            r["duration_seconds"] for r in _rows(legacy) if r["balanced_language_bucket"] == "en"
        )
        available_seconds = sum(r["duration_seconds"] for r in pool)
        if available_seconds < target_seconds:
            raise ValueError(
                "insufficient unused qualified English capacity: "
                f"{available_seconds / 3600:.6f}h available; "
                f"{target_seconds / 3600:.6f}h required"
            )
        selected, quotas = stratified_select(pool, target_seconds, cfg["validation_seed"])
        if {r["source"] for r in selected} != {r["source"] for r in pool}:
            raise ValueError("validation target is too small to represent all country sources")
    output.mkdir(parents=True)
    selected_ids = {r["id"] for r in selected}
    english = []
    with stage_progress("Write frozen new English and retained ES/PT manifests"):
        for row in _rows(released):
            if row["id"] in selected_ids:
                english.append({**row, "split": "validation", "dataset_version": VERSION})
                selected_ids.remove(row["id"])
        if selected_ids:
            raise ValueError("selected English IDs are missing from qualified pool")
        english.sort(key=lambda r: r["id"])
        parts = {lang: output / f"validation_{lang}.jsonl" for lang in ("en", "es", "pt")}
        write_rows(parts["en"], english)
        retained = output / "legacy_es_pt_rows.jsonl"
        counts: Counter[str] = Counter()
        with retained.open("x", encoding="utf-8") as handle:
            for line in legacy.read_text().splitlines(keepends=True):
                row = json.loads(line)
                lang = row["balanced_language_bucket"]
                if lang in {"es", "pt"}:
                    handle.write(line)
                    counts[lang] += 1
        for lang in ("es", "pt"):
            write_rows(
                parts[lang],
                [
                    {**r, "dataset_version": VERSION}
                    for r in _rows(retained)
                    if r["balanced_language_bucket"] == lang
                ],
            )
        if set(counts) != {"es", "pt"}:
            raise ValueError("legacy ES/PT validation is empty")
        combined = output / "full_ctc_validation.jsonl"
        validation_summary = _write_interleaved_manifest(
            combined,
            parts,
            dataset_version=VERSION,
            expected_split="validation",
            forbidden_ids=train_ids,
            forbidden_audio_paths=train_paths,
        )
        validation_stats, smoke_val, _, _ = scan_manifest(
            combined, "validation", per_group=8, expected=int(validation_summary["records"])
        )
        write_rows(output / "smoke_train.jsonl", smoke_train)
        write_rows(output / "smoke_validation.jsonl", smoke_val)
    options = {**training["training_options"], **cfg["training_overrides"]}
    for path in sorted(Path(__file__).parent.glob("english_newval*.py")):
        pin(path, identities)
    pin(Path(__file__).parents[3] / "scripts/run_english_newval.py", identities)
    manifests = {p.name: _sha(p) for p in output.glob("*.jsonl")}
    plan = {
        "schema_version": 1,
        "configuration": cfg,
        "input_sha256": identities,
        "manifest_sha256": manifests,
        "train_manifest": str(train),
        "train_cache": str(train_cache),
        "legacy_validation_manifest": str(legacy),
        "legacy_validation_cache": str(old_cache),
        "vocab": str(vocab),
        "workzone_config": training["workzone_config"],
        "model": a["model"],
        "training_options": options,
        "encoder_batch_size": training["encoder_batch_size"],
        "samples_per_shard": training["samples_per_shard"],
        "epochs": cfg["epochs"],
        "audio_isolation_policy": POLICY,
        "test_set_used": False,
    }
    report = {
        "status": "new_validation_prepared",
        "output_dir": str(output),
        "validation_target_english_hours": target_seconds / 3600,
        "unused_qualified_english": metrics(pool),
        "selected_english": metrics(selected),
        "stratum_quotas": quotas,
        "excluded_pool_rows": excluded,
        "retained_es_pt_counts": dict(counts),
        "validation": validation_stats,
        "train": train_stats,
        "training_options": options,
        "epochs": cfg["epochs"],
        "train_validation_path_overlaps": 0,
        "train_validation_id_overlaps": 0,
        "selection_from": "qualified_pool_minus_actual_training_and_protected_paths",
        "audio_isolation_policy": POLICY,
        "audio_byte_audit_performed": False,
        "speaker_disjoint_verified": False,
        "model_loaded": False,
        "test_labels_used": False,
        "training_started": False,
        "train_cache_reuse": "original location; tensor SHA deferred to Head trainer",
    }
    with stage_progress("Recheck input identities and seal prepared manifests"):
        for name, digest in identities.items():
            pin(Path(name), {}, digest)
        save(output / "plan.json", plan)
        save(output / "report.json", report)
        seal(output)
        small_bundle(output / "prepare_bundle.zip", output, ["report.json", "plan.json"])
    report["return_file"] = str(output / "prepare_bundle.zip")
    return report


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def small_bundle(path: Path, root: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
        hashes = []
        for name in names:
            file = root / name
            if file.is_file():
                archive.write(file, name)
                hashes.append(f"{_sha(file)}  {name}\n")
        archive.writestr("sha256.txt", "".join(hashes))
