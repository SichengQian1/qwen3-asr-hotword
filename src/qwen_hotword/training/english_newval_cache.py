"""Extract only new English validation; copy existing BF16 states for ES/PT and smoke."""

from __future__ import annotations

import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from qwen_hotword.config import load_workzone_config
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    ExperimentRecord,
    load_experiment_records,
)
from qwen_hotword.training.english_newval import pin, save
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    FeatureShardDescriptor,
    load_disk_feature_cache,
    load_feature_shard,
)
from qwen_hotword.training.stage_progress import stage_progress


def records(path: Path, split: str) -> list[ExperimentRecord]:
    with stage_progress(f"Load {path.name} labels (metadata only)") as progress:
        return load_experiment_records(
            path,
            num_classes=90,
            expected_experiment="full-ctc-v1",
            expected_split=split,
            prevalidated_audio_paths=True,
            progress=progress,
        )


def open_cache(folder: Path, split: str, manifest: Path, vocab: Path) -> DiskFeatureCache:
    return load_disk_feature_cache(
        folder,
        expected_split=split,
        source_manifest_path=manifest,
        vocab_path=vocab,
        verify_sha256=False,
    )


class FeatureLookup:
    """Read only requested source shards; check tensor SHA before first use, bounded RAM."""

    def __init__(self, caches: list[DiskFeatureCache]) -> None:
        self.index: dict[str, FeatureShardDescriptor] = {}
        self.loaded: OrderedDict[Path, dict[str, CachedSample]] = OrderedDict()
        self.verified: dict[str, str] = {}
        parameters = {c.encoder_frozen_parameters for c in caches}
        if len(parameters) != 1:
            raise ValueError("source caches have different frozen Encoder sizes")
        self.parameters = parameters.pop()
        for cache in caches:
            for shard in cache.shards:
                for sample_id in shard.sample_ids:
                    if sample_id in self.index:
                        raise ValueError("feature sources contain overlapping sample IDs")
                    self.index[sample_id] = shard

    def __call__(
        self, batch: list[ExperimentRecord], wrapper: Any, **kwargs: Any
    ) -> tuple[list[CachedSample], int, float]:
        result = []
        for record in batch:
            shard = self.index[record.sample_id]
            path = shard.feature_path
            if path not in self.loaded:
                if str(path) not in self.verified:
                    pin(path, self.verified, shard.feature_sha256)
                self.loaded[path] = {
                    sample.sample_id: sample for sample in load_feature_shard(shard, num_classes=90)
                }
                if len(self.loaded) > 4:
                    self.loaded.popitem(last=False)
            self.loaded.move_to_end(path)
            sample = self.loaded[path][record.sample_id]
            if sample.token_ids != record.token_ids:
                raise ValueError(f"copied feature labels differ: {record.sample_id}")
            result.append(sample)
        return result, self.parameters, 0.0


def materialize(
    manifest: Path, destination: Path, split: str, plan: dict[str, Any], lookup: FeatureLookup
) -> dict[str, Any]:
    with stage_progress(f"Copy/reuse {destination.name} features; no Encoder (see shard counts)"):
        summary = cache_feature_split(
            records(manifest, split),
            None,
            destination,
            split=split,
            source_manifest_path=manifest,
            model_path=plan["model"]["path"],
            model_dtype=plan["model"]["dtype"],
            vocab_path=plan["vocab"],
            encoder_batch_size=plan["encoder_batch_size"],
            samples_per_shard=plan["samples_per_shard"],
            extractor=lookup,
        ).to_dict()
    return summary


def verify_cache_seal(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text())
    with stage_progress(f"Verify cached feature SHA256 ({path.name})"):
        for name, digest in payload["sha256"].items():
            pin(Path(name), {}, digest)
    return payload


def publish_cache_seal(path: Path, folders: list[Path], payload: dict[str, Any]) -> dict[str, Any]:
    identities: dict[str, str] = {}
    # Only new/small caches are sealed here. Full original train cache is checked by Head trainer.
    with stage_progress("Seal materialized cache metadata and tensors"):
        for folder in folders:
            for name in sorted(folder.rglob("*")):
                if name.is_file():
                    pin(name, identities)
        payload["sha256"] = identities
        save(path, payload)
    return payload


def build_cache(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    marker = output / "cache_report.json"
    if marker.exists():
        return verify_cache_seal(marker)
    english_manifest = output / "validation_en.jsonl"
    english_cache = output / "cache/english"
    english_records = records(english_manifest, "validation")
    source_model = plan["model"]
    with stage_progress("Verify Qwen metadata and space for new English validation cache"):
        zone = load_workzone_config(plan["workzone_config"], require_existing_model=True)
        if (
            zone.model.path.resolve() != Path(source_model["path"]).resolve()
            or zone.model.dtype != source_model["dtype"]
            or zone.model.device != "cuda:0"
            or not zone.model.local_files_only
        ):
            raise ValueError("new validation Encoder model differs from existing cache")
        estimates = sum(
            int(r["estimated_ctc_input_length"]) for r in _manifest_rows(english_manifest)
        )
        # Reserve English plus composed validation and small smoke copies; train is referenced.
        old_bytes = json.loads(
            (Path(plan["legacy_validation_cache"]) / "cache_summary.json").read_text()
        )["feature_bytes"]
        needed = int(estimates * 2048 * 2.5 + old_bytes * 1.5 + 2 * 1024**3)
        stored = sum(p.stat().st_size for p in (output / "cache").rglob("*") if p.is_file())
        if shutil.disk_usage(output).free < max(0, needed - stored):
            raise ValueError(f"insufficient new cache space; need ~{needed / 1024**3:.1f} GiB")
    if (english_cache / "cache_summary.json").exists():
        with stage_progress("Verify completed English validation cache; no model load"):
            load_disk_feature_cache(
                english_cache,
                expected_split="validation",
                source_manifest_path=english_manifest,
                vocab_path=plan["vocab"],
                verify_sha256=True,
            )
    else:
        with stage_progress("Load frozen Qwen for NEW English validation only"):
            wrapper = load_asr_model(zone.model)
        with stage_progress("Extract NEW English validation only (see shard counts)"):
            cache_feature_split(
                english_records,
                wrapper,
                english_cache,
                split="validation",
                source_manifest_path=english_manifest,
                model_path=zone.model.path,
                model_dtype=zone.model.dtype,
                vocab_path=plan["vocab"],
                encoder_batch_size=plan["encoder_batch_size"],
                samples_per_shard=plan["samples_per_shard"],
            )
        del wrapper
        import torch

        torch.cuda.empty_cache()
    with stage_progress("Read old ES/PT and new English cache indices"):
        lookup = FeatureLookup(
            [
                open_cache(
                    Path(plan["legacy_validation_cache"]),
                    "validation",
                    Path(plan["legacy_validation_manifest"]),
                    Path(plan["vocab"]),
                ),
                open_cache(english_cache, "validation", english_manifest, Path(plan["vocab"])),
            ]
        )
    validation = materialize(
        output / "full_ctc_validation.jsonl",
        output / "cache/validation",
        "validation",
        plan,
        lookup,
    )
    return publish_cache_seal(
        marker,
        [english_cache, output / "cache/validation"],
        {
            "status": "new_validation_cache_completed",
            "validation": validation,
            "new_english_samples": len(english_records),
            "train_cache_copied": False,
            "es_pt_encoder_recomputed": False,
            "reused_source_tensor_sha256": lookup.verified,
            "test_set_used": False,
        },
    )


def build_smoke_cache(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    marker = output / "smoke_cache_report.json"
    if marker.exists():
        return verify_cache_seal(marker)
    summaries = {}
    sources = {}
    for split, folder, manifest in (
        ("train", Path(plan["train_cache"]), Path(plan["train_manifest"])),
        ("validation", output / "cache/validation", output / "full_ctc_validation.jsonl"),
    ):
        with stage_progress(f"Read {split} cache metadata for small smoke subset"):
            lookup = FeatureLookup([open_cache(folder, split, manifest, Path(plan["vocab"]))])
        summaries[split] = materialize(
            output / f"smoke_{split}.jsonl", output / "smoke_cache" / split, split, plan, lookup
        )
        sources.update(lookup.verified)
    return publish_cache_seal(
        marker,
        [output / "smoke_cache"],
        {
            "status": "smoke_cache_completed",
            "splits": summaries,
            "reused_source_tensor_sha256": sources,
            "encoder_loaded": False,
            "test_set_used": False,
        },
    )


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
