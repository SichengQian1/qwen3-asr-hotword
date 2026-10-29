"""Stream one mixed shard at a time; reuse hidden states but replace ALL labels."""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.phonemes.mfa_v03_review import rows
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    ExperimentRecord,
    load_experiment_records,
)
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.mfa_round_common import digest, old_noise_part, read, sources
from qwen_hotword.training.noise_cache import NoiseExtractor
from qwen_hotword.training.noise_prepare import language
from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    load_disk_feature_cache,
    load_feature_shard,
)
from qwen_hotword.training.stage_progress import stage_progress


class RelabeledExtractor:
    def __init__(
        self,
        root: Path,
        cfg: dict[str, Any],
        old: dict[str, Any],
        source: Any,
        part: dict[str, Any],
        manifest: Path,
    ) -> None:
        self.root, self.cfg, self.old, self.source = root, cfg, old, source
        self.part, self.manifest = part, manifest
        self.data = {r["id"]: r for r in rows(manifest)}
        self.prior_rows = {}
        if part["old_part"] is not None:
            path = root / cfg["noise_source"] / part["old_part"]["manifest"]
            pin(path, part["old_part"]["sha256"])
            self.prior_rows = {r["id"]: r for r in rows(path)}

    def reusable(self, row: dict[str, Any]) -> bool:
        prior = self.prior_rows.get(row["id"])
        return bool(
            row["noise_selected"]
            and prior is not None
            and prior["noise10db"] == row["noise10db"]
            and all(prior[k] == row[k] for k in ("audio_path", "text", "language"))
        )

    def needs_encoder(self) -> bool:
        return any(r["noise_selected"] and not self.reusable(r) for r in self.data.values())

    def __call__(
        self, records: list[ExperimentRecord], wrapper: Any, **kwargs: Any
    ) -> tuple[list[CachedSample], int, float]:
        started = time.monotonic()
        pin(self.source.feature_path, self.source.feature_sha256)
        original = {r.sample_id: r for r in load_feature_shard(self.source, num_classes=90)}
        previous: dict[str, CachedSample] = {}
        if any(self.reusable(self.data[r.sample_id]) for r in records):
            cached = old_noise_part(self.root, self.cfg, self.old, self.part["old_part"])
            metadata = read(cached.root / "cache_config.json")
            base_metadata = read(
                Path(self.old[f"{self.part['split']}_cache"]) / "cache_config.json"
            )
            if any(
                metadata[k] != base_metadata[k]
                for k in ("model", "tap_module", "hidden_size", "feature_dtype", "vocab")
            ):
                raise ValueError("old noisy features came from a different Encoder")
            shard = cached.shards[0]
            pin(shard.feature_path, shard.feature_sha256)
            previous = {r.sample_id: r for r in load_feature_shard(shard, num_classes=90)}
        missing = [
            r
            for r in records
            if self.data[r.sample_id]["noise_selected"]
            and not self.reusable(self.data[r.sample_id])
        ]
        fresh = {}
        mix_audit = self.manifest.parent / "new_mix_audit.json"
        if missing:
            if wrapper is None:
                raise ValueError("uncached selected noise needs the frozen work-zone Encoder")
            extractor = NoiseExtractor(self.manifest, mix_audit)
            values, parameters, _ = extractor(missing, wrapper, **kwargs)
            if parameters != 317477504:
                raise ValueError("unexpected frozen Encoder parameter count")
            fresh = {r.sample_id: r for r in values}
        else:
            write_json(mix_audit, [])
        result, audit = [], []
        for record in records:
            row = self.data[record.sample_id]
            base = original[row["base_sample_id"]]
            if digest(list(base.token_ids)) != row["legacy_label_sha256"]:
                raise ValueError("source cached labels differ from the pinned old manifest")
            if not row["noise_selected"]:
                value, mode = base, "original_hidden_reused"
            elif self.reusable(row):
                value, mode = previous[record.sample_id], "identical_old_noise_hidden_reused"
                if value.token_ids != base.token_ids:
                    raise ValueError("old noisy labels do not match their original sample")
            else:
                value, mode = fresh[record.sample_id], "new_noise_hidden_extracted"
            frames = int(value.hidden_states.shape[0])
            if abs(frames - int(base.hidden_states.shape[0])) > 1:
                raise ValueError(f"unexpected Encoder length change for {record.sample_id}")
            if record.ctc_minimum_input_length > frames * record.ctc_time_upsampling_factor:
                write_json(
                    self.manifest.parent / "infeasible.json",
                    {
                        "sample_id": record.sample_id,
                        "actual_frames": frames,
                        "minimum_target_frames": record.ctc_minimum_input_length,
                        "noise_selected": row["noise_selected"],
                    },
                )
                raise ValueError(
                    f"new labels exceed actual mixed-audio CTC frames: {record.sample_id}"
                )
            result.append(CachedSample(record.sample_id, value.hidden_states, record.token_ids))
            audit.append(
                {
                    "id": record.sample_id,
                    "backend": mode,
                    "frames": frames,
                    "original_frames": int(base.hidden_states.shape[0]),
                    "label_sha256": digest(list(record.token_ids)),
                }
            )
        write_json(self.manifest.parent / "reuse_audit.json", audit)
        return result, 317477504, time.monotonic() - started


def compose(
    output: Path, cfg: dict[str, Any], prepared: dict[str, Any]
) -> tuple[DiskFeatureCache, DiskFeatureCache, dict[str, str]]:
    groups: dict[str, str] = {}
    all_caches: dict[str, list[DiskFeatureCache]] = {"train": [], "validation": []}
    with stage_progress(
        "Verify new-label mixed cache parts", total=len(prepared["parts"]), every=25
    ) as tick:
        for number, part in enumerate(prepared["parts"], 1):
            manifest = output / part["manifest"]
            pin(manifest, part["sha256"])
            cached = load_disk_feature_cache(
                manifest.parent / "features",
                expected_split=part["split"],
                source_manifest_path=manifest,
                vocab_path=cfg["absolute_vocab"],
                verify_sha256=True,
            )
            all_caches[part["split"]].append(cached)
            if part["split"] == "validation":
                for row in rows(manifest):
                    if row["id"] in groups:
                        raise ValueError("duplicate validation view ID")
                    groups[row["id"]] = language(row) + (
                        "_noise10db" if row["noise_selected"] else "_original"
                    )
            tick(number)
    views = {}
    for split, caches in all_caches.items():
        shards = tuple(s for c in caches for s in c.shards)
        identifiers = [i for s in shards for i in s.sample_ids]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate new-cache IDs")
        views[split] = DiskFeatureCache(
            split=split,
            root=output / "cache_views" / split,
            sample_count=len(identifiers),
            shard_count=len(shards),
            feature_bytes=sum(s.feature_bytes for s in shards),
            encoder_frozen_parameters=317477504,
            fingerprint=digest([c.fingerprint for c in caches]),
            sha256_verified=True,
            ctc_time_upsampling_factor=2,
            shards=shards,
        )
    if set(groups) & {i for s in views["train"].shards for i in s.sample_ids}:
        raise ValueError("new training and validation IDs overlap")
    if set(groups.values()) != {
        lang + condition for lang in ("en", "es", "pt") for condition in ("_original", "_noise10db")
    }:
        raise ValueError("six validation groups required")
    return views["train"], views["validation"], groups


def build(
    root: Path, cfg: dict[str, Any], output: Path, prepared: dict[str, Any]
) -> dict[str, Any]:
    marker = output / "cache_completed.json"
    if marker.exists():
        report = read(marker)
        for name, checksum in report["files"].items():
            pin(output / name, checksum)
        return report
    old, originals = sources(root, cfg)
    wrapper = None
    # The new cache stores only ONE training input per record plus paired validation.
    estimated = sum(c.feature_bytes for c in originals.values()) * 1.15 + 2 * 1024**3
    stored = sum(p.stat().st_size for p in (output / "parts").rglob("*.pt"))
    if shutil.disk_usage(output).free < max(0, estimated - stored):
        raise ValueError(f"reserve ~{estimated / 1024**3:.1f} GiB for the separate new-label cache")
    counts: dict[str, int] = {}
    files = {}
    try:
        with stage_progress(
            "Build new-label 70% noise training features", total=len(prepared["parts"]), every=10
        ) as tick:
            for index, part in enumerate(prepared["parts"], 1):
                manifest = output / part["manifest"]
                pin(manifest, part["sha256"])
                records = load_experiment_records(
                    manifest,
                    num_classes=100,
                    expected_experiment="full-ctc-v1",
                    expected_split=part["split"],
                    prevalidated_audio_paths=True,
                )
                extractor = RelabeledExtractor(
                    root,
                    cfg,
                    old,
                    originals[part["split"]].shards[part["base_shard"]],
                    part,
                    manifest,
                )
                if (
                    not (manifest.parent / "features/cache_summary.json").exists()
                    and extractor.needs_encoder()
                    and wrapper is None
                ):
                    from qwen_hotword.config import load_workzone_config
                    from qwen_hotword.modeling.qwen_backbone import load_asr_model

                    zone = load_workzone_config(old["workzone_config"], require_existing_model=True)
                    if (
                        zone.model.path.resolve() != Path(old["model"]["path"]).resolve()
                        or zone.model.dtype != "bfloat16"
                        or zone.model.device != "cuda:0"
                        or not zone.model.local_files_only
                    ):
                        raise ValueError("work-zone model differs from inherited frozen Encoder")
                    with stage_progress(
                        "Load local Qwen3-ASR-1.7B frozen Encoder on selected H200"
                    ):
                        wrapper = load_asr_model(zone.model)
                cache_feature_split(
                    records,
                    wrapper,
                    manifest.parent / "features",
                    split=part["split"],
                    source_manifest_path=manifest,
                    model_path=old["model"]["path"],
                    model_dtype=old["model"]["dtype"],
                    vocab_path=root / cfg["vocab"],
                    encoder_batch_size=old["encoder_batch_size"],
                    samples_per_shard=len(records),
                    extractor=extractor,
                )
                audit = json.loads((manifest.parent / "reuse_audit.json").read_text())
                for row in audit:
                    counts[row["backend"]] = counts.get(row["backend"], 0) + 1
                for path in manifest.parent.rglob("*"):
                    if path.is_file() and not path.name.endswith(".tmp"):
                        files[str(path.relative_to(output))] = sha(path)
                tick(index)
    finally:
        wrapper = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "status": "new_100_class_caches_completed",
        "files": files,
        "backends": counts,
        "old_labels_reused": False,
        "encoder_retrained": False,
        "test_set_used_for_training": False,
    }
    write_json(marker, report)
    return report
