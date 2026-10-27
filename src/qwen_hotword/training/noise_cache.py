"""Cache only the 50% noise pool; compose two views over immutable original shards."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_hotword.config import load_workzone_config
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    ExperimentRecord,
    extract_frozen_features,
    load_experiment_records,
)
from qwen_hotword.training.english_newval import pin, save
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.noise_audio import mix_10db, read_audio
from qwen_hotword.training.noise_prepare import language, source_cache
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, load_disk_feature_cache
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def publish(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"existing artifact differs: {path}")
    else:
        save(path, value)


class NoiseExtractor:
    def __init__(self, manifest: Path, audit: Path) -> None:
        self.rows = {r["id"]: r for r in _rows(manifest)}
        self.loaded: OrderedDict[str, Any] = OrderedDict()
        self.audit_path = audit
        self.audit: list[dict[str, Any]] = []

    def waveform(self, record: ExperimentRecord) -> Any:
        import numpy as np

        noise = self.rows[record.sample_id]["noise10db"]
        path = noise["path"]
        if path not in self.loaded:
            pin(Path(path), {}, noise["sha256"])
            self.loaded[path] = read_audio(Path(path))
            if len(self.loaded) > 16:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(path)
        original_sha = _sha(record.audio_path)
        speech, background = read_audio(record.audio_path), self.loaded[path]
        fallback = False
        try:
            mixed, audit = mix_10db(speech, background, offset=noise["offset"])
        except ValueError as error:
            if "insufficient active-region energy" not in str(error):
                raise
            # A transient noise can have long silent regions. Deterministically align
            # its peak to an active original-speech frame; never drop the speech row.
            frame = int(
                np.argmax(
                    [
                        np.mean(speech[start : start + 320].astype(np.float64) ** 2)
                        for start in range(0, len(speech), 320)
                    ]
                )
            )
            offset = (int(np.argmax(np.abs(background))) - frame * 320) % len(background)
            mixed, audit = mix_10db(speech, background, offset=offset)
            fallback = True
        audit.update(planned_offset_16k=noise["offset"], silent_segment_realigned=fallback)
        self.audit.append({"id": record.sample_id, "original_audio_sha256": original_sha, **audit})
        return mixed

    def __call__(
        self, records: list[ExperimentRecord], wrapper: Any, **kwargs: Any
    ) -> tuple[list[CachedSample], int, float]:
        result = extract_frozen_features(records, wrapper, waveform_loader=self.waveform, **kwargs)
        publish(self.audit_path, self.audit)
        return result


def part_cache(
    output: Path, plan: dict[str, Any], part: dict[str, Any], *, verify: bool
) -> DiskFeatureCache:
    manifest = output / part["manifest"]
    pin(manifest, {}, part["sha256"])
    return load_disk_feature_cache(
        manifest.parent / "features",
        expected_split=part["split"],
        source_manifest_path=manifest,
        vocab_path=plan["vocab"],
        verify_sha256=verify,
    )


def build_noise_cache(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    marker = output / "cache_completed.json"
    if marker.exists():
        report: dict[str, Any] = json.loads(marker.read_text())
        for path, digest in report["sha256"].items():
            pin(Path(path), {}, digest)
        return report
    missing = [
        p
        for p in plan["parts"]
        if not (output / p["manifest"]).parent.joinpath("features/cache_summary.json").exists()
    ]
    wrapper = None
    if missing:
        from qwen_hotword.modeling.qwen_backbone import load_asr_model

        zone = load_workzone_config(plan["workzone_config"], require_existing_model=True)
        if (
            zone.model.path.resolve() != Path(plan["model"]["path"]).resolve()
            or zone.model.dtype != "bfloat16"
            or zone.model.device != "cuda:0"
            or not zone.model.local_files_only
        ):
            raise ValueError("noise Encoder configuration differs from the original frozen cache")
        hours = sum(plan["noisy_hours"]["50"].values()) + sum(
            value["hours"] for value in plan["validation"].values()
        )
        needed = int(hours * 3600 * 12.5 * 2048 * 1.35 + 2 * 1024**3)
        stored = sum(p.stat().st_size for p in (output / "noise").rglob("*.pt"))
        if shutil.disk_usage(output).free < max(0, needed - stored):
            raise ValueError(
                f"insufficient free space for noise features (~{needed / 1024**3:.1f} GiB)"
            )
        with stage_progress("Load local Qwen3-ASR-1.7B; frozen Encoder, no decoder generation"):
            wrapper = load_asr_model(zone.model)
    identities: dict[str, str] = {}
    mixing: dict[str, Any] = {
        "records": 0,
        "noise_looped": 0,
        "common_gain_applied": 0,
        "silent_segment_realigned": 0,
        "snr_db_min": 10.0,
        "snr_db_max": 10.0,
        "peak_max": 0.0,
    }
    with stage_progress("Noise feature parts", total=len(plan["parts"]), every=20) as tick:
        for n, part in enumerate(plan["parts"], 1):
            manifest = output / part["manifest"]
            pin(manifest, {}, part["sha256"])
            records = load_experiment_records(
                manifest,
                num_classes=90,
                expected_experiment="full-ctc-v1",
                expected_split=part["split"],
                prevalidated_audio_paths=True,
            )
            cache_feature_split(
                records,
                wrapper,
                manifest.parent / "features",
                split=part["split"],
                source_manifest_path=manifest,
                model_path=plan["model"]["path"],
                model_dtype=plan["model"]["dtype"],
                vocab_path=plan["vocab"],
                samples_per_shard=len(records),
                encoder_batch_size=plan["encoder_batch_size"],
                extractor=NoiseExtractor(manifest, manifest.parent / "mix_audit.json"),
            )
            part_cache(output, plan, part, verify=True)
            for audit in json.loads((manifest.parent / "mix_audit.json").read_text()):
                mixing["records"] += 1
                for key in ("noise_looped", "silent_segment_realigned"):
                    mixing[key] += int(audit[key])
                mixing["common_gain_applied"] += int(audit["common_gain"] < 1)
                mixing["snr_db_min"] = min(mixing["snr_db_min"], audit["snr_db"])
                mixing["snr_db_max"] = max(mixing["snr_db_max"], audit["snr_db"])
                mixing["peak_max"] = max(mixing["peak_max"], audit["peak"])
            for path in manifest.parent.rglob("*"):
                if path.is_file():
                    pin(path, identities)
            tick(n)
    del wrapper
    report = {
        "status": "noise_cache_completed",
        "parts": len(plan["parts"]),
        "sha256": identities,
        "test_set_used": False,
        "clean_encoder_recomputed": False,
        "materialized_noisy_wav_files": 0,
        "mixing": mixing,
    }
    save(marker, report)
    return report


def compose_views(
    output: Path, plan: dict[str, Any], ratio: int
) -> tuple[DiskFeatureCache, DiskFeatureCache, dict[str, str]]:
    if ratio not in {30, 50}:
        raise ValueError("only the frozen 30 and 50 percent views are supported")
    train, validation = (
        source_cache(plan, split, verify=True) for split in ("train", "validation")
    )
    original_train = json.loads((train.root / "cache_config.json").read_text())
    original_val = json.loads((validation.root / "cache_config.json").read_text())
    for key in ("model", "tap_module", "hidden_size", "feature_dtype", "vocab"):
        if original_train[key] != original_val[key]:
            raise ValueError(f"clean cache mismatch: {key}")
    train_shards = list(train.shards)
    val_shards = list(validation.shards)
    groups = {r["id"]: language(r) + "_original" for r in _rows(Path(plan["validation_manifest"]))}
    signature: list[Any] = [ratio, train.fingerprint, validation.fingerprint]
    validation_signature: list[Any] = [validation.fingerprint]
    extra_bytes = 0
    with stage_progress(
        "Verify shared noise feature parts", total=len(plan["parts"]), every=50
    ) as tick:
        for n, part in enumerate(plan["parts"], 1):
            cached = part_cache(output, plan, part, verify=True)
            metadata = json.loads((cached.root / "cache_config.json").read_text())
            if any(
                metadata[key] != original_train[key]
                for key in ("model", "tap_module", "hidden_size", "feature_dtype", "vocab")
            ):
                raise ValueError("noise features come from a different Encoder or vocabulary")
            if cached.encoder_frozen_parameters != train.encoder_frozen_parameters:
                raise ValueError("noise Encoder parameter count differs")
            shard = cached.shards[0]
            rows = list(_rows(output / part["manifest"]))
            if part["split"] == "train":
                ids = tuple(r["id"] for r in rows if ratio == 50 or r["selected_30"])
                if ids:
                    position = part["base_shard"]
                    train_shards[position] = replace(
                        train_shards[position], replacement=shard, replacement_ids=ids
                    )
                    extra_bytes += cached.feature_bytes
            else:
                ids = tuple(r["id"] for r in rows)
                val_shards.append(shard)
                groups.update({r["id"]: language(r) + "_noise10db" for r in rows})
                validation_signature.append((shard.feature_sha256, ids))
            signature.append((part["split"], shard.feature_sha256, ids))
            tick(n)
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    train = replace(
        train,
        root=output / f"view_{ratio}",
        shards=tuple(train_shards),
        feature_bytes=train.feature_bytes + extra_bytes,
        fingerprint=digest,
    )
    validation = replace(
        validation,
        root=output / "validation_original_and_noise10db",
        shards=tuple(val_shards),
        shard_count=len(val_shards),
        sample_count=sum(s.record_count for s in val_shards),
        feature_bytes=sum(s.feature_bytes for s in val_shards),
        fingerprint=hashlib.sha256(
            json.dumps(validation_signature, sort_keys=True).encode()
        ).hexdigest(),
    )
    if len(groups) != validation.sample_count:
        raise ValueError("validation view IDs or group coverage disagree")
    if set(groups) & {sample_id for shard in train.shards for sample_id in shard.sample_ids}:
        raise ValueError("noise validation IDs overlap training IDs")
    return train, validation, groups
