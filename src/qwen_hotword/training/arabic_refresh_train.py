"""Materialize relabeled features from the same r15 Encoder; extract only missing audio."""

from __future__ import annotations

import gc
import json
import shutil
import subprocess
import time
import zipfile
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_hotword.config import load_workzone_config
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.arabic_per_diagnostics import decoded_summary
from qwen_hotword.training.arabic_refresh import read, verified_labels
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    ExperimentRecord,
    extract_frozen_features,
    load_experiment_records,
)
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    FeatureShardDescriptor,
    load_disk_feature_cache,
    load_feature_shard,
)
from qwen_hotword.training.stage_progress import stage_progress


class ReuseOrExtract:
    def __init__(self, caches: list[DiskFeatureCache], audio: dict[str, str], model: Any):
        self.index = {
            sample: shard
            for cache in caches
            for shard in cache.shards
            for sample in shard.sample_ids
        }
        self.audio = audio
        self.model = model
        self.wrapper: Any = None
        self.loaded: OrderedDict[Path, dict[str, CachedSample]] = OrderedDict()
        self.verified: set[Path] = set()
        self.reused = self.extracted = 0
        # Opt-in only for content-addressed manifests verified against the old cache.
        self.audio_hashes: dict[str, str] = {}
        self.alias_log: Path | None = None
        self.aliases_verified = 0

    def verify_alias(self, record: ExperimentRecord) -> None:
        old = self.audio[record.sample_id]
        expected = self.audio_hashes.get(record.sample_id, "")
        detail = f"sample={record.sample_id}; cached={old}; selected={record.audio_path}"
        if (
            len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or record.sample_id != "en-ar-v1-" + expected
            or self.alias_log is None
        ):
            raise ValueError("reused feature audio path identity changed; " + detail)
        before = record.audio_path.stat()
        actual = sha(record.audio_path)
        after = record.audio_path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in fields) or actual != expected:
            raise ValueError(
                f"reused feature audio bytes mismatch; {detail}; "
                f"expected={expected}; actual={actual}"
            )
        evidence = {
            "sample_id": record.sample_id,
            "cached_audio_path": old,
            "selected_audio_path": str(record.audio_path),
            "source_manifest_audio_sha256": expected,
            "selected_audio_sha256": actual,
            "status": "byte_verified_path_alias",
        }
        with self.alias_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evidence, ensure_ascii=False) + "\n")
        self.aliases_verified += 1
        print(f"Verified cached audio path alias: {detail}", flush=True)

    def __call__(
        self, batch: list[ExperimentRecord], wrapper: Any, **kwargs: Any
    ) -> tuple[list[CachedSample], int, float]:
        result: dict[str, CachedSample] = {}
        missing = []
        for record in batch:
            shard: FeatureShardDescriptor | None = self.index.get(record.sample_id)
            if shard is None:
                missing.append(record)
                continue
            if record.audio_path.resolve() != Path(self.audio[record.sample_id]).resolve():
                self.verify_alias(record)
            path = shard.feature_path
            if path not in self.loaded:
                if path not in self.verified:
                    pin(path, shard.feature_sha256)
                    self.verified.add(path)
                self.loaded[path] = {
                    r.sample_id: r for r in load_feature_shard(shard, num_classes=167)
                }
                if len(self.loaded) > 2:
                    self.loaded.popitem(last=False)
            self.loaded.move_to_end(path)
            sample = self.loaded[path][record.sample_id]
            if (
                sample.hidden_states.shape[0] * record.ctc_time_upsampling_factor
                < record.ctc_minimum_input_length
            ):
                raise ValueError("repaired targets exceed actual Encoder length")
            result[record.sample_id] = replace(sample, token_ids=record.token_ids)
            self.reused += 1
        seconds = 0.0
        if missing:
            if self.wrapper is None:
                with stage_progress("Load frozen r15 only for uncached audio"):
                    self.wrapper = load_asr_model(self.model)
            samples, parameters, seconds = extract_frozen_features(missing, self.wrapper, **kwargs)
            if parameters != 317477504:
                raise ValueError("unexpected frozen Encoder size")
            result.update({r.sample_id: r for r in samples})
            self.extracted += len(samples)
        return [result[r.sample_id] for r in batch], 317477504, seconds


def run(root: Path, output: Path, gpu: int, factor: int) -> dict[str, Any]:
    import torch

    from qwen_hotword.training.ctc_diagnostics import diagnose_ctc_checkpoint
    from qwen_hotword.training.sharded_ctc import exclusive_training_run, train_sharded_ctc_head
    from scripts.arabic_diagnostic_common import (
        completed_run,
        load_cache,
        recheck,
        training_options,
    )
    from scripts.run_en_ar_merged import wait_for_gpu
    from scripts.run_en_ar_training import combine_caches

    prepared = read(output / "prepare_report.json")
    if prepared["status"] != "prepared" or read(output / "contract.json") != prepared["contract"]:
        raise ValueError("preparation is incomplete")
    contract = prepared["contract"]
    for name, digest in contract["review_files"].items():
        pin(Path(name), digest)
    if contract["quality_exclusions_path"]:
        pin(Path(contract["quality_exclusions_path"]), contract["quality_exclusions_sha256"])
    pin(Path(contract["config_path"]), contract["config_sha256"])
    pin(root / "src/qwen_hotword/training/arabic_refresh.py", contract["code_sha256"])
    labels = Path(prepared["labels_dir"])
    pin(labels / "report.json", contract["labels_report_sha256"])
    verified_labels(labels)
    source = completed_run(Path(prepared["source_run"]))
    if source["identity"] != contract["source_identity"]:
        raise ValueError("source training identity changed")
    pin(Path(contract["old_checkpoint"]["path"]), contract["old_checkpoint"]["sha256"])
    for item in prepared["manifests"].values():
        pin(Path(item["path"]), item["sha256"])
    vocab_path = Path(source["plan"]["vocabularies"]["ar"]["path"])
    vocab = load_phoneme_vocab(vocab_path)
    options = training_options(source["head"], factor)
    training_contract = {
        "prepare_report_sha256": sha(output / "prepare_report.json"),
        "options": options,
        "code_sha256": sha(Path(__file__)),
        "trainer_sha256": sha(root / "src/qwen_hotword/training/sharded_ctc.py"),
        "support_code_sha256": {
            name: sha(root / name)
            for name in (
                "src/qwen_hotword/training/ctc_overfit.py",
                "src/qwen_hotword/training/feature_cache.py",
                "src/qwen_hotword/training/ctc_diagnostics.py",
                "src/qwen_hotword/modeling/qwen_backbone.py",
                "scripts/arabic_diagnostic_common.py",
                "scripts/run_en_ar_training.py",
                "scripts/run_en_ar_merged.py",
            )
        },
    }
    seal = output / "training_contract.json"
    if seal.exists():
        if read(seal) != training_contract:
            raise ValueError("training identity changed")
    else:
        write_json(seal, training_contract)
    with stage_progress("Verify r15 model identity before cache reuse or extraction"):
        model = source["plan"]["model"]
        for entry in model["files"].values():
            pin(Path(entry["path"]), entry["sha256"])
        pin(Path(model["view"]) / "model.safetensors.index.json", model["index_sha256"])
    # Queue behind another GPU workload rather than sharing the ×3 experiment's GPU.
    with stage_progress(f"Wait until physical GPU {gpu} has no compute processes"):
        while True:
            processes = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            if not processes.stdout.strip():
                break
            print(f"GPU {gpu} active PIDs: {processes.stdout.strip()}", flush=True)
            time.sleep(10)
    wait_for_gpu(gpu, prepared["config"]["minimum_gpu_free_gib"])
    old_caches = [load_cache(source, split, output) for split in ("train", "validation")]
    audio = {
        row["id"]: row["audio_path"]
        for item in source["plan"]["manifests"].values()
        for row in iter_jsonl(Path(item["path"]))
    }
    zone = load_workzone_config(source["plan"]["cache_config"]["path"], require_existing_model=True)
    if str(zone.model.path) != model["view"] or zone.model.dtype != "bfloat16":
        raise ValueError("runtime Encoder config differs from frozen r15")
    lookup = ReuseOrExtract(old_caches, audio, zone.model)
    available = shutil.disk_usage(output).free
    existing = sum(p.stat().st_size for p in (output / "cache").rglob("*.pt"))
    expected = sum(
        r["duration_seconds"] * 13 * 2048
        for key, item in prepared["manifests"].items()
        if key.startswith("ar/") and not key.endswith("test")
        for r in iter_jsonl(Path(item["path"]))
    )
    if available + existing < expected * 1.2 + 2 * 1024**3:
        raise ValueError("insufficient space for new Arabic feature shards")
    views, metadata = {}, {}
    for split in ("train", "validation"):
        manifest = output / f"{split}.jsonl"
        with stage_progress(f"Load new Arabic {split} labels") as tick:
            records = load_experiment_records(
                manifest,
                num_classes=167,
                expected_experiment="full-ctc-v1",
                expected_split=split,
                prevalidated_audio_paths=True,
                progress=tick,
            )
        with stage_progress(
            f"Reuse r15 features / extract only missing Arabic {split}; shard progress"
        ):
            cache_feature_split(
                records,
                None,
                output / "cache" / split,
                split=split,
                source_manifest_path=manifest,
                model_path=model["view"],
                model_dtype="bfloat16",
                vocab_path=vocab_path,
                encoder_batch_size=8,
                samples_per_shard=512,
                extractor=lookup,
            )
        ar_cache = load_disk_feature_cache(
            output / "cache" / split,
            expected_split=split,
            source_manifest_path=manifest,
            vocab_path=vocab_path,
        )
        en_manifest = Path(source["plan"]["manifests"][f"en/{split}"]["path"])
        en_cache = load_disk_feature_cache(
            en_manifest.parent / "cache" / split,
            expected_split=split,
            source_manifest_path=en_manifest,
            vocab_path=source["plan"]["vocabularies"]["en"]["path"],
        )
        views[split] = combine_caches([en_cache, ar_cache], output)
        metadata[split] = [*iter_jsonl(en_manifest), *iter_jsonl(manifest)]
    lookup.wrapper = None
    lookup.loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()
    write_json(
        output / "feature_reuse.json",
        {
            "newly_reused_samples_this_invocation": lookup.reused,
            "newly_extracted_samples_this_invocation": lookup.extracted,
            "english_original_cache_referenced": True,
            "arabic_targets_replaced_in_new_shards": True,
            "resumed_shards_not_in_invocation_counts": True,
        },
    )
    train_ids = {r["id"] for r in metadata["train"]}
    groups = {r["id"]: r["language"] for r in metadata["validation"]}
    if train_ids.intersection(groups) or set(groups.values()) != {"en", "ar"}:
        raise ValueError("train/validation overlap or language missing")
    head = output / f"head_up{factor}"
    with exclusive_training_run(head), stage_progress("Train repaired English/Arabic Head"):
        if not (head / "report.json").exists():
            train_sharded_ctc_head(
                views["train"],
                views["validation"],
                vocab,
                head,
                vocab_path=vocab_path,
                device="cuda:0",
                validation_sample_groups=groups,
                resume=(head / "training_state_latest.pt").exists(),
                **options,
            )
    with stage_progress("Evaluate repaired Head by language, dialect and source"):
        checkpoint = head / "ctc_head_best.pt"
        result: dict[str, Any] = diagnose_ctc_checkpoint(
            checkpoint,
            views["validation"],
            vocab,
            device="cuda:0",
            batch_size=256,
            sample_groups=groups,
            sample_groupings={
                field: {r["id"]: str(r[field]) for r in metadata["validation"]}
                for field in ("language", "source_group", "source")
            },
            include_sample_metrics=True,
        )
        durations = {r["id"]: r["duration_seconds"] for r in metadata["validation"]}
        for row in result["sample_metrics"]:
            row["duration_seconds"] = durations[row["sample_id"]]
        result["sample_breakdown"] = decoded_summary(result["sample_metrics"])
        write_json(output / "validation_diagnostics.json", result)
    paired_manifest = output / "paired_ar_validation.jsonl"
    paired_metadata = list(iter_jsonl(paired_manifest))
    if not paired_metadata:
        raise ValueError("no common repaired reference subset for old/new Head comparison")
    with stage_progress("Build shared-reference validation cache; reuse old r15 states only"):
        records = load_experiment_records(
            paired_manifest,
            num_classes=167,
            expected_experiment="full-ctc-v1",
            expected_split="validation",
            prevalidated_audio_paths=True,
        )
        if any(record.sample_id not in lookup.index for record in records):
            raise ValueError("paired validation requires original cached audio")
        cache_feature_split(
            records,
            None,
            output / "cache/paired_validation",
            split="validation",
            source_manifest_path=paired_manifest,
            model_path=model["view"],
            model_dtype="bfloat16",
            vocab_path=vocab_path,
            encoder_batch_size=8,
            samples_per_shard=512,
            extractor=lookup,
        )
        paired_cache = load_disk_feature_cache(
            output / "cache/paired_validation",
            expected_split="validation",
            source_manifest_path=paired_manifest,
            vocab_path=vocab_path,
        )
    paired: dict[str, Any] = {
        "reference_policy": "identical_repaired_automatic_labels_for_both_heads",
        "samples": len(paired_metadata),
        "reference_sha256": sha(paired_manifest),
        "excluded_old_validation": prepared["manifests"]["ar/paired_validation"][
            "missing_or_review_samples"
        ],
        "human_gold": False,
        "old_checkpoint_sha256": contract["old_checkpoint"]["sha256"],
        "new_checkpoint_sha256": sha(checkpoint),
        "split_isolation": "new_head_recording_disjoint_old_head_legacy_split",
        "new_head_excludes_old_validation_recordings_from_training": True,
        "old_head_original_recording_isolation_verified": False,
    }
    with stage_progress("Compare old/new Heads on same audio and repaired reference"):
        for name, checkpoint_path in (
            ("old", source["head"]["best_checkpoint_path"]),
            ("new", str(checkpoint)),
        ):
            paired[name] = diagnose_ctc_checkpoint(
                checkpoint_path,
                paired_cache,
                vocab,
                device="cuda:0",
                batch_size=256,
                sample_groups={r["id"]: r["source_group"] for r in paired_metadata},
                sample_groupings={
                    "source_group": {r["id"]: r["source_group"] for r in paired_metadata}
                },
                include_sample_metrics=True,
            )
            paired[name]["sample_breakdown"] = decoded_summary(paired[name].pop("sample_metrics"))
    write_json(output / "paired_validation.json", paired)
    recheck(source)
    pin(Path(contract["old_checkpoint"]["path"]), contract["old_checkpoint"]["sha256"])
    report = {
        "status": "completed",
        "head": read(head / "report.json"),
        "checkpoint_sha256": sha(checkpoint),
        "vocab_sha256": sha(vocab_path),
        "prepare_report_sha256": training_contract["prepare_report_sha256"],
        "human_pronunciation_review_completed": False,
        "test_set_used": False,
        "old_new_PER_directly_comparable": False,
        "paired_reference_comparison": str(output / "paired_validation.json"),
        "validation_diagnostics": str(output / "validation_diagnostics.json"),
    }
    write_json(output / "report.json", report)
    with zipfile.ZipFile(output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "report.json",
            "prepare_report.json",
            "feature_reuse.json",
            "paired_validation.json",
        ):
            archive.write(output / name, name)
        archive.writestr(
            "validation_summary.json",
            json.dumps(
                {k: v for k, v in result.items() if k != "sample_metrics"}, ensure_ascii=False
            ),
        )
        archive.write(labels / "report.json", "labels_report.json")
        archive.write(labels / "review_delivery/summary.json", "pronunciation_review_summary.json")
    return report
