"""Full Chinese/US/UK preparation, 70% noisy frozen features, random Head and PER."""

from __future__ import annotations

import gc
import json
import shutil
import traceback
import zipfile
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import cache_feature_split, exclusive_feature_cache_run
from qwen_hotword.training.noise_audio import audit_noise, mix_10db, rank, read_audio
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import contract, digest, read, rows
from qwen_hotword.training.zh_en_data import (
    audio_bytes,
    inventory,
    manifests,
    split_inventory,
    verify_source_files,
    waveform,
)
from qwen_hotword.training.zh_en_foundation import verify_assets
from qwen_hotword.training.zh_en_labels import (
    generate_unique,
    install_frontend,
    label_part,
    make_worker,
    text_parts,
)


def prepare(
    root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]
) -> dict[str, Any]:
    verify_assets(root, foundation)
    receipt = output / "prepare_completed.json"
    if receipt.exists():
        verify_source_files(output, foundation)
        saved = read(receipt)
        for name, expected in saved["files"].items():
            pin(output / name, expected)
        # Validate all label and selected-manifest hashes; never regenerate completed G2P.
        done = manifests(output, sorted((output / "labels").glob("*/rows.jsonl")), cfg)
        if read(output / "prepared.json") != done:
            raise ValueError("prepared manifest summary changed")
        return done
    install_frontend(output, cfg["frontend_packages"])
    # API/environment failures are caught before the full audio audit, without a pilot experiment.
    worker, assets, prefix = make_worker(root, output, foundation, cfg)
    sources = inventory(output, foundation, cfg)
    separated = split_inventory(output, sources, cfg["seed"])
    with stage_progress(
        "Prepare Chinese context readings / English words", total=len(separated), every=1
    ) as tick:
        texts = text_parts(output, separated, tick)
    pronunciations = generate_unique(
        root, output, texts, worker, assets, prefix, cfg["g2p_workers"]
    )
    labeled = []
    with stage_progress(
        "Assemble full 183-class labels and check CTC lengths", total=len(texts), every=1
    ) as tick:
        for i, source in enumerate(texts):
            labeled.append(
                label_part(output, source, i, pronunciations, cfg["head"]["time_upsampling_factor"])
            )
            tick(i + 1)
    report = manifests(output, labeled, cfg)
    write_json(output / "prepared.json", report)
    write_json(
        receipt,
        {
            "files": {
                name: sha(output / name)
                for name in (
                    "prepared.json",
                    "label_report.json",
                    "g2p_report.json",
                    "source_identity.json",
                )
            }
        },
    )
    return report


class MixedExtractor:
    def __init__(
        self, items: list[dict[str, Any]], noises: dict[str, list[dict[str, Any]]], seed: int
    ) -> None:
        self.items = {r["id"]: r for r in items}
        self.noises = noises
        self.seed = seed
        self.loaded: OrderedDict[str, Any] = OrderedDict()
        self.audit: list[dict[str, Any]] = []

    def load(self, record: Any) -> Any:
        import numpy as np

        row = self.items[record.sample_id]
        speech = waveform(audio_bytes(row))
        if len(speech) != row["samples_16k"]:
            raise ValueError("decoded speech length changed")
        if not row["noisy"]:
            return speech
        pool = self.noises[row["split"]]
        choice = int(rank(self.seed, row["base_id"]), 16)
        noise = pool[choice % len(pool)]
        path = noise["path"]
        if path not in self.loaded:
            pin(Path(path), noise["sha256"])
            self.loaded[path] = read_audio(Path(path))
            if len(self.loaded) > 8:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(path)
        background = self.loaded[path]
        offset = (choice // len(pool)) % len(background)
        try:
            mixed, audit = mix_10db(speech, background, offset=offset, snr_db=10)
        except ValueError as error:
            if "insufficient active-region energy" not in str(error):
                raise
            active = (
                int(
                    np.argmax(
                        [
                            np.mean(speech[s : s + 320].astype("float64") ** 2)
                            for s in range(0, len(speech), 320)
                        ]
                    )
                )
                * 320
            )
            offset = (int(np.argmax(np.abs(background))) - active) % len(background)
            mixed, audit = mix_10db(speech, background, offset=offset, snr_db=10)
        self.audit.append(
            {"id": row["id"], "noise_path": path, "noise_sha256": noise["sha256"], **audit}
        )
        return mixed

    def __call__(self, records: Any, wrapper: Any, **kwargs: Any) -> Any:
        from qwen_hotword.training.ctc_overfit import extract_frozen_features

        return extract_frozen_features(records, wrapper, waveform_loader=self.load, **kwargs)


def records_from_rows(items: list[dict[str, Any]]) -> Any:
    from qwen_hotword.training.ctc_overfit import ExperimentRecord

    return [
        ExperimentRecord(
            r["id"],
            Path(r["audio"]["path"]),
            r["text"],
            "Chinese" if r["group"] == "zh" else "English",
            tuple(r["phoneme_token_ids"]),
            r["ctc_minimum_input_length"],
            r["ctc_time_upsampling_factor"],
        )
        for r in items
    ]


def model_identity(output: Path, model: Path) -> None:
    index = read(model / "model.safetensors.index.json")
    names = [
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        *sorted(set(index["weight_map"].values())),
    ]
    names += [p.name for p in model.glob("*token*.json")]
    identity = {}
    with stage_progress(
        "Verify original Qwen checkpoint identity", total=len(set(names)), every=1
    ) as tick:
        for i, name in enumerate(sorted(set(names)), 1):
            identity[name] = sha(model / name)
            tick(i)
    destination = output / "model_identity.json"
    if destination.exists() and read(destination) != identity:
        raise ValueError("Qwen weights/processor changed; no historical cache reuse allowed")
    write_json(destination, identity)


def cache(
    root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from qwen_hotword.config import ModelConfig
    from qwen_hotword.modeling.qwen_backbone import load_asr_model
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

    receipt = read(output / "prepare_completed.json")
    for name, expected in receipt["files"].items():
        pin(output / name, expected)
    prepared = manifests(output, sorted((output / "labels").glob("*/rows.jsonl")), cfg)
    model_identity(output, Path(cfg["model"]))
    noise_path = output / "noise_inventory.json"
    if not noise_path.exists():
        write_json(noise_path, audit_noise(Path(cfg["noise_root"]), cfg["seed"]))
    noise = read(noise_path)
    for entry in noise["entries"]:
        pin(Path(entry["path"]), entry["sha256"])
    pools = {s: [n for n in noise["entries"] if n["split"] == s] for s in ("train", "validation")}
    if any(not p for p in pools.values()):
        raise ValueError("both isolated noise pools are required")
    identity = {
        "prepared": sha(output / "prepared.json"),
        "noise": sha(noise_path),
        "model": sha(output / "model_identity.json"),
    }
    cache_contract = output / "cache_contract.json"
    if cache_contract.exists() and read(cache_contract) != identity:
        raise ValueError("cache audio/label/noise/model identity changed")
    write_json(cache_contract, identity)
    missing = []
    verified = {}
    with stage_progress(
        "Verify existing feature parts", total=len(prepared["parts"]), every=25
    ) as tick:
        for i, part in enumerate(prepared["parts"], 1):
            manifest = Path(part["path"])
            pin(manifest, part["sha256"])
            folder = manifest.parent / (manifest.stem + "_features")
            if (folder / "cache_summary.json").exists():
                verified[str(folder)] = load_disk_feature_cache(
                    folder,
                    expected_split=part["split"],
                    source_manifest_path=manifest,
                    vocab_path=root / foundation["vocab"],
                    verify_sha256=True,
                )
            else:
                missing.append(part)
            tick(i)
    wrapper = None
    if missing:
        if not torch.cuda.is_available():
            raise ValueError("GPU cache stage requires the allocated H200")
        free, _ = torch.cuda.mem_get_info()
        if free < cfg["minimum_free_gpu_gib"] * 1024**3:
            raise ValueError("insufficient free GPU memory; no other job is stopped")
        seconds = sum(sum(r["seconds"] for r in rows(Path(p["path"]))) for p in missing)
        required = int(seconds * 13 * 2048 * 1.3 + 2 * 1024**3)
        if shutil.disk_usage(output).free < required:
            raise ValueError(
                f"need approximately {required / 1024**3:.1f} GiB free for remaining features"
            )
        with stage_progress("Load original Qwen3-ASR-1.7B; freeze entire backbone"):
            wrapper = load_asr_model(
                ModelConfig(Path(cfg["model"]), "Qwen3-ASR-1.7B", "bfloat16", "cuda:0", True)
            )
            wrapper.model.eval()
            for parameter in wrapper.model.parameters():
                parameter.requires_grad_(False)
    summaries = []
    with stage_progress(
        "Cache 70% mixed TRAIN and paired validation", total=len(prepared["parts"]), every=1
    ) as tick:
        for i, part in enumerate(prepared["parts"], 1):
            manifest = Path(part["path"])
            pin(manifest, part["sha256"])
            items = list(rows(manifest))
            folder = manifest.parent / (manifest.stem + "_features")
            extractor = MixedExtractor(items, pools, cfg["seed"])
            if str(folder) in verified:
                loaded = verified[str(folder)]
            else:
                cache_feature_split(
                    records_from_rows(items),
                    wrapper,
                    folder,
                    split=part["split"],
                    source_manifest_path=manifest,
                    model_path=cfg["model"],
                    model_dtype="bfloat16",
                    vocab_path=root / foundation["vocab"],
                    encoder_batch_size=cfg["encoder_batch_size"],
                    samples_per_shard=cfg["part_records"],
                    extractor=extractor,
                )
                loaded = load_disk_feature_cache(
                    folder,
                    expected_split=part["split"],
                    source_manifest_path=manifest,
                    vocab_path=root / foundation["vocab"],
                    verify_sha256=True,
                )
            if extractor.audit:
                write_json(folder / "mix_audit.json", extractor.audit)
            summaries.append(
                {
                    "path": str(folder),
                    "manifest": str(manifest),
                    "split": part["split"],
                    "fingerprint": loaded.fingerprint,
                    "bytes": loaded.feature_bytes,
                }
            )
            tick(i)
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {"status": "frozen_183_class_caches_completed", "parts": summaries, "test_used": False}
    write_json(output / "cache_completed.json", result)
    return result


def cache_views(
    root: Path, output: Path, foundation: dict[str, Any]
) -> tuple[Any, Any, dict[str, str], dict[str, str]]:
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

    parts: dict[str, list[Any]] = {"train": [], "validation": []}
    groups, accents = {}, {}
    for part in read(output / "cache_completed.json")["parts"]:
        manifest = Path(part["manifest"])
        loaded = load_disk_feature_cache(
            part["path"],
            expected_split=part["split"],
            source_manifest_path=manifest,
            vocab_path=root / foundation["vocab"],
            verify_sha256=True,
        )
        if loaded.fingerprint != part["fingerprint"]:
            raise ValueError("cached features changed")
        parts[part["split"]].append(loaded)
        if part["split"] == "validation":
            for row in rows(manifest):
                groups[row["id"]] = (
                    ("zh" if row["group"] == "zh" else "en") + "/" + row["condition"]
                )
                accents[row["id"]] = row["group"] + "/" + row["condition"]
    combined = []
    for split, chunks in parts.items():
        if not chunks:
            raise ValueError("empty feature split")
        shards = tuple(s for chunk in chunks for s in chunk.shards)
        ids = [i for s in shards for i in s.sample_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate cache sample IDs")
        first = chunks[0]
        if any(
            c.encoder_frozen_parameters != first.encoder_frozen_parameters
            or c.ctc_time_upsampling_factor != 2
            for c in chunks
        ):
            raise ValueError("cache shape/backbone mismatch")
        combined.append(
            replace(
                first,
                root=output / "cache_views" / split,
                sample_count=len(ids),
                shard_count=len(shards),
                feature_bytes=sum(c.feature_bytes for c in chunks),
                fingerprint=digest([c.fingerprint for c in chunks]),
                shards=shards,
            )
        )
    train_ids = {i for s in combined[0].shards for i in s.sample_ids}
    if any(i.rsplit(":", 1)[0] in train_ids for i in groups):
        raise ValueError("train/validation identity overlap")
    return combined[0], combined[1], groups, accents


def train(
    root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]
) -> dict[str, Any]:
    from qwen_hotword.training.ctc_diagnostics import diagnose_ctc_checkpoint
    from qwen_hotword.training.sharded_ctc import exclusive_training_run, train_sharded_ctc_head

    training, validation, groups, accents = cache_views(root, output, foundation)
    vocab_path = root / foundation["vocab"]
    vocab = load_phoneme_vocab(vocab_path)
    folder = output / "head"
    with exclusive_training_run(folder):
        if (folder / "report.json").exists():
            report = read(folder / "report.json")
        else:
            report = train_sharded_ctc_head(
                training,
                validation,
                vocab,
                folder,
                vocab_path=vocab_path,
                device="cuda:0",
                epochs=cfg["training"]["epochs"],
                minimum_epochs=5,
                early_stopping_patience=6,
                early_stopping_min_delta=0.0005,
                early_stopping_metric="validation_macro_per",
                checkpoint_selection_metric="validation_macro_per",
                validation_sample_groups=groups,
                train_batch_size=cfg["training"]["batch_size"],
                learning_rate=cfg["training"]["learning_rate"],
                minimum_learning_rate=5e-6,
                seed=cfg["seed"],
                resume=(folder / "training_state_latest.pt").exists(),
                head_type="temporal_upsample",
                head_hidden_dimension=cfg["head"]["hidden_dimension"],
                head_kernel_size=cfg["head"]["kernel_size"],
                head_dropout=cfg["head"]["dropout"],
                head_time_upsampling_factor=2,
                initial_head_checkpoint=None,
            ).to_dict()
        if (
            report["status"] != "completed"
            or report["num_classes"] != 183
            or report["test_set_used"]
        ):
            raise ValueError("incompatible completed training report")
        marker = output / "training_completed.json"
        checkpoint = folder / "ctc_head_best.pt"
        identity = {
            "checkpoint_sha256": sha(checkpoint),
            "report_sha256": sha(folder / "report.json"),
            "train_fingerprint": training.fingerprint,
            "validation_fingerprint": validation.fingerprint,
        }
        if marker.exists() and read(marker)["identity"] != identity:
            raise ValueError("completed Head identity changed")
        per_path = output / "validation_per.json"
        if not marker.exists():
            per = diagnose_ctc_checkpoint(
                checkpoint,
                validation,
                vocab,
                device="cuda:0",
                sample_groups=accents,
                sample_groupings={"language_condition": groups},
            )
            write_json(per_path, per)
            write_json(
                marker,
                {
                    "status": "fresh_head_training_completed",
                    "identity": identity,
                    "validation_per_sha256": sha(per_path),
                    "num_classes": 183,
                    "best_epoch": report["best_epoch"],
                    "validation_per_by_group": {
                        name: values["phoneme_error_rate"]
                        for name, values in cast(dict[str, Any], per["validation_by_group"]).items()
                    },
                    "head_initialization": "random",
                    "encoder_frozen": True,
                    "test_used": False,
                },
            )
        result = read(marker)
        pin(per_path, result["validation_per_sha256"])
        return cast(dict[str, Any], result)


def bundle(output: Path) -> Path:
    path = output / "return_bundle.zip"
    names = [
        "report.json",
        "identity.json",
        "config.json",
        "source_identity.json",
        "split_report.json",
        "label_report.json",
        "g2p_report.json",
        "model_identity.json",
        "training_completed.json",
        "head/report.json",
        "head/metrics.jsonl",
        "validation_per.json",
        "logs/runtime_probe.json",
        "error_traceback.txt",
    ]
    with zipfile.ZipFile(path.with_suffix(".pending.zip"), "w", zipfile.ZIP_DEFLATED) as zipped:
        sums = []
        for name in names:
            file = output / name
            if file.is_file():
                zipped.write(file, name)
                sums.append(f"{sha(file)}  {name}")
        zipped.writestr("sha256.txt", "\n".join(sums) + "\n")
    path.with_suffix(".pending.zip").replace(path)
    return path


def status(output: Path) -> dict[str, Any]:
    result = (
        read(output / "report.json")
        if (output / "report.json").exists()
        else {"status": "not_started"}
    )
    result["completed_inventory_parts"] = len(list((output / "inventory").glob("*/completed.json")))
    result["completed_label_parts"] = len(list((output / "labels").glob("*/completed.json")))
    result["completed_feature_parts"] = len(
        list((output / "prepared").glob("*/*/*/*_features/cache_summary.json"))
    )
    metric = output / "head/metrics.jsonl"
    if metric.exists():
        for line in reversed(metric.read_text().splitlines()):
            try:
                result["last_epoch"] = json.loads(line)
                break
            except ValueError:
                continue
    return result


def run(root: Path, output: Path, cfg: dict[str, Any], action: str) -> dict[str, Any]:
    foundation = read(root / cfg["foundation_config"])
    with exclusive_feature_cache_run(output.parent / ("." + output.name + ".lock")):
        contract(root, output, cfg, foundation)
        write_json(output / "config.json", cfg)
        report: dict[str, Any] = {
            "status": "running",
            "scope": "zh_us_uk_prepare_cache_scratch_head_validation_PER",
            "output": str(output),
            "noise_percent": 70,
            "snr_db": 10,
            "num_classes": 183,
            "retrieval_run": False,
            "asr_run": False,
            "test_used_for_training": False,
            "stages": {},
        }
        try:
            for stage in ["prepare", "cache", "train"] if action == "all" else [action]:
                report["active_stage"] = stage
                write_json(output / "report.json", report)
                with stage_progress("Chinese / US / UK stage: " + stage):
                    value = {"prepare": prepare, "cache": cache, "train": train}[stage](
                        root, output, cfg, foundation
                    )
                report["stages"][stage] = {
                    k: v
                    for k, v in value.items()
                    if k not in {"parts", "groups", "training_token_counts"}
                }
            report.update(status="completed", active_stage=None)
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            (output / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        report["return_bundle"] = str(output / "return_bundle.zip")
        write_json(output / "report.json", report)
        bundle(output)
        return report
