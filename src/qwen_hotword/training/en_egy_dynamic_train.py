"""Single-GPU r15 cache reuse/extraction and fresh Head for dynamically selected EGY/EN."""

from __future__ import annotations

import gc
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.en_egy_dynamic import read, seal
from qwen_hotword.training.stage_progress import stage_progress


def run(
    root: Path, cfg: dict[str, Any], output: Path, source: dict[str, Any], gpu: int
) -> dict[str, Any]:
    import torch

    from qwen_hotword.config import load_workzone_config
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab
    from qwen_hotword.training.arabic_refresh_train import ReuseOrExtract
    from qwen_hotword.training.ctc_overfit import load_experiment_records
    from qwen_hotword.training.feature_cache import cache_feature_split
    from qwen_hotword.training.sharded_ctc import (
        exclusive_training_run,
        load_disk_feature_cache,
        train_sharded_ctc_head,
    )
    from scripts.arabic_diagnostic_common import load_cache, training_options
    from scripts.run_en_ar_merged import wait_for_gpu

    prepared = read(output / "prepare_report.json")
    for item in prepared["manifests"].values():
        pin(Path(item["path"]), item["sha256"])
    vocab_path = root / cfg["vocab"]
    pin(vocab_path, cfg["vocab_sha256"])
    options = training_options(source["head"], cfg["head_time_upsampling_factor"])
    seal(
        output / "training_contract.json",
        {
            "prepare_report_sha256": sha(output / "prepare_report.json"),
            "options": options,
            "initialization": "random",
            "encoder": source["plan"]["model"],
            "code_sha256": sha(Path(__file__)),
        },
    )
    final = output / "report.json"
    if final.exists():
        result = read(final)
        pin(output / "head/ctc_head_best.pt", result["checkpoint_sha256"])
        return dict(result)
    model = source["plan"]["model"]
    with stage_progress("Verify frozen r15 weights and preprocessor identities"):
        for value in model["files"].values():
            pin(Path(value["path"]), value["sha256"])
        pin(Path(model["view"]) / "model.safetensors.index.json", model["index_sha256"])
    with stage_progress(f"Wait for physical GPU {gpu} to be free; no other jobs terminated"):
        while True:
            result = subprocess.run(
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
            if not result.stdout.strip():
                break
            print(f"GPU {gpu} occupied: {result.stdout.strip()}", flush=True)
            time.sleep(10)
    wait_for_gpu(gpu, cfg["minimum_gpu_free_gib"])
    old = [load_cache(source, split, output) for split in ("train", "validation")]
    audio = {
        row["id"]: row["audio_path"]
        for item in source["plan"]["manifests"].values()
        for row in iter_jsonl(Path(item["path"]))
    }
    zone = load_workzone_config(source["plan"]["cache_config"]["path"], require_existing_model=True)
    if str(zone.model.path) != model["view"] or zone.model.dtype != "bfloat16":
        raise ValueError("cache extraction model/config differs from r15")
    lookup = ReuseOrExtract(old, audio, zone.model)
    expected = sum(
        r["duration_seconds"] * 13 * 2048
        for split in ("train", "validation")
        for r in iter_jsonl(output / f"{split}.jsonl")
    )
    existing = sum(p.stat().st_size for p in (output / "cache").rglob("*.pt"))
    if shutil.disk_usage(output).free + existing < expected * 1.2 + 2 * 1024**3:
        raise ValueError("insufficient disk space for new cache shards")
    caches = {}
    groups = {}
    for split in ("train", "validation"):
        manifest = output / f"{split}.jsonl"
        if split == "validation":
            groups = {r["id"]: r["language"] for r in iter_jsonl(manifest)}
        with stage_progress(f"Load {split} selected labels") as tick:
            records = load_experiment_records(
                manifest,
                num_classes=167,
                expected_experiment="full-ctc-v1",
                expected_split=split,
                prevalidated_audio_paths=True,
                progress=tick,
            )
        with stage_progress(f"Cache {split}: reuse r15 / extract missing audio; see shard counts"):
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
        caches[split] = load_disk_feature_cache(
            output / "cache" / split,
            expected_split=split,
            source_manifest_path=manifest,
            vocab_path=vocab_path,
        )
    lookup.wrapper = None
    lookup.loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()
    write_json(
        output / "feature_reuse.json",
        {
            "reused_this_invocation": lookup.reused,
            "extracted_this_invocation": lookup.extracted,
            "resumed_shards_not_counted_here": True,
            "old_caches_unchanged": True,
            "new_labels_in_new_shards": True,
        },
    )
    if set(groups.values()) != {"en", "ar"}:
        raise ValueError("validation must contain both English and Egyptian Arabic")
    head = output / "head"
    with (
        exclusive_training_run(head),
        stage_progress("Train fresh English / Egyptian Arabic CTC Head; epoch metrics"),
    ):
        if not (head / "report.json").exists():
            train_sharded_ctc_head(
                caches["train"],
                caches["validation"],
                load_phoneme_vocab(vocab_path),
                head,
                vocab_path=vocab_path,
                device="cuda:0",
                validation_sample_groups=groups,
                resume=(head / "training_state_latest.pt").exists(),
                **options,
            )
    head_report = read(head / "report.json")
    if head_report["status"] != "completed":
        raise ValueError("Head has not completed")
    report = {
        "status": "completed",
        "output_dir": str(output),
        "physical_gpu": gpu,
        "head": head_report,
        "selection": prepared,
        "test_set_used": False,
        "checkpoint_sha256": sha(head / "ctc_head_best.pt"),
        "model_weight_sha256": model["files"]["model.safetensors"]["sha256"],
        "vocab_sha256": cfg["vocab_sha256"],
        "Arabic_validation": "EGY declared sources only",
        "comparison_note": (
            "Arabic validation and labels changed; PER is not directly paired to old EGY+MSA PER"
        ),
    }
    write_json(final, report)
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report
