"""Additive Chinese preparation and two-H200 extraction, ending at validation PER."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.paired_gpu import (
    ResourceFailure,
    autotune,
    ensure_no_workers,
    run_workers,
)
from qwen_hotword.training.paired_labels import effective_workers, incremental_g2p, parallel_text
from qwen_hotword.training.paired_selection import manifests
from qwen_hotword.training.paired_sources import legacy_parts, merge_splits, new_inventory
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import completed, contract, read, rows
from qwen_hotword.training.zh_en_foundation import verify_assets
from qwen_hotword.training.zh_en_labels import install_frontend, label_part, make_worker


def establish(root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]) -> None:
    for value in (cfg["source_run"], cfg["added_chinese_root"]):
        source = (root / value).resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("new output overlaps a protected source")
    if cfg["gpu_ids"] != list(dict.fromkeys(cfg["gpu_ids"])) or len(cfg["gpu_ids"]) != 2:
        raise ValueError("two distinct allocated GPUs required")
    if cfg["encoder_process_candidates"] != [1, 2, 4] or cfg["benchmark_repeats"] != 2:
        raise ValueError("this workflow compares 1/2/4 workers per GPU in two reversed rounds")
    if (
        cfg["hours"]["train"] != {"zh": 480, "us": 336, "uk": 144}
        or cfg["minimum_hours"]["train"] != cfg["hours"]["train"]
    ):
        raise ValueError("480h Chinese and 336h US + 144h UK training required")
    if not 1 <= cfg["cpu_worker_limit"] <= 64 or cfg["encoder_batch_size"] != 1:
        raise ValueError("CPU ceiling 64 and precision-preserving per-audio Encoder required")
    # Keep the historical implementation unmodified; its contract remains independently verifiable.
    contract(root, output, cfg, foundation)
    paths = [
        *sorted((root / "src/qwen_hotword/training").glob("paired_*.py")),
        root / "scripts/run_zh_en_dual.py",
    ]
    identity = {str(p.relative_to(root)): sha(p) for p in paths}
    receipt = output / "dual_code_identity.json"
    if receipt.exists() and read(receipt) != identity:
        raise ValueError("dual-worker source changed; preserve run and use a new output")
    write_json(receipt, identity)


def verify_prepared(output: Path) -> dict[str, Any]:
    receipt = read(output / "prepare_completed.json")
    for name, expected in receipt["files"].items():
        pin(output / name, expected)
    marker = read(output / "prepared/completed.json")
    done = completed(output / "prepared", marker["identity"])
    if done != read(output / "prepared.json"):
        raise ValueError("prepared identities differ")
    return read(output / "prepared.json")  # type: ignore[no-any-return]


def prepare(
    root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]
) -> dict[str, Any]:
    verify_assets(root, foundation)
    if (output / "prepare_completed.json").exists():
        return verify_prepared(output)
    old = legacy_parts(root, output, cfg, foundation)
    install_frontend(output, cfg["frontend_packages"])
    worker, assets, prefix = make_worker(root, output, foundation, cfg)
    workers = effective_workers(cfg["cpu_worker_limit"])
    print(
        f"CPU frontend/MFA workers: {workers}; configured ceiling {cfg['cpu_worker_limit']}",
        flush=True,
    )
    write_json(
        output / "cpu_resources.json",
        {
            "configured_ceiling": cfg["cpu_worker_limit"],
            "effective_frontend_mfa_workers": workers,
            "nested_threads": 1,
            "memory_policy": "at most 60% available RAM at planning allowance 1GiB/worker",
        },
    )
    added = new_inventory(output, cfg)
    old_overlay, new = merge_splits(output, old, added, cfg["seed"])
    available = read(output / "merged_source_report.json")["by_split"]
    short = {
        g: available.get(g + "/train", {}).get("eligible_before_new_g2p_hours", 0)
        for g, minimum in cfg["minimum_hours"]["train"].items()
        if available.get(g + "/train", {}).get("eligible_before_new_g2p_hours", 0) < minimum
    }
    if short:
        raise ValueError(
            f"insufficient train hours before additional G2P: {short}; "
            "see merged_source_report.json"
        )
    texts = parallel_text(output, new, workers)
    db = incremental_g2p(
        root, output, texts, root / cfg["source_run"], worker, assets, prefix, workers
    )
    labeled = []
    with stage_progress("Assemble only added Chinese labels", total=len(texts), every=25) as tick:
        for i, source in enumerate(texts):
            labeled.append(label_part(output, source, i, db, cfg["head"]["time_upsampling_factor"]))
            tick(i + 1)
    report = manifests(output, old_overlay + labeled, cfg)
    write_json(output / "prepared.json", report)
    names = (
        "prepared.json",
        "prepared/completed.json",
        "label_report.json",
        "g2p_report.json",
        "legacy_reuse.json",
        "added_source_identity.json",
        "merged_source_report.json",
    )
    write_json(output / "prepare_completed.json", {"files": {n: sha(output / n) for n in names}})
    return report


def collect_jobs(
    root: Path, output: Path, foundation: dict[str, Any], prepared: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

    missing, done = [], []
    with stage_progress(
        "Verify completed feature parts; queue only unfinished parts",
        total=len(prepared["parts"]),
        every=25,
    ) as tick:
        for i, part in enumerate(prepared["parts"]):
            manifest = Path(part["path"])
            pin(manifest, part["sha256"])
            folder = output / "features" / f"part-{i:06d}"
            if (folder / "cache_summary.json").exists():
                loaded = load_disk_feature_cache(
                    folder,
                    expected_split=part["split"],
                    source_manifest_path=manifest,
                    vocab_path=root / foundation["vocab"],
                    verify_sha256=True,
                )
                done.append(
                    {
                        "path": str(folder),
                        "manifest": str(manifest),
                        "split": part["split"],
                        "fingerprint": loaded.fingerprint,
                        "bytes": loaded.feature_bytes,
                    }
                )
            else:
                missing.append(
                    {
                        "manifest": str(manifest),
                        "manifest_sha256": part["sha256"],
                        "destination": str(folder),
                        "split": part["split"],
                        "seconds": sum(r["seconds"] for r in rows(manifest)),
                    }
                )
            tick(i + 1)
    return missing, done


def cache(
    root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]
) -> dict[str, Any]:
    from qwen_hotword.training.noise_audio import audit_noise
    from qwen_hotword.training.zh_en_run import model_identity

    ensure_no_workers(output)
    prepared = verify_prepared(output)
    model_identity(output, Path(cfg["model"]))
    noise_path = output / "noise_inventory.json"
    if not noise_path.exists():
        with stage_progress("Audit and isolate training/validation noise"):
            write_json(noise_path, audit_noise(Path(cfg["noise_root"]), cfg["seed"]))
    noise = read(noise_path)
    with stage_progress("Verify noise identities", total=len(noise["entries"]), every=100) as tick:
        for i, entry in enumerate(noise["entries"], 1):
            pin(Path(entry["path"]), entry["sha256"])
            tick(i)
    if any(not any(n["split"] == s for n in noise["entries"]) for s in ("train", "validation")):
        raise ValueError("both isolated noise pools required")
    identity = {
        n: sha(output / n)
        for n in (
            "prepared.json",
            "noise_inventory.json",
            "model_identity.json",
            "contract.json",
            "dual_code_identity.json",
        )
    }
    receipt = output / "cache_contract.json"
    if receipt.exists() and read(receipt) != identity:
        raise ValueError("cache source identity changed")
    write_json(receipt, identity)
    missing, done = collect_jobs(root, output, foundation, prepared)
    if missing:
        required = int(sum(j["seconds"] for j in missing) * 13 * 2048 * 1.3 + 5 * 1024**3)
        if shutil.disk_usage(output).free < required:
            raise ValueError(
                f"need approximately {required / 1024**3:.1f} GiB free for remaining features"
            )
        selected = autotune(root, output, cfg, foundation, prepared)
        count = selected["selected_processes_per_gpu"]
        attempts = []
        while missing:
            try:
                result = run_workers(
                    root, output, cfg, foundation, missing, count, selected["reuse"]
                )
                attempts.append({"status": "completed", **result})
            except ResourceFailure as error:
                attempts.append(
                    {"status": "resource_unavailable", "per_gpu": count, "error": str(error)}
                )
                write_json(output / "production_workers.json", {"attempts": attempts})
                if count == 1:
                    raise
                count //= 2
            write_json(
                output / "production_workers.json", {"attempts": attempts, "final_per_gpu": count}
            )
            missing, done = collect_jobs(root, output, foundation, prepared)
    result = {"status": "frozen_183_class_caches_completed", "parts": done, "test_used": False}
    write_json(output / "cache_completed.json", result)
    return result


def train_process(root: Path, output: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    ensure_no_workers(output)
    log = output / "logs/train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg["gpu_ids"][0]), "PYTHONUNBUFFERED": "1"}
    with log.open("a", encoding="utf-8") as stream:
        command = [
            sys.executable,
            "-B",
            str(root / "scripts/run_zh_en_dual.py"),
            "train-worker",
            "--output-dir",
            str(output),
        ]
        process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait() != 0:
                raise RuntimeError(
                    f"Head training failed; see {log}; completed epochs are resumable"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    return read(output / "training_completed.json")  # type: ignore[no-any-return]


def bundle(output: Path) -> Path:
    names = [
        "report.json",
        "identity.json",
        "dual_code_identity.json",
        "config.json",
        "cpu_resources.json",
        "merged_source_report.json",
        "label_report.json",
        "g2p_report.json",
        "model_identity.json",
        "logs/runtime_probe.json",
        "training_completed.json",
        "head/report.json",
        "head/metrics.jsonl",
        "validation_per.json",
        "production_workers.json",
        "error_traceback.txt",
    ]
    for source, target, keys in (
        ("legacy_reuse.json", "legacy_reuse_summary.json", ("label_parts",)),
        ("encoder_autotune.json", "encoder_autotune_summary.json", ("reuse",)),
    ):
        if (output / source).exists():
            value = read(output / source)
            for key in keys:
                value[key + "_count"] = len(value.pop(key, {}))
            for trial in value.get("trials", []):
                trial.pop("cache_jobs", None)
            write_json(output / target, value)
            names.append(target)
    destination = output / "return_bundle.zip"
    with zipfile.ZipFile(destination.with_suffix(".pending.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        sums = []
        for name in names:
            p = output / name
            if p.is_file():
                z.write(p, name)
                sums.append(f"{sha(p)}  {name}")
        z.writestr("sha256.txt", "\n".join(sums) + "\n")
    destination.with_suffix(".pending.zip").replace(destination)
    return destination


def status(output: Path) -> dict[str, Any]:
    result = (
        read(output / "report.json")
        if (output / "report.json").exists()
        else {"status": "not_started"}
    )
    result["completed_new_audio_parts"] = len(list(output.glob("new_inventory/*/completed.json")))
    result["completed_new_label_parts"] = len(list(output.glob("labels/*/completed.json")))
    result["completed_feature_parts"] = len(list(output.glob("features/*/cache_summary.json")))
    metric = output / "head/metrics.jsonl"
    if metric.exists():
        import json

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
        establish(root, output, cfg, foundation)
        write_json(output / "config.json", cfg)
        report: dict[str, Any] = {
            "status": "running",
            "scope": "zh_us_uk_additive_dual_encoder_train_PER",
            "output": str(output),
            "gpu_ids": cfg["gpu_ids"],
            "cpu_worker_limit": cfg["cpu_worker_limit"],
            "noise_percent": 70,
            "snr_db": 10,
            "num_classes": 183,
            "retrieval_run": False,
            "asr_run": False,
            "test_used_for_training": False,
            "stages": {},
        }
        try:
            for stage in ("prepare", "cache", "train") if action == "all" else (action,):
                report["active_stage"] = stage
                write_json(output / "report.json", report)
                with stage_progress("Chinese / US / UK stage: " + stage):
                    value = (
                        prepare(root, output, cfg, foundation)
                        if stage == "prepare"
                        else (
                            cache(root, output, cfg, foundation)
                            if stage == "cache"
                            else train_process(root, output, cfg)
                        )
                    )
                report["stages"][stage] = {
                    k: v for k, v in value.items() if k not in ("parts", "failure_examples")
                }
            report.update(status="completed", active_stage=None)
        except (Exception, KeyboardInterrupt) as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            (output / "error_traceback.txt").write_text(traceback.format_exc())
        report["return_bundle"] = str(output / "return_bundle.zip")
        write_json(output / "report.json", report)
        bundle(output)
        return report
