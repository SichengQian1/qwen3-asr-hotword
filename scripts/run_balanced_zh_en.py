#!/usr/bin/env python3
"""Prepare the four-source Mandarin / two-source English run and train through PER."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.balanced_selection import select
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.paired_labels import effective_workers, incremental_g2p, parallel_text
from qwen_hotword.training.paired_run import bundle, cache, train_process, verify_prepared
from qwen_hotword.training.paired_sources import merge_splits, new_inventory
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import contract, read, rows, save_rows
from qwen_hotword.training.zh_en_foundation import verify_assets
from qwen_hotword.training.zh_en_labels import install_frontend, label_part, make_worker


def previous_labels(previous: Path, foundation: dict[str, Any], cfg: dict[str, Any]) -> list[Path]:
    old = read(previous / "contract.json")
    if old["foundation"] != foundation:
        raise ValueError("previous foundation mismatch")
    for key in ("frontend_policy", "frontend_packages", "seed", "head", "num_classes"):
        if old["config"][key] != cfg[key]:
            raise ValueError(f"previous label configuration mismatch: {key}")
    marker = previous / "prepared/completed.json"
    pin(marker, read(previous / "prepare_completed.json")["files"]["prepared/completed.json"])
    expected = read(marker)["identity"]["labels"]
    paths = sorted(previous.glob("legacy_labels/part-*.jsonl")) + sorted(
        previous.glob("labels/part-*/rows.jsonl")
    )
    if not paths or len(paths) != len(expected):
        raise ValueError("previous full-corpus label parts missing")
    with stage_progress("Verify complete previous label pool", total=len(paths)) as tick:
        for i, (path, checksum) in enumerate(zip(paths, expected, strict=True), 1):
            pin(path, checksum)
            tick(i)
    return paths


def namespaced(path: Path, name: str, directory: int) -> Iterator[dict[str, Any]]:
    for row in rows(path):
        row["id"] = f"{name}:{directory}:" + row["id"]
        row["source_group"] = name
        row["family"] = name + ":" + row["family"]
        yield row


def prepare(out: Path, cfg: dict[str, Any], foundation: dict[str, Any]) -> None:
    verify_assets(ROOT, foundation)
    if (out / "prepare_completed.json").is_file():
        verify_prepared(out)
        return
    previous = ROOT / cfg["previous_run"]
    receipt = read(previous / "prepare_completed.json")["files"]
    pin(previous / "label_report.json", receipt["label_report.json"])
    us_hours = read(previous / "label_report.json")["availability"]["train/us"]["hours"]
    print(
        f"Previous eligible US training pool: {us_hours:.3f} h (before new PCM checks)", flush=True
    )
    if not cfg["scale_to_us"] and us_hours < cfg["quotas"]["train"]["us"]:
        raise ValueError(f"Previous complete US labels already limit TRAIN to {us_hours:.3f} h")
    old = previous_labels(previous, foundation, cfg)
    added = []
    for name in ("0009", "0059"):
        folder = out / name
        root = Path(cfg["chinese_roots"][name])
        archives = sorted(root.rglob("*.tar"))
        if len(archives) != cfg["archive_counts"][name]:
            raise ValueError(
                f"{root}: expected {cfg['archive_counts'][name]} plain tar shards, "
                f"found {len(archives)}"
            )
        for j, parent in enumerate(sorted({p.parent for p in archives})):
            work = folder / f"dir-{j:04d}"
            parts = new_inventory(work, {**cfg, "added_chinese_root": str(parent)})
            for i, path in enumerate(parts):
                dest = work / "named" / f"part-{i:06d}.jsonl"
                save_rows(dest, namespaced(path, name, j))
                added.append(dest)
    old, new = merge_splits(out, old, added, cfg["seed"])
    install_frontend(out, cfg["frontend_packages"])
    worker, assets, prefix = make_worker(ROOT, out, foundation, cfg)
    workers = effective_workers(cfg["cpu_worker_limit"])
    texts = parallel_text(out, new, workers)
    db = incremental_g2p(
        ROOT, out, texts, ROOT / cfg["source_run"], worker, assets, prefix, workers
    )
    with stage_progress("Assemble new Mandarin labels", total=len(texts)) as tick:
        for i, path in enumerate(texts):
            old.append(label_part(out, path, i, db, cfg["head"]["time_upsampling_factor"]))
            tick(i + 1)
    report = select(out, old, cfg)
    write_json(out / "prepared.json", report)
    files = ("prepared.json", "prepared/completed.json", "label_report.json", "g2p_report.json")
    write_json(out / "prepare_completed.json", {"files": {k: sha(out / k) for k in files}})
    print(report["groups"], flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale-to-us",
        action="store_true",
        help="If eligible US training hours fall below 530, scale all train quotas proportionally.",
    )
    args = parser.parse_args()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = read(ROOT / "configs/zh_en_noise70_dual.workzone.json")
    extra = read(ROOT / "configs/zh_en_balanced.workzone.json")
    cfg.update(extra)
    cfg["scale_to_us"] = args.scale_to_us
    cfg["hours"] = {
        s: {"us": q["us"], "uk": q["uk"], "zh": q["us"] + q["uk"]} for s, q in cfg["quotas"].items()
    }
    cfg.pop("minimum_hours")
    if args.scale_to_us:
        cfg["output"] += "_uscap"
    foundation = read(ROOT / cfg["foundation_config"])
    out = ROOT / cfg["output"]
    with exclusive_feature_cache_run(out.parent / ("." + out.name + ".lock")):
        contract(ROOT, out, cfg, foundation)
        code = [
            Path(__file__),
            ROOT / "src/qwen_hotword/training/balanced_selection.py",
            ROOT / "scripts/run_zh_en_dual.py",
            *ROOT.glob("src/qwen_hotword/training/paired_*.py"),
        ]
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in code}
        identity = out / "dual_code_identity.json"
        if identity.exists() and read(identity) != hashes:
            raise ValueError("run code identity changed")
        write_json(identity, hashes)
        write_json(out / "config.json", cfg)
        result: dict[str, Any] = {"status": "running", "output": str(out)}
        try:
            for stage in ("prepare", "cache", "train"):
                result["active_stage"] = stage
                write_json(out / "report.json", result)
                with stage_progress("Balanced Chinese / English: " + stage):
                    if stage == "prepare":
                        prepare(out, cfg, foundation)
                    elif stage == "cache":
                        cache(ROOT, out, cfg, foundation)
                    else:
                        result["training"] = train_process(ROOT, out, cfg)
            result.update(status="completed", active_stage=None)
        except (Exception, KeyboardInterrupt) as error:
            result.update(status="failed", error=f"{type(error).__name__}: {error}")
            (out / "error_traceback.txt").write_text(traceback.format_exc())
        result["return_bundle"] = str(out / "return_bundle.zip")
        write_json(out / "report.json", result)
        bundle(out)
        print(result, flush=True)
        return int(result["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
