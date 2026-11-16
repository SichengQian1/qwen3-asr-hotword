#!/usr/bin/env python3
"""Label pure Egyptian pools, select equal English/Arabic hours, cache r15 and train."""

from __future__ import annotations

import argparse
import importlib.metadata
import multiprocessing
import os
import subprocess
import sys
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.en_egy_dynamic import (
    freeze,
    prepare_english,
    prepare_inputs,
    read,
    save_rows,
    seal,
)
from qwen_hotword.training.stage_progress import stage_progress
from scripts.label_arabic_native import exclusive_job

FRONTEND: Any = None


def label_job(job: tuple[str, str, str, int, list[dict[str, Any]]]) -> dict[str, Any]:
    from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
    from qwen_hotword.phonemes.arabic_pronunciation import PronunciationFrontend

    global FRONTEND
    destination, config_path, policy_path, number, rows = job
    output = Path(destination)
    folder = output / "parts" / f"{number:06d}"
    marker = folder / "completed.json"
    if marker.exists():
        result = read(marker)
        for key in ("labels", "review"):
            pin(output / result[key], result[key + "_sha256"])
        return dict(result)
    cfg = read(Path(config_path))
    tokens = read(ROOT / cfg["vocab"])["tokens"]
    frontend_cfg = read(ROOT / cfg["arabic_frontend_config"])
    if FRONTEND is None:
        FRONTEND = PronunciationFrontend(
            ROOT / frontend_cfg["assets"], "ar_egy", tokens, read(Path(policy_path))
        )
    ready, review = [], []
    reasons: Counter[str] = Counter()
    with stage_progress(f"EGY label part {number}", total=len(rows), every=250) as tick:
        for index, row in enumerate(rows, 1):
            try:
                generated = FRONTEND.label(row["text"], mode="sentence")
                if generated.get("unreviewed_exception_words"):
                    raise ValueError("unreviewed_pronunciation_exception")
                ids = generated["ids"]
                if not ids or any(type(i) is not int or not 72 <= i < len(tokens) for i in ids):
                    raise ValueError("invalid_or_empty_Arabic_ids")
                minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
                frames = qwen3_asr_audio_output_length(
                    round(row["duration_seconds"] * 16000) // 160
                )
                if minimum > 2 * frames:
                    raise ValueError("estimated_CTC_length_infeasible")
                ready.append(
                    {
                        **row,
                        **generated,
                        "vocab_sha256": cfg["vocab_sha256"],
                        "ctc_minimum_input_length": minimum,
                    }
                )
            except ValueError as error:
                reasons[str(error).split(":", 1)[0]] += 1
                review.append({**row, "label_issue": str(error)})
            tick(index)
    folder.mkdir(parents=True, exist_ok=True)
    save_rows(folder / "labels.jsonl", ready)
    save_rows(folder / "review.jsonl", review)
    result = {
        "rows": len(rows),
        "ready_rows": len(ready),
        "reasons": reasons,
        "by_bucket": {
            bucket: {
                "rows": sum(r["bucket"] == bucket for r in ready),
                "hours": sum(r["duration_seconds"] for r in ready if r["bucket"] == bucket) / 3600,
            }
            for bucket in sorted({r["bucket"] for r in rows})
        },
        **{key: str((folder / f"{key}.jsonl").relative_to(output)) for key in ("labels", "review")},
        **{key + "_sha256": sha(folder / f"{key}.jsonl") for key in ("labels", "review")},
    }
    write_json(marker, result)
    return result


def label(cfg: dict[str, Any], config_path: Path, output: Path, workers: int) -> None:
    from qwen_hotword.phonemes.arabic_pronunciation import validate_policy
    from scripts.label_arabic_native import verify_assets

    frontend = read(ROOT / cfg["arabic_frontend_config"])
    if importlib.metadata.version("camel-tools") != frontend["camel_tools_version"]:
        raise ValueError("CAMeL version changed")
    verify_assets(frontend)
    policy = ROOT / cfg["pronunciation_policy"]
    validate_policy(read(policy))
    destination = output / "labels"
    destination.mkdir(exist_ok=True)
    marker = destination / "report.json"
    if marker.exists():
        for piece in read(marker)["pieces"]:
            for key in ("labels", "review"):
                pin(destination / piece[key], piece[key + "_sha256"])
        return
    pieces = []
    with (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        stage_progress("Arabic labeling: completed candidate rows") as tick,
    ):
        pending: set[Any] = set()
        count = 0

        def drain() -> None:
            nonlocal pending, count
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                piece = future.result()
                pieces.append(piece)
                count += piece["rows"]
                tick(count)

        batch, number = [], 0
        for row in iter_jsonl(output / "arabic_candidates.jsonl"):
            batch.append(row)
            if len(batch) == 1000:
                pending.add(
                    pool.submit(
                        label_job, (str(destination), str(config_path), str(policy), number, batch)
                    )
                )
                batch, number = [], number + 1
                if len(pending) >= workers * 2:
                    drain()
        if batch:
            pending.add(
                pool.submit(
                    label_job, (str(destination), str(config_path), str(policy), number, batch)
                )
            )
        while pending:
            drain()
    pieces.sort(key=lambda p: p["labels"])
    buckets: dict[str, Any] = {}
    for piece in pieces:
        for bucket, value in piece["by_bucket"].items():
            item = buckets.setdefault(bucket, {"rows": 0, "hours": 0.0})
            for field in item:
                item[field] += value[field]
    write_json(
        marker,
        {
            "status": "automatic_labels_completed",
            "pieces": pieces,
            "by_bucket": buckets,
            "reasons": dict(sum((Counter(p["reasons"]) for p in pieces), Counter())),
            "vocab_sha256": cfg["vocab_sha256"],
            "policy_sha256": sha(policy),
            "human_pronunciation_review_completed": False,
            "MSA_used": False,
        },
    )


def identity(cfg: dict[str, Any], config_path: Path, output: Path, source: dict[str, Any]) -> None:
    files = [
        config_path,
        ROOT / cfg["vocab"],
        ROOT / cfg["pronunciation_policy"],
        ROOT / cfg["arabic_frontend_config"],
        ROOT / cfg["english_pool_config"],
        ROOT / cfg["english_inventory"],
        ROOT / cfg["english_g2p"],
        ROOT / cfg["arabic_inventory"],
        ROOT / "configs/mfa_v03_foundation.workzone.json",
        ROOT / "configs/phonemes/en_ar_mfa_g2p_vocab.v1.0.json",
    ]
    for name in (
        "en_egy_dynamic.py",
        "en_egy_dynamic_train.py",
        "arabic_refresh_train.py",
        "arabic_refresh.py",
        "mfa_round_labels.py",
        "mfa_round_common.py",
        "feature_cache.py",
        "ctc_overfit.py",
        "sharded_ctc.py",
    ):
        files.append(ROOT / "src/qwen_hotword/training" / name)
    files.extend(
        ROOT / "src/qwen_hotword/phonemes" / name
        for name in ("arabic_native.py", "arabic_pronunciation.py")
    )
    files.extend(
        ROOT / "scripts" / name
        for name in (
            "run_en_egy_dynamic.py",
            "run_en_ar_training.py",
            "run_en_ar_merged.py",
            "arabic_diagnostic_common.py",
            "label_arabic_native.py",
        )
    )
    pool = read(ROOT / cfg["english_pool_config"])["pools"]["en"]
    for split, checksum in pool["manifest_sha256"].items():
        p = ROOT / pool["pool"] / f"full_ctc_{split}.jsonl"
        pin(p, checksum)
        files.append(p)
    audit_contract = (ROOT / cfg["arabic_inventory"]).parent / "contract.json"
    files.append(audit_contract)
    for name, checksum in read(audit_contract)["sources"].items():
        pin(Path(name), checksum)
        files.append(Path(name))
    pin(ROOT / cfg["vocab"], cfg["vocab_sha256"])
    pin(ROOT / cfg["arabic_inventory"], cfg["arabic_inventory_sha256"])
    native = read(ROOT / cfg["vocab"])["tokens"]
    old = read(ROOT / "configs/phonemes/en_ar_mfa_g2p_vocab.v1.0.json")["tokens"]
    if len(native) != 167 or native[:72] != old[:72]:
        raise ValueError("English prefix or class count changed")
    protected = [
        Path(source["source"]),
        *(Path(i["path"]).parent for i in source["plan"]["manifests"].values()),
    ]
    for p in [*files, *protected]:
        p = p.resolve()
        if output == p or output in p.parents or p in output.parents:
            raise ValueError("output overlaps a protected input: " + str(p))
    current = {
        "configuration": cfg,
        "files": {str(p.resolve()): sha(p) for p in files},
        "source_identity": source["identity"],
        "schema_version": 1,
    }
    output.mkdir(parents=True, exist_ok=True)
    seal(output / "contract.json", current)


def recheck(output: Path) -> None:
    with stage_progress("Recheck sealed source/config/code identities"):
        contract = read(output / "contract.json")
        for name, checksum in contract["files"].items():
            pin(Path(name), checksum)
        for value in contract["source_identity"].values():
            pin(Path(value["path"]), value["sha256"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "prepare", "_label"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/en_egy_dynamic.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--io-workers", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64 or not 1 <= args.io_workers <= 64 or args.gpu < 0:
        parser.error("workers/io-workers must be 1..64; GPU must be nonnegative")
    cfg = read(args.config)
    output = (args.output_dir or ROOT / cfg["output_dir"]).resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        if args.action == "_label":
            recheck(output)
            with exclusive_job(output / "labels"):
                label(cfg, args.config.resolve(), output, args.workers)
            return 0
        from scripts.arabic_diagnostic_common import completed_run

        with exclusive_job(output):
            source = completed_run(ROOT / cfg["source_run"])
            with stage_progress("Verify source identities and seal dynamic bilingual run"):
                if output.exists() and not (output / "contract.json").exists():
                    raise ValueError("existing unsealed output; choose a new directory")
                identity(cfg, args.config.resolve(), output, source)
            prepare_inputs(ROOT, cfg, output, source)
            frontend = read(ROOT / cfg["arabic_frontend_config"])
            env = {
                **os.environ,
                "CAMELTOOLS_DATA": str(ROOT / frontend["assets"]),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            subprocess.run(
                [
                    str(ROOT / frontend["runtime"] / "bin/python"),
                    "-u",
                    "-B",
                    str(Path(__file__).resolve()),
                    "_label",
                    "--config",
                    str(args.config.resolve()),
                    "--output-dir",
                    str(output),
                    "--workers",
                    str(args.workers),
                ],
                env=env,
                check=True,
            )
            prepare_english(ROOT, cfg, output)
            prepared = freeze(ROOT, cfg, output, args.io_workers)
            recheck(output)
            print(__import__("json").dumps(prepared, ensure_ascii=False, indent=2), flush=True)
            if args.action == "all":
                from qwen_hotword.training.en_egy_dynamic_train import run

                run(ROOT, cfg, output, source, args.gpu)
                recheck(output)
            with zipfile.ZipFile(
                output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for name in (
                    "contract.json",
                    "training_contract.json",
                    "admission_report.json",
                    "capacity_report.json",
                    "prepare_report.json",
                    "english_labels_report.json",
                    "feature_reuse.json",
                    "head/report.json",
                    "report.json",
                ):
                    if (output / name).exists():
                        archive.write(output / name, name)
                summary = read(output / "labels/report.json")
                summary.pop("pieces")
                write_json(output / "arabic_labels_summary.json", summary)
                archive.write(output / "arabic_labels_summary.json", "arabic_labels_summary.json")
            print(f"Return summary: {output / 'return_summary.zip'}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(
            __import__("json").dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
