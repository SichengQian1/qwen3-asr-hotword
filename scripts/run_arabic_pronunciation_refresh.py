#!/usr/bin/env python3
"""Relabel Arabic with traceable rules, export review evidence, freeze and train on r15."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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

from qwen_hotword.phonemes.arabic_pronunciation import PronunciationFrontend, validate_policy
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.stage_progress import stage_progress
from scripts.label_arabic_native import (
    CONFIG,
    SOURCES,
    VOCAB,
    exclusive_job,
    label_part,
    verify_assets,
)

POLICY = ROOT / "configs/arabic_pronunciation_policy.json"
CONFIG_RUN = ROOT / "configs/arabic_pronunciation_refresh.workzone.json"
FRONTEND: PronunciationFrontend | None = None


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def label_job(job: tuple[str, int, list[dict[str, Any]], str, str]) -> dict[str, Any]:
    global FRONTEND
    group, number, rows, destination, policy_path = job
    config, policy = read(CONFIG), read(Path(policy_path))
    tokens = read(VOCAB)["tokens"]
    if FRONTEND is None or FRONTEND.group != group:
        FRONTEND = PronunciationFrontend(ROOT / config["assets"], group, tokens, policy)
    piece, _ = label_part(
        config,
        Path(destination),
        group,
        rows,
        number,
        tokens,
        FRONTEND,
        {"vocab_sha256": sha(VOCAB)},
    )
    return piece


def label(output: Path, policy_path: Path, workers: int, resume: bool) -> dict[str, Any]:
    config, policy = read(CONFIG), read(policy_path)
    validate_policy(policy)
    if importlib.metadata.version("camel-tools") != config["camel_tools_version"]:
        raise ValueError("CAMeL version changed")
    verify_assets(config)
    files = [
        CONFIG,
        VOCAB,
        policy_path,
        Path(__file__),
        ROOT / "scripts/label_arabic_native.py",
        ROOT / "src/qwen_hotword/phonemes/arabic_native.py",
        ROOT / "src/qwen_hotword/phonemes/arabic_pronunciation.py",
        *SOURCES.values(),
    ]
    for path in files:
        if output == path.resolve() or output in path.resolve().parents:
            raise ValueError("output overlaps protected input")
    with stage_progress("Seal full Arabic relabel inputs; no WAV reads"):
        identity = {str(path.resolve()): sha(path) for path in files}
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if read(output / "contract.json") != identity:
            raise ValueError("relabel inputs or code changed; choose a new output directory")
    else:
        write_json(output / "contract.json", identity)
    pieces = []
    with (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        stage_progress(f"Relabel all Arabic ({workers} processes); completed source rows") as tick,
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
                print(f"Completed {count:,} rows; {piece['group']} {piece['labels']}", flush=True)

        for group, source in SOURCES.items():
            batch, number = [], 0
            for row in iter_jsonl(source):
                if row["group"] != group:
                    continue
                batch.append(row)
                if len(batch) == 2000:
                    pending.add(
                        pool.submit(
                            label_job, (group, number, batch, str(output), str(policy_path))
                        )
                    )
                    batch, number = [], number + 1
                    if len(pending) >= workers * 2:
                        drain()
            if batch:
                pending.add(
                    pool.submit(label_job, (group, number, batch, str(output), str(policy_path)))
                )
        while pending:
            drain()
    pieces.sort(key=lambda p: p["labels"])
    totals = {}
    with stage_progress("Assemble deterministic labels and verify input identities"):
        pending_path = output / "labels.pending"
        with pending_path.open("wb") as target:
            for piece in pieces:
                with (output / piece["labels"]).open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
        destination = output / "labels.jsonl"
        if destination.exists():
            pin(destination, sha(pending_path))
            pending_path.unlink()
        else:
            pending_path.replace(destination)
        for group in SOURCES:
            selected = [p for p in pieces if p["group"] == group]
            totals[group] = {
                "source_rows": sum(p["rows"] for p in selected),
                "ready_rows": sum(p["ready_rows"] for p in selected),
                "ready_hours": sum(p["ready_hours"] for p in selected),
                "reasons": dict(sum((Counter(p["reasons"]) for p in selected), Counter())),
            }
        for input_name, digest in identity.items():
            pin(Path(input_name), digest)
    report = {
        "status": "automatic_labels_completed",
        "labels_sha256": sha(destination),
        "by_group": totals,
        "vocab_sha256": sha(VOCAB),
        "num_classes": 167,
        "english_prefix_unchanged": True,
        "identity": identity,
        "policy": policy,
        "workers": workers,
        "audio_read": False,
        "human_pronunciation_review_completed": False,
        "training_started": False,
        "review_directory": str(output / "review_delivery"),
        "return_file": str(output / "return_summary.zip"),
    }
    write_json(output / "report.json", report)
    from qwen_hotword.training.arabic_refresh import export_review

    export_review(output)
    with zipfile.ZipFile(output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "report.json",
            "contract.json",
            "review_delivery/summary.json",
            "review_delivery/blocking_issues.json",
        ):
            archive.write(output / name, name)
        archive.write(policy_path, "pronunciation_policy.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("label", "_label", "query", "_query", "freeze", "run", "review")
    )
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--config", type=Path, default=CONFIG_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--factor", type=int, choices=(2, 3), default=2)
    parser.add_argument("--text")
    parser.add_argument("--group", choices=("ar_egy", "ar_msa"), default="ar_egy")
    parser.add_argument("--mode", choices=("sentence", "isolated", "linked"), default="isolated")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("workers must be 1..64")
    config = read(args.config)
    output = (args.output_dir or ROOT / config["output_dir"]).resolve()
    labels = (args.labels or ROOT / config["labels_dir"]).resolve()
    if args.action in {"label", "_label"} and args.output_dir is None:
        output = labels
    if args.action in {"label", "query"}:
        runtime = read(CONFIG)
        command = [
            str(ROOT / runtime["runtime"] / "bin/python"),
            "-u",
            "-B",
            __file__,
            "_" + args.action,
            *sys.argv[2:],
        ]
        env = {
            **os.environ,
            "CAMELTOOLS_DATA": str(ROOT / runtime["assets"]),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        return subprocess.run(command, env=env).returncode
    try:
        if args.action == "_query":
            if not args.text:
                raise ValueError("query requires --text")
            runtime = read(CONFIG)
            verify_assets(runtime)
            frontend = PronunciationFrontend(
                ROOT / runtime["assets"], args.group, read(VOCAB)["tokens"], read(args.policy)
            )
            result = frontend.label(args.text, mode=args.mode)
        elif args.action == "_label":
            with exclusive_job(output):
                result = label(output, args.policy.resolve(), args.workers, args.resume)
        else:
            from qwen_hotword.training.arabic_refresh import export_review, freeze

            if args.action == "review":
                result = export_review(labels)
            else:
                with exclusive_job(output):
                    if args.action == "freeze":
                        result = freeze(ROOT, args.config.resolve(), labels, output, args.resume)
                    else:
                        if args.gpu is None or args.gpu < 0:
                            raise ValueError("run requires --gpu N")
                        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
                        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                        from qwen_hotword.training.arabic_refresh_train import run

                        result = run(ROOT, output, args.gpu, args.factor)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
