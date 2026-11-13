#!/usr/bin/env python3
"""Base-ASR agreement on Arabic validation and deterministic per-source training samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import correlation, mean
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.evaluation.casablanca_wer import inspect_full_model, load_qwen
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.arabic_per_diagnostics import agreement_scores, agreement_summary
from qwen_hotword.training.stage_progress import stage_progress
from scripts.arabic_diagnostic_common import (
    completed_run,
    manifest_rows,
    protect_output,
    protected_run_directories,
    read,
    recheck,
    rows,
)
from scripts.label_arabic_native import exclusive_job


def select_samples(train: list[Any], validation: list[Any], per_source: int) -> list[Any]:
    if per_source <= 0:
        raise ValueError("per-source sample count must be positive")
    by_source: dict[str, list[Any]] = defaultdict(list)
    for row in train:
        by_source[row["source"]].append(row)
    selected = list(validation)
    for _, group in sorted(by_source.items()):
        selected.extend(sorted(group, key=lambda r: (r["audio_sha256"], r["id"]))[:per_source])
    if len({r["id"] for r in selected}) != len(selected) or len(
        {r["audio_sha256"] for r in selected}
    ) != len(selected):
        raise ValueError("duplicate or cross-split audio in agreement subset")
    return sorted(selected, key=lambda r: (r["split"], r["source"], r["audio_sha256"], r["id"]))


def run_predictions(model: Any, samples: list[Any], output: Path, batch_size: int) -> list[Any]:
    """Completed batches are atomically committed; a partial pending file is never reused."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    result = []
    with stage_progress(
        "Base Arabic ASR and text agreement", total=len(samples), every=batch_size
    ) as tick:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            path = output / "parts" / f"{start:06d}.json"
            if path.exists():
                predictions = read(path)
                if len(predictions) != len(batch):
                    raise ValueError("resume batch count changed")
                for old, current in zip(predictions, batch, strict=True):
                    if any(
                        old[k] != current[k]
                        for k in (
                            "id",
                            "split",
                            "source",
                            "source_row",
                            "audio_path",
                            "audio_sha256",
                            "text",
                        )
                    ):
                        raise ValueError("resume sample identity changed")
                    if agreement_scores(old["text"], old["asr_text"]) != {
                        k: old[k] for k in ("cer", "cer_counts", "wer", "wer_counts")
                    }:
                        raise ValueError("resume score mismatch")
            else:
                decoded = model.transcribe(
                    audio=[r["audio_path"] for r in batch],
                    language="Arabic",
                    return_time_stamps=False,
                )
                if len(decoded) != len(batch):
                    raise ValueError("ASR result count mismatch")
                predictions = [
                    {
                        **{k: v for k, v in row.items() if k != "phoneme_token_ids"},
                        "asr_text": value.text,
                        **agreement_scores(row["text"], value.text),
                    }
                    for row, value in zip(batch, decoded, strict=True)
                ]
                write_json(path, predictions)
            result.extend(predictions)
            tick(len(result))
    return result


def per_agreement(predictions: list[Any], per_path: Path) -> dict[str, Any]:
    per_rows = list(rows(per_path))
    lookup = {r["sample_id"]: r for r in per_rows}
    if len(lookup) != len(per_rows):
        raise ValueError("PER file must contain one checkpoint and unique sample IDs")
    paired = []
    for row in predictions:
        if row["split"] != "validation":
            continue
        per = lookup.get(row["id"])
        if per is None or per["text"] != row["text"] or per["audio_sha256"] != row["audio_sha256"]:
            raise ValueError("validation PER/text-agreement sample mismatch")
        paired.append((row["cer"], per["phoneme_error_rate"]))
    pearson = None
    if len(paired) > 2 and len({p[0] for p in paired}) > 1 and len({p[1] for p in paired}) > 1:
        pearson = correlation([p[0] for p in paired], [p[1] for p in paired])
    buckets = {}
    for lo, hi in ((0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, float("inf"))):
        part = [per for cer, per in paired if lo <= cer < hi]
        if part:
            buckets[f"{lo}-{hi}"] = {"samples": len(part), "mean_PER": mean(part)}
    return {
        "paired_validation_samples": len(paired),
        "pearson_CER_PER": pearson,
        "CER_buckets": buckets,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume for the identical selection/model")
    if args.resume and not (output / "contract.json").is_file():
        raise ValueError("resume requires a complete contract.json")
    run = completed_run(args.source_run)
    protect_output(output, protected_run_directories(run) + [args.model])
    with stage_progress("Select all Arabic validation plus per-source training samples"):
        samples = select_samples(
            manifest_rows(run, "train", ("ar",)),
            manifest_rows(run, "validation", ("ar",)),
            args.per_source,
        )
    with stage_progress("Verify local base model files (no inference yet)"):
        model = inspect_full_model(args.model.resolve())
        model_files = {
            p.name: sha(p)
            for p in args.model.iterdir()
            if p.is_file() and (p.suffix in (".json", ".jinja") or p.name == "merges.txt")
        }
        model_files.update({r["file"]: sha(args.model / r["file"]) for r in model["weights"]})
    contract = {
        "source_identity": run["identity"],
        "model": model,
        "model_files_sha256": model_files,
        "selection_sha256": hashlib.sha256(
            json.dumps(samples, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "samples": len(samples),
        "per_source": args.per_source,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "language": "Arabic",
        "hotwords": False,
        "per_sample_sha256": sha(args.per_sample) if args.per_sample else None,
        "code": {
            str(p): sha(p)
            for p in (
                Path(__file__),
                ROOT / "src/qwen_hotword/training/arabic_per_diagnostics.py",
                ROOT / "src/qwen_hotword/evaluation/casablanca_wer.py",
                ROOT / "scripts/arabic_diagnostic_common.py",
            )
        },
    }
    if args.resume:
        if read(output / "contract.json") != contract:
            raise ValueError("resume model, selection, source evidence or code changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "contract.json", contract)
    with stage_progress("Load frozen BASE Qwen for text agreement (no hotwords)"):
        qwen = load_qwen(str(args.model.resolve()), args.gpu, args.batch_size, args.max_new_tokens)
    predictions = run_predictions(qwen, samples, output, args.batch_size)
    del qwen
    with stage_progress("Recheck identities and summarize text agreement"):
        recheck(run)
        for name, digest in model_files.items():
            pin(args.model / name, digest)
        if args.per_sample:
            pin(args.per_sample, contract["per_sample_sha256"])
        # This aggregate is derived only from committed batches in this run.
        pending = output / "predictions.pending"
        with pending.open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        pending.replace(output / "predictions.jsonl")
        report = {
            "status": "completed",
            "samples": len(predictions),
            "contract_sha256": sha(output / "contract.json"),
            "source_identity": run["identity"],
            "model": model,
            "by_split_group_source": agreement_summary(predictions),
            "test_set_used": False,
            "validation_PER_agreement": per_agreement(predictions, args.per_sample)
            if args.per_sample
            else None,
            "interpretation": (
                "ASR-reference disagreement is a screening signal, not verified annotation error; "
                "training subset is per-source, not prevalence weighted."
            ),
        }
        write_json(output / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "outputs/en_ar_r15_5100_ctc_v1")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--per-source", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--per-sample", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.model is None:
        args.model = Path(
            yaml.safe_load((ROOT / "configs/480h_cache.workzone.yaml").read_text())["model"]["path"]
        )
    if args.gpu < 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        parser.error("invalid GPU/batch/token limit")
    with exclusive_job(args.output_dir.resolve()):
        report = execute(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
