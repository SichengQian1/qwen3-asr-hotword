#!/usr/bin/env python3
"""Paired Head-only comparison on the completed pure-EGY/English run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.ctc_diagnostics import load_validation_sample_groups
from qwen_hotword.training.en_egy_dynamic import read, seal
from qwen_hotword.training.sharded_ctc import (
    exclusive_training_run,
    load_disk_feature_cache,
    train_sharded_ctc_head,
)
from qwen_hotword.training.stage_progress import stage_progress
from scripts.arabic_diagnostic_common import training_options
from scripts.label_arabic_native import exclusive_job
from scripts.run_en_ar_merged import wait_for_gpu


def source_contract(root: Path, source: Path, factor: int) -> tuple[dict[str, Any], Path]:
    report = read(source / "report.json")
    head = read(source / "head/report.json")
    prepared = read(source / "prepare_report.json")
    training = read(source / "training_contract.json")
    cfg = read(source / "contract.json")["configuration"]
    if report["status"] != "completed" or head["status"] != "completed":
        raise ValueError("new EGY source training is not completed")
    if report["head"] != head or report["selection"] != prepared:
        raise ValueError("source reports disagree")
    if report["test_set_used"] is not False or prepared["arabic_msa_used"] is not False:
        raise ValueError("requires the new pure-EGY run, not the old EGY+MSA run")
    if head["head_config"]["time_upsampling_factor"] != 2:
        raise ValueError("paired source must be the x2 baseline")
    vocab = root / cfg["vocab"]
    pin(vocab, report["vocab_sha256"])
    if prepared["vocab_sha256"] != report["vocab_sha256"]:
        raise ValueError("source vocabulary identities disagree")
    pin(source / "head/ctc_head_best.pt", report["checkpoint_sha256"])
    pin(source / "prepare_report.json", training["prepare_report_sha256"])
    model = training["encoder"]
    if model["files"]["model.safetensors"]["sha256"] != report["model_weight_sha256"]:
        raise ValueError("source Encoder identities disagree")
    files = {
        str(source / name): sha(source / name)
        for name in (
            "report.json",
            "head/report.json",
            "prepare_report.json",
            "contract.json",
            "training_contract.json",
            "head/ctc_head_best.pt",
        )
    }
    files[str(vocab)] = sha(vocab)
    for split in ("train", "validation"):
        item = prepared["manifests"][split]
        path = source / f"{split}.jsonl"
        if path.resolve() != Path(item["path"]).resolve():
            raise ValueError("source manifest must belong to selected new run")
        pin(path, item["sha256"])
        files[str(path)] = item["sha256"]
        folder = source / "cache" / split
        cached = read(folder / "cache_config.json")
        expected = {
            "path": model["view"],
            "dtype": "bfloat16",
            "config_sha256": model["files"]["config.json"]["sha256"],
            "weight_index_sha256": model["index_sha256"],
        }
        if cached["model"] != expected:
            raise ValueError("cache belongs to a different Encoder")
        for name in ("cache_config.json", "cache_index.json", "cache_summary.json"):
            files[str(folder / name)] = sha(folder / name)
    return {
        "source_run": str(source),
        "source_files": files,
        "training_options": training_options(head, factor),
        "model_weight_sha256": report["model_weight_sha256"],
        "vocab_sha256": report["vocab_sha256"],
        "initialization": "fresh_random_head",
        "encoder_recomputed": False,
        "arabic_msa_used": False,
        "test_set_used": False,
        "code": {
            str(p): sha(p)
            for p in (
                Path(__file__),
                root / "scripts/arabic_diagnostic_common.py",
                root / "src/qwen_hotword/training/sharded_ctc.py",
                root / "src/qwen_hotword/modeling/ctc_head.py",
            )
        },
    }, vocab


def run(root: Path, source: Path, output: Path, gpu: int, factor: int = 3) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("comparison output must be separate from the source run")
    with stage_progress("Verify completed NEW EGY x2 run, labels and Encoder identity"):
        contract, vocab = source_contract(root, source, factor)
    if output.exists() and not (output / "contract.json").exists():
        raise ValueError("existing unsealed output; preserve it and choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_job(output):
        seal(output / "contract.json", contract)
        if (output / "report.json").exists():
            report = read(output / "report.json")
            pin(output / "head/ctc_head_best.pt", report["checkpoint_sha256"])
            return dict(report)
        caches = {}
        for split in ("train", "validation"):
            with stage_progress(f"Verify {split} cache SHA256; no Encoder execution"):
                caches[split] = load_disk_feature_cache(
                    source / "cache" / split,
                    expected_split=split,
                    source_manifest_path=source / f"{split}.jsonl",
                    vocab_path=vocab,
                    verify_sha256=True,
                )
        groups = load_validation_sample_groups(
            source / "validation.jsonl",
            caches["validation"],
            group_column="language",
            expected_groups=("en", "ar"),
        )
        train_ids = {s for shard in caches["train"].shards for s in shard.sample_ids}
        if train_ids.intersection(groups):
            raise ValueError("train/validation sample overlap")
        head = output / "head"
        state = head / "training_state_latest.pt"
        if not state.exists() and any(
            (head / name).exists() for name in ("metrics.jsonl", "ctc_head_best.pt")
        ):
            raise ValueError("partial Head without resumable training state")
        wait_for_gpu(gpu, 20)
        with exclusive_training_run(head), stage_progress(f"Train NEW EGY paired x{factor} Head"):
            if not (head / "report.json").exists():
                train_sharded_ctc_head(
                    caches["train"],
                    caches["validation"],
                    load_phoneme_vocab(vocab),
                    head,
                    vocab_path=vocab,
                    device="cuda:0",
                    validation_sample_groups=groups,
                    resume=state.exists(),
                    **contract["training_options"],
                )
        for path, digest in {**contract["source_files"], **contract["code"]}.items():
            pin(Path(path), digest)
        head_report = read(head / "report.json")
        if head_report["status"] != "completed":
            raise ValueError("Head training not completed")
        report = {
            "status": "completed",
            "source_run": str(source),
            "physical_gpu": gpu,
            "head_time_upsampling_factor": factor,
            "encoder_recomputed": False,
            "contract_sha256": sha(output / "contract.json"),
            "head": head_report,
            "checkpoint_sha256": sha(head / "ctc_head_best.pt"),
            "test_set_used": False,
            "paired_manifest_sha256": {
                split: contract["source_files"][str(source / f"{split}.jsonl")]
                for split in ("train", "validation")
            },
        }
        write_json(output / "report.json", report)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "outputs/en_egy_480_ctc_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/en_egy_480_head_up3_v1")
    parser.add_argument("--gpu", type=int, default=5)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        result = run(ROOT, args.source_run, args.output_dir, args.gpu)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
