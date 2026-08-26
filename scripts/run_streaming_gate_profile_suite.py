#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.inference.streaming_gate_suite import (
    STREAMING_GATE_PROFILES,
    write_streaming_gate_suite_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the sealed five-profile 4k streaming prompt-filter suite."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--hotword-families", required=True)
    parser.add_argument("--ctc-report", required=True)
    parser.add_argument("--offline-rag-dir", required=True)
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--language", default="Portuguese")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser()
    if output.exists() and not args.resume:
        parser.error(f"output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "status": "running",
        "test_set_used": False,
        "profiles": [profile.__dict__ for profile in STREAMING_GATE_PROFILES],
        "anchor": {
            "shortlist_size": 64,
            "start_radius": 2,
            "ngram_sizes": [2, 3, 4],
            "anchors_per_entry": 24,
            "offset_tolerance": 1,
        },
        "inputs": {
            key: str(getattr(args, key))
            for key in (
                "model",
                "validation_manifest",
                "vocab",
                "hotwords",
                "cases",
                "hotword_families",
                "ctc_report",
                "offline_rag_dir",
                "ctc_checkpoint",
            )
        },
        "max_samples": args.max_samples,
        "runtime": {
            "language": args.language,
            "dtype": args.dtype,
            "device": args.device,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_new_tokens": args.max_new_tokens,
            "chunk_size_sec": 2.0,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
        },
    }
    config_path = output / "suite_config.json"
    previous = (
        json.loads(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else None
    )
    if args.resume and previous is not None:
        comparable_previous = dict(previous)
        comparable_previous["status"] = "running"
        if comparable_previous != config:
            parser.error("resume suite configuration differs from the existing run")
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_specs = (
        ("baseline_oracle", "C,E", 0.86, 5, 0.0),
        ("conservative", "D", 0.86, 5, 0.0),
        ("balanced", "D", 0.82, 7, 0.5),
        ("recall_first", "D", 0.75, 7, 0.5),
    )
    for subdir, groups, threshold, top_k, minimum_posterior in run_specs:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/run_streaming_rag_evaluation.py"),
            "--model",
            args.model,
            "--validation-manifest",
            args.validation_manifest,
            "--vocab",
            args.vocab,
            "--hotwords",
            args.hotwords,
            "--cases",
            args.cases,
            "--hotword-families",
            args.hotword_families,
            "--ctc-report",
            args.ctc_report,
            "--offline-rag-dir",
            args.offline_rag_dir,
            "--offline-format",
            "multi_nested_v3",
            "--offline-control-mode",
            "selection_only",
            "--ctc-checkpoint",
            args.ctc_checkpoint,
            "--output-dir",
            str(output / subdir),
            "--groups",
            groups,
            "--chunk-size-sec",
            "2.0",
            "--unfixed-chunk-num",
            "2",
            "--unfixed-token-num",
            "5",
            "--retrieval-backend",
            "anchor_guided",
            "--anchor-shortlist-size",
            "64",
            "--anchor-start-radius",
            "2",
            "--anchor-ngram-sizes",
            "2,3,4",
            "--anchors-per-entry",
            "24",
            "--anchor-offset-tolerance",
            "1",
            "--threshold",
            str(threshold),
            "--top-k",
            str(top_k),
            "--maximum-edit-ratio",
            "0.35",
            "--posterior-weight",
            "0.25",
            "--minimum-posterior-confidence",
            str(minimum_posterior),
            "--minimum-top1-margin",
            "0",
            "--language",
            args.language,
            "--dtype",
            args.dtype,
            "--device",
            args.device,
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
        if args.max_samples:
            command.extend(("--max-samples", str(args.max_samples)))
        if args.max_new_tokens is not None:
            command.extend(("--max-new-tokens", str(args.max_new_tokens)))
        if args.resume:
            command.append("--resume")
        if args.quiet_progress:
            command.append("--quiet-progress")
        subprocess.run(command, check=True)

    config["status"] = "pass"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = write_streaming_gate_suite_report(output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
