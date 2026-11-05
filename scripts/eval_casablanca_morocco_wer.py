#!/usr/bin/env python3
"""Evaluate a local Qwen3-ASR checkpoint on both Casablanca Morocco shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation.backend_comparison import read
from qwen_hotword.evaluation.casablanca_wer import inspect_checkpoint, inspect_shards, run
from qwen_hotword.training.stage_progress import stage_progress


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="metadata only, no CUDA or inference")
    parser.add_argument("--gpu", type=int, help="allocated physical GPU number")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--language", choices=("auto", "Arabic"), default="auto")
    parser.add_argument("--model", type=Path, help="local HF or LoRA directory, or optimizer.pt")
    parser.add_argument("--base-model", type=Path, help="local HF base used to train the LoRA")
    parser.add_argument("--parquet", type=Path, nargs="+", help="override the input Parquet paths")
    parser.add_argument("--output-dir", type=Path, help="new output directory")
    args = parser.parse_args()
    if not args.check and (args.gpu is None or args.gpu < 0):
        parser.error("inference requires --gpu N")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")

    config = read(ROOT / "configs/casablanca_morocco_wer.workzone.json")
    try:
        with stage_progress("Check local model and Parquet metadata"):
            checkpoint = inspect_checkpoint(args.model or Path(config["model"]), args.base_model)
            shards = inspect_shards(args.parquet or [Path(path) for path in config["parquet"]])
        if args.check:
            result = {
                "status": "checked_metadata",
                "checkpoint": checkpoint,
                "shards": shards,
                "expected_samples": sum(shard["rows"] for shard in shards),
            }
        else:
            run_name = f"casablanca_morocco_wer_{uuid4().hex[:8]}"
            output = args.output_dir or ROOT / "outputs" / run_name
            result = run(
                checkpoint,
                shards,
                output.resolve(),
                args.gpu,
                args.batch_size,
                args.max_new_tokens,
                None if args.language == "auto" else args.language,
            )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
