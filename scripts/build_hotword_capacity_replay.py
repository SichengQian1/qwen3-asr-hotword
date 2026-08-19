#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.capacity_replay import (
    build_offline_capacity_replay,
    build_streaming_capacity_replay,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable offline or causal 2-second CTC decoded replay input "
            "for Portuguese hotword capacity benchmarks."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "streaming"),
        required=True,
    )
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--ctc-checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-cache")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-cache-sha256-verification", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--language", default="Portuguese")
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--save-log-posteriors",
        action="store_true",
        help=(
            "For streaming mode, also save causal frame-level log-softmax values "
            "as validated float16 tensor shards."
        ),
    )
    parser.add_argument("--posterior-shard-size", type=int, default=32)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "offline":
            if not args.validation_cache:
                parser.error("--mode offline requires --validation-cache")
            summary = build_offline_capacity_replay(
                validation_cache_path=args.validation_cache,
                validation_manifest_path=args.validation_manifest,
                vocab_path=args.vocab,
                checkpoint_path=args.ctc_checkpoint,
                cases_path=args.cases,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                verify_cache_sha256=not args.skip_cache_sha256_verification,
                print_progress=not args.quiet_progress,
            )
        else:
            if not args.model:
                parser.error("--mode streaming requires --model")
            summary = build_streaming_capacity_replay(
                model_path=args.model,
                validation_manifest_path=args.validation_manifest,
                vocab_path=args.vocab,
                checkpoint_path=args.ctc_checkpoint,
                cases_path=args.cases,
                output_dir=args.output_dir,
                device=args.device,
                dtype=args.dtype,
                language=args.language,
                chunk_size_sec=args.chunk_size_sec,
                max_samples=args.max_samples,
                save_log_posteriors=args.save_log_posteriors,
                posterior_shard_size=args.posterior_shard_size,
                print_progress=not args.quiet_progress,
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
