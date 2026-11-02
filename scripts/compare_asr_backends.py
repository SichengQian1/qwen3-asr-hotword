#!/usr/bin/env python3
"""Install isolated vLLM0.23 and compare 30 shared audio inputs on an allocated H200."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("all", "status", "_worker"))
    parser.add_argument("--gpu", type=int, help="previously allocated physical H200 number")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/asr_backend_comparison_v023.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stage", choices=("probe", "prepare", "transformers", "vllm"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    from qwen_hotword.evaluation.backend_comparison import read

    if args.action == "_worker":
        if args.output_dir is None or args.stage is None:
            parser.error("internal worker needs output directory and stage")
        from qwen_hotword.evaluation.backend_compare_runtime import infer, prepare, probe

        cfg = read(args.output_dir / "run_contract.json")["config"]
        if args.stage == "probe":
            probe(args.output_dir)
        elif args.stage == "prepare":
            prepare(ROOT, cfg, args.output_dir)
        else:
            infer(args.stage, cfg, args.output_dir)
        return 0
    cfg = read(args.config.resolve())
    output = (args.output_dir or ROOT / cfg["output"]).resolve()
    if args.action == "status":
        report = read(output / "report.json") if (output / "report.json").exists() else None
        print(
            json.dumps(
                {
                    "output": str(output),
                    "report": report,
                    "requests_prepared": (output / "requests.json").exists(),
                    "saved_calls": {
                        name: len(list((output / name / "cells").glob("*.json")))
                        for name in ("transformers", "vllm")
                    },
                    "return_bundle": str(output / "return_bundle.zip"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.gpu is None or args.gpu < 0:
        parser.error("all requires --gpu N for an allocated physical H200 with enough free memory")
    from qwen_hotword.evaluation.backend_compare_run import run

    try:
        result = run(ROOT, cfg, output, args.gpu, Path(__file__).resolve())
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "return_bundle": str(output / "return_bundle.zip"),
                },
                ensure_ascii=False,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
