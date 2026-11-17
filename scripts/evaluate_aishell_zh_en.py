#!/usr/bin/env python3
"""Evaluate approved AISHELL 1054-word Top5/Top7 with the completed Chinese/US/UK Head."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/aishell_retrieval.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=4, help="Allocated physical GPU; default 4")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU ID must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from qwen_hotword.evaluation.aishell_run import run
    from qwen_hotword.training.zh_en_common import read

    try:
        output = (args.output_dir or ROOT / read(args.config)["output"]).resolve()
        result = run(ROOT, args.config.resolve(), output, args.gpu)
    except (Exception, KeyboardInterrupt) as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    brief = {k: v for k, v in result.items() if k != "metrics"}
    if "metrics" in result:
        brief["metrics"] = {
            mode: {k: v for k, v in values.items() if k != "by_word"}
            for mode, values in result["metrics"].items()
        }
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
