#!/usr/bin/env python3
"""Run real paired ASR on GPU, resume saved calls and package 50 recovered-hotword WAVs."""

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
        "--gpu", type=int, required=True, choices=range(8), help="allocated physical GPU"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # GPU selection precedes any module that could import Torch.
    from qwen_hotword.evaluation.recovered_audio_export import TABLE_RELATIVE
    from qwen_hotword.evaluation.recovered_case_assets import SOURCE_RUN
    from qwen_hotword.evaluation.recovered_case_run import run_case_hunt

    root = args.root.resolve()
    output = (args.output_dir or root / "outputs/recovered_hotword_cases_latest_v1").resolve()
    try:
        summary = run_case_hunt(
            root,
            (args.source_run or root / SOURCE_RUN).resolve(),
            (args.table or root / TABLE_RELATIVE).resolve(),
            output,
            count=args.count,
        )
    except KeyboardInterrupt:
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "output": str(output),
                    "resume": "repeat the identical command; completed ASR calls are retained",
                }
            )
        )
        return 130
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(error),
                    "output": str(output),
                    "progress": str(output / "progress.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
