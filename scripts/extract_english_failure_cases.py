#!/usr/bin/env python3
"""Export reproducible English shortlist/gate failure examples without GPU inference."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.english_failure_cases import SEED, collect_cases, write_cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--old-run", default="outputs/reviewed_joint_target_retrieval_480h_v1")
    parser.add_argument("--new-run", default="outputs/reviewed_joint_4000_retrieval_480h_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output_dir or root / "outputs" / ("english_failure_cases_v1_" + uuid.uuid4().hex[:10])
    ).resolve()
    try:
        if output.exists():
            raise FileExistsError(f"refusing existing output: {output}")
        report, examples = collect_cases(
            root, (root / args.old_run).resolve(), (root / args.new_run).resolve(), args.seed
        )
        write_cases(report, examples, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, IndexError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(output),
                "example_count": report["example_count"],
                "unique_audio_count": report["unique_audio_count"],
                "by_dataset": {k: v["strata"] for k, v in report["by_dataset"].items()},
                "return_file": str(output / "return_bundle.zip"),
                "return_file_bytes": (output / "return_bundle.zip").stat().st_size,
                "model_loaded": False,
                "audio_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
