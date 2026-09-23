#!/usr/bin/env python3
"""Inspect wave1-4 inputs on H200's mounted storage; no GPU or model needed."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.wave_inventory import inspect_waves
from qwen_hotword.training.spanish_capacity import _sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir or REPO / "outputs" / f"wave_keyword_inventory_v1_{uuid.uuid4().hex[:10]}"
    )
    if output.exists():
        parser.error("output directory already exists; previous outputs must be preserved")
    result = inspect_waves(
        args.root.resolve(), REPO / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
    )
    output.mkdir(parents=True)
    report = output / "report.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "sha256.txt").write_text(f"{_sha(report)}  report.json\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_unions": result["primary_unions"],
                "dataset_counts": {
                    k: {f: v.get(f) for f in ("audio_count", "transcript_count", "error")}
                    for k, v in result["datasets"].items()
                },
                "issues": result["issues"],
                "tables_created": False,
                "model_loaded": False,
                "return_files": [str(report), str(output / "sha256.txt")],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
