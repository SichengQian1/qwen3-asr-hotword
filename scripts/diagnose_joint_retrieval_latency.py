#!/usr/bin/env python3
"""Attribute saved EN6000 / ES400 / PT400 joint-table prefix retrieval latency on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.joint_latency import analyze_run, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step"),
    )
    parser.add_argument(
        "--assets", type=Path, default=Path("outputs/reviewed_joint_4000_retrieval_480h_v1")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run, assets = (root / args.run).resolve(), (root / args.assets).resolve()
    output = (
        root / (args.output_dir or Path(f"outputs/joint_retrieval_latency_{uuid.uuid4().hex[:10]}"))
    ).resolve()
    try:
        # Reject overlaps before consuming any large prefix inventory.
        if output.exists():
            raise FileExistsError(f"refusing existing output: {output}")
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (run, assets)):
            raise ValueError("diagnostic output must be separate from source evidence")
        report = analyze_run(run, assets)
        summary = write_report(report, output, (run, assets))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Return only: {output / 'return_bundle.zip'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
