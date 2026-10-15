#!/usr/bin/env python3
"""Reuse completed labels, skip WAV-byte scans, and run unchanged CTC training stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "cache", "pilot", "formal"))
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/english_countries_refresh.workzone.json")
    )
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/english_countries_refresh_fast_v1")
    )
    parser.add_argument("--gpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage != "prepare" and (args.gpu is None or not args.gpu.isdecimal()):
        parser.error("choose one allocated physical GPU with --gpu N")
    if args.resume and args.stage != "pilot":
        parser.error("--resume only applies to pilot; fast prepare requires a NEW directory")
    from qwen_hotword.training.english_refresh_fast import prepare, run_stage

    try:
        root = args.root.resolve()
        cfg = json.loads((root / args.config).read_text())
        output = (root / args.output_dir).resolve()
        result = (
            prepare(root, cfg, (root / (args.prepared_dir or cfg["output"])).resolve(), output)
            if args.stage == "prepare"
            else run_stage(root, output, args.stage, args.gpu or "", resume=args.resume)
        )
        if args.stage == "prepare":
            result = {
                k: result[k]
                for k in (
                    "status",
                    "selection",
                    "audio_isolation_policy",
                    "audio_byte_audit_performed",
                    "output_dir",
                )
            }
            result["return_file"] = str(output / "return_bundle.zip")
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
        subprocess.CalledProcessError,
    ) as error:
        result = {"status": "failed", "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
