#!/usr/bin/env python3
"""Pinned three-language MFA foundation and PT listening audit; CPU only, no training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.phonemes.mfa_v03_foundation import acquire_lock, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("all", "download", "setup", "smoke", "audit", "review", "worker")
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/mfa_v03_foundation.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--asset-archive", type=Path, help="optional pinned offline G2P asset ZIP")
    parser.add_argument("--mfa-python", help="optional existing MFA-3.4.0/3.4.3 Python; read only")
    parser.add_argument("--request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["_config_path"] = str(args.config.resolve())
    assets = (args.assets_dir or ROOT / config["assets_dir"]).resolve()
    if args.action == "worker":
        from qwen_hotword.phonemes.mfa_v03_runtime import run as worker

        if args.request is None or args.worker_output is None:
            parser.error("worker requires request and output")
        worker(config, assets, ROOT, args.request, args.worker_output)
        return 0
    output = (args.output_dir or ROOT / config["output"]).resolve()
    try:
        descriptor, lock = acquire_lock(output)
    except FileExistsError:
        parser.exit(
            1,
            f"Another/interrupted foundation process owns {output}.lock; inspect its PID first.\n",
        )
    try:
        report = run(
            ROOT,
            config,
            output,
            assets,
            args.action,
            mfa_python=args.mfa_python,
            asset_archive=args.asset_archive,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Foundation stopped; existing assets/outputs preserved: {error}\n")
    finally:
        os.close(descriptor)
        lock.unlink()
    summary = {
        k: report[k]
        for k in (
            "status",
            "scope",
            "output",
            "assets",
            "gpu_used",
            "encoder_loaded",
            "environment_modified",
            "test_set_used",
            "human_review",
            "training_started",
            "return_bundle",
        )
    }
    if report.get("error"):
        summary["error"] = report["error"]
    if "review" in report["stages"]:
        summary["review"] = report["stages"]["review"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
