#!/usr/bin/env python3
"""Prepare fresh MFA targets, audit JSON audio, or run fixed joint Top5/Top7 retrieval."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "audit", "run"))
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--config", type=Path, default=Path("configs/ziyan_english.workzone.json"))
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "run" and (args.gpu is None or args.gpu < 0):
        parser.error("run requires --gpu N for a free physical GPU")
    if args.resume and args.stage != "run":
        parser.error("--resume applies only to identical retrieval runs")
    if args.stage == "run":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from qwen_hotword.evaluation.ziyan_assets import build
    from qwen_hotword.inference.ziyan_retrieval import run

    try:
        root = args.root.resolve()
        config = (root / args.config).resolve()
        cfg = json.loads(config.read_text())
        assets = (root / (args.assets_dir or cfg["assets"])).resolve()
        output = (root / (args.output_dir or cfg["output"])).resolve()
        if args.stage == "prepare":
            mfa = shutil.which("mfa")
            if not mfa:
                raise FileNotFoundError("mfa not found; use conda run -n aligner for prepare")
            report = build(root, config, assets, args.jobs, mfa)
        else:
            report = run(
                root, config, assets, output, resume=args.resume, audit_only=args.stage == "audit"
            )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        report = {"status": "failed", "error": str(error)}
    summary = report
    if report["status"] == "completed":
        summary = {
            "status": report["status"],
            "report_file": str((assets if args.stage == "prepare" else output) / "report.json"),
            "target_presence": report["target_presence"],
        }
        if args.stage == "prepare":
            summary.update(table_ready=report["table_ready"], selection=report["selection"])
        else:
            fields = (
                "expected_target_sample_pairs",
                "gated_target_hits",
                "gated_target_recall",
                "raw_target_hits",
                "raw_target_recall",
                "selected_all_words",
                "selected_words_absent_from_transcript",
            )
            summary.update(
                sample_count=report["sample_count"],
                keyword_count=report["keyword_count"],
                metrics={k: {f: v[f] for f in fields} for k, v in report["metrics"].items()},
                delivery_directory=str(output / "delivery"),
                delivery_file_count=2,
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
