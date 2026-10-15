#!/usr/bin/env python3
"""Rebuild the pruned joint index and replay <=4-target Ziyan samples on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ziyan_control_le4_drop37.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from qwen_hotword.evaluation.ziyan_control import run
    from qwen_hotword.evaluation.ziyan_identity import digest

    try:
        root = args.root.resolve()
        config = root / args.config
        cfg = json.loads(config.read_text(encoding="utf-8"))
        output = (root / (args.output_dir or cfg["output"])).resolve()
        report = run(root, config, output, resume=args.resume)
        summary = {
            k: report[k]
            for k in (
                "status",
                "selection",
                "index_sizes",
                "baseline_same_subset_metrics",
                "pruned_same_subset_metrics",
                "paired_delta",
            )
        }
        summary.update(
            report_file=str(output / "report.json"),
            delivery_directory=str(output / "delivery"),
            return_file=str(output / "return_bundle.zip"),
            return_sha256=digest(output / "return_bundle.zip"),
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
