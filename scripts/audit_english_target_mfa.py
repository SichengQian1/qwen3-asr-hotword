#!/usr/bin/env python3
"""Compare retained English target labels with fresh MFA and the training dictionary."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.english_target_mfa import run_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/english_target_mfa_audit.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        root
        / (
            args.output_dir
            or Path("outputs") / ("english_target_mfa_audit_v1_" + uuid.uuid4().hex[:10])
        )
    ).resolve()
    try:
        mfa = shutil.which("mfa")
        if not mfa:
            raise FileNotFoundError("mfa not found; use conda run -n aligner")
        report = run_audit(root, (root / args.config).resolve(), output, args.jobs, mfa)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report = {"status": "failed", "error": str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
