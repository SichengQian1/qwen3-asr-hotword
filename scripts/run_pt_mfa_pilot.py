#!/usr/bin/env python3
"""Run the fixed Portuguese MFA pilot on H200; never trains a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.pt_mfa_runner import run_pilot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/pt_mfa_pilot.workzone.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Must not exist; default creates a fresh run."
    )
    parser.add_argument("--assets-dir", type=Path, help="Previously downloaded MFA asset directory")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = run_pilot(config, REPO_ROOT, args.output_dir, args.assets_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
