#!/usr/bin/env python3
"""Read-only metadata check; prints compact JSON and never writes output files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.pt_validation_metadata import (
    audit_pt_validation_metadata,
    inspect_mfa_environment,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pt_validation_metadata.workzone.json",
    )
    parser.add_argument(
        "--mfa-inventory", action="store_true",
        help="Inspect the active environment and local MFA assets; no corpus reads or MFA run.",
    )
    args = parser.parse_args()
    if args.mfa_inventory:
        try:
            report = inspect_mfa_environment(REPO_ROOT)
        except OSError as error:
            parser.error(str(error))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        required = {
            "validation_manifest", "validation_sha256", "common_voice_root", "fleurs_train_tsv"
        }
        if not isinstance(config, dict) or set(config) != required:
            raise ValueError(f"Config must contain exactly: {sorted(required)}")
        if any(not isinstance(value, str) or not value for value in config.values()):
            raise ValueError("Config values must be nonempty strings")
        report = audit_pt_validation_metadata(
            validation_manifest=REPO_ROOT / config["validation_manifest"],
            expected_sha256=config["validation_sha256"],
            common_voice_root=REPO_ROOT / config["common_voice_root"],
            fleurs_train_tsv=REPO_ROOT / config["fleurs_train_tsv"],
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
