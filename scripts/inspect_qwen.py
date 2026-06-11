#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.config import ConfigError, load_workzone_config
from qwen_hotword.modeling.qwen_backbone import (
    ModelValidationError,
    inspect_local_config,
    inspection_dict,
    validate_inspection_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local Qwen3-ASR-1.7B config.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "configs/models/qwen3-asr-1.7b.yaml"),
    )
    args = parser.parse_args()
    try:
        config = load_workzone_config(args.config, require_existing_model=True)
        inspection = inspect_local_config(config.model)
    except (ConfigError, ModelValidationError, OSError, ValueError) as error:
        print(f"MODEL INSPECTION FAILED: {error}", file=sys.stderr)
        return 1
    mismatches = validate_inspection_manifest(inspection, args.manifest)
    report = {
        "inspection": inspection_dict(inspection),
        "manifest_validation": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
