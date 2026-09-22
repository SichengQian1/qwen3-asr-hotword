#!/usr/bin/env python3
"""Read existing Spanish metadata and bound explicit Latin-American train capacity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.spanish_capacity import audit_spanish_capacity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-root", type=Path, default=REPO / "outputs/es_candidate_inventory_v1"
    )
    parser.add_argument(
        "--pool-root", type=Path, default=REPO / "outputs/es_combined_temporal2x_v1"
    )
    parser.add_argument("--target-hours", type=float, default=480.0)
    args = parser.parse_args()
    try:
        report = audit_spanish_capacity(
            args.inventory_root, args.pool_root, target_hours=args.target_hours
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Capacity audit stopped; files unchanged: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # An honest shortage is a completed inventory result, not a processing failure.
    return 1 if report["status"] == "blocked_existing_scope_audit" else 0


if __name__ == "__main__":
    raise SystemExit(main())
