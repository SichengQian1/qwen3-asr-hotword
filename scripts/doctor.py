#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.config import ConfigError, load_workzone_config
from qwen_hotword.diagnostics.environment import (
    collect_environment,
    overall_status,
    write_json_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the secure work-zone environment.")
    parser.add_argument("--config", required=True, help="Path to workzone.local.yaml")
    parser.add_argument("--output", help="Optional sanitized JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_workzone_config(args.config)
        report = collect_environment(config)
    except ConfigError as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return 2

    status = overall_status(report)
    report["overall_status"] = status
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if args.output:
        write_json_report(report, args.output)
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
