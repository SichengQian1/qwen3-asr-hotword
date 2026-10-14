#!/usr/bin/env python3
"""Verify existing Ziyan artifacts and package small identity/schema evidence on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.ziyan_identity import inspect, read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--config", type=Path, default=Path("configs/ziyan_english.workzone.json"))
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        config = (root / args.config).resolve()
        cfg = read_json(config)
        output = args.output_dir or Path("outputs") / ("ziyan_identity_v1_" + uuid.uuid4().hex[:10])
        result = inspect(
            root,
            config,
            root / (args.assets_dir or cfg["assets"]),
            root / (args.source_run or cfg["output"]),
            root / output,
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {"status": "failed", "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
