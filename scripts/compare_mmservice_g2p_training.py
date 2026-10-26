#!/usr/bin/env python3
"""Audit C++ G2P against every actual training word and frozen sentence label."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.phonemes.mmservice_training import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mmservice_g2p_training.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--release-config", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / (
        args.output_dir or Path("outputs") / ("mmservice_g2p_training_" + uuid.uuid4().hex[:10])
    )
    try:
        result = run(
            root,
            root / args.config,
            output,
            timeout=args.timeout,
            release_config=args.release_config,
            binary=args.binary,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        result = {"status": "failed", "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
