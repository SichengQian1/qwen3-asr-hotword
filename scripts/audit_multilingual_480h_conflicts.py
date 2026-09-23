#!/usr/bin/env python3
"""Read saved fingerprints and pending manifests to explain blocked 480h audio conflicts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.audio_conflict_audit import audit_conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/480h_conflict_audit.workzone.json"
    )
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = args.freeze_dir or Path(cfg["freeze_dir"])
    try:
        report = audit_conflicts(root, top_n=args.top_n, target_hours=cfg["target_hours"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Read-only conflict audit stopped: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
