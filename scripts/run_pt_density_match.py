#!/usr/bin/env python3
"""Build density-matched validation on CPU; evaluate existing Heads only on H200."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.training.pt_density_match import build_subset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "evaluate", "audit-cache"))
    parser.add_argument(
        "--config", type=Path, default=REPO / "configs/pt_es_density_match.workzone.json"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    try:
        if args.stage == "build":
            report = build_subset(config, REPO)
        elif args.stage == "audit-cache":
            from qwen_hotword.training.pt_density_evaluation import audit_reference_cache

            report = audit_reference_cache(config, REPO)
        else:
            from qwen_hotword.training.pt_density_evaluation import evaluate_subset

            report = evaluate_subset(config, REPO)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Density experiment stopped; existing outputs preserved: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"matched", "completed", "consistent"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
