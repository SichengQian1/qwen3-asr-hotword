#!/usr/bin/env python3
"""CPU-only project-policy comparison reusing the sealed MMService smoke outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mmservice_g2p_pipeline.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--probe-worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("model", "words", "output"):
        parser.add_argument(f"--probe-{name}", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    from qwen_hotword.phonemes.mmservice_pipeline import probe_spanish, run

    try:
        if args.probe_worker:
            if not all((args.probe_model, args.probe_words, args.probe_output)):
                raise ValueError("probe worker requires model, words and new output")
            if args.probe_output.exists():
                raise FileExistsError("probe output already exists")
            result = probe_spanish(args.probe_model, args.probe_words, args.probe_output)
        else:
            root = args.root.resolve()
            output = (
                root
                / (
                    args.output_dir
                    or Path("outputs") / ("mmservice_g2p_pipeline_v1_" + uuid.uuid4().hex[:10])
                )
            ).resolve()
            mfa = shutil.which("mfa")
            if not mfa:
                raise FileNotFoundError("mfa unavailable; run in the existing aligner environment")
            result = run(
                root, root / args.config, output, mfa=mfa, jobs=args.jobs, timeout=args.timeout
            )
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        result = {"status": "failed", "error": str(error)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
