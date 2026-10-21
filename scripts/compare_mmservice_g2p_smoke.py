#!/usr/bin/env python3
"""CPU-only unmodified MMService C++ versus official MFA Top1 smoke comparison."""

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
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mmservice_g2p_smoke.workzone.json")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    from qwen_hotword.phonemes.mmservice_smoke import run

    root = args.root.resolve()
    output = (
        root
        / (args.output_dir or Path("outputs") / ("mmservice_g2p_smoke_v1_" + uuid.uuid4().hex[:10]))
    ).resolve()
    try:
        mfa = shutil.which("mfa")
        if not mfa:
            raise FileNotFoundError("mfa unavailable; run in the existing aligner environment")
        result = run(
            root,
            root / args.config,
            root / args.binary,
            output,
            mfa=mfa,
            jobs=args.jobs,
            timeout=args.timeout,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        result = {"status": "failed", "error": str(error), "output_dir": str(output)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
