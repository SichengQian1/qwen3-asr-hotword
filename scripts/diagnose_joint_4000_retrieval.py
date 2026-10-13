#!/usr/bin/env python3
"""Compare saved reviewed-only and expanded joint retrieval results on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qwen_hotword.evaluation.joint_diagnostics import diagnose_joint_runs, write_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--old-run", default="outputs/reviewed_joint_target_retrieval_480h_v1")
    parser.add_argument("--new-run", default="outputs/reviewed_joint_4000_retrieval_480h_v1")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir
        or args.root / "outputs" / f"joint_4000_diagnostics_v1_{uuid.uuid4().hex[:10]}"
    ).resolve()
    try:
        if output.exists():
            raise FileExistsError(f"refusing existing output: {output}")
        report, examples = diagnose_joint_runs(
            (args.root / args.old_run).resolve(), (args.root / args.new_run).resolve()
        )
        write_diagnostics(report, examples, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False, indent=2))
        return 1
    summary = json.loads((output / "summary.json").read_text())
    print(
        json.dumps(
            {
                **summary,
                "output_dir": str(output),
                "return_files": [
                    str(output / name) for name in ("summary.json", "examples.json", "sha256.txt")
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
