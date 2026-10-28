#!/usr/bin/env python3
"""Find saved no-hotword/retrieved-ASR gains and package original WAVs plus Markdown."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from qwen_hotword.evaluation.recovered_audio_export import TABLE_RELATIVE, export_saved_cases

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--scan-root", action="append", type=Path)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (
        args.output_dir or root / "outputs" / f"hotword_recovered_cases_{uuid.uuid4().hex[:10]}"
    )
    try:
        summary = export_saved_cases(
            root,
            [p.resolve() for p in (args.scan_root or [root / "outputs"])],
            args.table or root / TABLE_RELATIVE,
            output.resolve(),
            count=args.count,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"status": "export_failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
