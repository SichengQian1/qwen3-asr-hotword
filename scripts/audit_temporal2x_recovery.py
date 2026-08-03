#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.temporal_recovery import (
    parse_corpus_spec,
    run_temporal_recovery_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of existing full-manifest review records under a "
            "Temporal CTC Head time-axis multiplier. Ready aggregates are read "
            "from summary.json; train_ready.jsonl content is not scanned."
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=MANIFEST_DIR",
        help=(
            "Corpus name and existing full-manifest directory. Repeat in the "
            "desired report priority order."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-upsampling-factor", type=int, default=2)
    parser.add_argument(
        "--release-max-effective-ratio",
        type=float,
        default=0.90,
        help=(
            "Recommended first-release ceiling among pure temporal recoveries; "
            "higher feasible records remain in the deferred high-pressure group."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Disable periodic corpus scan progress lines.",
    )
    args = parser.parse_args()

    try:
        corpora = [parse_corpus_spec(value) for value in args.corpus]
        summary = run_temporal_recovery_audit(
            corpora,
            args.output_dir,
            time_upsampling_factor=args.time_upsampling_factor,
            release_max_effective_ratio=args.release_max_effective_ratio,
            progress_every=args.progress_every,
            print_progress=not args.quiet_progress,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
