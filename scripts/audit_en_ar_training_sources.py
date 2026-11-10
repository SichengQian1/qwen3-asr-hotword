#!/usr/bin/env python3
"""Probe or audit English–Arabic source manifests and audio headers on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.training.en_ar_source_audit import audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="at most 32 rows per source")
    mode.add_argument("--audit", action="store_true", help="check every audio header")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--audio-fallback", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--output-dir", type=Path, help="new output directory")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "en_ar_training_sources.workzone.json",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in 1..32")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fallbacks = {Path(old): Path(new) for old, new in config["audio_fallbacks"].items()}
    for value in args.audio_fallback:
        old, separator, new = value.partition("=")
        if not separator or not Path(old).is_absolute() or not Path(new).is_absolute():
            parser.error("--audio-fallback requires absolute OLD=NEW paths")
        fallbacks[Path(old)] = Path(new)
    if Path("/") in fallbacks:
        parser.error("root fallback is too broad")
    swift = Path(config["swift_root"])
    sources = [
        (group, swift / relative)
        for group in ("en_egy", "ar_egy", "ar_msa")
        for relative in config["groups"][group]
    ]
    sources.append(("en_us_donor", ROOT / config["us_donor"]))
    run_name = f"en_ar_training_source_audit_{uuid4().hex[:8]}"
    output = args.output_dir or ROOT / "outputs" / run_name
    try:
        result = audit(
            sources,
            fallbacks,
            output.resolve(),
            limit=32 if args.check else None,
            workers=args.workers,
        )
    except OSError as error:
        print(f"Source audit failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "checked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
