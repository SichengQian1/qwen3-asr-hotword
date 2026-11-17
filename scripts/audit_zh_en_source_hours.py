#!/usr/bin/env python3
"""Count raw English and four Chinese source durations on CPU, without training filters."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation.source_hours import chinese_hours, english_hours, totals
from qwen_hotword.phonemes.mfa_v03_assets import write_json
from qwen_hotword.training.zh_en_common import read, save_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/zh_en_source_hours.workzone.json"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("workers must be in 1..64; Chinese archive concurrency is capped at 4")
    cfg = read(args.config)
    names = [s["name"] for s in cfg["sources"]]
    if len(names) != len(set(names)) or any(
        s["kind"] not in ("english_json", "chinese_shards") for s in cfg["sources"]
    ):
        parser.error("unique source names and supported kinds required")
    parent = ROOT / "outputs"
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="zh_en_source_hours_", dir=parent))
    write_json(output / "config.json", cfg)
    print(f"Duration audit output: {output}", flush=True)
    results = {s["name"]: {"status": "not_started", "source": s["path"]} for s in cfg["sources"]}
    try:
        for source in cfg["sources"]:
            name, path = source["name"], Path(source["path"])
            try:
                value = (
                    english_hours(path, cfg["audio_fallbacks"], args.workers)
                    if source["kind"] == "english_json"
                    else chinese_hours(path, args.workers)
                )
                details = value.pop("archive_details", [])
                if details:
                    save_rows(output / (name + "_archive_details.jsonl"), details)
                results[name] = value
            except Exception as error:
                results[name] = {
                    "status": "incomplete",
                    "source": str(path),
                    "scan_error": f"{type(error).__name__}: {error}",
                }
            write_json(output / "sources.json", results)
            print(json.dumps({name: results[name]}, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        print("Interrupted; preserving completed source reports and incomplete totals.", flush=True)
    english = [s["name"] for s in cfg["sources"] if s["kind"] == "english_json"]
    chinese = [s["name"] for s in cfg["sources"] if s["kind"] == "chinese_shards"]
    ok = all(r["status"] == "complete" for r in results.values())
    report = {
        "status": "completed" if ok else "incomplete",
        "scope": "raw_source_audio_header_durations_all_rows_no_training_selection",
        "by_source": results,
        "english_total": totals(results, english),
        "chinese_total": totals(results, chinese),
        "gpu_used": False,
        "mfa_used": False,
        "model_loaded": False,
        "training_started": False,
        "limitations": [
            "Hours are audio-header frame counts divided by sample rate; "
            "not full audio decode validation.",
            "Includes all source rows/shards, including holdout material; "
            "not eligible training hours.",
            "Totals sum references; no cross-source or audio-content deduplication.",
            "English US/UK labels identify source files, not verified speaker accents.",
            "Incomplete sources have known partial hours only, never certified total hours.",
            "No files are extracted; compressed tar may require "
            "additional read/decompression work.",
        ],
        "output": str(output),
        "return_bundle": str(output / "return_bundle.zip"),
    }
    write_json(output / "report.json", report)
    with zipfile.ZipFile(output / "return_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ("report.json", "config.json"):
            bundle.write(output / name, name)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
