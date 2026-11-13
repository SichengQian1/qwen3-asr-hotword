#!/usr/bin/env python3
"""Check fixed 183-class assets and inspect English JSON / Chinese shard schemas."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_foundation import (
    inspect_chinese,
    inspect_file,
    sha256,
    verify_assets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/zh_en_foundation.workzone.json"
    )
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or ROOT / "outputs" / f"zh_en_foundation_probe_{uuid4().hex[:10]}"
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "scope": "assets_and_bounded_schema_only",
        "gpu_used": False,
        "encoder_loaded": False,
        "labeling_started": False,
        "training_started": False,
        "config_sha256": sha256(args.config),
        "output": str(output.resolve()),
    }
    try:
        report["assets"] = verify_assets(ROOT, config)
    except (OSError, ValueError, KeyError) as error:
        report["assets"] = {"status": "failed", "error": str(error)}
    if not args.assets_only:
        english = {}
        with stage_progress("Read English JSON schema samples", total=2, every=1) as tick:
            for number, (lang, path) in enumerate(config["english_sources"].items(), 1):
                english[lang] = inspect_file(Path(path))
                tick(number)
        report["english_sources"] = english
        report["chinese_source"] = inspect_chinese(Path(config["chinese_root"]))
    report["status"] = "inspection_completed"
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with zipfile.ZipFile(output / "return_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(output / "report.json", "report.json")
        bundle.write(args.config, "config.json")
    print(
        json.dumps(
            {
                "status": report["status"],
                "assets": report["assets"],
                "scope": report["scope"],
                "labeling_started": False,
                "return_bundle": str((output / "return_bundle.zip").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["assets"]["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
