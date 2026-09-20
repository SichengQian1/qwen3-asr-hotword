#!/usr/bin/env python3
"""Run outside the container: download MFA assets only, no MFA/Conda required."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.pt_mfa_assets import DEFAULT_ASSETS, download_assets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / DEFAULT_ASSETS)
    args = parser.parse_args()
    try:
        print(json.dumps(download_assets(args.output_dir), ensure_ascii=False, indent=2))
    except (OSError, ValueError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Download failed; existing files preserved: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
