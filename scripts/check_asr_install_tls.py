#!/usr/bin/env python3
"""Check existing trusted CAs; optionally resume the same H200 ASR backend test."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation.backend_install_tls import main

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
