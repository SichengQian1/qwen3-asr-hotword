#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.config import ConfigError, load_workzone_config
from qwen_hotword.modeling.qwen_backbone import ModelValidationError, load_asr_model


def synthetic_tone(seconds: float = 1.0, sample_rate: int = 16_000) -> tuple[object, int]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for the smoke test") from error

    sample_count = int(seconds * sample_rate)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = (0.01 * np.sin(2.0 * math.pi * 440.0 * time)).astype(np.float32)
    return waveform, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="Load 1.7B and run synthetic audio.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        config = load_workzone_config(args.config, require_existing_model=True)
        wrapper = load_asr_model(config.model)
        result = wrapper.transcribe(audio=synthetic_tone(), language="English")
    except (ConfigError, ModelValidationError, RuntimeError, OSError) as error:
        print(f"SMOKE TEST FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    first = result[0]
    sanitized = {
        "model": config.model.expected_name,
        "device": config.model.device,
        "input": "synthetic_440hz_1s",
        "result_count": len(result),
        "detected_language": getattr(first, "language", None),
        "text_length": len(getattr(first, "text", "") or ""),
        "status": "pass",
    }
    print(json.dumps(sanitized, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
