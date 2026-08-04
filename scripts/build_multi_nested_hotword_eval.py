#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.multi_nested import DEFAULT_SEED, build_multi_nested_assets

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic validation-only v3 assets for independent, compound, "
            "and nested multi-hotword CTC evaluation. Existing outputs are not overwritten."
        )
    )
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        summary = build_multi_nested_assets(
            args.validation_manifest,
            args.dictionary,
            args.vocab,
            args.output_dir,
            seed=args.seed,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"V3 ASSET BUILD FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
