#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.training.spanish_mfa_repair import (
    finalize_shared_spanish_mfa_repair,
)

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Map one incremental Spanish proxy MFA dictionary back to original "
            "spellings, apply Spanish-only phone cleanup, and audit every corpus."
        )
    )
    parser.add_argument(
        "--corpus",
        action="append",
        required=True,
        metavar="NAME=G2P_DIR",
    )
    parser.add_argument(
        "--dictionary",
        action="append",
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--repair-root", required=True)
    parser.add_argument("--proxy-dictionary", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        g2p_dirs = _named_paths(args.corpus, "corpus")
        dictionaries = _named_paths(args.dictionary, "dictionary")
        if set(g2p_dirs) != set(dictionaries):
            raise ValueError("corpus and dictionary names must match exactly")
        corpora = {
            name: (g2p_dir, dictionaries[name])
            for name, g2p_dir in g2p_dirs.items()
        }
        summary = finalize_shared_spanish_mfa_repair(
            corpora,
            args.repair_root,
            args.proxy_dictionary,
            args.vocab,
            args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _named_paths(values: list[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        name = name.strip()
        raw_path = raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(f"{label} must use NAME=PATH: {value!r}")
        if name in parsed:
            raise ValueError(f"duplicate {label} name: {name}")
        parsed[name] = raw_path
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
