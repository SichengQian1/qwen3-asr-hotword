#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.phonemes.espeak_mfa_comparison import (  # noqa: E402
    ESPEAK_LANGUAGE_CODES,
    NamedPath,
    compare_espeak_mfa,
    environment_tool_metadata,
    parse_named_path,
)

DEFAULT_VOCAB = "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def _phonemize_batch(words: list[str], language: str) -> list[str]:
    try:
        from phonemizer import phonemize
    except ImportError as error:
        raise RuntimeError(
            "phonemizer is not installed; install the eval extra and espeak-ng"
        ) from error
    result = phonemize(
        words,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True,
        language_switch="keep-flags",
        njobs=1,
    )
    if not isinstance(result, list):
        raise RuntimeError("phonemizer did not return one IPA string per input word")
    return [str(value).strip() for value in result]


def _unique_paths(values: list[str], label: str) -> dict[str, Path]:
    parsed: list[NamedPath] = [parse_named_path(value) for value in values]
    result: dict[str, Path] = {}
    for item in parsed:
        if item.language in result:
            raise ValueError(f"duplicate {label} for language {item.language}")
        result[item.language] = item.path
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare eSpeak-ng and MFA on deterministic 500-word en/es/pt train samples "
            "without changing training manifests or the CTC vocabulary."
        )
    )
    parser.add_argument(
        "--language-manifest",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="Train JSONL; provide exactly en, es, and pt.",
    )
    parser.add_argument(
        "--mfa-dictionary",
        action="append",
        required=True,
        metavar="LANG=PATH",
        help="Existing MFA dictionary; provide exactly en, es, and pt.",
    )
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_260_829)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        manifests = _unique_paths(args.language_manifest, "manifest")
        dictionaries = _unique_paths(args.mfa_dictionary, "MFA dictionary")
        summary = compare_espeak_mfa(
            manifests,
            dictionaries,
            args.vocab,
            args.output_dir,
            phonemize_batch=_phonemize_batch,
            sample_size=args.sample_size,
            seed=args.seed,
            tool_metadata={
                **environment_tool_metadata(),
                "language_mapping": json.dumps(ESPEAK_LANGUAGE_CODES, sort_keys=True),
            },
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
