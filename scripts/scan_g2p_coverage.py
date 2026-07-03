#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qwen_hotword.evaluation.config import load_eval_asset_config
from qwen_hotword.evaluation.scanner import scan_sources
from qwen_hotword.phonemes.coverage import (
    coerce_record_language,
    coerce_record_text,
    espeak_language_code,
    load_phoneme_vocab,
    tokenize_ipa_to_vocab,
)

DEFAULT_VOCAB = "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan training manifests with G2P and report CTC phoneme vocab coverage."
    )
    parser.add_argument(
        "--input",
        action="append",
        help=(
            "Input manifest file. Supports .jsonl, .csv, and .tsv. Repeat for multiple files. "
            "Use PATH::LANG to set a per-file fallback language, for example train_en.jsonl::en."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "Evaluation source YAML config. Reuses the existing corpus scanners and source paths, "
            "for example configs/eval_sources.workzone.yaml."
        ),
    )
    parser.add_argument("--vocab", default=DEFAULT_VOCAB, help="Phoneme vocabulary JSON path.")
    parser.add_argument("--output-dir", required=True, help="Directory for scan reports.")
    parser.add_argument("--text-column", help="Text column name. Auto-detected if omitted.")
    parser.add_argument("--language-column", help="Language column name. Auto-detected if omitted.")
    parser.add_argument(
        "--default-language",
        help="Language to use when the input has no language column, e.g. en, es-419, pt-BR.",
    )
    parser.add_argument("--id-column", help="Utterance/sample id column. Auto-generated if omitted.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional cap for a quick sample scan. 0 means scan all records.",
    )
    parser.add_argument(
        "--max-issue-records",
        type=int,
        default=5000,
        help="Maximum records with OOV or G2P failures to write to JSONL.",
    )
    parser.add_argument(
        "--backend",
        choices=("espeak",),
        default="espeak",
        help="G2P backend. Currently only phonemizer/espeak is supported.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if isinstance(value, dict):
                    yield value
        return

    if path.suffix in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            yield from reader
        return

    raise ValueError(f"Unsupported manifest extension: {path}")


def parse_input_spec(spec: str) -> tuple[Path, str | None]:
    if "::" not in spec:
        return Path(spec).expanduser(), None
    path_text, language = spec.rsplit("::", maxsplit=1)
    return Path(path_text).expanduser(), language.strip() or None


def phonemize_text(text: str, language: str) -> str:
    try:
        from phonemizer import phonemize
    except ImportError as error:
        raise RuntimeError(
            "phonemizer is not installed. Install the eval extra and espeak-ng first, "
            "for example: python -m pip install '.[eval]' and brew/apt install espeak-ng."
        ) from error

    ipa = phonemize(
        text,
        language=espeak_language_code(language),
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=True,
    )
    return str(ipa).strip()


def sample_id(row: dict[str, Any], id_column: str | None, fallback: str) -> str:
    columns = (id_column,) if id_column else ("id", "utt_id", "audio_id", "key", "path", "audio_path")
    for column in columns:
        if column and row.get(column) not in {None, ""}:
            return str(row[column])
    return fallback


def iter_config_records(config_path: str | Path) -> Iterable[dict[str, Any]]:
    config = load_eval_asset_config(config_path)
    for utterance in scan_sources(config):
        yield {
            "id": utterance.utt_id,
            "dataset": utterance.dataset,
            "split": utterance.split,
            "language": utterance.language,
            "audio_path": utterance.audio_path,
            "text": utterance.text,
        }


def iter_input_records(args: argparse.Namespace) -> Iterable[tuple[str, dict[str, Any], str | None]]:
    if args.config:
        for row in iter_config_records(args.config):
            yield str(args.config), row, None
    for input_path_text in args.input or []:
        input_path, input_language = parse_input_spec(input_path_text)
        for row in read_manifest(input_path):
            yield str(input_path), row, input_language


def write_counter_csv(path: Path, header: tuple[str, str], counter: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key, count in counter.most_common():
            writer.writerow([key, count])


def write_language_summary(path: Path, by_language: dict[str, dict[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "language",
                "records",
                "g2p_failures",
                "records_with_oov",
                "phone_tokens",
                "oov_units",
                "coverage",
            ]
        )
        for language in sorted(by_language):
            stats = by_language[language]
            phone_tokens = stats["phone_tokens"]
            oov_units = stats["oov_units"]
            coverage = 1.0 if phone_tokens == 0 else (phone_tokens - oov_units) / phone_tokens
            writer.writerow(
                [
                    language,
                    stats["records"],
                    stats["g2p_failures"],
                    stats["records_with_oov"],
                    phone_tokens,
                    oov_units,
                    f"{coverage:.6f}",
                ]
            )


def main() -> int:
    args = parse_args()
    if not args.config and not args.input:
        raise SystemExit("Provide --config or at least one --input manifest.")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = load_phoneme_vocab(args.vocab)

    records_seen = 0
    g2p_failures = 0
    records_with_oov = 0
    phone_counts: Counter[str] = Counter()
    oov_counts: Counter[str] = Counter()
    by_language: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "records": 0,
            "g2p_failures": 0,
            "records_with_oov": 0,
            "phone_tokens": 0,
            "oov_units": 0,
        }
    )

    issue_path = output_dir / "records_with_oov_or_g2p_failure.jsonl"
    issue_count = 0
    with issue_path.open("w", encoding="utf-8") as issue_handle:
        for row_index, (source_name, row, input_language) in enumerate(iter_input_records(args)):
            if args.max_records > 0 and records_seen >= args.max_records:
                break
            text = coerce_record_text(row, args.text_column)
            language = coerce_record_language(
                row,
                args.language_column,
                input_language or args.default_language,
            )
            if not text or not language:
                continue

            record_id = sample_id(
                row,
                args.id_column,
                f"{source_name}:{row_index}",
            )
            records_seen += 1
            by_language[language]["records"] += 1

            try:
                ipa = phonemize_text(text, language)
            except Exception as error:  # noqa: BLE001 - report data/tooling failures per record
                g2p_failures += 1
                by_language[language]["g2p_failures"] += 1
                if issue_count < args.max_issue_records:
                    issue_handle.write(
                        json.dumps(
                            {
                                "id": record_id,
                                "language": language,
                                "text": text,
                                "error": str(error),
                                "issue_type": "g2p_failure",
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    issue_count += 1
                continue

            tokenized = tokenize_ipa_to_vocab(ipa, vocab)
            phone_counts.update(tokenized.tokens)
            oov_counts.update(tokenized.oov_units)
            by_language[language]["phone_tokens"] += len(tokenized.tokens) + len(
                tokenized.oov_units
            )
            by_language[language]["oov_units"] += len(tokenized.oov_units)

            if tokenized.oov_units:
                records_with_oov += 1
                by_language[language]["records_with_oov"] += 1
                if issue_count < args.max_issue_records:
                    issue_handle.write(
                        json.dumps(
                            {
                                "id": record_id,
                                "language": language,
                                "text": text,
                                "ipa": ipa,
                                "normalized_ipa": tokenized.normalized_ipa,
                                "phones": tokenized.tokens,
                                "phone_ids": tokenized.token_ids,
                                "oov_units": tokenized.oov_units,
                                "issue_type": "oov",
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    issue_count += 1

    total_phone_units = sum(phone_counts.values()) + sum(oov_counts.values())
    coverage = 1.0 if total_phone_units == 0 else sum(phone_counts.values()) / total_phone_units
    summary = {
        "vocab": str(Path(args.vocab)),
        "ctc_output_classes": len(vocab.tokens),
        "records_seen": records_seen,
        "g2p_failures": g2p_failures,
        "records_with_oov": records_with_oov,
        "phone_tokens_in_vocab": sum(phone_counts.values()),
        "oov_units": sum(oov_counts.values()),
        "phone_unit_coverage": coverage,
        "top_oov_units": oov_counts.most_common(30),
        "top_phone_tokens": phone_counts.most_common(30),
        "by_language": by_language,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_counter_csv(output_dir / "phone_counts.csv", ("phone", "count"), phone_counts)
    write_counter_csv(output_dir / "oov_counts.csv", ("oov_unit", "count"), oov_counts)
    write_language_summary(output_dir / "language_summary.csv", by_language)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
