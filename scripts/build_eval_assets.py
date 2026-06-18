#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from qwen_hotword.evaluation.cases import build_eval_cases
from qwen_hotword.evaluation.config import load_eval_asset_config
from qwen_hotword.evaluation.hotwords import build_hotwords
from qwen_hotword.evaluation.records import write_jsonl
from qwen_hotword.evaluation.scanner import scan_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ASR hotword evaluation manifests from local corpora."
    )
    parser.add_argument("--config", required=True, help="Evaluation source YAML config.")
    parser.add_argument("--output-dir", help="Override output directory from config.")
    parser.add_argument(
        "--phonemizer",
        choices=("none", "espeak"),
        default="none",
        help="IPA backend. Use 'espeak' after phonemizer/espeak-ng are available.",
    )
    parser.add_argument(
        "--require-ipa",
        action="store_true",
        help="Fail if IPA cannot be generated for every selected hotword.",
    )
    return parser.parse_args()


def _count_by(records: Sequence[object], *fields: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        key = " / ".join(str(getattr(record, field)) for field in fields)
        counter[key] += 1
    return dict(sorted(counter.items()))


def main() -> int:
    args = parse_args()
    config = load_eval_asset_config(args.config)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    utterances = scan_sources(config)
    hotword_utterances = [
        utterance
        for utterance in utterances
        if str(utterance.metadata.get("use_for_hotwords", "true")).lower() == "true"
    ]
    case_utterances = [
        utterance
        for utterance in utterances
        if str(utterance.metadata.get("use_for_cases", "true")).lower() == "true"
    ]
    hotwords = build_hotwords(
        hotword_utterances,
        config.hotwords,
        phonemizer_backend=str(args.phonemizer),
        require_ipa=bool(args.require_ipa),
    )
    ctc_cases = build_eval_cases(
        case_utterances,
        hotwords,
        config.cases,
        seed=config.sampling.seed,
        eval_stage="ctc",
    )
    asr_cases = build_eval_cases(
        case_utterances,
        hotwords,
        config.cases,
        seed=config.sampling.seed + 1009,
        eval_stage="asr_injection",
    )

    write_jsonl(output_dir / "utterances.jsonl", [item.to_json() for item in utterances])
    write_jsonl(output_dir / "hotwords.jsonl", [item.to_json() for item in hotwords])
    write_jsonl(output_dir / "ctc_eval_cases.jsonl", [item.to_json() for item in ctc_cases])
    write_jsonl(
        output_dir / "asr_injection_eval_cases.jsonl",
        [item.to_json() for item in asr_cases],
    )
    summary = {
        "output_dir": str(output_dir),
        "utterances": len(utterances),
        "hotword_utterances": len(hotword_utterances),
        "case_utterances": len(case_utterances),
        "hotwords": len(hotwords),
        "ctc_eval_cases": len(ctc_cases),
        "asr_injection_eval_cases": len(asr_cases),
        "utterances_by_dataset": _count_by(utterances, "dataset"),
        "utterances_by_dataset_split": _count_by(utterances, "dataset", "split"),
        "case_utterances_by_dataset": _count_by(case_utterances, "dataset"),
        "hotwords_by_language": _count_by(hotwords, "language"),
        "phonemizer": args.phonemizer,
        "require_ipa": args.require_ipa,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
