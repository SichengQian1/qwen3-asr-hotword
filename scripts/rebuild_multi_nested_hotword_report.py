#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.hotwords.multi_nested import (
    evaluate_multi_nested_case_scores,
    load_hotword_families,
    load_multi_nested_case_scores,
    load_multi_nested_cases,
)
from qwen_hotword.hotwords.registry import load_hotword_table
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute corrected v3 metrics from existing case scores without loading "
            "the CTC Head or validation feature cache. The source report is not overwritten."
        )
    )
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--hotwords", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--case-scores", required=True)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if output.exists():
        parser.error(f"refusing to overwrite corrected report: {output}")
    try:
        vocab = load_phoneme_vocab(args.vocab)
        hotwords = load_hotword_table(args.hotwords, vocab=vocab, blank_id=0)
        families = load_hotword_families(args.families)
        cases = load_multi_nested_cases(args.cases)
        scores = load_multi_nested_case_scores(args.case_scores)
        base = json.loads(Path(args.base_report).read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError("base report must be a JSON object")
        base["metrics"] = evaluate_multi_nested_case_scores(cases, hotwords, families, scores)
        base["metric_corrections"] = {
            "report_only_rebuild": True,
            "ctc_inference_repeated": False,
            "longest_match_redundant_family_hits_excluded_from_false_positives": True,
            "nested_member_metrics_restricted_to_each_case_family": True,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(base, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except (FileNotFoundError, KeyError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "pass", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
