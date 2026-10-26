"""Export complete saved native-label differences separately from small return evidence."""

from __future__ import annotations

import csv
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mmservice_smoke import _sha
from qwen_hotword.phonemes.mmservice_training_labels import LANGUAGES, rows
from qwen_hotword.training.stage_progress import stage_progress

FIELDS = (
    "word",
    "occurrences",
    "status",
    "source_tsv",
    "source_build_config",
    "reference_ipa",
    "cpp_native_ipa",
    "cpp_extra_es_cleanup_ipa",
    "reference_ids",
    "cpp_native_ids",
    "cpp_extra_es_cleanup_ids",
)


def export_differences(output: Path, statistics: dict[str, Any]) -> dict[str, Any]:
    """Keep source-specific references, failed rows and OOVs; never invent phones."""
    delivery = output / "delivery"
    delivery.mkdir()
    counts: Counter[str] = Counter()
    occurrences: Counter[str] = Counter()
    total = sum(item["source_word_pairs"] for item in statistics.values())
    number = 0
    with (
        ExitStack() as stack,
        stage_progress("Export ALL three-language native differences", total=total) as update,
    ):
        json_files = {
            lang: stack.enter_context(
                (delivery / f"{lang}_differences.jsonl").open("x", encoding="utf-8")
            )
            for lang in LANGUAGES
        }
        tsv_files = {
            lang: csv.writer(
                stack.enter_context(
                    (delivery / f"{lang}_differences.tsv").open("x", encoding="utf-8", newline="")
                ),
                delimiter="\t",
            )
            for lang in LANGUAGES
        }
        for writer in tsv_files.values():
            writer.writerow(FIELDS)
        for number, row in enumerate(rows(output / "words.jsonl"), 1):
            update(number)
            if row["status"]["native"] == "equal":
                continue
            lang = row["language"]
            counts[lang] += 1
            occurrences[lang] += row["occurrences"]
            json_files[lang].write(json.dumps(row, ensure_ascii=False) + "\n")
            labels = [
                row["saved_reference"],
                row["cpp"]["native"],
                row["cpp"]["existing_es_cleanup"],
            ]
            tsv_files[lang].writerow(
                [
                    row["word"],
                    row["occurrences"],
                    row["status"]["native"],
                    row["source_tsv"],
                    row["source_build_config"],
                    *(label["raw_ipa"] if label else None for label in labels),
                    *(json.dumps(label["token_ids"]) if label else "" for label in labels),
                ]
            )
    if number != total:
        raise ValueError("delivery scan does not match total source-word rows")
    for lang in LANGUAGES:
        expected = statistics[lang]["word_label_comparison"]["native"]
        if counts[lang] != sum(expected["source_word_pairs"].values()) - expected[
            "source_word_pairs"
        ].get("equal", 0):
            raise ValueError(f"{lang} delivery row count differs from report")
        if occurrences[lang] != sum(expected["word_occurrences"].values()) - expected[
            "word_occurrences"
        ].get("equal", 0):
            raise ValueError(f"{lang} delivery occurrence count differs from report")
    (delivery / "README.md").write_text(
        "# 新版C++ G2P全量标签差异\n\n"
        "每语种JSONL/TSV保留全部native标签不同、失败或OOV条目；同词不同来源不合并。\n"
        "reference为原训练词典标签，cpp_native为新版明确--lang的输出。\n"
        "extra_es_cleanup为额外诊断清洗分支，不冒充新版原生输出。\n"
        "空发音代表失败，不补造标签。无差异时JSONL为空、TSV仅有表头。\n"
        "这是标签兼容性比较，不等于真实发音准确率、检索召回或ASR WER。\n"
        "完整逐词及逐句结果见上一级words.jsonl和samples.jsonl。\n",
        encoding="utf-8",
    )
    return {
        "directory": str(delivery),
        "source_word_pairs": {lang: counts[lang] for lang in LANGUAGES},
        "word_occurrences": {lang: occurrences[lang] for lang in LANGUAGES},
        "files": {
            path.name: {"sha256": _sha(path), "bytes": path.stat().st_size}
            for path in sorted(delivery.iterdir())
        },
        "included_in_small_return_zip": False,
    }
