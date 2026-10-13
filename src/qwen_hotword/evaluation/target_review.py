"""Review-only multilingual target union; no fillers, audio or retrieval changes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _json, _key
from qwen_hotword.inference.external_keyword_retrieval import load_keyword_bias
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified

LANGUAGES = ("en", "es", "pt")
LABELS = {"en": "英语", "es": "西语", "pt": "葡语"}


def _checked(path: Path, identities: dict[str, Any], expected: str | None = None) -> Path:
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    identities[str(path)] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return path


def compile_policy(policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Only explicit, single-token rules; never apply stopwords inside a phrase."""
    if not policy.get("version") or set(policy["languages"]) != set(LANGUAGES):
        raise ValueError("policy needs a version and exactly en/es/pt")
    compiled = {}
    for language, config in policy["languages"].items():
        rules: dict[str, str] = {}
        for category, words in config["remove"].items():
            if category not in policy["categories"] or not isinstance(words, list):
                raise ValueError(f"unknown category/invalid word list: {language}/{category}")
            for word in words:
                key = _key(word)
                if len(key.split()) != 1 or key in rules:
                    raise ValueError(f"multi-token or duplicate basic-word rule: {language}/{word}")
                rules[key] = category
        overrides = config["keep_overrides"]
        if not isinstance(overrides, list):
            raise ValueError("keep_overrides must be a list")
        keep = {_key(word) for word in overrides}
        if len(keep) != len(overrides) or not keep <= rules.keys():
            raise ValueError(f"keep_overrides must uniquely name existing rules: {language}")
        compiled[language] = {"rules": rules, "keep": keep}
    return compiled


def _source_targets(
    root: Path,
    language: str,
    config: dict[str, Any],
    vocab: PhonemeVocab,
    identities: dict[str, Any],
) -> list[dict[str, Any]]:
    directory = root / config["directory"]
    _checked(directory / "sha256.txt", identities)
    report = _json(_verified(directory, "report.json", identities))
    if report.get("status") != "completed" or report.get("tables_created") is not True:
        raise ValueError(f"completed, sealed source table required: {language}")
    table = _verified(directory, config["table"], identities)
    groups_path = _verified(directory, config["target_groups"], identities)
    _checked(table, identities, config.get("table_sha256"))
    _checked(groups_path, identities, config.get("target_groups_sha256"))
    # Reject duplicate keys before using the existing loader for vocabulary checks.
    _json(table)
    bundle = load_keyword_bias(table, vocab=vocab, keyword_set="all_keywords", language=language)
    groups = _json(groups_path)
    if not isinstance(groups, dict) or set(groups) != set(config["expected_groups"]):
        raise ValueError(f"target dataset/wave groups changed: {language}")
    memberships: dict[str, list[str]] = defaultdict(list)
    for name, words in sorted(groups.items()):
        if not isinstance(words, list) or not words:
            raise ValueError(f"invalid/empty target group: {language}/{name}")
        keys = [_key(word) for word in words]
        if keys != words or len(keys) != len(set(keys)):
            raise ValueError(f"target keys must be canonical and unique: {language}/{name}")
        for key in keys:
            memberships[key].append(name)
    entries = {entry.normalized: entry for entry in bundle.entries}
    declared = (
        report.get("mandatory_union")
        if language == "en"
        else report.get("languages", {}).get(language, {}).get("mandatory")
    )
    if (
        len(entries) != config["expected_table_size"]
        or len(memberships) != config["expected_targets"]
        or declared != config["expected_targets"]
        or not memberships.keys() <= entries.keys()
    ):
        raise ValueError(f"target/table counts or declared membership mismatch: {language}")
    # Crucially iterate the sealed target memberships, not all 3k/4k table entries.
    return [
        {
            "language": language,
            "surface": entries[key].surface,
            "normalized": key,
            "pronunciation": entries[key].pronunciation,
            "phoneme_tokens": list(entries[key].phoneme_tokens),
            "token_ids": list(entries[key].token_ids),
            "source_groups": memberships[key],
            "source_table": str(table),
            "source_target_groups": str(groups_path),
        }
        for key in sorted(memberships)
    ]


def decide_targets(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    compiled = compile_policy(policy)
    decisions = []
    for row in rows:
        language, key = row["language"], row["normalized"]
        category = compiled[language]["rules"].get(key)
        override = key in compiled[language]["keep"]
        removed = category is not None and not override
        flags = []
        if len(key.split()) == 1 and len(key) <= 2:
            flags.append("短词/单字母：可能是缩写、字母或功能词，需审阅词义")
        if len(row["token_ids"]) <= 2:
            flags.append("音素数不超过2：仅提示检查，不因音素短而删除")
        if category:
            flags.append("完整词面命中基础词规则：无法自动区分同形专名或字母名称")
        if any(key in compiled[other]["rules"] for other in LANGUAGES if other != language):
            flags.append("也命中其他语言的基础词：仅按本条语言决定，不跨语言误删")
        decisions.append(
            {
                **row,
                "decision": "removed_basic" if removed else "retained",
                "rule_category": category,
                "reason": (
                    "用户配置保留例外"
                    if override
                    else policy["categories"][category]
                    if removed
                    else "未命中本语言完整基础词规则；保留待人工审阅"
                ),
                "review_flags": flags,
            }
        )
    return decisions


def combine_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One display entry per normalized surface, all language pronunciations retained."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["decision"] == "retained":
            grouped[row["normalized"]].append(row)
    return [
        {
            "id": "target_" + hashlib.sha256(key.encode()).hexdigest()[:16],
            "normalized": key,
            "surface": variants[0]["surface"],
            "surface_forms": sorted({r["surface"] for r in variants}),
            "languages": [r["language"] for r in variants],
            "variants": variants,
        }
        for key, variants in sorted(grouped.items())
    ]


def _cell(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _review_document(rows: list[dict[str, Any]], report: dict[str, Any], *, removed: bool) -> str:
    lines = [
        "# " + ("移出的基础词：请检查是否需要恢复" if removed else "三语需求热词合并审阅表"),
        "",
        "只取原始需求目标，不含填充词或近音补词。不补齐到固定数量；本轮未做推理。",
        "本表是人工审阅草案，不代表已确认每项都是业务关键词或发音正确。",
        "只匹配完整条目；保留完整短语。基础词的同形专名可能被移出，需核对下表。",
        "总词表按规范词面合并；下表按语言展开。JSON保留各语言词面、发音和来源。",
        "本版目标范围改变，未来召回不能直接与旧目标分母比较。",
        "",
        "| 语言 | 输入需求词 | 保留 | 移出 |",
        "|---|---:|---:|---:|",
    ]
    for lang, counts in report["by_language"].items():
        lines.append(
            f"| {LABELS[lang]} | {counts['input_targets']} | "
            f"{counts['retained']} | {counts['removed_basic']} |"
        )
    lines.extend(
        [
            "",
            f"保留的语言条目：{report['retained_language_entries']}；"
            f"合并后的规范唯一词面：{report['unique_retained_surfaces']}。",
            "",
            "| 序号 | 词面 | 语言 | 原有音素 | 来源组 | 处理说明 | 审阅提示 |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    selected = [r for r in rows if (r["decision"] == "removed_basic") == removed]
    for n, row in enumerate(sorted(selected, key=lambda r: (r["normalized"], r["language"])), 1):
        cells = (
            n,
            row["surface"],
            LABELS[row["language"]],
            row["pronunciation"],
            ", ".join(row["source_groups"]),
            row["reason"],
            "; ".join(row["review_flags"]),
        )
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    if not selected:
        lines.extend(["", "本清单为空。"])
    return "\n".join(lines) + "\n"


def build_target_review(
    root: Path, config_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    cfg = _json(config_path)
    output = (root / (output_dir or cfg["output"])).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}; choose a new --output-dir")
    if set(cfg["sources"]) != set(LANGUAGES):
        raise ValueError("exactly en/es/pt sources required")
    for source in cfg["sources"].values():
        if output.is_relative_to((root / source["directory"]).resolve()):
            raise ValueError("new review output must not be inside an existing source table")
    identities: dict[str, Any] = {}
    _checked(config_path, identities)
    policy = _json(_checked(root / cfg["basic_word_policy"], identities))
    compile_policy(policy)
    vocab = load_phoneme_vocab(_checked(root / cfg["vocab"], identities, cfg["vocab_sha256"]))
    rows = []
    for language in LANGUAGES:
        rows.extend(_source_targets(root, language, cfg["sources"][language], vocab, identities))
    decisions = decide_targets(rows, policy)
    combined = combine_variants(decisions)
    counts: dict[str, dict[str, Any]] = {}
    for language in LANGUAGES:
        selected = [r for r in decisions if r["language"] == language]
        outcomes = Counter(r["decision"] for r in selected)
        counts[language] = {
            "input_targets": len(selected),
            "retained": outcomes["retained"],
            "removed_basic": outcomes["removed_basic"],
            "removed_by_category": dict(
                Counter(r["rule_category"] for r in selected if r["decision"] == "removed_basic")
            ),
        }
    report = {
        "status": "completed",
        "scope": "multilingual_declared_targets_review_only",
        "human_approved": False,
        "output_dir": str(output),
        "by_language": counts,
        "input_language_entries": len(rows),
        "retained_language_entries": sum(x["retained"] for x in counts.values()),
        "removed_language_entries": sum(x["removed_basic"] for x in counts.values()),
        "unique_input_surfaces": len({r["normalized"] for r in rows}),
        "unique_retained_surfaces": len(combined),
        "cross_language_retained_surfaces": sum(len(e["variants"]) > 1 for e in combined),
        "filler_entries_added": 0,
        "model_loaded": False,
        "audio_read": False,
        "transcripts_or_predictions_read": False,
        "inference_run": False,
        "policy_version": policy["version"],
        "inputs": identities,
        "code_sha256": {str(Path(__file__).resolve()): _sha(Path(__file__))},
        "limitations": [
            "Review draft: exact grammatical-word rules do not establish business relevance.",
            "Homographic names/letters may be removed; inspect removed_basic_words.md.",
            "Phrases are kept whole; short terms and ordinary content words are not auto-deleted.",
            "Same surface across languages preserves all language/pronunciation variants.",
            "Existing phonemes are preserved, not newly generated or certified.",
            "New target scope changes evaluation denominators; no recall claim is made.",
            "keyword_table.json is a review schema, not an automatic inference configuration.",
        ],
        "return_files": [
            str(output / n)
            for n in (
                "review.md",
                "removed_basic_words.md",
                "keyword_table.json",
                "report.json",
                "sha256.txt",
            )
        ],
    }
    for name, identity in identities.items():
        if _sha(Path(name)) != identity["sha256"]:
            raise ValueError(f"input changed during build: {name}")
    # Refuse overwrite and do not create any directory until all source checks pass.
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("config.json", cfg),
        ("basic_word_policy.json", policy),
        (
            "keyword_table.json",
            {"schema_version": 1, "scope": report["scope"], "entries": combined},
        ),
        ("report.json", report),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    (output / "keywords.txt").write_text(
        "".join(e["surface"] + "\n" for e in combined), encoding="utf-8"
    )
    (output / "all_target_decisions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in decisions),
        encoding="utf-8",
    )
    for removed, name in ((False, "review.md"), (True, "removed_basic_words.md")):
        (output / name).write_text(
            _review_document(decisions, report, removed=removed), encoding="utf-8"
        )
    seal(output)
    return report
