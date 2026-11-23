"""Editorial Arabic candidate review; corpus evidence never implies semantic approval.

Only the frozen training manifest is read. No held-out text, audio, model inference,
G2P or retrieval is used. Attached-prefix matches are review evidence, not targets.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json

BUCKETS = ("open_source", "gaoshan_train", "coarse_train")


def tokens(text: str) -> tuple[str, ...]:
    # Keep hamza/alef/ya/ta-marbuta distinctions. Punctuation blocks phrase matches.
    normalized = "".join(
        c
        for c in unicodedata.normalize("NFC", text)
        if c != "ـ" and not unicodedata.category(c).startswith("M")
    )
    return tuple(re.findall(r"\w+|[^\w\s]", normalized))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("language") != "ar" or data.get("schema_version") != 1:
        raise ValueError("expected Arabic candidate schema 1")
    rows: list[dict[str, Any]] = data["candidates"]
    if not rows:
        raise ValueError("empty candidate list")
    seen_ids: set[str] = set()
    owners: dict[tuple[str, ...], str] = {}
    for row in rows:
        for key in ("id", "surface", "category", "gloss_zh"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"missing candidate {key}")
        if row["id"] in seen_ids or row.get("review_status") != "pending":
            raise ValueError("duplicate ID or non-pending candidate")
        seen_ids.add(row["id"])
        aliases = row.get("aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError("aliases must be strings")
        for surface in [row["surface"], *aliases]:
            key_tokens = tokens(surface)
            if not key_tokens or any(
                not all("ARABIC" in unicodedata.name(c, "") and c.isalpha() for c in t)
                for t in key_tokens
            ):
                raise ValueError("candidate must contain complete Arabic words: " + surface)
            if key_tokens in owners and owners[key_tokens] != row["id"]:
                raise ValueError("normalized surface/alias belongs to multiple candidates")
            owners[key_tokens] = row["id"]
    return rows


def nested_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    forms = {r["id"]: [tokens(s) for s in [r["surface"], *r.get("aliases", [])]] for r in rows}
    pairs = []
    for short in rows:
        for long in rows:
            if short["id"] == long["id"]:
                continue
            if any(
                len(a) < len(b) and any(b[i : i + len(a)] == a for i in range(len(b) - len(a) + 1))
                for a in forms[short["id"]]
                for b in forms[long["id"]]
            ):
                pairs.append(
                    {
                        "short_id": short["id"],
                        "long_id": long["id"],
                        "short": short["surface"],
                        "long": long["surface"],
                    }
                )
    return pairs


def scan_training(
    rows: Iterable[dict[str, Any]],
    candidates: list[dict[str, Any]],
    progress: Callable[[int], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact: dict[tuple[str, ...], set[str]] = defaultdict(set)
    attached: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for candidate in candidates:
        for surface in [candidate["surface"], *candidate.get("aliases", [])]:
            form = tokens(surface)
            exact[form].add(candidate["id"])
            # These may be clitics OR part of another word. Never accept as ground truth.
            for prefix in ("و", "ف", "ب", "ك", "ل", "وب", "فب", "ول", "فل", "وك", "فك"):
                attached[(prefix + form[0], *form[1:])].add(candidate["id"])
            if form[0].startswith("ال"):
                for prefix in ("لل", "ولل", "فلل"):
                    attached[(prefix + form[0][2:], *form[1:])].add(candidate["id"])
    lengths = sorted({len(f) for f in exact})
    counts: dict[str, dict[str, Any]] = {
        r["id"]: {
            "exact_samples": 0,
            "exact_occurrences": 0,
            "by_source_group": {},
            "by_source": {},
            "examples": [],
            "attached_prefix_review_samples": 0,
            "attached_prefix_examples": [],
        }
        for r in candidates
    }
    totals: Counter[str] = Counter()
    hours: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        progress(index)
        if row.get("split") != "train":
            raise ValueError("candidate mining accepts train records only")
        totals["rows"] += 1
        if row["language"] == "en":
            totals["english_rows_skipped"] += 1
            continue
        if row["language"] != "ar" or row["source_group"] not in BUCKETS:
            raise ValueError("unexpected language/source group")
        bucket = row["source_group"]
        totals["arabic_rows"] += 1
        hours[bucket] += row["duration_seconds"] / 3600
        text_tokens = tokens(row["text"])
        found: Counter[str] = Counter()
        possible: set[str] = set()
        for size in lengths:
            for start in range(len(text_tokens) - size + 1):
                window = text_tokens[start : start + size]
                found.update(exact.get(window, ()))
                possible.update(attached.get(window, ()))
        example = {k: row[k] for k in ("id", "source", "source_row", "source_group", "text")}
        # Full example text retained (at most two per candidate), no audio copies.
        for candidate_id, count in found.items():
            value = counts[candidate_id]
            value["exact_samples"] += 1
            value["exact_occurrences"] += count
            for field, name in (("by_source_group", bucket), ("by_source", row["source"])):
                value[field][name] = value[field].get(name, 0) + 1
            if len(value["examples"]) < 2:
                value["examples"].append(example)
        for candidate_id in possible:
            value = counts[candidate_id]
            value["attached_prefix_review_samples"] += 1
            if len(value["attached_prefix_examples"]) < 1:
                value["attached_prefix_examples"].append(example)
    result = []
    for row in candidates:
        value = counts[row["id"]]
        result.append(
            {
                **row,
                "training_evidence": value,
                "corpus_status": "observed_in_train"
                if value["exact_samples"]
                else "prefix_review_only"
                if value["attached_prefix_review_samples"]
                else "not_observed_under_conservative_match",
                "phoneme_status": "not_generated",
                "approved_for_retrieval": False,
            }
        )
    return result, {
        **totals,
        "arabic_hours_by_source": dict(hours),
        "candidates_with_exact_training_evidence": sum(
            bool(v["exact_samples"]) for v in counts.values()
        ),
    }


def verify_run(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    run = root / config["source_run"]
    expected = config["expected"]
    report_path = run / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "completed" or report["head"]["best_epoch"] != expected["best_epoch"]:
        raise ValueError("requires completed new EGY x3 epoch 29")
    for key in ("checkpoint_sha256", "vocab_sha256", "model_weight_sha256"):
        if report[key] != expected[key]:
            raise ValueError("selected run identity mismatch: " + key)
    for key in ("time_upsampling_factor", "num_classes"):
        if report["head"]["head_config"][key] != expected[key]:
            raise ValueError("selected Head shape mismatch: " + key)
    if report["selection"]["arabic_msa_used"] is not False:
        raise ValueError("requires new EGY source selection")
    if report["selection"]["manifests"]["train"]["sha256"] != expected["train_manifest_sha256"]:
        raise ValueError("selected training manifest mismatch")
    files = {
        str(report_path): sha(report_path),
        str(run / "head/ctc_head_best.pt"): expected["checkpoint_sha256"],
        str(root / config["vocab"]): expected["vocab_sha256"],
        str(run / "train.jsonl"): expected["train_manifest_sha256"],
    }
    for name, digest in files.items():
        pin(Path(name), digest)
    return {
        "files": files,
        "train_manifest": str(run / "train.jsonl"),
        "expected_rows": report["selection"]["manifests"]["train"]["samples"],
        "encoder_weight_sha256_reported_not_rehashed": expected["model_weight_sha256"],
    }


def write_review(output: Path, candidates: list[dict[str, Any]], report: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    pairs = nested_pairs(candidates)
    write_json(
        output / "candidates.json", {"status": "awaiting_user_review", "candidates": candidates}
    )
    write_json(output / "nested_pairs.json", pairs)
    report = {
        **report,
        "candidate_count": len(candidates),
        "nested_pairs": len(pairs),
        "categories": dict(Counter(r["category"] for r in candidates)),
        "test_set_read": False,
        "validation_set_read": False,
        "audio_read": False,
        "retrieval_started": False,
        "phonemization_started": False,
        "human_semantic_review_completed": False,
    }
    write_json(output / "report.json", report)
    lines = [
        "# 埃及阿语热词候选审阅表",
        "",
        "人工策划候选，不是最终词表；中文释义辅助审阅，尚未经阿语专家核准。",
        "无训练语料证据的草案不能被描述为从语料提取；词频也不等同业务价值。",
        "本阶段不读取验证/测试文本，不生成IPA，不运行检索。",
        "",
        "模型：新EGY ×3，第29轮；后续原始音频、整段Top5/Top7。",
        "后续评测按来源音频时长 open-source:gaoshan:coarse=8:1:1；本表类别数不受该比例约束。",
        "当前匹配保留完整词边界、标点边界和阿文字母差异，只去附标/延长线。",
        "前接成分只列为待审线索；零精确出现不代表该实体绝对不在语料中。",
        "",
        "| ID | 类别 | 阿语词面 | 中文释义 | 训练中精确出现句数 | 前接形式待审句数 |",
        "|---|---|---|---|---:|---:|",
    ]
    with (output / "candidates.tsv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "id",
                "category",
                "surface",
                "gloss_zh",
                "train_exact_samples",
                "attached_prefix_review_samples",
                "user_decision",
            ]
        )
        for row in candidates:
            evidence = row.get("training_evidence", {})
            fields = [
                row["id"],
                row["category"],
                row["surface"],
                row["gloss_zh"],
                evidence.get("exact_samples", "待工区核实"),
                evidence.get("attached_prefix_review_samples", "待工区核实"),
            ]
            writer.writerow([*fields, "待确认"])
            lines.append("| " + " | ".join(str(f).replace("|", "\\|") for f in fields) + " |")
    lines += [
        "",
        "## 嵌套关系：需确认计分和保留策略",
        "",
        "短词与长词均保留为审阅候选，不代表日后两者同时计入目标。",
    ]
    lines += [f"- {p['short']} ← {p['long']}" for p in pairs]
    (output / "候选词表.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
