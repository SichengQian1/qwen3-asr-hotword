"""Read-only Portuguese train-source audit and bounded, listenable label review kit."""

from __future__ import annotations

import hashlib
import heapq
import html
import json
import math
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, run_worker, sha, write_json
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text
from qwen_hotword.training.stage_progress import stage_progress

ALIASES = {"en": "en", "en-US": "en", "es": "es", "es-419": "es", "pt": "pt", "pt-BR": "pt"}


def input_issues(text: str) -> list[str]:
    issues = []
    if any(c.isdigit() for c in text):
        issues.append("numbers_require_explicit_verbalization")
    if any(c in "&+%$@#=*" for c in text):
        issues.append("symbols_require_explicit_verbalization")
    if not extract_word_tokens(text):
        issues.append("no_lexical_words")
    return issues


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def source_manifest(root: Path, config: dict[str, Any]) -> tuple[Path, str]:
    source = root / config["source_run"]
    plan_path = source / "plan.json"
    pin(plan_path, config["source_plan_sha256"])
    plan = json.loads(plan_path.read_text())
    train = root / plan["train_manifest"]
    expected = {
        os.path.abspath(root / name): digest for name, digest in plan["input_sha256"].items()
    }.get(os.path.abspath(train))
    if expected is None:
        raise ValueError("baseline plan does not pin its actual training manifest")
    with stage_progress("Verify actual baseline TRAIN manifest SHA (no heldout reads)"):
        pin(train, expected)
    return train, expected


def portuguese_targets(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / config["hotword_table"]
    pin(path, config["hotword_table_sha256"])
    table = json.loads(path.read_text())
    if len(table["entries"]) != config["max_hotword_entries"]:
        raise ValueError("joint table size differs")
    result = []
    for entry in table["entries"]:
        for variant in entry["variants"]:
            if variant["language"] in {"pt", "pt-BR"} and variant.get("role", "target") == "target":
                result.append(
                    {
                        "surface": variant["surface"],
                        "words": extract_word_tokens(variant["surface"]),
                        "old_pronunciation": variant.get("pronunciation"),
                        "old_token_ids": variant.get("token_ids"),
                        "surface_input_issues": input_issues(variant["surface"]),
                    }
                )
    if not result:
        raise ValueError("no Portuguese business targets found in pinned joint table")
    return result


def hints(words: list[str]) -> list[str]:
    text = " ".join(words)
    patterns = {
        "r": r"r",
        "nh": r"nh",
        "lh": r"lh",
        "nasal_spelling": r"[ãõ]",
        "qu_gu": r"[qg]u",
        "t_d_i_e": r"[td][ieíéê]",
        "vowel_quality": r"[éêóô]",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def scan_sources(
    root: Path,
    config: dict[str, Any],
    output: Path,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    train, digest = source_manifest(root, config)
    marker = output / "source_audit.completed.json"
    if marker.exists():
        done = json.loads(marker.read_text())
        if done["train_sha256"] != digest:
            raise ValueError("source manifest changed")
        for name, checksum in done["files"].items():
            pin(output / name, checksum)
        return list(rows(output / "selected_audio.jsonl"))
    size = int(config["review_samples"])
    if not 1 <= size <= 500:
        raise ValueError("foundation audio review is bounded to 1..500 samples")
    counts: Counter[str] = Counter()
    groups: dict[str, dict[str, Any]] = {}
    duplicates: Counter[str] = Counter()
    missing_metadata: Counter[str] = Counter()
    ids: set[str] = set()
    paths: set[str] = set()
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    retained: dict[str, dict[str, Any]] = {}
    references: Counter[str] = Counter()
    target_starts: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for target in targets:
        if target["words"] and not target["surface_input_issues"]:
            target_starts[target["words"][0]].append(tuple(target["words"]))

    def offer(key: str, priority: int, row: dict[str, Any]) -> None:
        heap, sample_id = heaps[key], row["id"]
        item = (-priority, sample_id)
        if len(heap) >= size and item <= heap[0]:
            return
        if len(heap) >= size:
            _, removed = heapq.heappop(heap)
            references[removed] -= 1
            if not references[removed]:
                retained.pop(removed, None)
        heapq.heappush(heap, item)
        retained[sample_id] = row
        references[sample_id] += 1

    with stage_progress(
        "Audit actual train composition; select PT source/risk review samples",
        total=sum(config["expected_train_counts"].values()),
    ) as update:
        for number, row in enumerate(rows(train), 1):
            lang = ALIASES.get(str(row.get("language", "")))
            if lang is None or row.get("split") != "train":
                raise ValueError("only known-language TRAIN records allowed")
            counts[lang] += 1
            update(number)
            if lang != "pt":
                continue
            sample_id, audio = row["id"], row["audio_path"]
            seconds = float(row["duration_seconds"])
            if not isinstance(sample_id, str) or not sample_id or not Path(audio).is_absolute():
                raise ValueError("invalid PT sample ID or non-absolute source audio")
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid source audio duration")
            lexical_path = os.path.normpath(audio)
            repeated = sample_id in ids or lexical_path in paths
            duplicates["repeated_ids"] += sample_id in ids
            duplicates["repeated_lexical_audio_paths"] += lexical_path in paths
            ids.add(sample_id)
            paths.add(lexical_path)
            source = str(row.get("source_corpus") or row.get("source") or "unknown")
            declared = str(row.get("dialect") or row.get("accent") or "unknown")
            group = groups.setdefault(
                source,
                {
                    "records": 0,
                    "seconds": 0.0,
                    "declared_languages": Counter(),
                    "declared_dialects": Counter(),
                    "release_sources": Counter(),
                },
            )
            group["records"] += 1
            group["seconds"] += seconds
            group["declared_languages"][row["language"]] += 1
            group["declared_dialects"][declared] += 1
            group["release_sources"][str(row.get("release_source", "unknown"))] += 1
            for field in ("speaker_id", "session_id", "dialect"):
                missing_metadata[field] += row.get(field) in (None, "")
            if repeated:
                continue
            words = extract_word_tokens(row["text"])
            business = any(
                tuple(words[i : i + len(term)]) == term
                for i, word in enumerate(words)
                for term in target_starts.get(word, [])
            )
            feature_hints = hints(words)
            selected_row = {
                **row,
                "audit_source": source,
                "orthographic_hints": feature_hints,
                "contains_business_target_text": business,
                "reference_input_issues": input_issues(row["text"]),
            }
            priority = int(hashlib.sha256(f"{config['seed']}:{sample_id}".encode()).hexdigest(), 16)
            for key in ["all", "source:" + source] + ["hint:" + h for h in feature_hints]:
                offer(key, priority, selected_row)
            if business:
                offer("business", priority, selected_row)
    if dict(counts) != config["expected_train_counts"]:
        raise ValueError(f"actual training language counts changed: {dict(counts)}")
    selected: dict[str, dict[str, Any]] = {}
    if "business" in heaps:
        for _, sample_id in sorted(heaps["business"], reverse=True)[: size // 2]:
            selected[sample_id] = retained[sample_id]
    queues = {k: iter(sorted(v, reverse=True)) for k, v in sorted(heaps.items()) if k != "business"}
    while len(selected) < size and queues:
        for key in list(queues):
            try:
                _, sample_id = next(queues[key])
                selected.setdefault(sample_id, retained[sample_id])
                if len(selected) == size:
                    break
            except StopIteration:
                del queues[key]
    if len(selected) != size:
        raise ValueError(f"not enough unique PT training audio: {len(selected)}/{size}")
    for group in groups.values():
        group["hours"] = group["seconds"] / 3600
        group["dialect_evidence"] = "declared metadata only; not listened"
    report = {
        "status": "completed_metadata_audit",
        "train_manifest": str(train),
        "train_sha256": digest,
        "train_counts": dict(counts),
        "pt_sources": groups,
        "duplicates": dict(duplicates),
        "pt_missing_metadata_records": dict(missing_metadata),
        "selected_audio": size,
        "selected_sources": dict(Counter(r["audit_source"] for r in selected.values())),
        "selected_business_target_audio": sum(
            r["contains_business_target_text"] for r in selected.values()
        ),
        "selection": "deterministic source/risk enrichment; not population evaluation",
        "dialect_verified": False,
        "speaker_disjoint_verified": False,
        "cross_split_audio_byte_audit": False,
        "test_set_used": False,
        "human_listening": "pending",
        "orthographic_hints_are_phonetic_truth": False,
    }
    write_json(output / "source_audit.json", report)
    ordered = sorted(selected.values(), key=lambda r: r["id"])
    write_rows(output / "selected_audio.jsonl", ordered)
    write_json(
        marker,
        {
            "train_sha256": digest,
            "files": {p: sha(output / p) for p in ("source_audit.json", "selected_audio.jsonl")},
        },
    )
    return ordered


def lexicon(path: Path, wanted: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Read optional MFA numeric columns as parameters, never as phone labels."""
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        word = normalize_training_text(fields[0])
        if word not in wanted:
            continue
        index, parameters = 1, []
        while index < len(fields):
            try:
                value = float(fields[index])
            except ValueError:
                break
            parameters.append(value)
            index += 1
        if index == len(fields):
            raise ValueError(f"dictionary entry has no phones: {word}")
        result[word].append({"ipa": " ".join(fields[index:]), "mfa_parameters": parameters})
    return dict(result)


def build_words(
    root: Path,
    config: dict[str, Any],
    assets: Path,
    output: Path,
    prefix: list[str],
    selected: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    words = {w for row in selected for w in extract_word_tokens(row["text"])}
    words.update(w for target in targets for w in target["words"])
    if len(words) > config["max_review_words"]:
        raise ValueError(f"bounded review exceeds max_review_words: {len(words)}")
    request = output / "review_words_request.jsonl"
    write_rows(request, [{"language": "pt", "word": w} for w in sorted(words)])
    prediction = output / "review_g2p.jsonl"
    run_worker(root, config, assets, prefix, request, prediction)
    generated = {r["word"]: r for r in rows(prediction)}
    spec = config["models"]["pt"]
    dictionary = assets / "pt/dictionary" / (spec["dialect"] + ".dict")
    supplied = lexicon(dictionary, words)
    vocab = load_phoneme_vocab(root / config["vocab"])
    combined = {}
    for word in sorted(words):
        variants = []
        for candidate in supplied.get(word, []):
            converted = tokenize_ipa_to_vocab(candidate["ipa"], vocab)
            variants.append(
                {
                    **candidate,
                    "tokens": converted.tokens,
                    "token_ids": converted.token_ids,
                    "oov_units": converted.oov_units,
                    "usable": bool(converted.token_ids) and not converted.oov_units,
                }
            )
        combined[word] = {
            "word": word,
            "g2p": generated[word],
            "official_dictionary_candidates": variants,
            "training_pronunciation_selected": False,
            "dictionary_file_order_is_ranking": False,
        }
    write_rows(output / "word_pronunciations.jsonl", list(combined.values()))
    write_rows(
        output / "business_hotword_review.jsonl",
        [
            {
                **t,
                "new_words": [combined[w] for w in t["words"]],
                "human_pronunciation_audit": "pending",
                "training_or_runtime_table_updated": False,
                "whole_surface_phonemization_ready": False,
            }
            for t in targets
        ],
    )
    return combined


def render_review(cases: list[dict[str, Any]]) -> str:
    sections = []

    def escape(value: Any) -> str:
        return html.escape(str(value))

    for row in cases:
        words = []
        for item in row["words"]:
            g2p = item["g2p"]
            native = [c["ipa"] for c in g2p["candidates"]]
            supplied = [c["ipa"] for c in item["official_dictionary_candidates"]]
            words.append(
                f"<tr><td>{escape(item['word'])}</td><td>{escape(' / '.join(native))}"
                f"<br>{escape(g2p['status'])}</td><td>{escape(' / '.join(supplied[:8]))}"
                f"<br>共{len(supplied)}条，完整数据见review.jsonl</td></tr>"
            )
        audio = (
            f'<audio controls preload="none" src="{escape(row["audio_file"])}"></audio>'
            if row["audio_file"]
            else "原音频复制失败：" + escape(row["audio_error"])
        )
        sections.append(
            f'<article data-id="{escape(row["id"])}"><h2>{row["number"]}. '
            f"{escape(row['id'])}</h2>{audio}<p>来源：{escape(row['source'])}</p>"
            f"<p>参考文字（不保证与口语完全一致）：{escape(row['text'])}</p>"
            f"<p>旧训练标签：{escape(' '.join(row['old_tokens']))}</p>"
            f"<p>新版G2P第一候选整句预览（仅供听审）：{escape(row['new_preview'])}</p>"
            "<table><tr><th>原词</th><th>新版G2P候选</th><th>官方词典候选（未选定）</th></tr>"
            + "".join(words)
            + '</table><p>听审：<select class="judgment">'
            '<option value="pending">待审核</option><option value="new_better">新版较好</option>'
            '<option value="old_better">旧版较好</option>'
            '<option value="both_plausible">两者均可</option>'
            '<option value="neither">两者均不符</option><option value="uncertain">无法判断</option>'
            '</select> 方言：<select class="dialect"><option value="unknown">未知／待确认</option>'
            '<option value="br">巴葡</option><option value="pt">葡葡</option>'
            '<option value="mixed">混语／混合</option></select></p>'
            '<textarea placeholder="记录实际发音、专名纠正、口语省略、音文不一致等">'
            "</textarea></article>"
        )
    script = """<script>
document.querySelector('button').onclick=()=>{
const rows=[...document.querySelectorAll('article')].map(a=>({id:a.dataset.id,
judgment:a.querySelector('.judgment').value,dialect:a.querySelector('.dialect').value,
notes:a.querySelector('textarea').value}));
const link=document.createElement('a');link.href=URL.createObjectURL(new Blob(
[JSON.stringify({status:'human_review',cases:rows},null,2)],{type:'application/json'}));
link.download='human_review.json';link.click();};</script>"""
    return (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>巴葡发音听审</title>'
        "<style>body{font:17px system-ui;max-width:1150px;margin:30px auto;padding:20px;"
        "line-height:1.6}article{border-top:2px solid #ddd;padding:20px 0}table{border-collapse:"
        "collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}"
        "textarea{width:95%;height:80px}button{padding:12px;position:sticky;top:5px}</style>"
        "<h1>巴葡原音频与新旧标签听审</h1><p>仅训练集抽样；自动标签不是正确答案。"
        "G2P和词典提供候选，尚未根据音频选定。请由熟悉巴葡并能核对音标的同事听审。"
        "填写结果只在当前页面内存中，关闭前点击导出；不会自动保存或上传。</p>"
        "<button>导出本次人工审核 JSON</button>" + "".join(sections) + script + "</html>"
    )


def export_review(
    root: Path,
    config: dict[str, Any],
    output: Path,
    selected: list[dict[str, Any]],
    words: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    package = output / "review_package"
    (package / "audio").mkdir(parents=True, exist_ok=True)
    old = load_phoneme_vocab(root / config["old_vocab"])
    cases, errors = [], []
    with stage_progress(
        "Copy unchanged selected PT originals; build listening kit", total=len(selected), every=20
    ) as update:
        for number, row in enumerate(selected, 1):
            source = Path(row["audio_path"])
            extension = source.suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,8}", extension):
                raise ValueError("unsafe/unsupported audio file suffix")
            name = f"audio/{number:04d}{extension}"
            case_words = [words[w] for w in extract_word_tokens(row["text"])]
            preview: list[str] = []
            for item in case_words:
                usable = [c for c in item["g2p"]["candidates"] if c["usable"]]
                if not usable:
                    preview = []
                    break
                preview.extend(usable[0]["tokens"])
            if input_issues(row["text"]):
                preview = []
            ids = row["phoneme_token_ids"]
            if not ids or any(type(i) is not int or not 2 <= i < len(old.tokens) for i in ids):
                raise ValueError("invalid saved old labels")
            case = {
                "number": number,
                "id": row["id"],
                "source": row["audit_source"],
                "text": row["text"],
                "original_audio_path": str(source),
                "old_token_ids": ids,
                "old_tokens": [old.tokens[i] for i in ids],
                "new_preview": " ".join(preview) if preview else "UNRESOLVED（不生成部分整句标签）",
                "words": case_words,
                "reference_input_issues": input_issues(row["text"]),
                "audio_file": name,
                "audio_error": None,
                "human_review": "pending",
                "new_preview_is_training_label": False,
            }
            try:
                checksum = sha(source)
                target = package / name
                if target.exists():
                    pin(target, checksum)
                else:
                    temporary = target.with_name(target.name + ".pending")
                    shutil.copyfile(source, temporary)
                    pin(temporary, checksum)
                    temporary.rename(target)
                case["audio_sha256"] = checksum
            except OSError as error:
                case.update(audio_file=None, audio_error=str(error))
                errors.append({"id": row["id"], "error": str(error)})
            cases.append(case)
            update(number)
    write_rows(package / "review.jsonl", cases)
    (package / "review.html").write_text(render_review(cases), encoding="utf-8")
    with zipfile.ZipFile(output / "review_audio.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file() and not path.name.endswith(".pending"):
                archive.write(path, path.relative_to(package))
    result = {
        "status": "ready_for_human_review" if not errors else "audio_copy_incomplete",
        "cases": len(cases),
        "audio_copied": len(cases) - len(errors),
        "audio_errors": errors,
        "human_listened": False,
        "pronunciation_accuracy_verified": False,
        "review_html": str(package / "review.html"),
        "review_audio_zip": str(output / "review_audio.zip"),
        "new_word_coverage": {
            "unique_words": len(words),
            "native_g2p_usable": sum(v["g2p"]["status"] == "ok" for v in words.values()),
            "dictionary_has_usable_candidate": sum(
                any(c["usable"] for c in v["official_dictionary_candidates"])
                for v in words.values()
            ),
            "g2p_statuses": dict(Counter(v["g2p"]["status"] for v in words.values())),
        },
        "ctc_per": None,
        "retrieval_recall": None,
        "asr_wer": None,
        "gpu_used": False,
        "encoder_loaded": False,
        "test_set_used": False,
    }
    write_json(output / "review_report.json", result)
    return result
