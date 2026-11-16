"""Egyptian-source hours audit: audio headers only, no pronunciation or training decisions."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import wave
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl, read_records, transcript_key
from qwen_hotword.training.experiment_a import read_audio_metadata
from qwen_hotword.training.stage_progress import stage_progress

TAG = re.compile(r"^\s*language\s+(\w+)\s*<asr_text>\s*", re.IGNORECASE)
AGREED = {"exact", "punctuation_or_spacing_only"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def text_field(value: Any) -> tuple[str | None, str]:
    if not isinstance(value, str) or not value.strip():
        return None, "missing"
    match = TAG.match(value)
    if match:
        if match[1].casefold() != "arabic":
            return None, "wrong_language_tag"
        text = value[match.end() :].strip()
        return (text or None), "tagged"
    if "<asr_text>" in value or value.lstrip().lower().startswith("language "):
        return None, "malformed_tag"
    return value.strip(), "plain"


def comparison_key(text: str) -> str:
    """Diagnostic only. NEVER use this spelling-normalized text for phoneme labeling."""
    text = re.sub("[ً-ْٰـ]", "", transcript_key(text))
    return re.sub("[أإآٱ]", "ا", text).replace("ى", "ي").replace("ة", "ه")


def inspect(row: dict[str, Any], fallbacks: dict[str, str]) -> dict[str, Any]:
    response, response_format = text_field(row.get("response"))
    assistant, assistant_format = None, "missing"
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                assistant, assistant_format = text_field(message.get("content"))
                break
    if response is None or assistant is None:
        relation = "one_or_both_text_fields_invalid"
    elif response == assistant:
        relation = "exact"
    elif transcript_key(response) == transcript_key(assistant):
        relation = "punctuation_or_spacing_only"
    elif comparison_key(response) == comparison_key(assistant):
        relation = "arabic_spelling_normalization_equivalent"
    else:
        relation = "content_difference"
    issues = []
    if response is None:
        issues.append("response_" + response_format)
    if assistant is None:
        issues.append("assistant_" + assistant_format)
    if relation not in AGREED:
        issues.append(relation)
    if str(row.get("language", "")).casefold() not in {"ar", "arabic"}:
        issues.append("missing_or_non_arabic_language_field")
    dialect = row.get("dialect")
    if dialect is not None and str(dialect).casefold() not in {"egy", "egyptian", "ar-eg"}:
        issues.append("conflicting_dialect_field")
    if not any("ARABIC" in unicodedata.name(c, "") for c in (response or assistant or "")):
        issues.append("no_arabic_characters")
    audios = row.get("audios")
    original = audios[0] if isinstance(audios, list) and len(audios) == 1 else None
    audio = None
    seconds = None
    route = "original"
    if isinstance(original, str) and Path(original).is_absolute():
        try:
            path = Path(original)
            if not path.is_file():
                for old, new in fallbacks.items():
                    if path.is_relative_to(old):
                        candidate = Path(new) / path.relative_to(old)
                        if candidate.is_file():
                            path, route = candidate, "verified_fallback"
                            break
            audio = str(path.resolve())
            if not path.is_file():
                issues.append("audio_missing")
            else:
                before = path.stat()
                metadata = read_audio_metadata(path)
                if (
                    metadata.frames <= 0
                    or metadata.sample_rate <= 0
                    or not math.isfinite(metadata.duration_seconds)
                ):
                    raise ValueError("invalid audio header")
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("audio changed during header read")
                seconds = metadata.duration_seconds
        except (OSError, RuntimeError, ValueError, ZeroDivisionError, EOFError, wave.Error):
            issues.append("audio_header_unreadable_or_changed")
    else:
        issues.append("expected_one_absolute_audio_path")
    return {
        "response_text": response,
        "response_format": response_format,
        "assistant_text": assistant,
        "assistant_format": assistant_format,
        "text_relation": relation,
        "text_for_labeling": None,
        "audio_original": original if isinstance(original, str) else None,
        "audio_path": audio,
        "path_source": route,
        "duration_seconds": seconds,
        "declared_dialect": dialect,
        "embedded_source": row.get("source"),
        "issues": issues,
    }


def discover(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(config["root"])
    if not root.is_absolute():
        raise ValueError("source root must be absolute")
    sources, directories = [], []
    for group, spec in sorted(config["groups"].items(), key=lambda p: (p[1]["priority"], p[0])):
        if spec["role"] not in {"train_pool", "original_test"} or spec["priority"] < 1:
            raise ValueError("invalid source role/priority")
        directory = root / spec["directory"]
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("source directory escapes configured Egyptian root")
        paths = sorted(set(directory.glob("*.json")) | set(directory.glob("*.jsonl")))
        actual = {p.name for p in paths}
        directories.append(
            {
                "group": group,
                "directory": str(directory),
                "exists": directory.is_dir(),
                "missing_expected_files": sorted(set(spec["expected"]) - actual),
                "additional_files": sorted(actual - set(spec["expected"])),
            }
        )
        for path in paths:
            sources.append(
                {
                    "path": str(path),
                    "group": group,
                    "role": spec["role"],
                    "priority": spec["priority"],
                }
            )
    return sources, directories


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "readable_headers": sum(r["duration_seconds"] is not None for r in rows),
        "readable_hours": sum(r["duration_seconds"] or 0 for r in rows) / 3600,
        "strict_text_agreement_hours": sum(
            (r["duration_seconds"] or 0) for r in rows if not r["issues"]
        )
        / 3600,
        "issue_counts": dict(Counter(i for r in rows for i in r["issues"])),
        "text_relation_counts": dict(Counter(r["text_relation"] for r in rows)),
        "text_relation_hours": dict(
            sum(
                (Counter({r["text_relation"]: (r["duration_seconds"] or 0) / 3600}) for r in rows),
                Counter(),
            )
        ),
        "response_formats": dict(Counter(r["response_format"] for r in rows)),
        "dialect_fields": dict(Counter(str(r["declared_dialect"]) for r in rows)),
        "embedded_sources": dict(Counter(str(r["embedded_source"]) for r in rows)),
    }


def add_summary(target: dict[str, Any], part: dict[str, Any]) -> None:
    for key, value in part.items():
        if isinstance(value, dict):
            target[key] = dict(Counter(target.get(key, {})) + Counter(value))
        else:
            target[key] = target.get(key, 0) + value


def audit(
    root: Path, config_path: Path, output: Path, workers: int, resume: bool
) -> dict[str, Any]:
    if not 1 <= workers <= 64:
        raise ValueError("workers must be 1..64")
    config = read(config_path)
    with stage_progress("Discover Egyptian JSON/JSONL and fingerprint text inputs (not WAV)"):
        sources, directories = discover(config)
        source_hashes = {s["path"]: sha(Path(s["path"])) for s in sources}
        contract: dict[str, Any] = {
            "config_sha256": sha(config_path),
            "sources": source_hashes,
            "directories": directories,
            "code": {
                name: sha(root / name)
                for name in (
                    "src/qwen_hotword/training/egyptian_source_audit.py",
                    "scripts/audit_egyptian_sources.py",
                    "src/qwen_hotword/training/en_ar_source_audit.py",
                    "src/qwen_hotword/training/noah_source_inspection.py",
                    "src/qwen_hotword/training/experiment_a.py",
                )
            },
        }
    for name in [str(config_path), *source_hashes]:
        if output == Path(name).resolve() or output in Path(name).resolve().parents:
            raise ValueError("output overlaps protected input")
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if read(output / "contract.json") != contract:
            raise ValueError("audit identity changed; use a new output directory")
        if (output / "completed.json").exists():
            for name, digest in read(output / "completed.json").items():
                pin(output / name, digest)
            result = read(output / "report.json")
            for name, digest in result["artifacts_sha256"].items():
                pin(output / name, digest)
            return dict(result)
    else:
        write_json(output / "contract.json", contract)
    reports, parts = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for source_index, source in enumerate(sources):
            path = Path(source["path"])
            folder = output / "parts" / f"{source_index:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            report = {**source, "status": "checked", "sha256": source_hashes[str(path)]}
            processed = 0
            try:
                report["format"], records = read_records(path)
                part_number = 0
                with stage_progress(
                    f"{source_index + 1}/{len(sources)} {source['group']} {path.name}: headers",
                    every=256,
                ) as tick:
                    while batch := list(islice(records, 1024)):
                        marker = folder / f"{part_number:06d}.json"
                        data_path = marker.with_suffix(".jsonl")
                        if marker.exists():
                            completed = read(marker)
                            pin(data_path, completed["sha256"])
                            if completed["summary"]["rows"] != len(batch):
                                raise ValueError("saved part row count differs")
                            processed += len(batch)
                            tick(processed)
                        else:
                            checked = []
                            for item in executor.map(
                                partial(inspect, fallbacks=config["audio_fallbacks"]), batch
                            ):
                                processed += 1
                                checked.append(
                                    {**source, "source": source["path"], "row": processed, **item}
                                )
                                tick(processed)
                            pending = data_path.with_suffix(".pending")
                            with pending.open("w", encoding="utf-8") as stream:
                                for item in checked:
                                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                            pending.replace(data_path)
                            completed = {"sha256": sha(data_path), "summary": summarize(checked)}
                            write_json(marker, completed)
                        add_summary(report, completed["summary"])
                        parts.append(data_path)
                        part_number += 1
                pin(path, source_hashes[str(path)])
            except (OSError, ValueError) as error:
                report.update(status="failed", error=str(error), rows_parsed=processed)
            reports.append(report)
    # Aggregate only validated complete parts, counting every resolved path once across files.
    unique: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    inventory = output / "source_inventory.jsonl"
    with (
        inventory.with_suffix(".pending").open("w", encoding="utf-8") as stream,
        stage_progress("Summarize path duplicates, original-test overlap and hours") as tick,
    ):
        n = 0
        for part in parts:
            for row in iter_jsonl(part):
                n += 1
                tick(n)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                key = row["group"] + "/" + row["text_relation"]
                if len(examples.setdefault(key, [])) < 3:
                    examples[key].append(row)
                if row["duration_seconds"] is None:
                    continue
                path = row["audio_path"]
                text_digest = hashlib.sha256(
                    transcript_key(row["response_text"] or row["assistant_text"] or "").encode()
                ).hexdigest()
                if path not in unique:
                    unique[path] = {
                        "seconds": row["duration_seconds"],
                        "group": row["group"],
                        "priority": row["priority"],
                        "seen_test": False,
                        "seen_train": False,
                        "occurrences": 0,
                        "text_digest": text_digest,
                        "text_conflict": False,
                    }
                item = unique[path]
                if not math.isclose(item["seconds"], row["duration_seconds"], abs_tol=1e-8):
                    raise ValueError("same audio path has inconsistent header durations")
                item["seen_test"] |= row["role"] == "original_test"
                item["seen_train"] |= row["role"] == "train_pool"
                item["occurrences"] += 1
                item["text_conflict"] |= item["text_digest"] != text_digest
                if row["priority"] < item["priority"]:
                    item.update(priority=row["priority"], group=row["group"])
    inventory.with_suffix(".pending").replace(inventory)
    groups: dict[str, Any] = {}
    for report in reports:
        total = groups.setdefault(report["group"], {})
        add_summary(
            total,
            {
                k: report[k]
                for k in (
                    "rows",
                    "readable_headers",
                    "readable_hours",
                    "strict_text_agreement_hours",
                    "issue_counts",
                    "text_relation_counts",
                    "text_relation_hours",
                )
                if k in report
            },
        )
    unique_by_group: dict[str, Any] = {}
    for item in unique.values():
        if item["seen_test"]:
            continue
        summary = unique_by_group.setdefault(item["group"], {"paths": 0, "hours": 0.0})
        summary["paths"] += 1
        summary["hours"] += item["seconds"] / 3600
    write_json(output / "examples.json", examples)
    pin(config_path, contract["config_sha256"])
    for name, digest in source_hashes.items():
        pin(Path(name), digest)
    result = {
        "status": "checked"
        if sources
        and all(s["status"] == "checked" for s in reports)
        and all(d["exists"] and not d["missing_expected_files"] for d in directories)
        else "review",
        "output_dir": str(output),
        "mode": "full_egyptian_header_audit",
        "source_files": len(sources),
        "directories": directories,
        "sources": reports,
        "by_group": groups,
        "totals": {
            "rows": sum(s.get("rows", 0) for s in reports),
            "readable_hours_including_duplicates_and_original_test": sum(
                s.get("readable_hours", 0) for s in reports
            ),
            "unique_readable_paths": len(unique),
            "path_unique_hours_including_original_test": sum(v["seconds"] for v in unique.values())
            / 3600,
            "duplicate_path_extra_rows": sum(v["occurrences"] - 1 for v in unique.values()),
            "train_original_test_overlapping_paths": sum(
                v["seen_train"] and v["seen_test"] for v in unique.values()
            ),
            "path_unique_train_pool_hours_excluding_original_test": sum(
                v["seconds"] for v in unique.values() if not v["seen_test"]
            )
            / 3600,
            "duplicate_paths_with_text_difference": sum(
                v["text_conflict"] for v in unique.values()
            ),
        },
        "path_unique_train_pool_attribution": unique_by_group,
        "priority_policy": (
            "open_source_first; gaoshan/coarse_equal_priority; original_test_protected"
        ),
        "hours_basis": "audio headers; path dedup only; not verified usable training hours",
        "dialect_verified": False,
        "audio_text_alignment_verified": False,
        "audio_content_hash_computed": False,
        "labels_generated": False,
        "encoder_loaded": False,
        "training_started": False,
        "msa_supplement_used": False,
        "artifacts_sha256": {
            name: sha(output / name)
            for name in ("source_inventory.jsonl", "examples.json", "contract.json")
        },
        "return_file": str(output / "return_summary.zip"),
    }
    write_json(output / "report.json", result)
    with zipfile.ZipFile(output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("report.json", "contract.json", "examples.json"):
            archive.write(output / name, name)
    write_json(
        output / "completed.json",
        {name: sha(output / name) for name in ("report.json", "return_summary.zip")},
    )
    return result
