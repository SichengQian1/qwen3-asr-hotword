"""Read completed Chinese/English label metadata without rerunning preparation."""

from __future__ import annotations

import io
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows
from qwen_hotword.training.zh_en_data import audio_bytes


def inspect_preparation(source: Path, *, headers_per_group: int = 3) -> dict[str, Any]:
    if not 0 <= headers_per_group <= 20:
        raise ValueError("header samples per group must be between 0 and 20")
    identity = read(source / "identity.json")
    pin(source / "contract.json", identity["contract_sha256"])
    config = read(source / "config.json")
    if config != read(source / "contract.json")["config"]:
        raise ValueError("saved configuration differs from the original run contract")
    previous = read(source / "label_report.json")
    parts = sorted((source / "labels").glob("*/rows.jsonl"))
    if not parts:
        raise ValueError("no completed label parts")
    g2p_sha = sha(source / "g2p_report.json")
    totals: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = {}
    groups: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    snapshots = {}
    with stage_progress(
        "Read completed labels: duration / exclusions / all splits", total=len(parts), every=25
    ) as tick:
        for number, path in enumerate(parts, 1):
            marker = read(path.parent / "completed.json")
            if marker["identity"]["g2p_report"] != g2p_sha:
                raise ValueError("label part belongs to a different G2P result")
            expected = marker["files"]["rows.jsonl"]
            pin(path, expected)
            snapshots[str(path.relative_to(source))] = expected
            seen = 0
            for row in rows(path):
                seen += 1
                group, split = row["group"], row["split"]
                totals[group] += 1
                sources.setdefault(group, Counter())[row["source"]] += 1
                stat = groups.setdefault(
                    group + "/" + split,
                    {
                        "records": 0,
                        "known_seconds": 0.0,
                        "duration_missing_records": 0,
                        "usable_records": 0,
                        "usable_seconds": 0.0,
                        "excluded_records": 0,
                        "excluded_known_seconds": 0.0,
                        "failure_occurrences": Counter(),
                        "failure_known_seconds": Counter(),
                    },
                )
                stat["records"] += 1
                seconds = row.get("seconds")
                if seconds is None:
                    stat["duration_missing_records"] += 1
                else:
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError("invalid saved audio duration")
                    stat["known_seconds"] += seconds
                if row["issues"]:
                    stat["excluded_records"] += 1
                    stat["excluded_known_seconds"] += seconds or 0.0
                    stat["failure_occurrences"].update(row["issues"])
                    for reason in set(row["issues"]):
                        stat["failure_known_seconds"][reason] += seconds or 0.0
                    bucket = examples.setdefault(group, [])
                    if len(bucket) < 8:
                        bucket.append({k: row[k] for k in ("id", "text", "issues")})
                elif split in {"train", "validation"}:
                    ids = row.get("phoneme_token_ids")
                    if not ids or any(type(i) is not int or i < 2 or i >= 183 for i in ids):
                        raise ValueError("unflagged row does not have a complete 183-class label")
                    if seconds is None:
                        raise ValueError("usable row is missing duration")
                    stat["usable_records"] += 1
                    stat["usable_seconds"] += seconds
                # Test rows deliberately have no labels: never classify them as usable training.
                if (
                    split != "test"
                    and not row["issues"]
                    and seconds is not None
                    and len(samples.setdefault(group, [])) < headers_per_group
                ):
                    samples[group].append(row)
            if seen != marker["summary"]["rows"]:
                raise ValueError("label part row count differs from receipt")
            tick(number)
    if dict(totals) != previous["totals"]:
        raise ValueError("not all original label rows are present")
    availability = {}
    for split, targets in config["hours"].items():
        for group, target in targets.items():
            key = group + "/" + split
            stat = groups.get(key, {})
            hours = stat.get("usable_seconds", 0.0) / 3600
            minimum = config["minimum_hours"][split][group]
            availability[key] = {
                "usable_records": stat.get("usable_records", 0),
                "available_hours": hours,
                "target_hours": target,
                "minimum_hours": minimum,
                "minimum_met": hours >= minimum,
                "shortfall_to_target_hours": max(0.0, target - hours),
            }
    by_source_group = {}
    for group in totals:
        stats = [s for k, s in groups.items() if k.startswith(group + "/")]
        by_source_group[group] = {
            "records": totals[group],
            "known_original_hours": sum(s["known_seconds"] for s in stats) / 3600,
            "duration_missing_records": sum(s["duration_missing_records"] for s in stats),
            "source_files": len(sources[group]),
        }
    header_checks = []
    selected = [r for group in sorted(samples) for r in samples[group]]
    with stage_progress(
        "Check bounded original WAV headers (no Encoder / no waveform decode)",
        total=len(selected),
        every=1,
    ) as tick:
        for number, row in enumerate(selected, 1):
            try:
                import soundfile as sf

                info = sf.info(io.BytesIO(audio_bytes(row)))
                seconds = info.frames / info.samplerate
                difference = abs(seconds - row["seconds"])
                header_checks.append(
                    {
                        "id": row["id"],
                        "group": row["group"],
                        "audio": row["audio"],
                        "sample_rate": info.samplerate,
                        "channels": info.channels,
                        "source_frames": info.frames,
                        "source_seconds": seconds,
                        "saved_seconds": row["seconds"],
                        "absolute_difference_seconds": difference,
                        "status": "pass" if difference <= 1 / 16000 + 1e-9 else "mismatch",
                    }
                )
            except (OSError, RuntimeError, ValueError) as error:
                header_checks.append({"id": row["id"], "status": "error", "error": str(error)})
            tick(number)
    return {
        "status": "inspection_completed",
        "source_run": str(source),
        "source_git_commit": identity["git_commit"],
        "source_contract_sha256": identity["contract_sha256"],
        "all_label_parts_verified": len(parts),
        "label_parts_sha256": digest(snapshots),
        "by_source_group": by_source_group,
        "by_split": groups,
        "availability": availability,
        "failure_examples_by_group": examples,
        "source_file_record_counts": sources,
        "header_checks": header_checks,
        "header_check_scope": "bounded diagnostic; not verification of every original WAV",
        "gpu_used": False,
        "g2p_rerun": False,
        "training_started": False,
        "source_run_modified": False,
        "test_audio_read": False,
        "test_metadata_only": True,
    }


def save_inspection(source: Path, output: Path, *, headers_per_group: int = 3) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("diagnostic output must be separate from the training run")
    output.mkdir(parents=True, exist_ok=False)
    report = inspect_preparation(source, headers_per_group=headers_per_group)
    write_json(output / "report.json", report)
    checks = f"{sha(output / 'report.json')}  report.json\n"
    with zipfile.ZipFile(output / "return_bundle.zip", "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(output / "report.json", "report.json")
        zipped.writestr("sha256.txt", checks)
    return {
        k: report[k] for k in ("status", "by_source_group", "availability", "header_checks")
    } | {
        "return_bundle": str(output / "return_bundle.zip"),
        "source_run_modified": False,
        "g2p_rerun": False,
        "training_started": False,
    }
