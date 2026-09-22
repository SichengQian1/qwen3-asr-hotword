"""Prepare an auditable staging inventory and word list, not a training split."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_hotword.training.experiment_a import read_audio_metadata
from qwen_hotword.training.g2p_prep import prepare_mfa_wordlist
from qwen_hotword.training.noah_source_inspection import (
    _hash,
    iter_json_array,
    resolve_audio_path,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric(groups: dict[str, Any], key: str, row: dict[str, Any]) -> None:
    metric = groups.setdefault(key, {"records": 0, "hours": 0.0})
    metric["records"] += 1
    metric["hours"] += row.get("duration_seconds", 0) / 3600


def _probe(item: tuple[int, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    index, raw = item
    row: dict[str, Any] = {
        "id": f"noah_es_{config['expected_sha256'][:12]}_{index}",
        "source_row": index,
        "source_corpus": "noah_es_mobile",
        "speaker_id": None,
        "split": "unassigned",
        "language": "es-419",
        "language_evidence": "user_confirmed_source_provenance",
    }
    try:
        audios, response, messages = raw.get("audios"), raw.get("response"), raw.get("messages")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            raise ValueError("expected exactly one audio path")
        if not Path(audios[0]).is_absolute():
            raise ValueError("audio path must be absolute")
        row["original_audio_path"] = audios[0]
        row["original_response"] = response
        assistants = (
            [
                m.get("content")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if isinstance(messages, list)
            else []
        )
        if len(assistants) != 1 or assistants[0] != response:
            raise ValueError("assistant/response conflict or unsupported conversation")
        match = (
            re.fullmatch(r"language Spanish<asr_text>([\s\S]*)", response)
            if isinstance(response, str)
            else None
        )
        if raw.get("language") != "Spanish" or not match or not match.group(1).strip():
            raise ValueError("invalid language/transcript contract")
        row["text"] = match.group(1).strip()
        parts = Path(audios[0]).parts
        if "noah_esZ00" not in parts[:-1]:
            raise ValueError("unrecognized source folder")
        row["source_batch"] = parts[parts.index("noah_esZ00") + 1]
        row["directory_group_hint"] = (
            parts[parts.index("category") + 1] if "category" in parts[:-1] else None
        )
        # Preserve any future source identity fields, without treating path hints as speakers.
        row["original_identity_metadata"] = {
            key: raw[key]
            for key in ("speaker_id", "speaker", "country", "accent", "dialect", "split")
            if key in raw
        }
        audio, path_status = resolve_audio_path(audios[0], config["audio_prefix_rewrites"])
        if audio is None:
            raise FileNotFoundError("mapped audio file missing")
        row["audio_path"] = str(audio.resolve())
        row["path_status"] = path_status
        before = audio.stat()
        info = read_audio_metadata(audio)
        if info.frames <= 0 or info.sample_rate <= 0 or not math.isfinite(info.duration_seconds):
            raise ValueError("invalid audio metadata")
        checksum = _hash(audio)
        after = audio.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("audio changed during inspection")
        row.update(
            {
                "audio_file_sha256": checksum,
                "audio_size_bytes": after.st_size,
                "duration_seconds": info.duration_seconds,
                "audio_frames": info.frames,
                "sample_rate": info.sample_rate,
                "status": "audio_text_candidate",
            }
        )
    except Exception as error:
        row["status"] = "needs_review"
        row["issue"] = f"{type(error).__name__}: {error}"[:500]
    return row


def prepare_noah_source(
    config: dict[str, Any],
    output: Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Hash and inspect every file, quarantine conflicting copies, preserve all source rows."""
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    source = Path(config["input"])
    if _hash(source) != config["expected_sha256"]:
        raise ValueError("source JSON SHA256 mismatch")
    output.mkdir(parents=True, exist_ok=False)
    print(f"New staging output: {output}", file=sys.stderr, flush=True)
    _write_json(output / "input_config.json", config)
    try:
        report = _prepare(config, output, workers)
    except Exception as error:
        _write_json(
            output / "report.json",
            {
                "status": "failed",
                "output_dir": str(output),
                "error": f"{type(error).__name__}: {error}",
                "training_ready": False,
                "partial_outputs_preserved": True,
            },
        )
        raise
    _write_json(output / "report.json", report)
    checks = [
        f"{_hash(path)}  {path.relative_to(output)}\n"
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    (output / "sha256.txt").write_text("".join(checks), encoding="utf-8")
    return report


def _prepare(config: dict[str, Any], output: Path, workers: int) -> dict[str, Any]:
    scanned: dict[str, Any] = {}
    readable: dict[str, Any] = {}
    sample_rates: Counter[str] = Counter()
    hashes: dict[str, str] = {}
    conflict_hashes: set[str] = set()
    inventory = output / "inventory.jsonl"
    records = enumerate(iter_json_array(Path(config["input"])), 1)
    count = 0
    with inventory.open("w", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as executor:
        while batch := list(islice(records, 256)):
            for row in executor.map(lambda item: _probe(item, config), batch):
                count += 1
                _metric(scanned, row.get("source_batch", "unknown"), row)
                if row["status"] == "audio_text_candidate":
                    _metric(readable, row["source_batch"], row)
                    sample_rates[str(row["sample_rate"])] += 1
                    checksum = row["audio_file_sha256"]
                    text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
                    if checksum in hashes and hashes[checksum] != text_hash:
                        conflict_hashes.add(checksum)
                    hashes[checksum] = text_hash
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if count % 5120 == 0:
                print(f"Audio metadata + file SHA256 checked: {count}", file=sys.stderr, flush=True)
    if count == 0:
        raise ValueError("source contains no records")
    if _hash(Path(config["input"])) != config["expected_sha256"]:
        raise ValueError("source changed during preparation")

    candidate_totals: dict[str, Any] = {}
    review_totals: dict[str, Any] = {}
    seen: set[str] = set()
    source_tsv = output / "source.tsv"
    with (
        inventory.open(encoding="utf-8") as handle,
        source_tsv.open("w", encoding="utf-8", newline="") as tsv_handle,
        (output / "candidates.jsonl").open("w", encoding="utf-8") as candidate_handle,
        (output / "needs_review.jsonl").open("w", encoding="utf-8") as review_handle,
    ):
        fields = [
            "audio",
            "text",
            "source_id",
            "source_corpus",
            "source_batch",
            "language",
            "speaker_id",
            "directory_group_hint",
            "source_split",
        ]
        writer = csv.DictWriter(tsv_handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for line in handle:
            row = json.loads(line)
            checksum = row.get("audio_file_sha256")
            reason = "audio_or_structure_issue" if row["status"] != "audio_text_candidate" else None
            if not reason and checksum in conflict_hashes:
                reason = "identical_file_conflicting_transcripts"
            elif not reason and checksum in seen:
                reason = "duplicate_file_same_transcript"
            if reason:
                row["preparation_reason"] = reason
                _metric(review_totals, reason, row)
                review_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            seen.add(checksum)
            _metric(candidate_totals, row["source_batch"], row)
            candidate_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            writer.writerow(
                {
                    "audio": row["audio_path"],
                    "text": row["text"],
                    "source_id": row["id"],
                    "source_corpus": row["source_corpus"],
                    "source_batch": row["source_batch"],
                    "language": row["language"],
                    "speaker_id": "",
                    "directory_group_hint": row["directory_group_hint"] or "",
                    "source_split": "unsplit",
                }
            )
    kept = sum(m["records"] for m in candidate_totals.values())
    reviewed = sum(m["records"] for m in review_totals.values())
    if kept + reviewed != count:
        raise RuntimeError("source record partition is inconsistent")
    wordlist = None
    if kept:
        wordlist = prepare_mfa_wordlist(source_tsv, output / "wordlist").to_dict()
        _write_json(output / "wordlist/summary.json", wordlist)
    return {
        "status": "completed" if kept else "no_candidates",
        "scope": "source_inventory_and_wordlist_not_training_manifest",
        "output_dir": str(output),
        "source_json_sha256": config["expected_sha256"],
        "source_record_count": count,
        "source_partition_by_batch": scanned,
        "readable_audio_by_batch_before_deduplication": readable,
        "sample_rate_counts": dict(sample_rates),
        "unique_file_hashes_before_conflict_exclusion": len(hashes),
        "conflicting_file_hash_groups": len(conflict_hashes),
        "candidate_records": kept,
        "candidate_hours": sum(m["hours"] for m in candidate_totals.values()),
        "candidate_by_batch": candidate_totals,
        "review_records": reviewed,
        "review_by_reason": review_totals,
        "wordlist": wordlist,
        "speaker_identity_confirmed": False,
        "split_assigned": False,
        "existing_holdout_overlap_checked": False,
        "phoneme_labels_prepared": False,
        "ctc_feasibility_checked": False,
        "training_ready": False,
        "model_loaded": False,
        "test_set_content_read": False,
        "limitations": [
            "Candidate hours are staging audio/text hours, before G2P/CTC and split filtering.",
            "Audio metadata and file bytes were read, not a full waveform decoding/label audit.",
            "File hashes detect identical bytes, not re-encoding or overlapping segments.",
            "All conflicting copies are quarantined, including the first occurrence.",
            "Speaker and recording group identity remains unverified; no speaker-disjoint claim.",
            "Cross-split overlap must be resolved before final training selection.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
