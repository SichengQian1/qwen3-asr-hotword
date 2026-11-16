"""Persistent byte hashes with explicit header precision and per-record review reasons."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.experiment_a import read_audio_metadata
from qwen_hotword.training.stage_progress import stage_progress

# Legacy full manifests round seconds to six decimal places. Never allow millisecond drift.
DURATION_ROUNDING_SECONDS = 0.5e-6 + 1e-12


def header_and_hash(path: str, cached: tuple[Any, ...] | None) -> tuple[Any, ...]:
    audio = Path(path)
    before = audio.stat()
    digest = (
        cached[2] if cached and cached[:2] == (before.st_size, before.st_mtime_ns) else sha(audio)
    )
    header = read_audio_metadata(audio)
    after = audio.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"audio changed while reading: {audio}")
    if not math.isfinite(header.duration_seconds) or header.duration_seconds <= 0:
        raise ValueError(f"invalid actual audio duration: {audio}")
    return path, after.st_size, after.st_mtime_ns, digest, header.duration_seconds


def fingerprint(rows: list[dict[str, Any]], output: Path, workers: int) -> None:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    with sqlite3.connect(output / "audio_hashes.sqlite") as db, ThreadPoolExecutor(workers) as pool:
        db.execute(
            "CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY,size INT,mtime INT,sha TEXT)"
        )
        db.execute(
            "CREATE TABLE IF NOT EXISTS headers("
            "path TEXT PRIMARY KEY,size INT,mtime INT,sha TEXT,seconds REAL)"
        )
        with stage_progress(
            "Verify audio bytes and header durations (persistent cache)", total=len(rows)
        ) as tick:
            for start in range(0, len(rows), 512):
                batch = rows[start : start + 512]
                pending = {}
                for row in batch:
                    name = row["audio_path"]
                    stat = Path(name).stat()
                    cached_header = db.execute(
                        "SELECT size,mtime FROM headers WHERE path=?", (name,)
                    ).fetchone()
                    if cached_header == (stat.st_size, stat.st_mtime_ns) or name in pending:
                        continue
                    old = db.execute(
                        "SELECT size,mtime,sha FROM hashes WHERE path=?", (name,)
                    ).fetchone()
                    pending[name] = pool.submit(header_and_hash, name, old)
                for future in as_completed(pending.values()):
                    entry = future.result()
                    db.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?,?)", entry[:4])
                    db.execute("INSERT OR REPLACE INTO headers VALUES (?,?,?,?,?)", entry)
                    db.commit()
                for row in batch:
                    digest, actual = db.execute(
                        "SELECT sha,seconds FROM headers WHERE path=?", (row["audio_path"],)
                    ).fetchone()
                    recorded = row.get("audited_duration_seconds", row["duration_seconds"])
                    expected_sha = row.get("expected_audio_sha256", row.get("audio_sha256"))
                    issue = None
                    if expected_sha and expected_sha != digest:
                        issue = "audio_bytes_changed"
                    elif abs(actual - recorded) > DURATION_ROUNDING_SECONDS:
                        issue = "audio_duration_mismatch"
                    ids = row.get("ids", row.get("phoneme_token_ids"))
                    if ids:
                        minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
                        frames = qwen3_asr_audio_output_length(round(actual * 16000) // 160)
                        if minimum > 2 * frames:
                            issue = issue or "actual_duration_CTC_infeasible"
                    row.update(
                        audio_sha256=digest,
                        duration_seconds=actual,
                        audited_duration_seconds=recorded,
                        expected_audio_sha256=expected_sha,
                    )
                    if issue:
                        row["audio_review_issue"] = issue
                    else:
                        row.pop("audio_review_issue", None)
                    category = issue or ("rounding_only" if actual != recorded else "exact")
                    counts[category] += 1
                    if category != "exact" and counts[category] <= 20:
                        examples.append(
                            {
                                "audio_path": row["audio_path"],
                                "issue": category,
                                "recorded_seconds": recorded,
                                "actual_seconds": actual,
                                "difference_seconds": actual - recorded,
                            }
                        )
                    if issue:
                        print(
                            f"Audio review: {issue}; stored={recorded:.12f}s "
                            f"actual={actual:.12f}s; "
                            f"{row['audio_path']}",
                            flush=True,
                        )
                tick(start + len(batch))
    write_json(
        output / "audio_duration_report.json",
        {
            "row_counts": counts,
            "rounding_tolerance_seconds": DURATION_ROUNDING_SECONDS,
            "examples": examples,
            "actual_header_seconds_used_for_selection": True,
            "mismatching_candidates_quarantined_protected_hashes_retained": True,
        },
    )
