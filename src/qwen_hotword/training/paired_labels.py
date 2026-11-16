"""Parallel frontend and incremental MFA requests; reuse old pronunciations exactly."""

from __future__ import annotations

import json
import multiprocessing
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, run_worker, sha, write_json
from qwen_hotword.training.paired_sources import cpu_limit
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows, save_rows
from qwen_hotword.training.zh_en_labels import text_parts


def frontend_part(job: tuple[Path, Path]) -> Path:
    output, source = job
    return text_parts(output, [source])[0]


def parallel_text(output: Path, sources: list[Path], workers: int) -> list[Path]:
    jobs = [(output / "frontend_jobs" / f"{i:06d}", p) for i, p in enumerate(sources)]
    result = []
    with (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        stage_progress(
            "Parallel Chinese context readings (new rows only)", total=len(jobs), every=25
        ) as tick,
    ):
        for i, path in enumerate(pool.map(frontend_part, jobs), 1):
            result.append(path)
            tick(i)
    return result


def effective_workers(limit: int) -> int:
    cores = cpu_limit(limit)
    memory = Path("/proc/meminfo")
    if memory.exists():
        available = next(
            int(s.split()[1]) * 1024
            for s in memory.read_text().splitlines()
            if s.startswith("MemAvailable:")
        )
        cap, used = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if cap.exists() and used.exists() and cap.read_text().strip() != "max":
            available = min(available, int(cap.read_text()) - int(used.read_text()))
        # Conservative planning allowance, not a measured per-FST RSS claim.
        cores = min(cores, max(1, int(available * 0.6) // (1024**3)))
    return cores


def incremental_g2p(
    root: Path,
    output: Path,
    sources: list[Path],
    old_run: Path,
    worker: dict[str, Any],
    assets: Path,
    prefix: list[str],
    workers: int,
) -> Path:
    db_path = output / "pronunciations.sqlite"
    db = sqlite3.connect(db_path)
    db.executescript(
        "DROP TABLE IF EXISTS words; CREATE TABLE words "
        "(lang TEXT, word TEXT, payload TEXT, PRIMARY KEY(lang,word));"
    )
    with stage_progress("Collect new-corpus unique words") as tick:
        for i, source in enumerate(sources, 1):
            for row in rows(source):
                if not row["issues"] and row["split"] != "test":
                    db.executemany(
                        "INSERT OR IGNORE INTO words VALUES (?,?,NULL)",
                        ((u["language"], u["word"]) for u in row["frontend"]),
                    )
            tick(i)
    db.commit()
    reused = 0
    old_files = sorted((old_run / "g2p").glob("*/predictions.jsonl"))
    hashes = []
    old_config_sha = sha(old_run / "g2p_worker_config.json")
    old_config = read(old_run / "g2p_worker_config.json")
    for key in ("models", "vocab_sha256", "g2p_num_pronunciations", "environment"):
        if old_config[key] != worker[key]:
            raise ValueError("legacy G2P policy differs: " + key)
    with stage_progress(
        "Reuse verified old G2P words; generate only missing requests",
        total=len(old_files),
        every=20,
    ) as tick:
        for i, path in enumerate(old_files, 1):
            marker = read(path.with_suffix(".completed.json"))
            if marker["identity"]["worker_config_sha256"] != old_config_sha:
                raise ValueError("old pronunciation worker configuration differs")
            pin(path, marker["output_sha256"])
            hashes.append(sha(path))
            for r in rows(path):
                reused += db.execute(
                    "UPDATE words SET payload=? WHERE lang=? AND word=? AND payload IS NULL",
                    (json.dumps(r, ensure_ascii=False), r["language"], r["word"]),
                ).rowcount
            tick(i)
    if digest(hashes) != read(old_run / "g2p_report.json")["prediction_sha256"]:
        raise ValueError("old pronunciation parts incomplete")
    db.commit()
    jobs: list[tuple[Path, Path]] = []
    cursor = db.execute("SELECT lang,word FROM words WHERE payload IS NULL ORDER BY lang,word")
    while batch := cursor.fetchmany(2000):
        folder = output / "g2p" / f"part-{len(jobs):06d}"
        request = folder / "request.jsonl"
        save_rows(request, ({"language": lang, "word": word} for lang, word in batch))
        jobs.append((request, folder / "predictions.jsonl"))

    def task(job: tuple[Path, Path]) -> Path:
        run_worker(root, worker, assets, prefix, *job)
        return job[1]

    with (
        ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool,
        stage_progress("MFA Top1: missing new words only", total=len(jobs), every=1) as tick,
    ):
        for i, path in enumerate(pool.map(task, jobs), 1):
            for r in rows(path):
                changed = db.execute(
                    "UPDATE words SET payload=? WHERE lang=? AND word=? AND payload IS NULL",
                    (json.dumps(r, ensure_ascii=False), r["language"], r["word"]),
                ).rowcount
                if changed != 1:
                    raise ValueError("unexpected or duplicate G2P output")
            db.commit()
            tick(i)
    if db.execute("SELECT count(*) FROM words WHERE payload IS NULL").fetchone()[0]:
        raise ValueError("incomplete new G2P output")
    counts: Counter[str] = Counter()
    for lang, payload in db.execute("SELECT lang,payload FROM words"):
        counts[lang + "/" + json.loads(payload)["status"]] += 1
    db.close()
    write_json(
        output / "g2p_report.json",
        {
            "counts": counts,
            "reused_unique_words": reused,
            "new_jobs": len(jobs),
            "prediction_sha256": digest({"old": hashes, "new": [sha(p) for _, p in jobs]}),
            "test_used": False,
        },
    )
    return db_path
