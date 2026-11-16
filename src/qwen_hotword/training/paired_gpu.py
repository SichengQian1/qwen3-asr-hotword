"""Two-GPU queue with equal-work, warmed 1/2/4-process throughput trials."""

from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows, save_rows


class ResourceFailure(RuntimeError):
    pass


def gpu_inventory(gpus: list[int]) -> dict[int, dict[str, Any]]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = {}
    for line in raw.splitlines():
        index, uid, name, free, total = [v.strip() for v in line.split(",")]
        if int(index) in gpus:
            if "H200" not in name:
                raise ValueError("selected GPU is not H200: " + name)
            result[int(index)] = {
                "uuid": uid,
                "name": name,
                "free_gib": float(free) / 1024,
                "total_gib": float(total) / 1024,
            }
    if set(result) != set(gpus) or len(set(gpus)) != 2:
        raise ValueError("exactly two distinct allocated physical H200 IDs required")
    return result


def ensure_no_workers(output: Path) -> None:
    for path in output.glob("gpu_runs/*/launch.json"):
        launch = read(path)
        if all(
            (path.parent / f"worker-{i}/exit.json").exists() for i in range(len(launch["pids"]))
        ):
            continue
        if launch["host"] != socket.gethostname():
            raise ValueError("prior GPU workers belong to another host; verify shutdown first")
        for i, pid in enumerate(launch["pids"]):
            if (path.parent / f"worker-{i}/exit.json").exists():
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(
                f"prior worker PID {pid} may still be active; preserve run until it exits"
            )


def run_workers(
    root: Path,
    output: Path,
    cfg: dict[str, Any],
    foundation: dict[str, Any],
    jobs: list[dict[str, Any]],
    per_gpu: int,
    reuse: dict[str, str] | None = None,
    name: str = "production",
) -> dict[str, Any]:
    resources = gpu_inventory(cfg["gpu_ids"])
    need = cfg["gpu_headroom_gib"] + per_gpu * cfg["estimated_worker_gpu_gib"]
    if any(r["free_gib"] < need for r in resources.values()):
        raise ResourceFailure(f"{per_gpu} workers/GPU need planning budget {need} GiB free/card")
    folder = output / "gpu_runs" / (name + "_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    with sqlite3.connect(folder / "queue.sqlite") as db:
        db.execute(
            "CREATE TABLE jobs (id INTEGER PRIMARY KEY, payload TEXT, seconds REAL, state TEXT)"
        )
        db.executemany(
            "INSERT INTO jobs VALUES (?,?,?,?)",
            ((i, json.dumps(job), job["seconds"], "pending") for i, job in enumerate(jobs)),
        )
    spec = {
        "root": str(root),
        "config": cfg,
        "foundation": foundation,
        "noise_inventory": str(output / "noise_inventory.json"),
        "warmup_manifest": jobs[0]["manifest"],
        "parent_pid": os.getpid(),
        "reuse": reuse or {},
    }
    write_json(folder / "spec.json", spec)
    procs: list[subprocess.Popen[Any]] = []
    logs = []
    started = time.monotonic()
    try:
        for gpu in cfg["gpu_ids"]:
            for _ in range(per_gpu):
                i = len(procs)
                log = (folder / f"worker-{i}.log").open("w")
                logs.append(log)
                env = {
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "PYTHONUNBUFFERED": "1",
                }
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        str(root / "scripts/run_zh_en_dual.py"),
                        "gpu-worker",
                        "--spec",
                        str(folder / "spec.json"),
                        "--worker-id",
                        str(i),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                procs.append(process)
                write_json(
                    folder / "launch.json",
                    {
                        "host": socket.gethostname(),
                        "pids": [p.pid for p in procs],
                        "gpus": cfg["gpu_ids"],
                        "per_gpu": per_gpu,
                    },
                )

        def failure_check() -> None:
            for i, process in enumerate(procs):
                code = process.poll()
                if code is not None and code != 0:
                    exit_path = folder / f"worker-{i}/exit.json"
                    reason = read(exit_path) if exit_path.exists() else {"error": f"exit {code}"}
                    if reason.get("oom"):
                        raise ResourceFailure(f"GPU worker OOM; details: {exit_path}")
                    raise RuntimeError(f"GPU worker failed: {reason.get('error')}; {folder}")

        with stage_progress(f"Load/warm {len(procs)} Encoder workers; excluded from throughput"):
            while not all((folder / f"worker-{i}/ready.json").exists() for i in range(len(procs))):
                failure_check()
                if any(
                    p.poll() is not None and not (folder / f"worker-{i}/ready.json").exists()
                    for i, p in enumerate(procs)
                ):
                    raise RuntimeError("Encoder worker exited before becoming ready")
                if time.monotonic() - started > 1800:
                    raise RuntimeError("Encoder worker readiness timed out; inspect worker logs")
                time.sleep(0.5)
        ready_seconds = time.monotonic() - started
        with stage_progress(f"{name}: {per_gpu} processes/GPU", total=len(jobs), every=1) as tick:
            measured = time.monotonic()
            write_json(folder / "go.json", {"started": time.time()})
            previous = -1
            while True:
                failure_check()
                with sqlite3.connect(folder / "queue.sqlite", timeout=60) as db:
                    done = db.execute("SELECT count(*) FROM jobs WHERE state='done'").fetchone()[0]
                if done != previous:
                    tick(done)
                    previous = done
                if done == len(jobs):
                    break
                if all(p.poll() is not None for p in procs):
                    raise RuntimeError("workers exited before all claimed work completed")
                time.sleep(0.2)
            elapsed = time.monotonic() - measured
        for process in procs:
            process.wait(timeout=120)
        failure_check()
        result = {
            "per_gpu": per_gpu,
            "measured_seconds": elapsed,
            "startup_seconds": ready_seconds,
            "audio_seconds": sum(j["seconds"] for j in jobs),
            "jobs": len(jobs),
            "hardware": resources,
            "folder": str(folder),
            "workers": [read(folder / f"worker-{i}/exit.json") for i in range(len(procs))],
        }
        write_json(folder / "result.json", result)
        return result
    finally:
        for i, process in enumerate(procs):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            exit_path = folder / f"worker-{i}/exit.json"
            if not exit_path.exists():
                write_json(
                    exit_path, {"status": "coordinator_stopped", "returncode": process.returncode}
                )
        for log in logs:
            log.close()


def probe_jobs(output: Path, prepared: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    strata: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    with stage_progress("Select fixed throughput cohort across language, noise and duration"):
        for part in prepared["parts"]:
            if part["split"] != "train":
                continue
            path = Path(part["path"])
            pin(path, part["sha256"])
            for row in rows(path):
                if row["seconds"] > 30:
                    continue
                duration = (
                    "short" if row["seconds"] <= 4 else "medium" if row["seconds"] <= 10 else "long"
                )
                key = f"{row['group']}/{row['noisy']}/{duration}"
                values = strata.setdefault(key, [])
                candidate = (rank(cfg["seed"] + 7, row["id"]), row)
                if len(values) < cfg["benchmark_per_stratum"]:
                    values.append(candidate)
                elif candidate[0] < max(v[0] for v in values):
                    values.remove(max(values, key=lambda v: v[0]))
                    values.append(candidate)
    selected = [r for key in sorted(strata) for _, r in sorted(strata[key], key=lambda v: v[0])]
    selected.sort(key=lambda r: rank(cfg["seed"] + 8, r["id"]))
    if len(selected) < 8:
        raise ValueError("not enough training records for a two-card concurrency comparison")
    jobs = []
    size = cfg["benchmark_records_per_job"]
    for start in range(0, len(selected), size):
        items = selected[start : start + size]
        manifest = output / "probe_manifests" / f"part-{start // size:05d}.jsonl"
        save_rows(manifest, items)
        jobs.append(
            {
                "manifest": str(manifest),
                "manifest_sha256": sha(manifest),
                "split": "train",
                "seconds": sum(r["seconds"] for r in items),
            }
        )
    write_json(
        output / "probe_cohort.json",
        {
            "ids": [r["id"] for r in selected],
            "manifest_sha256": digest([j["manifest_sha256"] for j in jobs]),
            "strata": {k: len(v) for k, v in strata.items()},
            "max_duration_seconds": 30,
        },
    )
    return jobs


def choose(trials: list[dict[str, Any]], repeats: int) -> int:
    scores = {}
    for per_gpu in (1, 2, 4):
        passed = [t for t in trials if t["per_gpu"] == per_gpu and t["status"] == "completed"]
        if len(passed) == repeats:
            scores[per_gpu] = statistics.median(
                t["audio_seconds"] / t["measured_seconds"] for t in passed
            )
    if not scores:
        raise ResourceFailure("no process count completed all throughput trials")
    return max(scores, key=lambda k: (scores[k], -k))


def autotune(
    root: Path,
    output: Path,
    cfg: dict[str, Any],
    foundation: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache

    cohort = probe_jobs(output, prepared, cfg)
    identity = {
        "cohort": sha(output / "probe_cohort.json"),
        "cache": sha(output / "cache_contract.json"),
        "gpus": {str(k): v["uuid"] for k, v in gpu_inventory(cfg["gpu_ids"]).items()},
        "candidates": cfg["encoder_process_candidates"],
        "repeats": cfg["benchmark_repeats"],
    }
    receipt = output / "encoder_autotune.json"
    saved: dict[str, Any] = (
        read(receipt) if receipt.exists() else {"identity": identity, "trials": []}
    )
    if saved["identity"] != identity:
        raise ValueError("benchmark identity changed")
    trials = saved["trials"]
    for repeat in range(cfg["benchmark_repeats"]):
        order = cfg["encoder_process_candidates"][:: 1 if repeat % 2 == 0 else -1]
        for count in order:
            prior = [t for t in trials if t["repeat"] == repeat and t["per_gpu"] == count]
            if prior:
                continue
            destinations = (
                output
                / "probe_features"
                / f"repeat-{repeat}-processes-{count}"
                / uuid.uuid4().hex[:8]
            )
            jobs = [
                {**j, "destination": str(destinations / f"part-{i:05d}")}
                for i, j in enumerate(cohort)
            ]
            try:
                result = run_workers(
                    root, output, cfg, foundation, jobs, count, name=f"benchmark-r{repeat}-p{count}"
                )
                trial = {**result, "repeat": repeat, "status": "completed", "cache_jobs": jobs}
            except ResourceFailure as error:
                trial = {
                    "repeat": repeat,
                    "per_gpu": count,
                    "status": "resource_unavailable",
                    "error": str(error),
                }
            trials.append(trial)
            write_json(receipt, saved)
    selected = choose(trials, cfg["benchmark_repeats"])
    print(
        f"Selected {selected} Encoder processes/GPU on {cfg['gpu_ids']}; "
        "see encoder_autotune.json for measured throughput",
        flush=True,
    )
    winner = next(t for t in trials if t["per_gpu"] == selected and t["status"] == "completed")
    reuse = {}
    for job in winner["cache_jobs"]:
        summary_path = Path(job["destination"]) / "job_completed.json"
        summary = read(summary_path)
        loaded = load_disk_feature_cache(
            summary["path"],
            expected_split="train",
            source_manifest_path=summary["manifest"],
            vocab_path=root / foundation["vocab"],
            verify_sha256=True,
        )
        if loaded.fingerprint != summary["fingerprint"]:
            raise ValueError("winning probe features changed")
        for sample_id in summary["sample_ids"]:
            if sample_id in reuse:
                raise ValueError("duplicate benchmark sample")
            reuse[sample_id] = str(summary_path)
    saved.update(
        selected_processes_per_gpu=selected,
        reuse=reuse,
        status="completed",
        selection_metric="median audio seconds / warmed wall seconds; includes read/mix/write",
    )
    write_json(receipt, saved)
    return saved
