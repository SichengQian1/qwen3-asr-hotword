"""Independent utterance workers for the single-H200 480h retrieval replay."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.phonemes.mfa_v03_assets import write_json
from qwen_hotword.training.paired_sources import cpu_limit
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest

# Only the child touches this process-local detector. Spawn keeps CUDA contexts independent.
_state: tuple[Any, ...] = ()
_detector: Any = None


class WorkerCapacityError(RuntimeError):
    pass


def _initialize(model: Path, checkpoint: Path, vocab: Any, entries: Any, owners: Any) -> None:
    global _state
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _state = (model, checkpoint, vocab, entries, owners)


def _infer(sample: dict[str, Any]) -> dict[str, Any]:
    global _detector
    import torch

    model, checkpoint, vocab, entries, owners = _state
    try:
        if _detector is None:
            torch.set_num_threads(1)
            _detector = external._load_detector(
                model=model,
                checkpoint=checkpoint,
                vocab=vocab,
                hotwords=entries,
                device="cuda:0",
                dtype="bfloat16",
                language="English",
            )
        return base.retrieve(_detector, vocab, entries, owners, sample)
    except (torch.OutOfMemoryError, MemoryError) as exc:
        raise WorkerCapacityError(f"memory exhausted for {sample['sample_id']}: {exc}") from exc


def worker_limit(requested: int, gpu: int) -> int:
    free = (
        int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu),
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        / 1024
    )
    if free < 10:
        raise ValueError(f"GPU {gpu} has only {free:.1f} GiB free for retrieval")
    # Scheduling estimates, not an allocation or measurement of per-worker peak.
    cores = cpu_limit(requested)
    available = next(
        int(line.split()[1]) / 1024**2
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    cap, used = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    if cap.exists() and used.exists() and cap.read_text().strip() != "max":
        available = min(available, (int(cap.read_text()) - int(used.read_text())) / 1024**3)
    count = max(1, min(cores, int((free - 8) / 10), int((available - 4) / 6)))
    print(
        f"GPU {gpu}: {free:.1f} GiB free; host available {available:.1f} GiB; "
        f"CPU limit {cores}; starting up to {count}/{requested} retrieval workers",
        flush=True,
    )
    return count


def _completed(
    pool: ProcessPoolExecutor, jobs: Iterator[tuple[Path, dict[str, Any]]], count: int
) -> Iterator[tuple[Path, dict[str, Any], dict[str, Any]]]:
    pending: dict[Future[dict[str, Any]], tuple[Path, dict[str, Any]]] = {}
    while True:
        while len(pending) < count:
            job = next(jobs, None)
            if job is None:
                break
            path, sample = job
            pending[pool.submit(_infer, sample)] = (path, sample)
        if not pending:
            return
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            path, sample = pending.pop(future)
            yield path, sample, future.result()


def prefill(
    out: Path,
    groups: dict[str, list[dict[str, Any]]],
    run_id: str,
    workers: int,
    gpu: int,
    initargs: tuple[Any, ...],
    cached: Callable[[Path, dict[str, Any]], Path],
    load: Callable[[Path, str, dict[str, Any]], dict[str, Any]],
) -> None:
    unique = {cached(out, s): s for samples in groups.values() for s in samples}
    missing = {p: s for p, s in unique.items() if not p.is_file()}
    print(
        f"Unique audio jobs: {len(unique)}; already on disk: {len(unique) - len(missing)}; "
        f"remaining: {len(missing)}",
        flush=True,
    )
    if not missing:
        return
    # Protect the worker pool from independent BLAS/tokenizer thread pools at import time.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    count = min(len(missing), worker_limit(workers, gpu))
    # Re-evaluate one saved utterance without writing it. Compare all public candidates/steps.
    anchor = next(((p, s) for p, s in unique.items() if p not in missing), None)
    total = len(missing)
    completed = 0
    with stage_progress("Parallel retrieval: newly saved audio", total=total, every=100) as tick:
        while missing:
            started = time.monotonic()
            initial = completed
            prefixes = 0
            try:
                with ProcessPoolExecutor(
                    max_workers=count,
                    mp_context=mp.get_context("spawn"),
                    initializer=_initialize,
                    initargs=initargs,
                ) as pool:
                    if anchor is not None:
                        p, s = anchor
                        expected = load(p, run_id, s)
                        actual = pool.submit(_infer, s).result()
                        if actual != expected:
                            raise ValueError(
                                "parallel/previous raw Top7 or delivered timeline differs: "
                                f"{s['sample_id']}; use --workers 1"
                            )
                        print(f"Parallel replay matches saved result: {s['sample_id']}", flush=True)
                    for path, sample, result in _completed(
                        pool, iter(list(missing.items())), count
                    ):
                        write_json(
                            path,
                            {
                                "run": run_id,
                                "audio": sample["audio_sha256"],
                                "language": sample["language"],
                                "digest": digest(result),
                                "result": result,
                            },
                        )
                        del missing[path]
                        completed += 1
                        prefixes += sample["step_count"]
                        tick(completed)
                        if (completed - initial) % 100 == 0 or not missing:
                            elapsed = time.monotonic() - started
                            print(
                                f"{count} worker(s), "
                                f"{(completed - initial) * 60 / elapsed:.1f} new audio/min, "
                                f"{prefixes / elapsed:.1f} prefixes/s (includes startup)",
                                flush=True,
                            )
            except (WorkerCapacityError, BrokenProcessPool) as exc:
                if count == 1:
                    raise
                count = max(1, count // 2)
                print(
                    f"Worker capacity/process failure: {exc}; retrying {len(missing)} pending "
                    f"jobs with {count} worker(s)",
                    flush=True,
                )
