"""One independent Encoder process; CUDA device is selected before importing torch."""

from __future__ import annotations

import gc
import os
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, write_json
from qwen_hotword.training.zh_en_common import read, rows


def claim(queue: Path) -> dict[str, Any] | None:
    import json

    with sqlite3.connect(queue, timeout=60) as db:
        db.execute("BEGIN IMMEDIATE")
        found = db.execute(
            "SELECT id,payload FROM jobs WHERE state='pending' ORDER BY seconds DESC,id LIMIT 1"
        ).fetchone()
        if not found:
            return None
        db.execute("UPDATE jobs SET state='running' WHERE id=?", (found[0],))
        return {"job_id": found[0], **json.loads(found[1])}


def process_job(
    job: dict[str, Any],
    wrapper: Any,
    cfg: dict[str, Any],
    foundation: dict[str, Any],
    root: Path,
    pools: dict[str, Any],
    reuse: dict[str, Any],
) -> dict[str, Any]:
    from qwen_hotword.training.ctc_overfit import (
        extract_frozen_features,
        validate_actual_encoder_ctc_length,
    )
    from qwen_hotword.training.feature_cache import cache_feature_split
    from qwen_hotword.training.sharded_ctc import load_disk_feature_cache, load_feature_shard
    from qwen_hotword.training.zh_en_run import MixedExtractor, records_from_rows

    manifest = Path(job["manifest"])
    pin(manifest, job["manifest_sha256"])
    items = list(rows(manifest))
    records = records_from_rows(items)
    reused = {}
    sources = {reuse[r.sample_id] for r in records if r.sample_id in reuse}
    needed = {r.sample_id for r in records}
    for path in sources:
        spec = read(Path(path))
        loaded = load_disk_feature_cache(
            spec["path"],
            expected_split="train",
            source_manifest_path=spec["manifest"],
            vocab_path=root / foundation["vocab"],
            verify_sha256=True,
        )
        if loaded.fingerprint != spec["fingerprint"]:
            raise ValueError("benchmark cache identity changed")
        for descriptor in loaded.shards:
            for sample in load_feature_shard(descriptor, num_classes=183):
                if sample.sample_id in needed:
                    reused[sample.sample_id] = sample
    for record in records:
        if record.sample_id in reused:
            sample = reused[record.sample_id]
            if sample.token_ids != record.token_ids:
                raise ValueError("benchmark labels do not match production labels")
            validate_actual_encoder_ctc_length(record, input_length=sample.hidden_states.shape[0])
    mixer = MixedExtractor(items, pools, cfg["seed"])

    def extractor(batch: Any, wrapper: Any, **kwargs: Any) -> Any:
        pending = [r for r in batch if r.sample_id not in reused]
        started = time.monotonic()
        computed: list[Any] = []
        frozen = 317477504
        if pending:
            # One producer keeps noise-cache mutation sequential while decoding ahead of GPU work.
            with ThreadPoolExecutor(max_workers=1) as pool:
                futures = {r.sample_id: pool.submit(mixer.load, r) for r in pending[:2]}
                positions = {r.sample_id: i for i, r in enumerate(pending)}

                def load(record: Any) -> Any:
                    audio = futures.pop(record.sample_id).result()
                    next_index = positions[record.sample_id] + 2
                    if next_index < len(pending):
                        next_record = pending[next_index]
                        futures[next_record.sample_id] = pool.submit(mixer.load, next_record)
                    return audio

                computed, frozen, _ = extract_frozen_features(
                    pending, wrapper, waveform_loader=load, **kwargs
                )
        lookup = {s.sample_id: s for s in computed} | reused
        return [lookup[r.sample_id] for r in batch], frozen, time.monotonic() - started

    folder = Path(job["destination"])
    cache_feature_split(
        records,
        wrapper,
        folder,
        split=job["split"],
        source_manifest_path=manifest,
        model_path=cfg["model"],
        model_dtype="bfloat16",
        vocab_path=root / foundation["vocab"],
        encoder_batch_size=1,
        samples_per_shard=cfg["part_records"],
        extractor=extractor,
    )
    loaded = load_disk_feature_cache(
        folder,
        expected_split=job["split"],
        source_manifest_path=manifest,
        vocab_path=root / foundation["vocab"],
        verify_sha256=True,
    )
    summary = {
        "path": str(folder),
        "manifest": str(manifest),
        "split": job["split"],
        "fingerprint": loaded.fingerprint,
        "bytes": loaded.feature_bytes,
        "sample_ids": [r.sample_id for r in records],
        "probe_samples_reused": len(reused),
    }
    write_json(folder / "mix_audit.json", mixer.audit)
    write_json(folder / "job_completed.json", summary)
    return summary


def worker(spec_path: Path, worker_id: int) -> None:
    spec = read(spec_path)
    folder = spec_path.parent / f"worker-{worker_id}"
    folder.mkdir(exist_ok=True)
    try:
        import torch

        from qwen_hotword.config import ModelConfig
        from qwen_hotword.modeling.qwen_backbone import load_asr_model
        from qwen_hotword.training.ctc_overfit import extract_frozen_features
        from qwen_hotword.training.zh_en_run import MixedExtractor, records_from_rows

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cfg, foundation = spec["config"], spec["foundation"]
        root = Path(spec["root"])
        wrapper = load_asr_model(
            ModelConfig(Path(cfg["model"]), "Qwen3-ASR-1.7B", "bfloat16", "cuda:0", True)
        )
        wrapper.model.eval()
        for p in wrapper.model.parameters():
            p.requires_grad_(False)
        noise = read(Path(spec["noise_inventory"]))
        pools = {
            s: [n for n in noise["entries"] if n["split"] == s] for s in ("train", "validation")
        }
        warm = next(rows(Path(spec["warmup_manifest"])))
        mixer = MixedExtractor([warm], pools, cfg["seed"])
        samples, _, _ = extract_frozen_features(
            records_from_rows([warm]), wrapper, encoder_batch_size=1, waveform_loader=mixer.load
        )
        del samples
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        write_json(folder / "ready.json", {"pid": os.getpid()})
        while not (spec_path.parent / "go.json").exists():
            if os.getppid() != spec["parent_pid"]:
                raise RuntimeError("coordinator exited")
            time.sleep(0.2)
        while os.getppid() == spec["parent_pid"]:
            job = claim(spec_path.parent / "queue.sqlite")
            if job is None:
                break
            process_job(job, wrapper, cfg, foundation, root, pools, spec.get("reuse", {}))
            torch.cuda.synchronize()
            with sqlite3.connect(spec_path.parent / "queue.sqlite", timeout=60) as db:
                db.execute("UPDATE jobs SET state='done' WHERE id=?", (job["job_id"],))
        write_json(
            folder / "exit.json",
            {
                "status": "completed",
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        )
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        write_json(
            folder / "exit.json",
            {
                "status": "failed",
                "error": str(error),
                "oom": "out of memory" in str(error).lower(),
                "traceback": traceback.format_exc(),
            },
        )
        raise
