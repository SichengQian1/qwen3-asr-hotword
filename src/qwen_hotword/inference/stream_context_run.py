"""Generate resumable causal RAG time lines; downstream owns streaming ASR decoding."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.stream_context_assets import prepare_stream
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.stream_context_contract import (
    EFFECTS,
    MODES,
    audio_prefixes,
    boundaries,
    candidates,
    contexts_for_steps,
    digest,
    validate_contexts,
)
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress


def _check_inputs(plan: dict[str, Any]) -> None:
    for filename, identity in plan["inputs"].items():
        if _sha(Path(filename)) != identity["sha256"]:
            raise ValueError(f"input changed: {filename}")


def _smoke_gate(smoke: Path, plan: dict[str, Any]) -> None:
    inputs: dict[str, Any] = {}
    report = _json(_verified(smoke, "report.json", inputs))
    previous = _json(_verified(smoke, "run_config.json", inputs))
    delivery = _json(_verified(smoke, "delivery/stream_contexts.json", inputs))
    durations = _json(_verified(smoke, "delivery/durations.json", inputs))
    validate_contexts(delivery, durations)
    if (
        report["status"] != "completed"
        or previous["stage"] != "smoke"
        or report["downstream_validator"]["returncode"] != 0
        or previous["checkpoint"] != plan["checkpoint"]
        or previous["configuration"] != plan["configuration"]
        or previous["source_configuration"] != plan["source_configuration"]
        or previous["code_sha256"] != plan["code_sha256"]
        or previous["effect"] != plan["effect"]
    ):
        raise ValueError("completed, compatible smoke with the same timing policy required")
    full_audio = {(s["group"], s["sample_id"]): s["audio_sha256"] for s in plan["samples"]}
    if any(
        full_audio.get((s["group"], s["sample_id"])) != s["audio_sha256"]
        for s in previous["samples"]
    ):
        raise ValueError("smoke audio differs from full-run audio")
    # Pin every smoke input file actually consumed, without inspecting model tensors.
    plan["inputs"].update(inputs)


def _bundle(output: Path) -> None:
    names = [
        "report.json",
        "run_config.json",
        "checkpoint_metadata.json",
        "causality_audit.jsonl",
        "delivery/stream_contexts.json",
        "delivery/durations.json",
        "delivery/audio_manifest.json",
    ]
    data = {name: (output / name).read_bytes() for name in names}
    data["sha256.txt"] = "".join(
        f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in sorted(data.items())
    ).encode()
    final = output / "return_bundle.zip"
    if final.exists():
        with zipfile.ZipFile(final) as archive:
            if set(archive.namelist()) != set(data) or any(
                archive.read(k) != v for k, v in data.items()
            ):
                raise ValueError("existing return bundle differs")
        return
    with tempfile.TemporaryDirectory(dir=output) as directory:
        temporary = Path(directory) / "bundle.zip"
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
            for name, content in data.items():
                archive.writestr(name, content)
        os.link(temporary, final)


def _run_prefixes(
    output: Path,
    plan: dict[str, Any],
    runtime: dict[str, Any],
    detector_factory: Callable[..., Any] | None,
    retrieve_function: external.RetrieveFunction | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    contexts: dict[str, Any] = {mode: {} for mode in MODES}
    durations: dict[str, Any] = {}
    audit, all_steps = [], []
    detector = None
    completed = 0
    try:
        with stage_progress("Causal prefix retrieval", total=plan["prefix_count"], every=1) as tick:
            for sample in plan["samples"]:
                group, key, utt = sample["group"], sample["key"], sample["sample_id"]
                print(f"Audio {group}/{utt}: {sample['step_count']} prefixes", flush=True)
                steps = []
                folder = (
                    output / "prefix_shards" / group / hashlib.sha256(utt.encode()).hexdigest()[:20]
                )
                folder.mkdir(parents=True, exist_ok=True)
                for prefix in audio_prefixes(sample["audio_path"], sample):
                    waveform = prefix.pop("waveform")
                    expected = {
                        "group": group,
                        "key": key,
                        "sample_id": utt,
                        "audio_sha256": sample["audio_sha256"],
                        **{k: v for k, v in prefix.items() if k != "prefix_preparation_seconds"},
                    }
                    shard = folder / f"{prefix['step']:06d}.json"
                    if shard.exists():
                        row = _json(shard)
                        if row.get("sha256") != digest(row) or any(
                            row.get(k) != v for k, v in expected.items()
                        ):
                            raise ValueError(f"corrupted prefix shard: {shard}")
                    else:
                        if detector is None:
                            with stage_progress("Load frozen Qwen and final best Head"):
                                detector = (detector_factory or external._load_detector)(
                                    model=runtime["model"],
                                    checkpoint=runtime["checkpoint"],
                                    vocab=runtime["vocab"],
                                    hotwords=runtime["variants"],
                                    device="cuda:0",
                                    dtype="bfloat16",
                                    language="English",
                                )
                        started = time.perf_counter()
                        raw = (
                            retrieve_function(detector, waveform, runtime["vocab"])
                            if retrieve_function
                            else external._retrieve_waveform(
                                detector, waveform, runtime["vocab"], saved_raw_rank_depth=64
                            )
                        )
                        merged = collapse_variants(
                            raw, runtime["variants"], runtime["by_variant"], plan["gate"][group]
                        )
                        qualified = _select(
                            merged["raw_ranked_matches"],
                            gate=plan["gate"][group],
                            top_k=len(merged["raw_ranked_matches"]),
                        )
                        row = {
                            **expected,
                            "prefix_preparation_seconds": prefix["prefix_preparation_seconds"],
                            "retrieval_wall_seconds": time.perf_counter() - started,
                            "candidates": candidates(qualified),
                            "retrieval": merged,
                        }
                        row["sha256"] = digest(row)
                        external._write_json(shard, row)
                    steps.append(row)
                    # Details are already persisted; retain only timing across utterances.
                    all_steps.append(
                        {
                            "prefix_preparation_seconds": row["prefix_preparation_seconds"],
                            "retrieval_wall_seconds": row["retrieval_wall_seconds"],
                        }
                    )
                    completed += 1
                    tick(completed)
                if len(steps) != sample["step_count"]:
                    raise ValueError(f"prefix count differs: {group}/{utt}")
                durations.setdefault(key, {})[utt] = sample["duration_seconds"]
                for mode, top_k in MODES.items():
                    contexts[mode].setdefault(key, {})[utt] = contexts_for_steps(
                        steps, plan["effect"], top_k
                    )
                for index, end in enumerate(boundaries(sample["frames"], sample["sample_rate"])):
                    source_index = index if plan["effect"] == "same-step" else index - 1
                    source = steps[source_index] if source_index >= 0 else None
                    cutoff = source["source_end_frame"] if source else 0
                    if cutoff > end or (source and source["step"] != source_index):
                        raise ValueError("future audio entered a delivered context")
                    audit.append(
                        {
                            "key": key,
                            "sample_id": utt,
                            "effective_step": index,
                            "available_audio_end_frame": end,
                            "source_sample_rate": sample["sample_rate"],
                            "context_source_step": source_index if source else None,
                            "retrieval_input_end_frame": cutoff,
                            "prefix_shard_sha256": source["sha256"] if source else None,
                        }
                    )
    finally:
        detector = None
        gc.collect()
        if detector_factory is None:
            import torch

            torch.cuda.empty_cache()
    return contexts, durations, audit, all_steps


def run_stream_contexts(
    root: Path,
    config: Path,
    output: Path,
    *,
    stage: str,
    effect: str,
    resume: bool = False,
    audit_only: bool = False,
    smoke_run: Path | None = None,
    detector_factory: Callable[..., Any] | None = None,
    retrieve_function: external.RetrieveFunction | None = None,
) -> dict[str, Any]:
    if stage not in {"smoke", "full"} or effect not in EFFECTS:
        raise ValueError("explicit smoke/full stage and same-step/next-step effect required")
    if stage == "full" and smoke_run is None:
        raise ValueError("full run requires --smoke-run from a completed matching smoke")
    cfg = _json(config)
    limit = cfg["smoke_samples"] if stage == "smoke" else 0
    if output.exists() and not resume and not audit_only:
        raise FileExistsError(f"output exists; --resume only for identical inputs: {output}")
    with stage_progress("Verify fixed assets and select stream samples"):
        plan, runtime = prepare_stream(root, config, limit)
    plan.update(stage=stage, effect=effect)
    if stage == "full" and smoke_run is not None:
        _smoke_gate(smoke_run, plan)
    for filename in plan["inputs"]:
        path = Path(filename)
        if path.is_relative_to(output) or (
            path.parent != root and output.is_relative_to(path.parent)
        ):
            raise ValueError("output overlaps a protected input directory")
    summary = {
        "status": "audit_pass",
        "selected_samples": plan["selected_samples"],
        "inventory_samples": plan["inventory_samples"],
        "prefix_count": plan["prefix_count"],
        "selected_duration_buckets": plan["selected_duration_buckets"],
        "checkpoint": plan["checkpoint"],
        "effect": effect,
        "model_loaded": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if audit_only:
        return summary
    lock = output.parent / f".{output.name}.orchestration"
    with exclusive_feature_cache_run(lock):
        external._prepare_output(output, plan, resume=resume)
        if (output / "sha256.txt").exists():
            with stage_progress("Verify completed stream output; no model reload"):
                seal(output)
            completed_report: dict[str, Any] = _json(output / "report.json")
            return completed_report
        external._write_or_verify_json(output / "checkpoint_metadata.json", plan["checkpoint"])
        contexts, durations, audit, steps = _run_prefixes(
            output, plan, runtime, detector_factory, retrieve_function
        )
        with stage_progress("Validate and publish stream interface files"):
            validation = validate_contexts(contexts, durations)
            (output / "delivery").mkdir(exist_ok=True)
            external._write_or_verify_json(output / "delivery/stream_contexts.json", contexts)
            external._write_or_verify_json(output / "delivery/durations.json", durations)
            manifest: dict[str, Any] = {}
            for sample in plan["samples"]:
                manifest.setdefault(sample["key"], {})[sample["sample_id"]] = sample
            external._write_or_verify_json(output / "delivery/audio_manifest.json", manifest)
            external._write_or_verify_jsonl(output / "causality_audit.jsonl", audit)
            upstream = (
                Path(__file__).resolve().parents[3]
                / "tests/fixtures/stream_interface_v1_validator.py.txt"
            )
            validated = subprocess.run(
                [
                    sys.executable,
                    str(upstream),
                    "--ctx",
                    str(output / "delivery/stream_contexts.json"),
                    "--durations",
                    str(output / "delivery/durations.json"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if validated.returncode:
                raise ValueError(f"downstream validator rejected delivery: {validated.stderr}")
            _check_inputs(plan)
            times = [s["prefix_preparation_seconds"] + s["retrieval_wall_seconds"] for s in steps]
            report = {
                "status": "completed",
                "stage": stage,
                "effect": effect,
                "checkpoint": plan["checkpoint"],
                "selected_samples": plan["selected_samples"],
                "inventory_samples": plan["inventory_samples"],
                "prefix_count": len(steps),
                "scope": "full_inventory" if stage == "full" else "selected_subset",
                "dataset_counts": {k: len(v) for k, v in durations.items()},
                "schema_validation": validation,
                "causality_checked": True,
                "downstream_validator": {
                    "returncode": validated.returncode,
                    "stdout": validated.stdout,
                },
                "prefix_compute_seconds": {
                    "sum": sum(times),
                    "max": max(times),
                    "over_2s_count": sum(t > 2 for t in times),
                },
                "candidate_pool_surfaces": len(runtime["bundle"].entries),
                "candidate_pronunciations": len(runtime["variants"]),
                "qwen_decoder_used": False,
                "retention_policy": "current_prefix_only",
                "parameters_tuned": False,
                "snr_verified": False,
                "delivery": "delivery/stream_contexts.json",
                "return_file": str(output / "return_bundle.zip"),
                "limitations": [
                    "Causal recorded-audio replay; live network and end-to-end latency unmeasured.",
                    "Both K values share each prefix retrieval; quality gates precede truncation.",
                    "Next-step does not deliver the last prefix; no extra tail slot is added.",
                    "Smoke covers the audio_manifest subset, not the complete source directories.",
                    "Retrieval uses no reference, ASR text, whole-audio features or future audio.",
                    "SNR is unverified; original reference-conditioned table limits remain.",
                ],
            }
            external._write_or_verify_json(output / "report.json", report)
            _bundle(output)
            seal(output)
    return report
