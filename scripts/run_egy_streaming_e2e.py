#!/usr/bin/env python3
"""One command: engineering smoke -> causal prefix retrieval -> three-way streaming ASR."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from importlib import import_module
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.evaluation.backend_comparison import block_legacy_qwen_vllm_registration
from qwen_hotword.evaluation.casablanca_wer import NORMALIZATION
from qwen_hotword.evaluation.egy_corpus_hotwords import (
    assert_disjoint,
    expected_ids,
    order_key,
    select_audio,
)
from qwen_hotword.evaluation.egy_hotword_review import verify_run
from qwen_hotword.evaluation.egy_streaming_e2e import (
    MODES,
    PrefixReplay,
    aggregate,
    score_row,
    smoke_rows,
    summary,
)
from qwen_hotword.inference.streaming_core import (
    StreamingSample,
    refresh_streaming_prompt,
    run_streaming_sample,
    schedule_stream_chunks,
)
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.stage_progress import stage_progress
from scripts.evaluate_egy_corpus_hotwords import fingerprint, load_detector, read, sealed
from scripts.label_arabic_native import exclusive_job


def save_cell(path: Path, value: dict[str, Any]) -> None:
    write_json(path, {"payload": value, "sha256": fingerprint(value)})


def load_cell(path: Path, identity: str) -> Any:
    if not path.exists():
        return None
    saved = read(path)
    value = saved["payload"]
    if saved["sha256"] != fingerprint(value) or value["identity"] != identity:
        raise ValueError("completed result changed: " + str(path))
    return value


# Reviewed upgrade: allocation and batched scheduling; data, prompts and scoring unchanged.
LEGACY_MEMORY_SCRIPT_SHA256 = "9c274bf1874081ba5957c4f3772181fb3a27459841cf0f9e1af91b5bc543f5f3"
BATCH_SCRIPT_SHA256 = "c85042699e96bae4dbb6194a09f1dd46b39b68d87029371ed246f94be5b646fc"
REVIEWED_SCRIPTS = {LEGACY_MEMORY_SCRIPT_SHA256, BATCH_SCRIPT_SHA256}


def seal_stream_contract(path: Path, current: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        previous: dict[str, Any] = read(path)
        upgraded = json.loads(json.dumps(previous))
        script = str(Path(__file__))
        old = upgraded.get("inputs", {}).get(script)
        if old in REVIEWED_SCRIPTS:
            upgraded["inputs"][script] = current["inputs"][script]
            if upgraded == current:
                prior_path = path.parent / "memory_runtime_upgrade.json"
                if prior_path.exists():
                    prior = read(prior_path)
                    if (
                        prior["original_contract_sha256"] != sha(path)
                        or prior["recorded_script_sha256"] != old
                        or prior["reviewed_script_sha256"] not in REVIEWED_SCRIPTS
                    ):
                        raise ValueError("previous runtime upgrade identity changed")
                sealed(
                    path.parent / "token_limit_runtime_upgrade.json",
                    {
                        "original_contract_sha256": sha(path),
                        "recorded_script_sha256": old,
                        "reviewed_script_sha256": current["inputs"][script],
                        "reason": (
                            "Retry token-limited unfinished requests; retain successful results. "
                            "Model, prompts, seed, temperature and scoring unchanged"
                        ),
                    },
                )
                print(
                    "Reviewed token-limit recovery upgrade; "
                    "retain original contract and completed cells",
                    flush=True,
                )
                return previous
    sealed(path, current)
    return current


def memory_budget(
    free_bytes: int, total_bytes: int, mode: str, fixed_utilization: float
) -> dict[str, Any]:
    if not 0 < free_bytes <= total_bytes:
        raise ValueError("invalid free/total GPU memory snapshot")
    if mode == "auto":
        reserve = max(2 * 1024**3, math.ceil(free_bytes * 0.1))
        if free_bytes <= reserve:
            raise ValueError("GPU has insufficient free memory after the 2 GiB minimum reserve")
        utilization = math.floor((free_bytes - reserve) / total_bytes * 10000) / 10000
    elif mode == "fixed":
        utilization = fixed_utilization
    else:
        raise ValueError("decoder memory mode must be auto or fixed")
    if not math.isfinite(utilization) or not 0 < utilization < 1:
        raise ValueError("invalid GPU memory utilization")
    requested = math.floor(total_bytes * utilization)
    if requested >= free_bytes:
        raise ValueError("requested decoder memory exceeds currently free memory")
    return {
        "mode": mode,
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "requested_bytes": requested,
        "reserved_free_bytes": free_bytes - requested,
        "effective_gpu_memory_utilization": utilization,
        "original_config_utilization": fixed_utilization,
        "policy": "auto: use 90% of free memory, leave at least 2 GiB; snapshot at engine startup",
    }


def runtime_info(decoder: bool) -> dict[str, Any]:
    packages: tuple[str, ...] = (
        "qwen-asr",
        "torch",
        "transformers",
        "numpy",
        "librosa",
        "soundfile",
    )
    if decoder:
        packages += ("vllm",)
    versions = {name: importlib.metadata.version(name) for name in packages}
    if versions["qwen-asr"] != "0.0.6":
        raise ValueError("streaming state requires installed qwen-asr 0.0.6")
    if decoder and versions["vllm"].split("+")[0] != "0.23.0":
        raise ValueError("requires previously verified native vLLM 0.23.0 environment")
    sources = [("qwen-asr", "qwen_asr/inference/qwen3_asr.py")]
    if decoder:
        sources.append(("vllm", "vllm/model_executor/models/qwen3_asr.py"))
        # Import native engine in its own process, without importing qwen_asr.
        import_module("vllm")
    else:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401

        with block_legacy_qwen_vllm_registration():
            from qwen_asr import Qwen3ASRModel
        for name in ("init_streaming_state", "streaming_transcribe", "finish_streaming_transcribe"):
            if not hasattr(Qwen3ASRModel, name):
                raise ValueError("official streaming method missing: " + name)
    return {
        "python": sys.executable,
        "versions": versions,
        "sources": {
            rel: sha(Path(str(importlib.metadata.distribution(pkg).locate_file(rel))))
            for pkg, rel in sources
        },
    }


def prepare(config_path: Path, output: Path) -> dict[str, Any]:
    cfg = read(config_path)
    source_dir = ROOT / cfg["source_output"]
    original = read(ROOT / cfg["source_config"])
    review = read(ROOT / original["source_config"])
    report = read(source_dir / "report.json")
    if report["status"] != "completed":
        raise ValueError("whole-audio source evaluation is incomplete")
    verified = verify_run(ROOT, review)
    pin(source_dir / "table.json", cfg["expected_table_sha256"])
    rows, table = read(source_dir / "selected_audio.json"), read(source_dir / "table.json")
    test = ROOT / original["test_manifest"]
    val = ROOT / original["validation_manifest"]
    pin(test, original["test_sha256"])
    pin(val, original["validation_sha256"])
    selected, selection = select_audio(list(iter_jsonl(test)))
    if rows != selected or len(rows) != cfg["expected_samples"]:
        raise ValueError("evaluation audio differs from frozen whole-audio test selection")
    targets = sum(len(expected_ids(row["text"], table)) for row in rows)
    if targets != cfg["expected_target_pairs"]:
        raise ValueError("gold target denominator changed")
    assert_disjoint(rows, iter_jsonl(val))
    assert_disjoint(rows, iter_jsonl(Path(verified["train_manifest"])))
    plan = read(ROOT / original["encoder_plan"])
    model = Path(plan["model"]["source"])
    model_inputs = {
        str(model / name): value["sha256"] for name, value in plan["model"]["files"].items()
    }
    model_inputs[str(model / "model.safetensors")] = review["expected"]["model_weight_sha256"]
    for name, digest in model_inputs.items():
        pin(Path(name), digest)
    if read(model / "config.json")["architectures"] != ["Qwen3ASRForConditionalGeneration"]:
        raise ValueError("wrong Qwen architecture")
    decoder = ROOT / cfg["decoder_python"]
    if not decoder.is_file():
        raise ValueError("existing vLLM environment not found: " + str(decoder))
    runtime_files = []
    for kind, python in (("main", sys.executable), ("decoder", str(decoder))):
        dest = output / ("runtime_" + kind + ".json")
        subprocess.run(
            [python, "-u", "-B", str(Path(__file__)), "--probe", kind, "--output-dir", str(output)],
            check=True,
            cwd=ROOT,
        )
        runtime_files.append(dest)
    paths = [
        config_path,
        ROOT / cfg["source_config"],
        ROOT / original["source_config"],
        ROOT / original["encoder_plan"],
        test,
        val,
        Path(__file__),
        *runtime_files,
    ]
    paths += [
        source_dir / name
        for name in (
            "report.json",
            "contract.json",
            "table.json",
            "selected_audio.json",
            "selection.json",
        )
    ]
    paths += [
        ROOT / ("src/qwen_hotword/" + name)
        for name in (
            "evaluation/egy_streaming_e2e.py",
            "evaluation/egy_corpus_hotwords.py",
            "evaluation/casablanca_wer.py",
            "inference/streaming_core.py",
            "inference/streaming_backends.py",
            "inference/hotword_prompt.py",
            "inference/external_keyword_retrieval.py",
            "modeling/ctc_head.py",
            "modeling/audio_encoder.py",
            "hotwords/scoring.py",
            "hotwords/anchor_index.py",
        )
    ]
    paths += [ROOT / "scripts/evaluate_egy_corpus_hotwords.py"]
    contract = {
        "schema_version": 1,
        "config": cfg,
        "model": str(model),
        "checkpoint": str(ROOT / review["source_run"] / "head/ctc_head_best.pt"),
        "vocab": str(ROOT / review["vocab"]),
        "expected": review["expected"],
        "inputs": {**verified["files"], **model_inputs, **{str(p): sha(p) for p in paths}},
        "selection": selection,
        "target_pairs": targets,
        "smoke_ids": [r["id"] for r in smoke_rows(rows, table)],
        "audio": "original_no_added_noise",
        "effect": "same_step_state_refresh",
        "inference_reference_access": False,
        "evaluation_scope": "corpus_conditioned_frozen_table_streaming_simulation",
    }
    return seal_stream_contract(output / "contract.json", contract)


def inputs_unchanged(contract: dict[str, Any]) -> None:
    for path, digest in contract["inputs"].items():
        if Path(path) == Path(__file__) and digest in REVIEWED_SCRIPTS:
            # Only this exact historical runner is accepted; all other input pins remain strict.
            continue
        pin(Path(path), digest)


def selection(output: Path, scope: str) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    contract = read(output / "contract.json")
    source = ROOT / contract["config"]["source_output"]
    rows = read(source / "selected_audio.json")
    if scope == "smoke":
        chosen = set(contract["smoke_ids"])
        rows = [r for r in rows if r["id"] in chosen]
    return contract, rows, read(source / "table.json")


def audio(row: dict[str, Any]) -> Any:
    from qwen_hotword.inference.external_keyword_retrieval import _load_audio

    pin(Path(row["audio_path"]), row["audio_sha256"])
    wave, metadata = _load_audio(Path(row["audio_path"]))
    if abs(metadata["duration_seconds"] - row["duration_seconds"]) > 0.02:
        raise ValueError("audio duration changed: " + row["audio_path"])
    return wave


def row_identity(output: Path, row: dict[str, Any]) -> str:
    return fingerprint([sha(output / "contract.json"), row["id"], row["audio_sha256"]])


def prefix_worker(output: Path, scope: str) -> None:
    from qwen_hotword.inference.external_keyword_retrieval import _retrieve_waveform
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab

    contract, rows, table = selection(output, scope)
    inputs_unchanged(contract)
    vocab = load_phoneme_vocab(Path(contract["vocab"]))
    detector = None
    with stage_progress(
        "Causal 2s prefixes: r15 Encoder + epoch29 x3 CTC", total=len(rows), every=1
    ) as tick:
        for n, row in enumerate(rows, 1):
            path = output / "prefixes" / (order_key(row["id"]) + ".json")
            identity = row_identity(output, row)
            if load_cell(path, identity) is None:
                if detector is None:
                    with (
                        stage_progress("Load frozen r15 Encoder and x3 CTC"),
                        block_legacy_qwen_vllm_registration(),
                    ):
                        detector = load_detector(
                            Path(contract["model"]), Path(contract["checkpoint"]), table, vocab
                        )
                wave = audio(row)
                prefixes = []
                for chunk in schedule_stream_chunks(
                    len(wave), chunk_size_sec=contract["config"]["chunk_seconds"]
                ):
                    pred = _retrieve_waveform(
                        detector, wave[: chunk.end_sample], vocab, saved_raw_rank_depth=64
                    )
                    prefixes.append(
                        {
                            "end_sample": chunk.end_sample,
                            "selected_matches": pred["selected_matches"],
                            "raw_ranked_matches": pred["raw_ranked_matches"],
                            "timing": pred["timing"],
                        }
                    )
                pin(Path(row["audio_path"]), row["audio_sha256"])
                save_cell(path, {"identity": identity, "prefixes": prefixes})
            tick(n)


def receive(channel: Connection, child: subprocess.Popen[Any], timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if channel.poll(1):
            value = channel.recv()
            if "error" in value:
                raise RuntimeError(value["error"])
            return value
        if child.poll() is not None:
            raise RuntimeError("native decoder exited: " + str(child.returncode))
    raise TimeoutError("native decoder response timed out; existing cells retained")


class TokenLimitError(RuntimeError):
    def __init__(self, detail: dict[str, Any]):
        super().__init__("decoder still reached token limit after bounded retries")
        self.detail = detail


def bounded_batch_generate(
    engine: Any,
    sampling_factory: Any,
    inputs: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    cfg: dict[str, Any],
    output: Path,
    launch_id: str,
    batch_id: int,
) -> list[dict[str, Any]]:
    """Only a capped request is retried; never pass its truncated text to the ASR state."""
    base_cap = int(cfg["max_new_tokens"])

    def generate(items: list[dict[str, Any]], cap: int) -> Any:
        return engine.generate(
            items,
            sampling_params=sampling_factory(temperature=0, max_tokens=cap, seed=cfg["seed"]),
            use_tqdm=False,
        )

    generated = generate(inputs, base_cap)
    if len(generated) != len(inputs):
        raise ValueError("native decoder response count differs from batch")
    replies = []
    for index, (item, request, result) in enumerate(zip(inputs, requests, generated, strict=True)):
        attempts = []
        cap = base_cap
        for attempt in range(3):
            prediction = result.outputs[0]
            prompt_ids = getattr(result, "prompt_token_ids", None)
            record = {
                "max_tokens": cap,
                "finish_reason": prediction.finish_reason,
                "generated_tokens": len(getattr(prediction, "token_ids", [])),
                "expanded_prompt_tokens": len(prompt_ids) if prompt_ids else None,
            }
            attempts.append(record)
            if prediction.finish_reason == "stop":
                break
            if prediction.finish_reason != "length":
                raise RuntimeError(
                    "unexpected decoder finish reason: " + str(prediction.finish_reason)
                )
            event_path = (
                output / "token_limit_events" / f"{launch_id}_{batch_id}_{index}_{attempt}.json"
            )
            write_json(
                event_path,
                {
                    "request": request.get("metadata", {}),
                    "audio_samples": len(request["wave"]) // 4,
                    "prompt_ids_sha256": fingerprint(request["ids"]),
                    "launch_id": launch_id,
                    "batch_id": batch_id,
                    "attempt": attempt,
                    **record,
                    "truncated_text": prediction.text,
                },
            )
            record["diagnostic_file"] = str(event_path)
            print(
                "Token limit: " + json.dumps({**request.get("metadata", {}), **record}), flush=True
            )
            if attempt == 2 or not prompt_ids:
                break
            headroom = int(cfg["max_model_len"]) - len(prompt_ids)
            next_cap = min(base_cap * (2 ** (attempt + 1)), headroom)
            if next_cap <= cap:
                break
            cap = next_cap
            # Same prefix, same prompt, no truncated continuation and no reference access.
            retried = generate([item], cap)
            if len(retried) != 1:
                raise ValueError("token-limit retry response count mismatch")
            result = retried[0]
        failed = prediction.finish_reason == "length"
        replies.append(
            {
                "text": "" if failed else prediction.text,
                "finish_reason": prediction.finish_reason,
                "error_type": "token_limit" if failed else None,
                "truncated_text_preview": prediction.text[:800] + prediction.text[-800:]
                if failed
                else None,
                "generation_attempts": attempts,
                "prompt_ids_sha256": fingerprint(request["ids"]),
                "audio_samples": len(request["wave"]) // 4,
                "batch_id": batch_id,
                "batch_size": len(inputs),
            }
        )
    return replies


def decoder_server(fd: int, output: Path) -> None:
    """Native vLLM subprocess deliberately never imports qwen_asr."""
    import numpy as np

    vllm = import_module("vllm")
    LLM, SamplingParams = vllm.LLM, vllm.SamplingParams

    contract = read(output / "contract.json")
    cfg = contract["config"]
    with Connection(fd) as channel:
        try:
            # CUDA is inspected in the launcher; force spawn before engine workers start.
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            import torch

            free, total = torch.cuda.mem_get_info(0)
            allocation = memory_budget(
                free,
                total,
                os.environ.get("EGY_DECODER_MEMORY", "auto"),
                cfg["gpu_memory_utilization"],
            )
            launch_id = uuid4().hex
            allocation.update(
                {
                    "launch_id": launch_id,
                    "created_unix_seconds": time.time(),
                    "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "script_sha256": sha(Path(__file__)),
                    "status": "starting",
                    "max_num_seqs": 3,
                }
            )
            launch_path = output / "decoder_launches" / (launch_id + ".json")
            write_json(launch_path, allocation)
            print("Decoder memory: " + json.dumps(allocation, ensure_ascii=False), flush=True)
            engine = LLM(
                model=contract["model"],
                model_impl="vllm",
                dtype="float16",
                enforce_eager=True,
                max_num_seqs=3,
                max_model_len=cfg["max_model_len"],
                gpu_memory_utilization=allocation["effective_gpu_memory_utilization"],
                seed=cfg["seed"],
                generation_config="vllm",
                enable_prefix_caching=False,
            )
            allocation["status"] = "ready"
            write_json(launch_path, allocation)
            channel.send({"ready": True, "decoder_launch_id": launch_id})
            batch_id = 0
            while True:
                packet = channel.recv()
                requests = packet.get("requests", [packet])
                if not 1 <= len(requests) <= 3:
                    raise ValueError("decoder batch must contain 1-3 independent streaming states")
                waves = [np.frombuffer(request["wave"], dtype="<f4") for request in requests]
                started = time.perf_counter()
                inputs = [
                    {
                        "prompt_token_ids": request["ids"],
                        "multi_modal_data": {"audio": (wave, 16000)},
                    }
                    for request, wave in zip(requests, waves, strict=True)
                ]
                replies = bounded_batch_generate(
                    engine, SamplingParams, inputs, requests, cfg, output, launch_id, batch_id
                )
                seconds = time.perf_counter() - started
                for reply in replies:
                    reply["batch_seconds"] = seconds
                if batch_id < 3 or batch_id % 50 == 0:
                    print(
                        f"Native decoder batch={batch_id} requests={len(requests)} "
                        f"elapsed={seconds:.3f}s",
                        flush=True,
                    )
                batch_id += 1
                channel.send({"results": replies} if "requests" in packet else replies[0])
        except EOFError:
            pass
        except Exception as exc:
            channel.send({"error": str(exc)})
            raise


class BatchBroker:
    """One socket owner coalesces up to three independent state requests, never their prompts."""

    def __init__(
        self,
        channel: Connection,
        child: subprocess.Popen[Any],
        maximum: int = 3,
        wait_seconds: float = 0.02,
    ):
        self.channel, self.child = channel, child
        self.maximum, self.wait_seconds = maximum, wait_seconds
        self.queue: queue.Queue[Any] = queue.Queue()
        self.lock = threading.Lock()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        future: Future[dict[str, Any]] = Future()
        with self.lock:
            if self.error is not None:
                raise self.error
            self.queue.put((value, future))
        return future.result(timeout=650)

    def _serve(self) -> None:
        while True:
            first = self.queue.get()
            if first is None:
                return
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.maximum:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                batch.append(item)
            try:
                self.channel.send({"requests": [value for value, _ in batch]})
                replies = receive(self.channel, self.child, 600)["results"]
                if len(replies) != len(batch):
                    raise ValueError("batch broker response count mismatch")
                for (_, future), reply in zip(batch, replies, strict=True):
                    future.set_result(reply)
            except BaseException as exc:
                with self.lock:
                    self.error = exc
                    while not self.queue.empty():
                        item = self.queue.get_nowait()
                        if item is not None:
                            batch.append(item)
                for _, future in batch:
                    if not future.done():
                        future.set_exception(exc)
                return

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=2)


class DecoderBridge:
    def __init__(
        self,
        channel: Connection,
        child: subprocess.Popen[Any],
        tokenizer: Any,
        broker: BatchBroker | None = None,
    ):
        self.broker = broker
        self.channel, self.child, self.tokenizer = channel, child, tokenizer
        self.expected_prompt = ""
        self.metadata: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def generate(self, inputs: list[dict[str, Any]], **kwargs: Any) -> list[Any]:
        (request,) = inputs
        (wave,) = request["multi_modal_data"]["audio"]
        prompt = request["prompt"]
        if not self.expected_prompt or not prompt.startswith(self.expected_prompt):
            raise ValueError("official streaming request did not use refreshed prompt")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        value = {
            "ids": ids,
            "wave": wave.astype("<f4", copy=False).tobytes(),
            "metadata": self.metadata,
        }
        if self.broker is None:
            self.channel.send(value)
            reply = receive(self.channel, self.child, 600)
        else:
            reply = self.broker.request(value)
        if reply["prompt_ids_sha256"] != fingerprint(ids) or reply["audio_samples"] != len(wave):
            raise ValueError("native decoder did not acknowledge the exact causal request")
        self.calls.append(
            {
                "prompt": prompt,
                "audio_samples": len(wave),
                "finish_reason": reply["finish_reason"],
                "prompt_ids_sha256": reply["prompt_ids_sha256"],
                "batch_id": reply.get("batch_id"),
                "batch_size": reply.get("batch_size"),
                "batch_seconds": reply.get("batch_seconds"),
                "generation_attempts": reply.get("generation_attempts", []),
            }
        )
        if reply.get("error_type") == "token_limit":
            raise TokenLimitError(reply)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=reply["text"])])]


class AuditedBackend:
    def __init__(self, wrapper: Any, bridge: DecoderBridge):
        self.wrapper, self.bridge = wrapper, bridge
        self.tokenizer = wrapper.processor.tokenizer

    def init_streaming_state(self, **kwargs: Any) -> Any:
        return self.wrapper.init_streaming_state(**kwargs)

    def streaming_transcribe(self, wave: Any, state: Any) -> None:
        self.bridge.expected_prompt = state.prompt_raw
        self.wrapper.streaming_transcribe(wave, state)

    def finish_streaming_transcribe(self, state: Any) -> None:
        self.bridge.expected_prompt = state.prompt_raw
        self.wrapper.finish_streaming_transcribe(state)


def prompt_probe(backend: Any, cfg: dict[str, Any]) -> None:
    args = dict(
        language="Arabic",
        chunk_size_sec=cfg["chunk_seconds"],
        unfixed_chunk_num=cfg["unfixed_chunk_num"],
        unfixed_token_num=cfg["unfixed_token_num"],
    )
    state = backend.init_streaming_state(context="", **args)
    before = state.prompt_raw
    refresh_streaming_prompt(backend, state, prompt="القاهرة", **args)
    if (
        state.context != "القاهرة"
        or state.prompt_raw == before
        or "القاهرة" not in state.prompt_raw
    ):
        raise ValueError("dynamic context refresh is not supported by this installed runtime")


def asr_worker(output: Path, scope: str) -> None:
    contract, rows, table = selection(output, scope)
    inputs_unchanged(contract)
    cfg = contract["config"]
    pending = []
    for row in rows:
        prefix_path = output / "prefixes" / (order_key(row["id"]) + ".json")
        if load_cell(prefix_path, row_identity(output, row)) is None:
            raise ValueError("missing causal prefixes: " + row["id"])
        for mode in MODES:
            identity = fingerprint([row_identity(output, row), sha(prefix_path), mode])
            path = output / "asr" / mode / (order_key(row["id"]) + ".json")
            if load_cell(path, identity) is None:
                pending.append((row, mode, path, identity))
    print(
        f"Streaming resume: completed={len(rows) * 3 - len(pending)}/{len(rows) * 3}; "
        f"pending={len(pending)}; decoder concurrency=3",
        flush=True,
    )
    if not pending:
        print("All streaming cells verified; no GPU reload needed", flush=True)
        return
    with block_legacy_qwen_vllm_registration():
        from qwen_asr import Qwen3ASRModel

        Qwen3ASRProcessor = import_module("qwen_asr.core.transformers_backend").Qwen3ASRProcessor
    processor = Qwen3ASRProcessor.from_pretrained(
        contract["model"], local_files_only=True, fix_mistral_regex=True
    )
    left, right = socket.socketpair()
    channel = Connection(left.detach())
    try:
        child = subprocess.Popen(
            [
                str(ROOT / cfg["decoder_python"]),
                "-u",
                "-B",
                str(Path(__file__)),
                "--stage",
                "decoder",
                "--fd",
                str(right.fileno()),
                "--output-dir",
                str(output),
            ],
            pass_fds=(right.fileno(),),
            cwd=ROOT,
        )
    finally:
        right.close()
    try:
        with stage_progress("Load native vLLM r15 Decoder; Encoder extraction process has exited"):
            ready = receive(channel, child, 1200)
        surfaces = {t["id"]: t["surface"] for t in table}
        active = tuple(t["id"] for t in table if t["pronunciation"]["status"] == "active")
        broker = BatchBroker(channel, child)

        def decode_mode(item: Any, wave: Any, prefixes: Any) -> None:
            row, mode, path, identity = item
            bridge = DecoderBridge(channel, child, processor.tokenizer, broker)
            bridge.metadata = {"sample_id": row["id"], "mode": mode}
            wrapper = Qwen3ASRModel(backend="vllm", model=bridge, processor=processor)
            backend = AuditedBackend(wrapper, bridge)
            prompt_probe(backend, cfg)
            replay = None if mode == "no_rag" else PrefixReplay(prefixes, active, int(mode[3:]))
            # Reference and expected targets are withheld from ALL model/prompt calls.
            sample = StreamingSample(
                case_id=row["id"],
                sample_id=row["id"],
                reference_text="",
                language="ar",
                expected_hotword_ids=(),
                expected_surfaces=(),
                active_hotword_ids=active,
            )
            bridge.calls.clear()
            try:
                result, timeline = run_streaming_sample(
                    backend=backend,
                    waveform=wave,
                    sample=sample,
                    group="C" if replay is None else "D",
                    hotword_surfaces=surfaces,
                    ctc_detector=replay,
                    prompt_template=cfg["prompt_template"],
                    chunk_size_sec=cfg["chunk_seconds"],
                    unfixed_chunk_num=cfg["unfixed_chunk_num"],
                    unfixed_token_num=cfg["unfixed_token_num"],
                    asr_language="Arabic",
                )
            except TokenLimitError as exc:
                failure = {
                    "identity": identity,
                    "id": row["id"],
                    "mode": mode,
                    "source_group": row["source_group"],
                    "audio_path": row["audio_path"],
                    "duration_seconds": row["duration_seconds"],
                    "reference": row["text"],
                    "status": "token_limit_unresolved",
                    "decoder_launch_id": ready["decoder_launch_id"],
                    "detail": exc.detail,
                    "actual_decoder_requests": list(bridge.calls),
                }
                save_cell(output / "failures" / mode / path.name, failure)
                print(
                    f"Recorded unresolved token limit: {row['id']} {mode}; continue other cells",
                    flush=True,
                )
                return
            if replay is not None:
                replay.finish()
            if not bridge.calls:
                raise ValueError("no actual decoder generation was performed")
            injected = {
                str(i) for step in timeline for i in cast(list[str], step["injected_hotword_ids"])
            }
            prediction = str(result["prediction"])
            pin(Path(row["audio_path"]), row["audio_sha256"])
            save_cell(
                path,
                {
                    "identity": identity,
                    "id": row["id"],
                    "mode": mode,
                    "decoder_launch_id": ready["decoder_launch_id"],
                    "source_group": row["source_group"],
                    "prediction": prediction,
                    "reference": row["text"],
                    "timeline": timeline,
                    "actual_decoder_requests": list(bridge.calls),
                    "inference_seconds": result["inference_seconds"],
                    "metrics": score_row(row["text"], prediction, table, injected),
                },
            )

        try:
            grouped: dict[str, list[Any]] = {}
            for item in pending:
                grouped.setdefault(item[0]["id"], []).append(item)
            with stage_progress(
                "Streaming ASR: batched no-RAG / Top5 / Top7", total=len(pending), every=1
            ) as tick:
                completed = 0
                with ThreadPoolExecutor(max_workers=3) as executor:
                    for items in grouped.values():
                        row = items[0][0]
                        wave = audio(row)
                        prefixes = load_cell(
                            output / "prefixes" / (order_key(row["id"]) + ".json"),
                            row_identity(output, row),
                        )["prefixes"]
                        futures = [
                            executor.submit(decode_mode, item, wave, prefixes) for item in items
                        ]
                        for future in as_completed(futures):
                            future.result()
                            completed += 1
                            tick(completed)
        finally:
            broker.close()
    finally:
        channel.close()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def collect(output: Path, scope: str) -> dict[str, Any]:
    contract, rows, table = selection(output, scope)
    cells = []
    failures = []
    for row in rows:
        prefix = output / "prefixes" / (order_key(row["id"]) + ".json")
        load_cell(prefix, row_identity(output, row))
        for mode in MODES:
            identity = fingerprint([row_identity(output, row), sha(prefix), mode])
            cell = load_cell(output / "asr" / mode / (order_key(row["id"]) + ".json"), identity)
            if cell is None:
                failed = load_cell(
                    output / "failures" / mode / (order_key(row["id"]) + ".json"), identity
                )
                if failed is None:
                    raise ValueError("incomplete evaluation; no final report will be emitted")
                failures.append(
                    {
                        key: failed[key]
                        for key in (
                            "id",
                            "mode",
                            "audio_path",
                            "duration_seconds",
                            "status",
                            "detail",
                        )
                    }
                )
            else:
                cells.append(cell)
    failed_ids = {f["id"] for f in failures}
    complete_ids = {r["id"] for r in rows} - failed_ids
    paired = [cell for cell in cells if cell["id"] in complete_ids]
    event_paths = sorted((output / "token_limit_events").glob("*.json"))
    examples = []
    for event_path in event_paths[:10]:
        event = read(event_path)
        event["truncated_text"] = event["truncated_text"][:1600]
        examples.append(event)
    report = {
        "token_limit_event_count": len(event_paths),
        "token_limit_examples": examples,
        "status": "needs_review" if failures else "completed",
        "expected_combinations": len(rows) * 3,
        "completed_combinations": len(cells),
        "unresolved_combinations": failures,
        "expected_target_pairs_full_set": sum(
            len(expected_ids(row["text"], table)) for row in rows
        ),
        "paired_completed_subset_only": summary(paired, complete_ids) if failures else None,
        "scope": scope,
        "output_dir": str(output),
        "contract_sha256": sha(output / "contract.json"),
        "samples": len(rows),
        "table_entries": len(table),
        "active_table_entries": sum(t["pronunciation"]["status"] == "active" for t in table),
        "prompt_audit": {
            mode: {
                "decoder_requests": sum(
                    len(c["actual_decoder_requests"]) for c in cells if c["mode"] == mode
                ),
                "nonempty_prompt_steps": sum(
                    bool(step["injected_hotword_ids"])
                    for c in cells
                    if c["mode"] == mode
                    for step in c["timeline"]
                ),
            }
            for mode in MODES
        },
        "by_mode": None if failures else summary(cells, {r["id"] for r in rows}),
        "by_source": {
            source: {
                mode: aggregate(
                    [c for c in cells if c["mode"] == mode and c["source_group"] == source]
                )
                for mode in MODES
            }
            for source in sorted({r["source_group"] for r in rows})
        }
        if not failures
        else None,
        "token_limit_retry_policy": (
            "512 -> 1024 -> 2048, bounded by context headroom; same prefix/prompt"
        ),
        "recovered_requests": sum(
            len(call.get("generation_attempts", [])) > 1
            for cell in cells
            for call in cell["actual_decoder_requests"]
        ),
        "model": contract["model"],
        "checkpoint": contract["checkpoint"],
        "expected": contract["expected"],
        "selection": contract["selection"],
        "table_sha256": contract["config"]["expected_table_sha256"],
        "decoder_launches": [
            read(path) for path in sorted((output / "decoder_launches").glob("*.json"))
        ],
        "legacy_cells_without_launch_id": sum("decoder_launch_id" not in cell for cell in cells),
        "WER_CER_normalization": NORMALIZATION,
        "notes": [
            "Final transcript recall, not retrieval candidate recall.",
            "Same-step causal prefix prompts; original audio, no added noise.",
            "Offline streaming simulation, not concurrent service latency.",
            "Independent states batch up to 3 requests; retained legacy cells may be serial. "
            "Batching can introduce floating-point differences; launch IDs record execution.",
            "Corpus-conditioned table; not unseen-business-vocabulary evaluation.",
            "Prompt adoption is co-occurrence, not a causal attribution.",
            "False words are measured against reference text, not human audio review.",
        ],
    }
    inputs_unchanged(contract)
    write_json(output / ("smoke_report.json" if scope == "smoke" else "report.json"), report)
    return report


def run(config_path: Path, output: Path, gpu: int, check: bool) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_job(output):
        with stage_progress("Verify frozen 2078 audio, table, epoch29 and installed runtimes"):
            contract = prepare(config_path, output)
        if check:
            return {"status": "checked", "output_dir": str(output)}
        for scope in ("smoke", "full"):
            for stage in ("prefix", "asr"):
                with stage_progress(f"{scope} {stage}; see child sample/chunk progress"):
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-B",
                            str(Path(__file__)),
                            "--stage",
                            stage,
                            "--scope",
                            scope,
                            "--output-dir",
                            str(output),
                        ],
                        check=True,
                        cwd=ROOT,
                    )
            report = collect(output, scope)
            if scope == "smoke" and report["status"] != "completed":
                write_json(output / "report.json", report)
                break
            if scope == "smoke":
                print(
                    "20-sample three-way engineering smoke passed; continue full frozen set",
                    flush=True,
                )
        with stage_progress("Recheck input identities and package small report"):
            inputs_unchanged(contract)
            with zipfile.ZipFile(
                output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for name in (
                    "report.json",
                    "smoke_report.json",
                    "contract.json",
                    "runtime_main.json",
                    "runtime_decoder.json",
                ):
                    archive.write(output / name, name)
                for name in ("memory_runtime_upgrade.json", "token_limit_runtime_upgrade.json"):
                    upgrade = output / name
                    if upgrade.exists():
                        archive.write(upgrade, upgrade.name)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/egy_streaming_e2e.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--decoder-memory",
        choices=("auto", "fixed"),
        default="auto",
        help="startup allocation: auto uses free memory minus reserve; fixed uses config",
    )
    parser.add_argument("--stage", choices=("prefix", "asr", "decoder"))
    parser.add_argument("--scope", choices=("smoke", "full"), default="full")
    parser.add_argument("--probe", choices=("main", "decoder"))
    parser.add_argument("--fd", type=int)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / read(args.config)["output"]).resolve()
    try:
        if args.probe:
            sealed(
                output / ("runtime_" + args.probe + ".json"), runtime_info(args.probe == "decoder")
            )
        elif args.stage == "decoder":
            if args.fd is None:
                raise ValueError("internal decoder requires socket")
            decoder_server(args.fd, output)
        elif args.stage == "prefix":
            prefix_worker(output, args.scope)
        elif args.stage == "asr":
            asr_worker(output, args.scope)
        else:
            os.environ["EGY_DECODER_MEMORY"] = args.decoder_memory
            print(
                json.dumps(
                    run(args.config.resolve(), output, args.gpu, args.check),
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error": str(exc), "output_dir": str(output)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
