"""Real-device serial smoke entry. Explicit opt-in; no device use on import."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path
from typing import Any

from qwen_rag_vllm_sample import ARCHITECTURE, register
from qwen_rag_vllm_sample.contracts import build_context, build_prompt, output_length, owner_tag
from qwen_rag_vllm_sample.retrieval import ProjectRetriever, checked_path


def one_worker(llm: Any, method: str, *args: Any) -> Any:
    replies = llm.collective_rpc(method, args=args)
    if len(replies) != 1:
        raise RuntimeError("sample requires TP=PP=DP=1 with exactly one model worker")
    return replies[0]


def verify_prompt(actual: Any, original: list[int], audio_token: int, rows: int) -> None:
    if original.count(audio_token) != 1:
        raise ValueError("expected one original audio placeholder")
    offset = original.index(audio_token)
    expected = original[:offset] + [audio_token] * rows + original[offset + 1 :]
    if list(actual) != expected:
        raise RuntimeError("final expanded Prompt token IDs differ; stop integration test")


def prepare_acoustics(processor: Any, audio: Any, language: str) -> tuple[Any, int]:
    import numpy as np

    # Match vLLM 0.23.0 Qwen3Omni processor's pad-to-hop-length behavior.
    hop = processor.feature_extractor.hop_length
    padded = np.pad(audio, (0, (-len(audio)) % hop))
    batch = processor(
        text=[build_prompt("", language)],
        audio=[padded],
        return_tensors="pt",
        padding=True,
    )
    length = len(padded) // hop
    if int(batch["feature_attention_mask"][0].sum().item()) != length:
        raise ValueError("feature mask differs from expected acoustic length")
    return batch["input_features"][0, :, :length].contiguous().cpu(), length


def run(config: dict[str, Any], audio_path: Path, language: str) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    import torch
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor

    versions = {name: importlib.metadata.version(name) for name in ("vllm", "vllm-ascend")}
    if any(version.split("+")[0] != "0.23.0" for version in versions.values()):
        raise RuntimeError(f"source-based sample targets both 0.23.0 packages: {versions}")
    for name in ("head", "vocab", "index"):
        checked_path(config, name)
    model = Path(config["model_path"]).expanduser().resolve(strict=True)
    if model.name != "Qwen3-ASR-1.7B" or not (model / "config.json").is_file():
        raise ValueError("model_path must be a complete local Qwen3-ASR-1.7B snapshot")
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1 or not 0 < len(audio) <= 30 * 16000:
        raise ValueError("PoC input must be mono 16kHz, duration (0,30]s; no silent conversion")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) <= 0:
        raise ValueError("audio must be finite and not all zeros")
    retriever = ProjectRetriever(config)
    register()
    print("[1/4] Load one Qwen into one vLLM instance; native engine logs follow", flush=True)
    llm = LLM(
        model=str(model),
        hf_overrides={"architectures": [ARCHITECTURE]},
        worker_extension_cls="qwen_rag_vllm_sample.model.RagWorkerExtension",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype="float16",
        enforce_eager=True,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=float(config["gpu_memory_utilization"]),
        additional_config={
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
                "enable_npu_graph_ex": False,
            }
        },
    )
    processor = Qwen3ASRProcessor.from_pretrained(str(model), local_files_only=True)
    tokenizer = processor.tokenizer
    features, feature_length = prepare_acoustics(processor, audio, language)
    rows = output_length(feature_length)
    sampling = SamplingParams(temperature=0, max_tokens=512)
    audio_token = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    results = []

    # No ordinary generate() is issued before preparation for either cached
    # route. raw_baseline is a separate comparison experiment, not phase one.
    for mode in ("raw_baseline", "cached_no_prompt", "cached_ctc_prompt"):
        print(f"[2/4] {mode}: start audio preparation", flush=True)
        before = one_worker(llm, "rag_stats")
        prepared = None
        try:
            matches = []
            if mode != "raw_baseline":
                request_owner = owner_tag(uuid.uuid4().hex)
                prepared = one_worker(
                    llm, "rag_prepare_audio", features, feature_length, request_owner
                )
                if mode == "cached_ctc_prompt":
                    matches = retriever.search(prepared)
            words = [match["surface"] for match in matches]
            context = build_context(language, words)
            text = build_prompt(context, language)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) - 1 + rows + 512 > 4096:
                raise ValueError("final Prompt + audio + output exceeds the fixed context budget")
            audio_data: Any
            if prepared is None:
                audio_data = (audio, sample_rate)
            else:
                audio_data = {
                    "rag_audio_handle": torch.tensor([prepared["handle"]], dtype=torch.int64),
                    "rag_audio_owner": torch.tensor([prepared["owner"]], dtype=torch.int64),
                    "audio_feature_lengths": torch.tensor([feature_length], dtype=torch.int64),
                }
            print(
                f"[3/4] {mode}: final Prompt ready; generate with {len(words)} hotwords", flush=True
            )
            output = llm.generate(
                {"prompt_token_ids": ids, "multi_modal_data": {"audio": audio_data}},
                sampling_params=sampling,
                use_tqdm=False,
            )[0]
            verify_prompt(output.prompt_token_ids, ids, audio_token, rows)
            during = one_worker(llm, "rag_stats")
            if during["encoder_calls"] - before["encoder_calls"] != 1:
                raise RuntimeError("each route must execute Encoder exactly once")
            if prepared is not None and during["cache_reads"] <= before["cache_reads"]:
                raise RuntimeError("Decoder route did not read the prepared audio representation")
            if output.outputs[0].finish_reason == "length":
                raise RuntimeError(
                    "generation hit length limit; do not accept truncated smoke result"
                )
            results.append(
                {
                    "mode": mode,
                    "selected_hotwords": words,
                    "context": context,
                    "raw_text": output.outputs[0].text,
                    "text": output.outputs[0].text.rsplit("<asr_text>", 1)[-1],
                    "prompt_tokens_verified": True,
                    "final_prompt_token_sha256": hashlib.sha256(
                        json.dumps(
                            output.prompt_token_ids,
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "encoder_calls_for_route": during["encoder_calls"] - before["encoder_calls"],
                    "cache_reads_for_route": during["cache_reads"] - before["cache_reads"],
                }
            )
        finally:
            # generate() here is synchronous: on a completed/error-returned call
            # the serial worker RPC is ordered after execution. A timed-out async
            # serving request needs abort acknowledgement BEFORE release.
            if prepared is not None:
                one_worker(llm, "rag_release_audio", prepared["handle"], prepared["owner"])
        after = one_worker(llm, "rag_stats")
        if after["owned_audio_cache_entries"] != 0:
            raise RuntimeError("sample-owned audio entry leaked")
        if mode == "cached_no_prompt" and results[0]["raw_text"] != results[1]["raw_text"]:
            break  # Isolate the audio bypass before adding prompt changes.
    equal = results[0]["raw_text"] == results[1]["raw_text"]
    print(
        "[4/4] Comparison complete; this run is functional validation, not a latency benchmark",
        flush=True,
    )
    status = "baseline_cache_mismatch"
    if equal:
        status = "smoke_passed" if results[2]["selected_hotwords"] else "cache_passed_no_hotwords"
    return {
        "status": status,
        "device_validation": "operator executed; record real NPU model and image separately",
        "versions": versions,
        "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
        "feature_length": feature_length,
        "audio_embedding_rows": rows,
        "raw_vs_cached_no_prompt_exact_text_equal": equal,
        "results": results,
        "memory_saving_measured": False,
        "latency_benchmarked": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--language", choices=("en", "es", "pt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-on-device", action="store_true")
    args = parser.parse_args()
    if not args.execute_on_device:
        parser.error("real NPU/model use requires the explicit --execute-on-device flag")
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    os.environ["QWEN_RAG_SAMPLE_CONFIG"] = str(config_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Never replace a prior output. There is no resume, no training and no bulk evaluation.
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        report = run(config, args.audio.resolve(strict=True), args.language)
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if report["status"] != "smoke_passed":
        raise SystemExit(f"not a full CTC-injection pass: {report['status']}; inspect report.json")


if __name__ == "__main__":
    main()
