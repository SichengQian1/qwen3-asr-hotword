"""Isolated CPU preparation / Transformers / native vLLM subprocesses for H200."""

from __future__ import annotations

import hashlib
import importlib.metadata
import random
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.backend_comparison import (
    CONDITIONS,
    LANGUAGES,
    block_legacy_qwen_vllm_registration,
    context,
    digest,
    public_wrapper_eos_ids,
    read,
    saved_source_rows,
    sha,
    write_once,
)
from qwen_hotword.training.stage_progress import stage_progress


def versions() -> dict[str, str]:
    result = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "torch", "torchaudio", "torchvision", "transformers", "qwen-asr")
    }
    expected = {
        "vllm": "0.23.0+cu129",
        "torch": "2.11.0+cu129",
        "torchaudio": "2.11.0+cu129",
        "torchvision": "0.26.0+cu129",
        "transformers": "4.57.6",
        "qwen-asr": "0.0.6",
    }
    if result != expected:
        raise ValueError(f"test environment version drift: {result}; expected {expected}")
    for package, relative in (
        ("vllm", "vllm/model_executor/models/qwen3_asr.py"),
        ("qwen-asr", "qwen_asr/inference/qwen3_asr.py"),
        ("qwen-asr", "qwen_asr/core/transformers_backend/modeling_qwen3_asr.py"),
    ):
        path = Path(str(importlib.metadata.distribution(package).locate_file(relative)))
        result[f"source:{relative}"] = sha(path)
    return result


def prepare(root: Path, cfg: dict[str, Any], output: Path) -> None:
    import librosa
    import numpy as np
    import soundfile as sf

    with block_legacy_qwen_vllm_registration():
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    ready = output / "requests.json"
    if ready.exists():
        plan = read(ready)
        if plan["versions"] != versions():
            raise ValueError("installed code or packages changed after sample preparation")
        for request in plan["requests"]:
            if sha(output / request["waveform_file"]) != request["waveform_file_sha256"]:
                raise ValueError("saved canonical audio changed; refusing reuse")
        print(f"Reuse {len(plan['requests'])} frozen requests; no new sample selection", flush=True)
        return
    source = root / cfg["source_run"]
    processor = Qwen3ASRProcessor.from_pretrained(
        cfg["model"],
        local_files_only=True,
        fix_mistral_regex=True,
    )
    if processor.feature_extractor.hop_length != cfg["pad_waveform_to_samples"]:
        raise ValueError("configured shared waveform padding differs from model hop length")
    counts: Counter[tuple[str, str]] = Counter()
    seen_audio: set[str] = set()
    seen_waveforms: set[str] = set()
    requests = []
    rejected: Counter[str] = Counter()
    selected = 0
    target = 6 * cfg["samples_per_language_condition"]
    (output / "waveforms").mkdir(exist_ok=True)
    with stage_progress(
        "Select shared audio and freeze both Prompt conditions", total=target, every=1
    ) as tick:
        for row in saved_source_rows(source, cfg):
            lang, condition = row["language"], row["condition"]
            if lang not in LANGUAGES or condition not in CONDITIONS:
                raise ValueError("unexpected source language/condition")
            bucket = lang, condition
            if counts[bucket] >= cfg["samples_per_language_condition"]:
                continue
            audio_path = Path(row["audio_path"])
            info = sf.info(audio_path)
            if not 0 < info.duration <= cfg["maximum_audio_seconds"]:
                rejected["duration_outside_short_smoke_range"] += 1
                continue
            audio_sha = sha(audio_path)
            if audio_sha != row["identity"]["audio_sha256"]:
                raise ValueError(f"source audio changed: {audio_path}")
            if audio_sha in seen_audio:
                rejected["duplicate_file_sha256"] += 1
                continue
            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
            if sample_rate != cfg["sample_rate"]:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=cfg["sample_rate"],
                )
            if not np.isfinite(waveform).all() or not np.any(waveform):
                raise ValueError(f"nonfinite/empty/silent input: {audio_path}")
            original_samples = len(waveform)
            waveform = np.pad(waveform, (0, (-len(waveform)) % cfg["pad_waveform_to_samples"]))
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
            wave_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
            if wave_hash in seen_waveforms:
                rejected["duplicate_normalized_waveform_sha256"] += 1
                continue
            seen_waveforms.add(wave_hash)
            wave_file = f"waveforms/{wave_hash}.npy"
            destination = output / wave_file
            if not destination.exists():
                np.save(destination, waveform, allow_pickle=False)
            elif not np.array_equal(np.load(destination, allow_pickle=False), waveform):
                raise ValueError("partially prepared waveform differs; use a new output")
            seen_audio.add(audio_sha)
            words = row["top7"]["words"]
            for mode, selected_words in (("no_hotwords", []), ("new70_top7", words)):
                prompt_context = context(lang, selected_words)
                messages = [
                    {"role": "system", "content": prompt_context},
                    {"role": "user", "content": [{"type": "audio", "audio": ""}]},
                ]
                prompt = (
                    processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    + f"language {LANGUAGES[lang]}<asr_text>"
                )
                request = {
                    "request_id": digest([row["identity"], mode]),
                    "sample_id": row["sample_id"],
                    "language": lang,
                    "condition": condition,
                    "mode": mode,
                    "audio_path": str(audio_path),
                    "audio_sha256": audio_sha,
                    "waveform_file": wave_file,
                    "waveform_file_sha256": sha(destination),
                    "waveform_sha256": wave_hash,
                    "original_normalized_samples": original_samples,
                    "shared_padded_samples": len(waveform),
                    "reference_text": row["reference_text"],
                    "words": selected_words,
                    "context": prompt_context,
                    "prompt": prompt,
                    "initial_prompt_ids": processor.tokenizer(prompt)["input_ids"],
                    "source_cell": row["source_cell"],
                    "source_cell_sha256": row["source_cell_sha256"],
                }
                requests.append(request)
            counts[bucket] += 1
            selected += 1
            tick(selected)
            if selected == target:
                break
    if selected != target:
        raise ValueError(f"insufficient short samples: {selected}/{target}; buckets={dict(counts)}")
    write_once(
        ready,
        {
            "requests": requests,
            "audio_samples": selected,
            "selection": "first eligible saved-cell SHA order; no selection by transcript success",
            "rejected": dict(rejected),
            "versions": versions(),
            "shared_waveform_padding": "zero-pad to hop multiple BEFORE both backends",
            "empty_top7_audio": sum(not r["words"] for r in requests if r["mode"] == "new70_top7"),
        },
    )


def wave(request: dict[str, Any], output: Path) -> Any:
    import numpy as np

    values = np.load(output / request["waveform_file"], allow_pickle=False)
    if values.ndim != 1 or values.dtype != np.float32:
        raise ValueError("canonical waveform shape/dtype changed")
    if hashlib.sha256(values.tobytes()).hexdigest() != request["waveform_sha256"]:
        raise ValueError("canonical waveform identity changed")
    return values


def content_tokens(tokens: list[int], stop_ids: list[int]) -> list[int]:
    return tokens[:-1] if tokens and tokens[-1] in stop_ids else tokens


def probe(output: Path) -> None:
    """Cheap real CUDA/native-kernel check after installation; no Qwen model load."""
    import torch
    from vllm import _custom_ops as ops

    if not torch.cuda.is_available() or torch.version.cuda != "12.9":
        raise ValueError("installed CUDA12.9 Torch cannot initialize on this driver")
    with torch.inference_mode():
        values = torch.randn(16, 1024, device="cuda:0", dtype=torch.float16)
        result = torch.empty_like(values)
        ops.rms_norm(result, values, torch.ones(1024, device="cuda:0", dtype=torch.float16), 1e-6)
        matmul = values @ values.T
        torch.cuda.synchronize()
        if not torch.isfinite(result).all().item() or not torch.isfinite(matmul).all().item():
            raise ValueError("CUDA/native vLLM kernel probe produced nonfinite values")
    write_once(
        output / "gpu_probe.json",
        {
            "status": "basic_cuda_and_vllm_rmsnorm_passed",
            "versions": versions(),
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "qwen_model_loaded": False,
            "all_attention_or_jit_kernels_proven": False,
        },
    )


def infer(backend: str, cfg: dict[str, Any], output: Path) -> None:
    import numpy as np
    import torch

    runtime = versions()
    if not torch.cuda.is_available() or torch.version.cuda != "12.9":
        raise ValueError("requires visible H200 and matching CUDA 12.9 PyTorch build")
    if "H200" not in torch.cuda.get_device_name(0):
        raise ValueError("this experiment targets the user's H200 work zone")
    runtime.update({"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})
    if backend == "transformers":
        # This caps PyTorch's caching allocator, not all CUDA/context allocations.
        # Parent preflight separately reserves headroom for the shared GPU.
        limit_bytes = int(float(cfg["transformers_allocator_limit_gib"]) * 1024**3)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(limit_bytes / total_bytes, device=0)
        runtime["transformers_allocator_limit_bytes"] = str(limit_bytes)
    plan = read(output / "requests.json")
    if plan["versions"] != versions():
        raise ValueError("backend packages/code differ from frozen preparation")
    requests = plan["requests"]
    stage = output / backend
    (stage / "cells").mkdir(parents=True, exist_ok=True)
    write_once(stage / "runtime.json", runtime)
    completed = {}
    for path in (stage / "cells").glob("*.json"):
        value = read(path)
        if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
            raise ValueError("saved inference cell changed")
        completed[path.stem] = value
    for request in requests:
        saved = completed.get(request["request_id"])
        if saved and saved["request_sha256"] != digest(request):
            raise ValueError("saved backend result belongs to a different request")
    if set(completed) == {r["request_id"] for r in requests}:
        print(f"Reuse all {len(requests)} completed {backend} calls; do not load model", flush=True)
        return
    if backend == "transformers":
        with block_legacy_qwen_vllm_registration():
            from qwen_asr import Qwen3ASRModel

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        with stage_progress("Load Transformers Qwen3-ASR-1.7B FP16"):
            engine = Qwen3ASRModel.from_pretrained(
                cfg["model"],
                dtype=torch.float16,
                device_map="cuda:0",
                local_files_only=True,
                attn_implementation="eager",
                max_inference_batch_size=1,
                max_new_tokens=cfg["max_new_tokens"],
            )
        engine.model.eval()
        generation = engine.model.thinker.generation_config
        unsupported = {
            key: getattr(generation, key, None)
            for key in (
                "bad_words_ids",
                "force_words_ids",
                "forced_bos_token_id",
                "forced_eos_token_id",
                "suppress_tokens",
                "begin_suppress_tokens",
                "sequence_bias",
                "stop_strings",
                "no_repeat_ngram_size",
                "encoder_no_repeat_ngram_size",
                "min_length",
                "min_new_tokens",
            )
            if getattr(generation, key, None) not in (None, 0, [], {})
        }
        if unsupported:
            raise ValueError(f"unmapped Transformers generation constraints: {unsupported}")
        stop_ids = public_wrapper_eos_ids(engine.model.generate)
        write_once(
            output / "decode_policy.json",
            {
                "stop_token_ids": stop_ids,
                "max_new_tokens": cfg["max_new_tokens"],
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.0,
            },
        )
        sampling = None
    else:
        # IMPORTANT: do NOT import qwen_asr in this subprocess. Its 0.0.6 import
        # attempts to register its older vLLM architecture over the native model.
        from vllm import LLM, SamplingParams

        policy = read(output / "decode_policy.json")
        stop_ids = policy["stop_token_ids"]
        with stage_progress("Load NATIVE vLLM 0.23.0 Qwen3-ASR-1.7B FP16"):
            engine = LLM(
                model=cfg["model"],
                dtype="float16",
                tensor_parallel_size=1,
                enforce_eager=True,
                enable_prefix_caching=False,
                max_num_seqs=1,
                max_model_len=cfg["max_model_len"],
                generation_config="vllm",
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                seed=cfg["seed"],
            )
        implementation = engine.apply_model(lambda model: type(model).__module__)
        if implementation != ["vllm.model_executor.models.qwen3_asr"]:
            raise ValueError(f"not the intended native vLLM model: {implementation}")
        write_once(stage / "implementation.json", {"model_modules": implementation})
        if engine.get_tokenizer().eos_token_id not in stop_ids:
            raise ValueError("vLLM's implicit EOS is outside the Transformers EOS policy")
        sampling = SamplingParams(
            temperature=0,
            max_tokens=cfg["max_new_tokens"],
            stop_token_ids=stop_ids,
            repetition_penalty=1.0,
            seed=cfg["seed"],
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
        )
    with stage_progress(f"{backend} matched requests", total=len(requests), every=1) as tick:
        for number, request in enumerate(requests, 1):
            if request["request_id"] in completed:
                tick(number)
                continue
            values = wave(request, output)
            random.seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            torch.manual_seed(cfg["seed"])
            if backend == "transformers":
                batch = engine.processor(
                    text=[request["prompt"]],
                    audio=[values],
                    return_tensors="pt",
                    padding=True,
                )
                ids = batch["input_ids"][0].tolist()
                if len(ids) + cfg["max_new_tokens"] > cfg["max_model_len"]:
                    raise ValueError("shared context budget exceeded")
                batch = batch.to("cuda:0").to(torch.float16)
                with torch.inference_mode():
                    # The public wrapper itself hard-codes return_dict=True in
                    # its thinker call. Invoke that same thinker directly with
                    # the PUBLIC wrapper's EOS defaults, avoiding duplicate
                    # return_dict_in_generate keyword forwarding.
                    generated = engine.model.thinker.generate(
                        **batch,
                        do_sample=False,
                        num_beams=1,
                        num_return_sequences=1,
                        repetition_penalty=1.0,
                        max_new_tokens=cfg["max_new_tokens"],
                        eos_token_id=stop_ids,
                        return_dict_in_generate=True,
                    )
                tokens = generated.sequences[0, len(ids) :].detach().cpu().tolist()
                text = engine.processor.tokenizer.decode(
                    tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                truncated = len(tokens) >= cfg["max_new_tokens"] and tokens[-1] not in stop_ids
            else:
                result = engine.generate(
                    {
                        "prompt_token_ids": request["initial_prompt_ids"],
                        "multi_modal_data": {"audio": (values, cfg["sample_rate"])},
                    },
                    sampling_params=sampling,
                    use_tqdm=False,
                )[0]
                ids = list(result.prompt_token_ids)
                reference_input = read(
                    output / "transformers/cells" / f"{request['request_id']}.json"
                )
                if ids != reference_input["input_token_ids"]:
                    write_once(
                        output / "input_mismatch.json",
                        {
                            "request_id": request["request_id"],
                            "transformers_input_ids": reference_input["input_token_ids"],
                            "vllm_input_ids": ids,
                        },
                    )
                    raise ValueError(
                        "expanded Prompt IDs differ; stopping before claiming comparability"
                    )
                tokens = list(result.outputs[0].token_ids)
                text = result.outputs[0].text
                truncated = result.outputs[0].finish_reason == "length"
            if truncated:
                raise ValueError(f"generation truncated at output limit: {request['request_id']}")
            cell = {
                "request_sha256": digest(request),
                "text": text,
                "input_token_ids": ids,
                "content_token_ids": content_tokens(tokens, stop_ids),
                "waveform_sha256": request["waveform_sha256"],
                "generation_limit_reached": False,
            }
            cell["sha256"] = digest(cell)
            write_once(stage / "cells" / f"{request['request_id']}.json", cell)
            tick(number)
