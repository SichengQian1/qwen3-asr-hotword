"""Audited Qwen-ASR 0.0.6 offline context calls, with no retrieval or training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
import platform
import random
from importlib import import_module
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.prompt_order import digest
from qwen_hotword.training.spanish_capacity import _sha

# Inspected from the published qwen-asr 0.0.6 wheel, not inferred from API names.
SOURCE_HASHES = {
    "inference/qwen3_asr.py": "0b1770f8e907b6c5a0a1e9ebce037cb63f48555f3cd15eaf6ea2078e9df41a7b",
    "inference/utils.py": "b37fb4140e6d9be85f2ecb1552ed16136eb98a57f44b402f8b0b213cb6df2a82",
    "core/transformers_backend/modeling_qwen3_asr.py": (
        "2fb5d98da1933748f5117ee05ce4e7150c9ead8154fb8e25f7af3968b853adc7"
    ),
    "core/transformers_backend/processing_qwen3_asr.py": (
        "a61f1fbfc06e3dfc63a2b8d15e3b29e98b032792c3fb7d0e01c04ca9c2181ffb"
    ),
}


def inspect_runtime() -> dict[str, Any]:
    dist = importlib.metadata.distribution("qwen-asr")
    if dist.version != "0.0.6":
        raise ValueError("this audited experiment requires qwen-asr==0.0.6; do not auto-upgrade")
    base = Path(str(dist.locate_file("qwen_asr")))
    hashes = {str(p.relative_to(base)): _sha(p) for p in sorted(base.rglob("*.py"))}
    for name, expected in SOURCE_HASHES.items():
        if hashes.get(name) != expected:
            raise ValueError(f"installed Qwen API source differs from audited 0.0.6: {name}")
    packages = {
        p: importlib.metadata.version(p)
        for p in (
            "qwen-asr",
            "torch",
            "transformers",
            "accelerate",
            "numpy",
            "soundfile",
            "librosa",
        )
    }
    return {
        "python": platform.python_version(),
        "packages": packages,
        "qwen_source_sha256": hashes,
        "backend": "transformers",
        "api": "transcribe(audio, context, language, return_time_stamps)",
        "decode_policy": "thinker GenerationConfig greedy; max_new_tokens wrapper argument",
        "max_unsegmented_audio_seconds": 1200,
    }


def collapse_audio_tokens(ids: list[int], audio_token: int) -> list[int]:
    result: list[int] = []
    for token in ids:
        if token != audio_token or not result or result[-1] != audio_token:
            result.append(token)
    return result


class QwenPromptRuntime:
    def __init__(self, model: Path, cfg: dict[str, Any]) -> None:
        import numpy as np
        import torch
        from qwen_asr import Qwen3ASRModel

        utils = import_module("qwen_asr.inference.utils")

        self.torch, self.np = torch, np
        self.normalize_audios, self.sample_rate = utils.normalize_audios, utils.SAMPLE_RATE
        if not torch.cuda.is_available():
            raise RuntimeError("H200 CUDA is required for full-model inference")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.seed(cfg["seed"])
        self.wrapper = Qwen3ASRModel.from_pretrained(
            str(model),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
            attn_implementation="eager",
            max_inference_batch_size=1,
            max_new_tokens=cfg["max_new_tokens"],
        )
        self.wrapper.model.eval()
        if self.wrapper.backend != "transformers":
            raise RuntimeError("unexpected Qwen backend")
        self.thinker = self.wrapper.model.thinker
        gc = self.thinker.generation_config
        gc.do_sample = False
        gc.num_beams = 1
        gc.num_return_sequences = 1
        self.generation_config = gc.to_dict()
        signature = inspect.signature(self.wrapper.transcribe)
        if not {"audio", "context", "language", "return_time_stamps"} <= set(signature.parameters):
            raise RuntimeError("installed transcribe signature differs")
        self.metadata = {
            "runtime": inspect_runtime(),
            "backend": self.wrapper.backend,
            "transcribe_signature": str(signature),
            "generation_config": self.generation_config,
            "max_new_tokens": self.wrapper.max_new_tokens,
            "max_inference_batch_size": self.wrapper.max_inference_batch_size,
            "attention_implementation": "eager",
            "dtype": str(self.wrapper.dtype),
            "device": "cuda:0",
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32": False,
            "model_training": self.wrapper.model.training,
            "seed_policy": "reset Python/NumPy/Torch/CUDA to the same seed per sample/request",
        }
        self.cached_path: str | None = None
        self.waveform: Any = None

    def seed(self, value: int) -> None:
        random.seed(value)
        self.np.random.seed(value)
        self.torch.manual_seed(value)
        self.torch.cuda.manual_seed_all(value)

    def transcribe(self, request: dict[str, Any]) -> dict[str, Any]:
        self.seed(request["seed"])
        if self.thinker.generation_config.to_dict() != self.generation_config:
            raise RuntimeError("generation configuration changed during experiment")
        if self.cached_path != request["audio_path"]:
            self.waveform = self.normalize_audios(request["audio_path"])[0]
            self.cached_path = request["audio_path"]
        duration = len(self.waveform) / self.sample_rate
        if not 0 < duration <= 1200:
            raise ValueError("experiment requires one full-audio chunk, with duration in (0,1200]s")
        actual_contexts: list[str] = []
        serialized: list[str] = []
        generations: list[dict[str, Any]] = []
        wrapper = self.wrapper
        original_build = wrapper._build_text_prompt
        original_generate = self.thinker.generate

        def audited_build(context: str, force_language: str | None) -> str:
            if context != request["context"] or force_language != "English":
                raise RuntimeError("context/language modified before prompt construction")
            messages = wrapper._build_messages(context=context, audio_payload="")
            if (
                len(messages) != 2
                or messages[0] != {"role": "system", "content": context}
                or messages[1]["role"] != "user"
            ):
                raise RuntimeError("unexpected message history or system prompt rewriting")
            text: str = original_build(context=context, force_language=force_language)
            if context not in text or not text.endswith("language English<asr_text>"):
                raise RuntimeError("serialized prompt lost context or language suffix")
            actual_contexts.append(context)
            serialized.append(text)
            return text

        def audited_generate(*args: Any, **kwargs: Any) -> Any:
            if (
                args
                or len(serialized) != 1
                or generations
                or kwargs.get("past_key_values") is not None
            ):
                raise RuntimeError(
                    "unexpected chunking, generation calls or inherited decoder state"
                )
            ids = kwargs["input_ids"]
            if ids.shape[0] != 1:
                raise RuntimeError("expected one independent audio request")
            input_ids: list[int] = ids[0].detach().cpu().tolist()
            tokenizer = wrapper.processor.tokenizer
            expected = tokenizer(serialized[0], truncation=False)["input_ids"]
            audio_id = tokenizer.convert_tokens_to_ids(wrapper.processor.audio_token)
            if collapse_audio_tokens(input_ids, audio_id) != collapse_audio_tokens(
                expected, audio_id
            ):
                raise RuntimeError(
                    "model input tokens differ from complete prompt; truncation/rewrite"
                )
            effective = {
                "do_sample": kwargs.get("do_sample", self.generation_config["do_sample"]),
                "num_beams": kwargs.get("num_beams", self.generation_config["num_beams"]),
                "max_new_tokens": kwargs["max_new_tokens"],
                "eos_token_id": kwargs.get("eos_token_id"),
                "return_dict_in_generate": kwargs.get("return_dict_in_generate"),
            }
            if (
                effective["do_sample"] is not False
                or effective["num_beams"] != 1
                or effective["max_new_tokens"] != wrapper.max_new_tokens
            ):
                raise RuntimeError("effective greedy decoding changed")
            result = original_generate(**kwargs)
            generated = int(result.sequences.shape[1] - ids.shape[1])
            generations.append(
                {
                    "effective_generate_arguments": effective,
                    "input_token_count": len(input_ids),
                    "input_token_ids_sha256": digest(input_ids),
                    "generated_tokens": generated,
                    "generation_limit_reached": generated >= wrapper.max_new_tokens,
                }
            )
            return result

        wrapper._build_text_prompt = audited_build
        self.thinker.generate = audited_generate
        try:
            with self.torch.inference_mode():
                result = wrapper.transcribe(
                    audio=(self.waveform.copy(), self.sample_rate),
                    context=request["context"],
                    language="English",
                    return_time_stamps=False,
                )
        finally:
            wrapper._build_text_prompt = original_build
            self.thinker.generate = original_generate
        if len(result) != 1 or len(actual_contexts) != 1 or len(generations) != 1:
            raise RuntimeError("request did not produce exactly one audited full-audio result")
        if not isinstance(result[0].text, str):
            raise RuntimeError("ASR result lacks string text")
        return {
            "prediction": result[0].text,
            "returned_language": result[0].language,
            "actual_context": actual_contexts[0],
            "serialized_prompt": serialized[0],
            "prompt_tokens_verified": True,
            "duration_seconds": duration,
            "normalized_audio_sha256": hashlib.sha256(self.waveform.tobytes()).hexdigest(),
            **generations[0],
        }
