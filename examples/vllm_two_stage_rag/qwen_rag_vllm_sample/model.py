"""Source-based vLLM 0.23.0 PoC; NOT tested on an Ascend 310P.

Normal raw-audio/profile calls retain the upstream path. Only explicit cache
handle inputs bypass the audio tower. No core vLLM files are overwritten.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from transformers.feature_extraction_utils import BatchFeature
from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalDataParser,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import DictEmbeddingItems

from qwen_rag_vllm_sample.contracts import AudioCache, output_length


def cache_fields(_: Any = None) -> dict[str, Any]:
    return {
        key: MultiModalFieldConfig.batched("audio")
        for key in ("rag_audio_handle", "rag_audio_owner", "audio_feature_lengths")
    }


class CacheParser(Qwen3ASRMultiModalDataParser):
    def _parse_audio_data(self, data: Any) -> Any:
        if isinstance(data, dict) and "rag_audio_handle" in data:
            if set(data) != set(cache_fields()):
                raise ValueError("cached audio accepts exactly handle, owner and acoustic length")
            if any(tensor.shape != (1,) or tensor.dtype != torch.int64 for tensor in data.values()):
                raise ValueError("sample supports one audio item with int64 tensors of shape [1]")
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields=set(cache_fields()),
                fields_factory=cache_fields,
            )
        return super()._parse_audio_data(data)


class CacheProcessingInfo(Qwen3ASRProcessingInfo):
    def get_data_parser(self) -> CacheParser:
        return CacheParser(
            target_sr=self.get_feature_extractor().sampling_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class CacheProcessor(Qwen3ASRMultiModalProcessor):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Any,
        mm_kwargs: Any,
        tok_kwargs: Any,
    ) -> Any:
        if not mm_data:
            # Cache handles are passthrough data, not actual waveforms. Calling
            # Qwen's audio processor with an audio placeholder but no waveform
            # can exhaust its audio-length iterator. Tokenize text only here;
            # the inherited vLLM prompt update expands audio_pad afterwards.
            return BatchFeature(self.info.get_tokenizer()([prompt], **dict(tok_kwargs)))
        return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

    def _get_mm_fields_config(self, hf_inputs: Any, hf_processor_mm_kwargs: Any) -> Any:
        if "rag_audio_handle" in hf_inputs:
            return cache_fields()
        return super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)

    # Inherit the ORIGINAL placeholder expansion: it uses acoustic frame length.
    # Inherit Qwen's MRoPE implementation on the model; no old text KV is reused.


class WorkerState:
    """Plain object keeps separately loaded Head out of Qwen weight-key matching.

    Construct during model load, BEFORE engine memory profiling. An nn.Module
    attached directly as model.ctc_head would add absent keys to Qwen loading.
    """

    def __init__(self, device: Any) -> None:
        from qwen_hotword.modeling.ctc_head import (
            TemporalUpsampleCtcHead,
            build_ctc_head_from_checkpoint,
        )
        from qwen_hotword.phonemes.coverage import load_phoneme_vocab
        from qwen_rag_vllm_sample.retrieval import checked_path

        config_path = os.environ.get("QWEN_RAG_SAMPLE_CONFIG")
        if not config_path:
            raise ValueError("set QWEN_RAG_SAMPLE_CONFIG to an absolute sample config path")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        vocab = load_phoneme_vocab(checked_path(config, "vocab"))
        payload = torch.load(checked_path(config, "head"), map_location="cpu", weights_only=True)
        if tuple(payload["vocab_tokens"]) != vocab.tokens or len(vocab.tokens) != 100:
            raise ValueError("this sample requires the matching 100-class v0.3 Head/vocabulary")
        # vLLM may set the default dtype/device during model construction.
        # Build in FP32 BEFORE loading, so FP32 checkpoint values are not first
        # rounded to FP16 then expanded back to FP32.
        with torch.device("cpu"):
            head = build_ctc_head_from_checkpoint(payload).float()
        if (
            not isinstance(head, TemporalUpsampleCtcHead)
            or head.time_upsampling_factor != 2
            or head.num_classes != 100
            or head.input_dimension != 1024
        ):
            raise ValueError("expected Temporal 2x CTC Head")
        head.load_state_dict(payload["state_dict"], strict=True)
        self.head = head.to(device=device, dtype=torch.float32).eval()
        self.head.requires_grad_(False)
        self.cache = AudioCache(max_bytes=int(config.get("audio_cache_bytes", 33554432)))
        self.encoder_calls = 0
        self.cache_reads = 0


@MULTIMODAL_REGISTRY.register_processor(
    CacheProcessor,
    info=CacheProcessingInfo,
    dummy_inputs=Qwen3ASRDummyInputsBuilder,
)
class Qwen3ASRTwoStageForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.rag = WorkerState(next(self.audio_tower.parameters()).device)

    def _process_audio_input(self, audio_input: Any) -> Any:
        self.rag.encoder_calls += 1
        return super()._process_audio_input(audio_input)

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def rag_prepare(self, input_features: Any, feature_length: int, owner: int) -> dict[str, Any]:
        from qwen_hotword.hotwords.scoring import decode_ctc_posterior

        if input_features.ndim != 2 or input_features.shape != (128, feature_length):
            raise ValueError("expected unpadded acoustic input [128, acoustic_frames]")
        captured = []

        def capture(_module: Any, _args: Any, hidden: Any) -> None:
            captured.append(hidden)

        # Local, short-lived hook is ONLY for this serial/eager PoC. For a serving
        # runner use an explicit (ln_post, projected_audio) return contract.
        hook = self.audio_tower.ln_post.register_forward_hook(capture)
        device = next(self.audio_tower.parameters()).device
        try:
            embeddings = self._process_audio_input(
                {
                    "input_features": input_features.to(device),
                    "audio_feature_lengths": torch.tensor([feature_length], device=device),
                }
            )
        finally:
            hook.remove()
        if len(captured) != 1 or len(embeddings) != 1:
            raise RuntimeError("expected one audio tower call and one ln_post tensor")
        hidden, projected = captured.pop(), embeddings[0]
        rows = output_length(feature_length)
        if hidden.shape != (rows, 1024) or projected.shape != (rows, 2048):
            raise ValueError("Qwen hidden/projected shape differs; stop, do not reshape blindly")
        lengths = torch.tensor([rows], device=device, dtype=torch.long)
        logits = self.rag.head(hidden.float().unsqueeze(0), input_lengths=lengths)
        effective = int(self.rag.head.output_lengths(lengths)[0].item())
        decoded = decode_ctc_posterior(logits[0], input_length=effective, blank_id=0)
        key = self.rag.cache.put(
            projected.detach(),
            owner=owner,
            feature_length=feature_length,
            byte_size=projected.numel() * projected.element_size(),
        )
        return {
            "handle": key,
            "owner": owner,
            "feature_length": feature_length,
            "embedding_rows": rows,
            "ctc_steps": effective,
            "decoded": [phone.to_dict() for phone in decoded],
        }

    def embed_multimodal(self, **kwargs: Any) -> Any:
        if "rag_audio_handle" not in kwargs:
            return super().embed_multimodal(**kwargs)
        if "input_audio_features" in kwargs:
            raise ValueError("cannot combine cached audio and raw acoustic features")
        values = {
            name: kwargs[name].reshape(-1).tolist()
            for name in ("rag_audio_handle", "rag_audio_owner", "audio_feature_lengths")
        }
        if any(len(value) != 1 for value in values.values()):
            raise ValueError("sample is serial, single-audio only")
        projected = self.rag.cache.get(
            int(values["rag_audio_handle"][0]),
            owner=int(values["rag_audio_owner"][0]),
            feature_length=int(values["audio_feature_lengths"][0]),
        )
        self.rag.cache_reads += 1
        return (projected,)


class RagWorkerExtension:
    """vLLM worker_extension_cls methods, called through LLM.collective_rpc."""

    def rag_prepare_audio(self, features: Any, feature_length: int, owner: int) -> Any:
        return cast(Any, self).get_model().rag_prepare(features, feature_length, owner)

    def rag_release_audio(self, handle: int, owner: int) -> None:
        cast(Any, self).get_model().rag.cache.release(handle, owner=owner)

    def rag_stats(self) -> Mapping[str, int]:
        state = cast(Any, self).get_model().rag
        return {
            "encoder_calls": state.encoder_calls,
            "cache_reads": state.cache_reads,
            "owned_audio_cache_bytes": state.cache.bytes_used,
            "owned_audio_cache_entries": len(state.cache.entries),
        }
