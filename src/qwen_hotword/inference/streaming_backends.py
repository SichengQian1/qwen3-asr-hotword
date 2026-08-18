from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_hotword.config import EXPECTED_MODEL_NAME, ModelConfig
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import (
    HotwordMatch,
    HotwordScoringConfig,
    HotwordScoringResult,
    score_hotwords,
)
from qwen_hotword.inference.streaming_core import StreamingCandidate
from qwen_hotword.modeling.qwen_backbone import load_asr_model
from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import build_audio_prompt, freeze_module


class OfficialVllmStreamingBackend:
    """Thin adapter around Qwen's official vLLM streaming interface."""

    def __init__(self, wrapper: Any) -> None:
        if getattr(wrapper, "backend", None) != "vllm":
            raise ValueError("official streaming evaluation requires qwen-asr vLLM backend")
        processor = getattr(wrapper, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("qwen-asr streaming wrapper does not expose processor.tokenizer")
        self.wrapper = wrapper
        self.tokenizer = tokenizer

    def init_streaming_state(self, **kwargs: Any) -> Any:
        return self.wrapper.init_streaming_state(**kwargs)

    def streaming_transcribe(self, audio: Any, state: Any) -> None:
        self.wrapper.streaming_transcribe(audio, state)

    def finish_streaming_transcribe(self, state: Any) -> None:
        self.wrapper.finish_streaming_transcribe(state)


def load_official_vllm_streaming_backend(
    model_path: str | Path,
    *,
    gpu_memory_utilization: float = 0.70,
    max_new_tokens: int = 128,
) -> OfficialVllmStreamingBackend:
    model = Path(model_path).expanduser()
    if model.name != EXPECTED_MODEL_NAME or not model.is_dir():
        raise ValueError(f"model path must be an existing {EXPECTED_MODEL_NAME} directory: {model}")
    if not 0.0 < gpu_memory_utilization < 1.0:
        raise ValueError("gpu_memory_utilization must be between zero and one")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise RuntimeError("qwen-asr with its vLLM extra is required") from error
    wrapper = Qwen3ASRModel.LLM(
        model=str(model),
        gpu_memory_utilization=gpu_memory_utilization,
        max_new_tokens=max_new_tokens,
    )
    return OfficialVllmStreamingBackend(wrapper)


class CumulativeAudioCtcDetector:
    """Run the frozen Qwen encoder and fixed CTC Head on causal cumulative audio."""

    def __init__(
        self,
        *,
        encoder_wrapper: Any,
        head: Any,
        hotwords: tuple[HotwordEntry, ...],
        language: str,
        scoring_config: HotwordScoringConfig,
        retrieval_mode: str,
        device: str,
    ) -> None:
        self.wrapper = encoder_wrapper
        self.head = head
        self.hotword_by_id = {entry.hotword_id: entry for entry in hotwords}
        if len(self.hotword_by_id) != len(hotwords):
            raise ValueError("hotword IDs must be unique")
        self.language = language
        self.scoring_config = scoring_config
        if retrieval_mode not in {"operating", "forced_topk"}:
            raise ValueError(f"unknown CTC retrieval mode: {retrieval_mode}")
        self.retrieval_mode = retrieval_mode
        self.device = device

    def __call__(
        self,
        cumulative_audio: Any,
        active_hotword_ids: tuple[str, ...],
    ) -> tuple[StreamingCandidate, ...]:
        import torch

        from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post

        active = []
        for hotword_id in active_hotword_ids:
            try:
                active.append(self.hotword_by_id[hotword_id])
            except KeyError as error:
                raise ValueError(f"unknown active hotword ID: {hotword_id}") from error
        prompt = build_audio_prompt(self.wrapper.processor, self.language)
        processor_batch = self.wrapper.processor(
            text=[prompt],
            audio=[cumulative_audio],
            return_tensors="pt",
            padding=True,
        )
        input_features = processor_batch["input_features"].to(
            device=self.wrapper.model.device,
            dtype=self.wrapper.model.dtype,
        )
        feature_attention_mask = processor_batch["feature_attention_mask"].to(
            device=self.wrapper.model.device
        )
        encoder = extract_padded_ln_post(
            self.wrapper.model.thinker.audio_tower,
            input_features,
            feature_attention_mask,
            no_grad=True,
        )
        hidden = encoder.hidden_states.to(device=self.device, dtype=torch.float32)
        lengths = encoder.input_lengths.to(device=self.device)
        with torch.no_grad():
            logits = self.head(hidden, input_lengths=lengths)
            effective = self.head.output_lengths(lengths)
        scored = score_hotwords(
            logits[0],
            input_length=int(effective[0].item()),
            hotwords=tuple(active),
            config=self.scoring_config,
            blank_id=0,
        )
        matches = select_streaming_ctc_matches(
            scored,
            scoring_config=self.scoring_config,
            retrieval_mode=self.retrieval_mode,
        )
        return tuple(
            StreamingCandidate(
                hotword_id=match.hotword_id,
                surface=match.surface,
                score=match.score,
                edit_ratio=match.edit_ratio,
                posterior_confidence=match.posterior_confidence,
            )
            for match in matches
        )


def select_streaming_ctc_matches(
    scored: HotwordScoringResult,
    *,
    scoring_config: HotwordScoringConfig,
    retrieval_mode: str,
) -> tuple[HotwordMatch, ...]:
    if retrieval_mode == "operating":
        return scored.selected_matches
    if retrieval_mode == "forced_topk":
        return scored.ranked_matches[: scoring_config.top_k]
    raise ValueError(f"unknown CTC retrieval mode: {retrieval_mode}")


def load_cumulative_ctc_detector(
    *,
    model_path: str | Path,
    checkpoint_path: str | Path,
    vocab: PhonemeVocab,
    hotwords: tuple[HotwordEntry, ...],
    language: str,
    device: str,
    dtype: str,
    scoring_config: HotwordScoringConfig,
    retrieval_mode: str = "operating",
) -> CumulativeAudioCtcDetector:
    import torch

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
    )

    model = Path(model_path).expanduser()
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CTC checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("CTC checkpoint must be a mapping")
    if tuple(payload.get("vocab_tokens", ())) != tuple(vocab.tokens):
        raise ValueError("CTC checkpoint vocabulary differs from the requested vocabulary")
    head = build_ctc_head_from_checkpoint(payload)
    if not isinstance(head, TemporalUpsampleCtcHead) or head.time_upsampling_factor != 2:
        raise ValueError("streaming RAG requires the sealed Temporal 2x CTC Head")
    head.load_state_dict(payload["state_dict"], strict=True)
    head = head.to(device=device, dtype=torch.float32)
    freeze_module(head)
    config = ModelConfig(
        path=model,
        expected_name=EXPECTED_MODEL_NAME,
        dtype=dtype,
        device=device,
        local_files_only=True,
    )
    encoder_wrapper = load_asr_model(config)
    freeze_module(encoder_wrapper.model.thinker.audio_tower)
    return CumulativeAudioCtcDetector(
        encoder_wrapper=encoder_wrapper,
        head=head,
        hotwords=hotwords,
        language=language,
        scoring_config=scoring_config,
        retrieval_mode=retrieval_mode,
        device=device,
    )
