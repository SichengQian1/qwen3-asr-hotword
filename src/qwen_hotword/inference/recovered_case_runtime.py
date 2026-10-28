"""Independent EN/ES/PT full-audio calls for hunting real final-ASR recovery cases."""

from __future__ import annotations

import hashlib
from typing import Any

from qwen_hotword.evaluation.prompt_order import digest
from qwen_hotword.inference.prompt_order_runtime import QwenPromptRuntime, collapse_audio_tokens

LANGUAGES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}


class RecoveredCaseRuntime(QwenPromptRuntime):
    """Reuse the audited loader; leave the historical English experiment untouched."""

    def transcribe(self, request: dict[str, Any]) -> dict[str, Any]:
        language = request["language"]
        if language not in LANGUAGES.values():
            raise ValueError("explicit English/Spanish/Portuguese language required")
        self.seed(request["seed"])
        if self.thinker.generation_config.to_dict() != self.generation_config:
            raise RuntimeError("generation configuration changed during case hunt")
        if self.cached_path != request["audio_path"]:
            self.waveform = self.normalize_audios(request["audio_path"])[0]
            self.cached_path = request["audio_path"]
        duration = len(self.waveform) / self.sample_rate
        if not 0 < duration <= 1200 or not self.np.isfinite(self.waveform).all():
            raise ValueError("one finite full-audio chunk in (0,1200]s required")
        actual_contexts: list[str] = []
        serialized: list[str] = []
        generations: list[dict[str, Any]] = []
        wrapper = self.wrapper
        original_build = wrapper._build_text_prompt
        original_generate = self.thinker.generate

        def audited_build(context: str, force_language: str | None) -> str:
            if context != request["context"] or force_language != language:
                raise RuntimeError("context/language changed before prompt construction")
            messages = wrapper._build_messages(context=context, audio_payload="")
            if (
                len(messages) != 2
                or messages[0] != {"role": "system", "content": context}
                or messages[1]["role"] != "user"
            ):
                raise RuntimeError("unexpected message history or system prompt rewriting")
            text: str = original_build(context=context, force_language=force_language)
            if context not in text or not text.endswith(f"language {language}<asr_text>"):
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
                raise RuntimeError("unexpected chunking or inherited decoder state")
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
                raise RuntimeError("model input tokens differ; prompt truncation/rewrite")
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
                    language=language,
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
