from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_hotword.inference.prompt_order_runtime import QwenPromptRuntime, collapse_audio_tokens


class Tensor:
    def __init__(self, values):
        self.values = np.array(values)
        self.shape = self.values.shape

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


class Tokenizer:
    def __call__(self, text, truncation=False):
        assert truncation is False
        # Character-level mock; '@' represents the special audio token.
        return {"input_ids": [ord(c) for c in text]}

    def convert_tokens_to_ids(self, token):
        return ord(token)


class Wrapper:
    def __init__(self, thinker, mode):
        self.thinker = thinker
        self.mode = mode
        self.processor = SimpleNamespace(tokenizer=Tokenizer(), audio_token="@")
        self.max_new_tokens = 512

    def _build_messages(self, context, audio_payload):
        messages = [{"role": "system", "content": context}, {"role": "user", "content": "audio"}]
        if self.mode == "history":
            messages.append({"role": "assistant", "content": "previous sample"})
        return messages

    def _build_text_prompt(self, context, force_language):
        return f"system:{context} user:@ assistant:language {force_language}<asr_text>"

    def transcribe(self, audio, context, language, return_time_stamps):
        assert language == "English" and return_time_stamps is False
        assert audio[1] == 16000
        context = context.replace("Alpha", "Rewritten") if self.mode == "context" else context
        prompt = self._build_text_prompt(context=context, force_language=language)
        ids = self.processor.tokenizer(prompt.replace("@", "@@@"))["input_ids"]
        if self.mode == "truncation":
            ids = ids[2:]
        if self.mode == "rewrite":
            ids[9] += 1
        kwargs = {
            "input_ids": Tensor([ids]),
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": [151645, 151643],
            "return_dict_in_generate": True,
        }
        if self.mode == "cache":
            kwargs["past_key_values"] = "state from another sample"
        if self.mode == "sampling":
            kwargs["do_sample"] = True
        self.thinker.generate(**kwargs)
        if self.mode == "chunking":
            self.thinker.generate(**kwargs)
        return [SimpleNamespace(text="Alpha", language="English")]


def runtime(mode="pass"):
    engine = QwenPromptRuntime.__new__(QwenPromptRuntime)
    engine.seed = lambda _: None
    engine.generation_config = {"do_sample": False, "num_beams": 1}
    generated = []

    def generate(**kwargs):
        generated.append(kwargs)
        length = kwargs["input_ids"].shape[1] + 5
        return SimpleNamespace(sequences=Tensor([[1] * length]))

    engine.thinker = SimpleNamespace(
        generation_config=SimpleNamespace(to_dict=lambda: engine.generation_config),
        generate=generate,
    )
    engine.wrapper = Wrapper(engine.thinker, mode)
    engine.cached_path = None
    engine.normalize_audios = lambda _: [np.zeros(16000, dtype=np.float32)]
    engine.sample_rate = 16000
    engine.torch = SimpleNamespace(inference_mode=nullcontext)
    return engine, generated


def test_actual_context_serialized_prompt_token_check_and_independent_calls():
    engine, calls = runtime()
    request = {"seed": 12, "audio_path": "mock.wav", "context": "Alpha\nBeta"}
    original = engine.wrapper._build_text_prompt
    result = engine.transcribe(request)
    assert result["actual_context"] == request["context"]
    assert result["serialized_prompt"] == original(
        context=request["context"], force_language="English"
    )
    assert result["prompt_tokens_verified"]
    assert result["generated_tokens"] == 5 and not result["generation_limit_reached"]
    assert result["effective_generate_arguments"]["do_sample"] is False
    engine.transcribe({**request, "context": ""})
    assert len(calls) == 2
    assert engine.wrapper._build_text_prompt == original


@pytest.mark.parametrize(
    "mode", ["context", "history", "truncation", "rewrite", "cache", "sampling", "chunking"]
)
def test_runtime_rejects_mixed_variables_and_restores_hooks(mode):
    engine, calls = runtime(mode)
    original_builder, original_generate = engine.wrapper._build_text_prompt, engine.thinker.generate
    with pytest.raises(RuntimeError):
        engine.transcribe({"seed": 1, "audio_path": "mock", "context": "Alpha"})
    assert engine.wrapper._build_text_prompt == original_builder
    assert engine.thinker.generate == original_generate
    assert len(calls) == (1 if mode == "chunking" else 0)


def test_only_audio_token_repetition_is_ignored():
    assert collapse_audio_tokens([1, 2, 2, 3, 3, 2], 2) == [1, 2, 3, 3, 2]
