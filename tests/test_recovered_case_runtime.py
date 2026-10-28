from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from test_prompt_order_runtime import Tensor, Wrapper, runtime

from qwen_hotword.inference.recovered_case_runtime import LANGUAGES, RecoveredCaseRuntime


class MultilingualWrapper(Wrapper):
    def transcribe(self, audio, context, language, return_time_stamps):
        assert language in LANGUAGES.values() and return_time_stamps is False
        assert audio[1] == 16000
        if self.mode == "context":
            context += " unwanted"
        actual_language = "Chinese" if self.mode == "language" else language
        prompt = self._build_text_prompt(context=context, force_language=actual_language)
        ids = self.processor.tokenizer(prompt.replace("@", "@@@"))["input_ids"]
        if self.mode == "truncation":
            ids = ids[2:]
        kwargs = {
            "input_ids": Tensor([ids]),
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": [151645, 151643],
            "return_dict_in_generate": True,
        }
        if self.mode == "cache":
            kwargs["past_key_values"] = "old state"
        if self.mode == "sampling":
            kwargs["do_sample"] = True
        self.thinker.generate(**kwargs)
        if self.mode == "chunking":
            self.thinker.generate(**kwargs)
        return [SimpleNamespace(text="Alpha", language=language)]


def engine(mode="pass"):
    base, calls = runtime(mode)
    base.__class__ = RecoveredCaseRuntime
    base.np = np
    base.wrapper = MultilingualWrapper(base.thinker, mode)
    return base, calls


@pytest.mark.parametrize("language", LANGUAGES.values())
@pytest.mark.parametrize("context", ["", "Alpha\nBeta"])
def test_all_languages_and_contexts_reach_real_model_token_boundary(language, context):
    model, calls = engine()
    request = {"language": language, "context": context, "seed": 1, "audio_path": "mock.wav"}
    result = model.transcribe(request)
    assert result["returned_language"] == language
    assert result["actual_context"] == context
    assert result["serialized_prompt"].endswith(f"language {language}<asr_text>")
    assert result["prompt_tokens_verified"] and not result["generation_limit_reached"]
    model.transcribe({**request, "context": ""})
    assert len(calls) == 2  # no inherited history or shared decoder request


@pytest.mark.parametrize(
    "mode", ["context", "language", "history", "cache", "truncation", "sampling", "chunking"]
)
def test_multilingual_runtime_refuses_prompt_state_and_decode_drift(mode):
    model, _ = engine(mode)
    builder, generate = model.wrapper._build_text_prompt, model.thinker.generate
    with pytest.raises(RuntimeError):
        model.transcribe(
            {"language": "Spanish", "context": "Alpha", "seed": 1, "audio_path": "mock.wav"}
        )
    assert model.wrapper._build_text_prompt == builder
    assert model.thinker.generate == generate


def test_explicit_supported_language_required():
    model, calls = engine()
    with pytest.raises(ValueError, match="explicit"):
        model.transcribe({"language": "Arabic"})
    assert not calls
