"""Qwen3-ASR hotword prompt construction and validation smoke tests."""

from qwen_hotword.inference.hotword_prompt import (
    DEFAULT_PT_BR_PROMPT_TEMPLATE,
    build_hotword_prompt,
    strict_phrase_match,
)

__all__ = [
    "DEFAULT_PT_BR_PROMPT_TEMPLATE",
    "build_hotword_prompt",
    "strict_phrase_match",
]
