"""Runtime hotword registry, scoring, and validation tooling."""

from qwen_hotword.hotwords.registry import HotwordEntry, load_hotword_table
from qwen_hotword.hotwords.scoring import (
    HotwordMatch,
    HotwordScoringConfig,
    HotwordScoringResult,
    score_hotwords,
)

__all__ = [
    "HotwordEntry",
    "HotwordMatch",
    "HotwordScoringConfig",
    "HotwordScoringResult",
    "load_hotword_table",
    "score_hotwords",
]
