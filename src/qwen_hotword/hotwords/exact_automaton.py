from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from qwen_hotword.hotwords.registry import HotwordEntry


@dataclass(frozen=True)
class ExactHotwordMatch:
    hotword_id: str
    surface: str
    start_token: int
    end_token: int
    phone_count: int
    mean_confidence: float
    minimum_confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "hotword_id": self.hotword_id,
            "surface": self.surface,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "phone_count": self.phone_count,
            "mean_confidence": self.mean_confidence,
            "minimum_confidence": self.minimum_confidence,
        }


class IntegerAhoCorasick:
    """A deterministic Aho-Corasick index over integer CTC token sequences."""

    def __init__(self, entries: Sequence[HotwordEntry]) -> None:
        if not entries:
            raise ValueError("cannot build an exact matcher without hotwords")
        self._entries = tuple(entries)
        self._transitions: list[dict[int, int]] = [{}]
        self._failure: list[int] = [0]
        self._outputs: list[list[int]] = [[]]
        seen_ids: set[str] = set()
        for entry_index, entry in enumerate(self._entries):
            if entry.hotword_id in seen_ids:
                raise ValueError(f"duplicate hotword ID: {entry.hotword_id}")
            if not entry.token_ids:
                raise ValueError(f"hotword {entry.hotword_id} has no token IDs")
            seen_ids.add(entry.hotword_id)
            state = 0
            for token_id in entry.token_ids:
                next_state = self._transitions[state].get(token_id)
                if next_state is None:
                    next_state = len(self._transitions)
                    self._transitions[state][token_id] = next_state
                    self._transitions.append({})
                    self._failure.append(0)
                    self._outputs.append([])
                state = next_state
            self._outputs[state].append(entry_index)
        self._build_failure_links()

    @property
    def node_count(self) -> int:
        return len(self._transitions)

    @property
    def transition_count(self) -> int:
        return sum(len(value) for value in self._transitions)

    @property
    def pattern_count(self) -> int:
        return len(self._entries)

    def find(
        self,
        token_ids: Sequence[int],
        *,
        confidences: Sequence[float] | None = None,
        active_hotword_ids: Iterable[str] | None = None,
        longest_only: bool = True,
    ) -> tuple[ExactHotwordMatch, ...]:
        if confidences is None:
            resolved_confidences = (1.0,) * len(token_ids)
        else:
            if len(confidences) != len(token_ids):
                raise ValueError("confidence and token sequences must have equal length")
            resolved_confidences = tuple(float(value) for value in confidences)
            if any(not 0.0 <= value <= 1.0 for value in resolved_confidences):
                raise ValueError("exact matcher confidences must be in [0, 1]")
        active = set(active_hotword_ids) if active_hotword_ids is not None else None
        state = 0
        matches: list[ExactHotwordMatch] = []
        for end_index, token_id in enumerate(token_ids, start=1):
            while state and token_id not in self._transitions[state]:
                state = self._failure[state]
            state = self._transitions[state].get(token_id, 0)
            for entry_index in self._outputs[state]:
                entry = self._entries[entry_index]
                if active is not None and entry.hotword_id not in active:
                    continue
                start_index = end_index - len(entry.token_ids)
                span_confidences = resolved_confidences[start_index:end_index]
                matches.append(
                    ExactHotwordMatch(
                        hotword_id=entry.hotword_id,
                        surface=entry.surface,
                        start_token=start_index,
                        end_token=end_index,
                        phone_count=len(entry.token_ids),
                        mean_confidence=sum(span_confidences) / len(span_confidences),
                        minimum_confidence=min(span_confidences),
                    )
                )
        if longest_only:
            matches = list(filter_longest_exact_matches(matches))
        return tuple(matches)

    def _build_failure_links(self) -> None:
        pending: deque[int] = deque()
        for state in self._transitions[0].values():
            pending.append(state)
        while pending:
            state = pending.popleft()
            for token_id, next_state in self._transitions[state].items():
                pending.append(next_state)
                failure = self._failure[state]
                while failure and token_id not in self._transitions[failure]:
                    failure = self._failure[failure]
                self._failure[next_state] = self._transitions[failure].get(token_id, 0)
                inherited = self._outputs[self._failure[next_state]]
                if inherited:
                    self._outputs[next_state].extend(inherited)


def rank_unique_exact_matches(
    matches: Sequence[ExactHotwordMatch],
) -> tuple[ExactHotwordMatch, ...]:
    best_by_id: dict[str, ExactHotwordMatch] = {}
    for match in matches:
        previous = best_by_id.get(match.hotword_id)
        if previous is None or _rank_key(match) < _rank_key(previous):
            best_by_id[match.hotword_id] = match
    return tuple(sorted(best_by_id.values(), key=_rank_key))


def filter_longest_exact_matches(
    matches: Sequence[ExactHotwordMatch],
) -> tuple[ExactHotwordMatch, ...]:
    kept: list[ExactHotwordMatch] = []
    for match in matches:
        covered = any(
            other.start_token <= match.start_token
            and other.end_token >= match.end_token
            and (
                other.start_token < match.start_token
                or other.end_token > match.end_token
            )
            for other in matches
        )
        if not covered:
            kept.append(match)
    return tuple(kept)


def _rank_key(match: ExactHotwordMatch) -> tuple[float, int, float, int, str]:
    return (
        -match.mean_confidence,
        -match.phone_count,
        -match.minimum_confidence,
        match.start_token,
        match.hotword_id,
    )
