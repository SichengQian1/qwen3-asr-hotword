from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from qwen_hotword.hotwords.exact_automaton import (
    IntegerAhoCorasick,
    rank_unique_exact_matches,
)
from qwen_hotword.hotwords.registry import HotwordEntry


@dataclass(frozen=True)
class AnchorIndexConfig:
    ngram_sizes: tuple[int, ...] = (2, 3, 4)
    anchors_per_entry: int = 24
    offset_tolerance: int = 1

    def validate(self) -> None:
        if not self.ngram_sizes or any(size <= 0 for size in self.ngram_sizes):
            raise ValueError("anchor n-gram sizes must be positive")
        if tuple(sorted(set(self.ngram_sizes))) != self.ngram_sizes:
            raise ValueError("anchor n-gram sizes must be unique and strictly increasing")
        if self.anchors_per_entry <= 0:
            raise ValueError("anchors_per_entry must be positive")
        if self.offset_tolerance < 0:
            raise ValueError("offset_tolerance must not be negative")


@dataclass(frozen=True)
class AnchorCandidate:
    hotword_id: str
    surface: str
    exact_match: bool
    alignment_score: float
    matched_weight: float
    total_anchor_weight: float
    matched_anchors: int
    selected_anchors: int
    longest_anchor: int
    best_offset: int | None
    phone_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorQueryResult:
    candidates: tuple[AnchorCandidate, ...]
    exact_hotword_ids: tuple[str, ...]
    anchored_hotword_ids: tuple[str, ...]
    total_candidate_count: int
    postings_visited: int

    @property
    def no_anchor(self) -> bool:
        return not self.anchored_hotword_ids


@dataclass(frozen=True)
class _AnchorSpec:
    ngram: tuple[int, ...]
    position: int
    weight: float


@dataclass(frozen=True)
class _Posting:
    entry_index: int
    entry_position: int
    ngram_length: int
    weight: float


class PhonemeAnchorIndex:
    """Rare positional phoneme n-gram index for bounded hotword shortlists."""

    def __init__(
        self,
        entries: Sequence[HotwordEntry],
        *,
        config: AnchorIndexConfig | None = None,
    ) -> None:
        if not entries:
            raise ValueError("cannot build an anchor index without hotwords")
        self.config = config or AnchorIndexConfig()
        self.config.validate()
        self._entries = tuple(entries)
        self._entry_index_by_id: dict[str, int] = {}
        for entry_index, entry in enumerate(self._entries):
            if entry.hotword_id in self._entry_index_by_id:
                raise ValueError(f"duplicate hotword ID: {entry.hotword_id}")
            if not entry.token_ids:
                raise ValueError(f"hotword {entry.hotword_id} has no token IDs")
            self._entry_index_by_id[entry.hotword_id] = entry_index
        self._all_hotword_ids = frozenset(self._entry_index_by_id)

        document_frequency: Counter[tuple[int, ...]] = Counter()
        entry_occurrences: list[tuple[tuple[tuple[int, ...], int], ...]] = []
        for entry in self._entries:
            occurrences = tuple(self._entry_ngrams(entry.token_ids))
            entry_occurrences.append(occurrences)
            document_frequency.update({ngram for ngram, _ in occurrences})

        postings: defaultdict[tuple[int, ...], list[_Posting]] = defaultdict(list)
        entry_anchors: list[tuple[_AnchorSpec, ...]] = []
        total_anchor_weights: list[float] = []
        entry_count = len(self._entries)
        for entry_index, occurrences in enumerate(entry_occurrences):
            ranked = sorted(
                occurrences,
                key=lambda item: (
                    document_frequency[item[0]],
                    -len(item[0]),
                    item[0],
                    item[1],
                ),
            )
            selected = ranked[: self.config.anchors_per_entry]
            anchors = tuple(
                _AnchorSpec(
                    ngram=ngram,
                    position=position,
                    weight=self._anchor_weight(
                        entry_count=entry_count,
                        document_frequency=document_frequency[ngram],
                        ngram_length=len(ngram),
                    ),
                )
                for ngram, position in selected
            )
            entry_anchors.append(anchors)
            total_anchor_weights.append(sum(anchor.weight for anchor in anchors))
            for anchor in anchors:
                postings[anchor.ngram].append(
                    _Posting(
                        entry_index=entry_index,
                        entry_position=anchor.position,
                        ngram_length=len(anchor.ngram),
                        weight=anchor.weight,
                    )
                )

        self._entry_anchors = tuple(entry_anchors)
        self._total_anchor_weights = tuple(total_anchor_weights)
        self._postings = {
            ngram: tuple(sorted(values, key=lambda item: (item.entry_index, item.entry_position)))
            for ngram, values in postings.items()
        }
        self._exact_matcher = IntegerAhoCorasick(self._entries)

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def unique_anchor_ngrams(self) -> int:
        return len(self._postings)

    @property
    def posting_count(self) -> int:
        return sum(len(postings) for postings in self._postings.values())

    @property
    def selected_anchor_count(self) -> int:
        return sum(len(anchors) for anchors in self._entry_anchors)

    @property
    def entries_without_anchors(self) -> int:
        return sum(not anchors for anchors in self._entry_anchors)

    def query(
        self,
        token_ids: Sequence[int],
        *,
        confidences: Sequence[float] | None = None,
        active_hotword_ids: Iterable[str] | None = None,
        maximum_candidates: int = 256,
    ) -> AnchorQueryResult:
        if maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")
        if confidences is not None and len(confidences) != len(token_ids):
            raise ValueError("confidence and token sequences must have equal length")
        active_indexes, active_ids = self._active_indexes(active_hotword_ids)

        exact_matches = rank_unique_exact_matches(
            self._exact_matcher.find(
                token_ids,
                confidences=confidences,
                active_hotword_ids=active_ids,
                longest_only=False,
            )
        )
        exact_ids = tuple(match.hotword_id for match in exact_matches)

        by_entry_offset: defaultdict[int, dict[int, list[float | int]]] = defaultdict(dict)
        postings_visited = 0
        for query_position, ngram in self._query_ngrams(token_ids):
            for posting in self._postings.get(ngram, ()):
                postings_visited += 1
                if posting.entry_index not in active_indexes:
                    continue
                offset = query_position - posting.entry_position
                state = by_entry_offset[posting.entry_index].setdefault(offset, [0.0, 0, 0])
                state[0] = float(state[0]) + posting.weight
                state[1] = int(state[1]) + 1
                state[2] = max(int(state[2]), posting.ngram_length)

        anchored: list[AnchorCandidate] = []
        for entry_index, offset_values in by_entry_offset.items():
            best: tuple[float, int, int, int] | None = None
            for center in sorted(offset_values):
                window = [
                    offset_values[offset]
                    for offset in range(
                        center - self.config.offset_tolerance,
                        center + self.config.offset_tolerance + 1,
                    )
                    if offset in offset_values
                ]
                alignment = (
                    sum(float(value[0]) for value in window),
                    sum(int(value[1]) for value in window),
                    max(int(value[2]) for value in window),
                    center,
                )
                if best is None or self._alignment_key(alignment) < self._alignment_key(best):
                    best = alignment
            if best is None:
                continue
            matched_weight, matched_anchors, longest_anchor, best_offset = best
            entry = self._entries[entry_index]
            total_weight = self._total_anchor_weights[entry_index]
            anchored.append(
                AnchorCandidate(
                    hotword_id=entry.hotword_id,
                    surface=entry.surface,
                    exact_match=entry.hotword_id in exact_ids,
                    alignment_score=(
                        min(1.0, matched_weight / total_weight) if total_weight else 0.0
                    ),
                    matched_weight=matched_weight,
                    total_anchor_weight=total_weight,
                    matched_anchors=matched_anchors,
                    selected_anchors=len(self._entry_anchors[entry_index]),
                    longest_anchor=longest_anchor,
                    best_offset=best_offset,
                    phone_count=len(entry.token_ids),
                )
            )

        anchored_ids = tuple(candidate.hotword_id for candidate in anchored)
        anchored_by_id = {candidate.hotword_id: candidate for candidate in anchored}
        ordered: list[AnchorCandidate] = []
        for hotword_id in exact_ids:
            exact_candidate = anchored_by_id.pop(hotword_id, None)
            if exact_candidate is None:
                entry_index = self._entry_index_by_id[hotword_id]
                entry = self._entries[entry_index]
                exact_candidate = AnchorCandidate(
                    hotword_id=hotword_id,
                    surface=entry.surface,
                    exact_match=True,
                    alignment_score=1.0,
                    matched_weight=0.0,
                    total_anchor_weight=self._total_anchor_weights[entry_index],
                    matched_anchors=0,
                    selected_anchors=len(self._entry_anchors[entry_index]),
                    longest_anchor=0,
                    best_offset=None,
                    phone_count=len(entry.token_ids),
                )
            ordered.append(exact_candidate)
        remaining_slots = max(0, maximum_candidates - len(ordered))
        ordered.extend(
            heapq.nsmallest(
                remaining_slots,
                anchored_by_id.values(),
                key=self._candidate_key,
            )
        )
        return AnchorQueryResult(
            candidates=tuple(ordered[:maximum_candidates]),
            exact_hotword_ids=exact_ids,
            anchored_hotword_ids=anchored_ids,
            total_candidate_count=len(anchored_by_id) + len(exact_ids),
            postings_visited=postings_visited,
        )

    def _active_indexes(
        self, active_hotword_ids: Iterable[str] | None
    ) -> tuple[set[int], frozenset[str]]:
        if active_hotword_ids is None:
            return set(range(len(self._entries))), self._all_hotword_ids
        active = tuple(active_hotword_ids)
        active_ids = frozenset(active)
        if len(active_ids) != len(active):
            raise ValueError("active hotword IDs contain duplicates")
        unknown = sorted(active_ids - self._all_hotword_ids)
        if unknown:
            raise ValueError(f"active hotword IDs are absent from the index: {unknown[:5]}")
        return {self._entry_index_by_id[hotword_id] for hotword_id in active}, active_ids

    def _entry_ngrams(
        self, token_ids: Sequence[int]
    ) -> Iterable[tuple[tuple[int, ...], int]]:
        for size in self.config.ngram_sizes:
            for position in range(len(token_ids) - size + 1):
                yield tuple(token_ids[position : position + size]), position

    def _query_ngrams(
        self, token_ids: Sequence[int]
    ) -> Iterable[tuple[int, tuple[int, ...]]]:
        for size in self.config.ngram_sizes:
            for position in range(len(token_ids) - size + 1):
                yield position, tuple(token_ids[position : position + size])

    @staticmethod
    def _anchor_weight(
        *, entry_count: int, document_frequency: int, ngram_length: int
    ) -> float:
        inverse_document_frequency = math.log((entry_count + 1) / (document_frequency + 1)) + 1.0
        return inverse_document_frequency * ngram_length

    @staticmethod
    def _alignment_key(value: tuple[float, int, int, int]) -> tuple[float, int, int, int, int]:
        weight, matched, longest, offset = value
        return (-weight, -matched, -longest, abs(offset), offset)

    @staticmethod
    def _candidate_key(
        candidate: AnchorCandidate,
    ) -> tuple[int, float, float, int, int, int, str]:
        return (
            -int(candidate.exact_match),
            -candidate.alignment_score,
            -candidate.matched_weight,
            -candidate.matched_anchors,
            -candidate.longest_anchor,
            -candidate.phone_count,
            candidate.hotword_id,
        )
