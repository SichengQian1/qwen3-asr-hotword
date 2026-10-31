"""Reuse the project's Anchor/re-rank code WITHOUT loading another Qwen model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def checked_path(config: dict[str, Any], name: str) -> Path:
    asset = config[name]
    path = Path(asset["path"]).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != asset["sha256"]:
        raise ValueError(f"{name} SHA256 differs: {path}")
    return path


class ProjectRetriever:
    def __init__(self, config: dict[str, Any]) -> None:
        from qwen_hotword.hotwords.anchor_index import AnchorIndexConfig, PhonemeAnchorIndex
        from qwen_hotword.hotwords.registry import HotwordEntry
        from qwen_hotword.hotwords.scoring import HotwordScoringConfig
        from qwen_hotword.phonemes.coverage import load_phoneme_vocab

        vocab = load_phoneme_vocab(checked_path(config, "vocab"))
        entries = []
        with checked_path(config, "index").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("words", "phoneme_tokens", "token_ids"):
                    row[key] = tuple(row[key])
                entry = HotwordEntry(**row)
                if not entry.token_ids or any(i < 2 or i >= 100 for i in entry.token_ids):
                    raise ValueError("index contains empty, unknown or invalid phone IDs")
                if tuple(vocab.tokens[i] for i in entry.token_ids) != entry.phoneme_tokens:
                    raise ValueError("index uses a different phone mapping")
                entries.append(entry)
        self.entries = {entry.hotword_id: entry for entry in entries}
        if not entries or len(self.entries) != len(entries):
            raise ValueError("index must be nonempty with unique pronunciation IDs")
        self.anchor = PhonemeAnchorIndex(
            tuple(entries),
            config=AnchorIndexConfig(
                ngram_sizes=(2, 3, 4),
                anchors_per_entry=24,
                offset_tolerance=1,
            ),
        )
        self.scoring = HotwordScoringConfig(**config["scoring"])
        self.scoring.validate()
        if self.scoring.top_k > 7:
            raise ValueError("this sample limits the final prompt to seven surfaces")

    def search(self, prepared: dict[str, Any]) -> list[dict[str, Any]]:
        from qwen_hotword.hotwords.scoring import (
            DecodedPhoneme,
            profile_anchor_guided_decoded_hotwords,
            select_operating_matches,
        )

        decoded = tuple(DecodedPhoneme(**phone) for phone in prepared["decoded"])
        shortlist = self.anchor.query(
            tuple(phone.token_id for phone in decoded),
            confidences=tuple(phone.confidence for phone in decoded),
            maximum_candidates=64,
        )
        scored = profile_anchor_guided_decoded_hotwords(
            decoded,
            effective_time_steps=prepared["ctc_steps"],
            hotwords=tuple(self.entries[item.hotword_id] for item in shortlist.candidates),
            start_hints={item.hotword_id: item.best_offset for item in shortlist.candidates},
            maximum_start_delta=2,
            config=self.scoring,
        ).result
        # A word may have multiple language pronunciations. Deduplicate surfaces
        # BEFORE gates/top-k, so multiple pronunciations do not take extra slots.
        seen: set[str] = set()
        merged = []
        for match in scored.ranked_matches:
            normalized = self.entries[match.hotword_id].normalized
            if normalized not in seen:
                seen.add(normalized)
                merged.append(match)
        selected, _reason = select_operating_matches(tuple(merged), config=self.scoring)
        return [match.to_dict() for match in selected]
