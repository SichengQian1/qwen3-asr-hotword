from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

JsonRecord = dict[str, Any]


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    dataset: str
    split: str
    language: str
    audio_path: str
    text: str
    duration_sec: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> JsonRecord:
        return asdict(self)


@dataclass(frozen=True)
class Hotword:
    hotword_id: str
    language: str
    surface: str
    normalized: str
    ipa: str | None
    phoneme_tokens: list[str]
    phoneme_source: str
    source_dataset: str
    source_utt_ids: list[str]
    hotword_type: str
    frequency: int

    def to_json(self) -> JsonRecord:
        return asdict(self)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    eval_stage: str
    case_type: str
    utt_id: str
    dataset: str
    split: str
    language: str
    audio_path: str
    reference_text: str
    active_hotword_ids: list[str]
    expected_hotword_ids: list[str]
    distractor_hotword_ids: list[str]

    def to_json(self) -> JsonRecord:
        return asdict(self)


def write_jsonl(path: str | Path, records: list[JsonRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[JsonRecord]:
    input_path = Path(path)
    records: list[JsonRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL record is not an object in {input_path}")
                records.append(value)
    return records

