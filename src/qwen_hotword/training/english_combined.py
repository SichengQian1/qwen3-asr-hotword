from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_combined import (
    SpanishCorpusInput,
    build_speaker_temporal2x_training,
)

DATASET_VERSION = "english-us-temporal2x-combined-v1"
CORPUS_NAME = "swift_us_english"


def build_english_us_temporal2x_training(
    manifest_dir: str | Path,
    speaker_inventory_tsv: str | Path,
    speaker_audit_summary: str | Path,
    output_dir: str | Path,
    *,
    allowed_speaker_first_components: Iterable[str] = ("US",),
    train_fraction: float = 0.96,
    validation_fraction: float = 0.02,
    test_fraction: float = 0.02,
    time_upsampling_factor: int = 2,
    release_max_effective_ratio: float = 0.90,
    speaker_split_seed: int = 20_260_824,
) -> dict[str, Any]:
    """Build the US-only Swift English pool from an audited speaker inventory."""

    manifest_root = Path(manifest_dir).expanduser()
    inventory_path = Path(speaker_inventory_tsv).expanduser()
    audit_path = Path(speaker_audit_summary).expanduser()
    for path in (manifest_root, inventory_path, audit_path):
        if not path.exists():
            raise FileNotFoundError(f"required input does not exist: {path}")
    audit = _load_object(audit_path)
    _validate_speaker_audit(audit, inventory_path)
    allowed = tuple(allowed_speaker_first_components)
    summary = build_speaker_temporal2x_training(
        [SpanishCorpusInput(CORPUS_NAME, manifest_root, inventory_path)],
        output_dir,
        dataset_version=DATASET_VERSION,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        time_upsampling_factor=time_upsampling_factor,
        release_max_effective_ratio=release_max_effective_ratio,
        speaker_split_seed=speaker_split_seed,
        allowed_speaker_first_components=allowed,
        expected_language="en-US",
    )
    expected_included = _allowed_record_count(audit, allowed)
    observed_partition = (
        int(summary["released_records"])
        + int(summary["review_not_released_records"])
    )
    if observed_partition != expected_included:
        raise RuntimeError(
            "US-only manifest partition differs from the audited speaker inventory"
        )
    return summary


def _validate_speaker_audit(
    audit: Mapping[str, Any],
    inventory_path: Path,
) -> None:
    if audit.get("status") != "pass":
        raise ValueError("speaker audit status is not pass")
    zero_fields = (
        "parse_failure_records",
        "manifest_missing_records",
        "manifest_extra_records",
        "duplicate_speaker_utterance_keys",
    )
    if any(audit.get(field) != 0 for field in zero_fields):
        raise ValueError("speaker audit contains parse, join, or duplicate failures")
    source_records = audit.get("source_records")
    parsed_records = audit.get("parsed_records")
    manifest_records = audit.get("manifest_records")
    if (
        not isinstance(source_records, int)
        or isinstance(source_records, bool)
        or source_records <= 0
        or parsed_records != source_records
        or manifest_records != source_records
    ):
        raise ValueError("speaker audit record counts are inconsistent")
    recorded_inventory = audit.get("speaker_inventory_path")
    if not isinstance(recorded_inventory, str) or (
        Path(recorded_inventory).expanduser().resolve() != inventory_path.resolve()
    ):
        raise ValueError("speaker audit does not describe the requested inventory")
    counts = audit.get("first_component_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("speaker audit has no first-component counts")
    if sum(_required_count(value) for value in counts.values()) != source_records:
        raise ValueError("speaker audit first-component counts are inconsistent")


def _allowed_record_count(
    audit: Mapping[str, Any],
    allowed: Iterable[str],
) -> int:
    normalized = {value.strip().casefold() for value in allowed if value.strip()}
    if not normalized:
        raise ValueError("allowed speaker components must not be empty")
    counts = audit["first_component_counts"]
    assert isinstance(counts, dict)
    result = sum(
        _required_count(value)
        for key, value in counts.items()
        if str(key).casefold() in normalized
    )
    if result <= 0:
        raise ValueError("speaker filter selects no audited records")
    return result


def _required_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("speaker audit contains an invalid count")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file is not an object: {path}")
    return value
