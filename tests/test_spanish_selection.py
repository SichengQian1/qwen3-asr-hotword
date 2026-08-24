from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from qwen_hotword.training.spanish_selection import (
    assign_spanish_speaker_split,
    select_spanish_auxiliary_pool,
)


def _write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _inventory_row(
    index: int,
    *,
    speaker: str,
    accent: str,
    seconds: float = 1.0,
    core_overlap: str = "",
) -> dict[str, object]:
    return {
        "audio": f"/audio/{index}.mp3",
        "source_id": str(index),
        "speaker_id": speaker,
        "official_split": "train",
        "locale": "es",
        "accent": accent,
        "accent_tier": "ignored_and_recomputed",
        "core_overlap_split": core_overlap,
        "duration_seconds": seconds,
        "audio_status": "ok",
        "metadata_text_match": "true",
    }


def test_speaker_split_is_deterministic_and_speaker_based() -> None:
    first = assign_spanish_speaker_split(
        "speaker-a",
        seed=20_260_824,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    assert first == assign_spanish_speaker_split(
        "speaker-a",
        seed=20_260_824,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )
    assert first in {"train", "validation", "test"}


def test_selection_uses_explicit_latam_and_blocks_core_holdout_speaker(
    tmp_path: Path,
) -> None:
    inventory_rows = [
        _inventory_row(1, speaker="rio", accent="Rioplatense: Argentina"),
        _inventory_row(2, speaker="mex", accent="México"),
        _inventory_row(3, speaker="central", accent="América central"),
        _inventory_row(4, speaker="spain", accent="España: Centro-Sur peninsular"),
        _inventory_row(5, speaker="unknown", accent=""),
        _inventory_row(6, speaker="core-test", accent="México"),
        _inventory_row(
            7,
            speaker="core-train",
            accent="México",
            core_overlap="train",
        ),
        _inventory_row(8, speaker="core-train", accent="México"),
    ]
    source = tmp_path / "source.tsv"
    _write_tsv(
        source,
        ("audio", "text"),
        [
            {"audio": row["audio"], "text": f"Texto {index}"}
            for index, row in enumerate(inventory_rows, start=1)
        ],
    )
    inventory = tmp_path / "inventory.tsv"
    _write_tsv(inventory, tuple(inventory_rows[0]), inventory_rows)
    core = tmp_path / "core.tsv"
    _write_tsv(
        core,
        ("audio", "speaker_id", "source_split"),
        [
            {"audio": "/core/test.mp3", "speaker_id": "core-test", "source_split": "test"},
            {
                "audio": "/core/train.mp3",
                "speaker_id": "core-train",
                "source_split": "train",
            },
        ],
    )

    output = tmp_path / "selected"
    summary = select_spanish_auxiliary_pool(
        source,
        inventory,
        core,
        output,
        target_hours=3.0 / 3600.0,
        train_fraction=1.0,
        validation_fraction=0.0,
        test_fraction=0.0,
        maximum_latin_american_speaker_hours=1.0,
    )

    assert summary["status"] == "pass"
    assert summary["selected_records"] == 3
    assert summary["selected_tier_counts"] == {
        "argentinian_rioplatense_metadata": 1,
        "latin_american_metadata": 2,
    }
    assert summary["eligible_tier_counts"] == {
        "argentinian_rioplatense_metadata": 1,
        "latin_american_metadata": 3,
    }
    assert summary["decision_counts"]["exclude_accent_tier_peninsular_metadata"] == 1
    assert summary["decision_counts"]["exclude_accent_tier_unknown"] == 1
    assert summary["decision_counts"]["exclude_core_holdout_speaker_overlap"] == 1
    assert summary["decision_counts"]["exclude_core_audio_overlap"] == 1
    selected = _read_tsv(output / "source.tsv")
    assert {row["speaker_id"] for row in selected} == {"rio", "mex", "central"}
    assert {row["source_split"] for row in selected} == {"train"}

    for line in (output / "sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected


def test_selection_rejects_core_speaker_cross_split(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    _write_tsv(source, ("audio", "text"), [{"audio": "a.mp3", "text": "Texto"}])
    inventory = tmp_path / "inventory.tsv"
    row = _inventory_row(1, speaker="speaker", accent="México")
    row["audio"] = "a.mp3"
    _write_tsv(inventory, tuple(row), [row])
    core = tmp_path / "core.tsv"
    _write_tsv(
        core,
        ("audio", "speaker_id", "source_split"),
        [
            {"audio": "one.mp3", "speaker_id": "speaker", "source_split": "train"},
            {"audio": "two.mp3", "speaker_id": "speaker", "source_split": "test"},
        ],
    )

    with pytest.raises(ValueError, match="multiple splits"):
        select_spanish_auxiliary_pool(
            source,
            inventory,
            core,
            tmp_path / "output",
            target_hours=1.0,
        )
