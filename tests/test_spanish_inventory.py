from __future__ import annotations

import csv
import hashlib
import wave
from pathlib import Path

import pytest

from qwen_hotword.training.spanish_inventory import (
    audit_spanish_candidate_inventory,
    classify_spanish_accent,
)


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def _write_source_tsv(path: Path, audio_paths: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio", "text"), delimiter="\t")
        writer.writeheader()
        for audio_path in audio_paths:
            writer.writerow({"audio": str(audio_path), "text": f"Texto {audio_path.stem}"})


def _write_cv_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ("client_id", "path", "sentence", "sentence_id", "locale", "accents")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _metadata_row(audio_path: Path, accent: str) -> dict[str, str]:
    return {
        "client_id": f"speaker-{audio_path.stem}",
        "path": audio_path.name,
        "sentence": f"Texto {audio_path.stem}",
        "sentence_id": f"sentence-{audio_path.stem}",
        "locale": "es",
        "accents": accent,
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_classify_spanish_accent_is_conservative() -> None:
    assert (
        classify_spanish_accent("Rioplatense: Argentina, Uruguay")
        == "argentinian_rioplatense_metadata"
    )
    assert classify_spanish_accent("México|Caribe") == "latin_american_metadata"
    assert classify_spanish_accent("España: Centro-Sur peninsular") == "peninsular_metadata"
    assert (
        classify_spanish_accent("España|Rioplatense")
        == "mixed_latin_american_peninsular"
    )
    assert classify_spanish_accent("") == "unknown"
    assert classify_spanish_accent("acento personal") == "other_unclassified_metadata"


def test_candidate_inventory_joins_cv_metadata_and_blocks_core_overlap(
    tmp_path: Path,
) -> None:
    mls_audio = [tmp_path / "mls" / "100_200_000001.wav"]
    _write_wav(mls_audio[0], 2.0)
    mls_tsv = tmp_path / "mls.tsv"
    _write_source_tsv(mls_tsv, mls_audio)

    cv_audio = [tmp_path / "cv" / f"clip-{index}.wav" for index in range(8)]
    for audio_path in cv_audio:
        _write_wav(audio_path, 1.0)
    cv_tsv = tmp_path / "cv.tsv"
    _write_source_tsv(cv_tsv, cv_audio)

    accents = [
        "Rioplatense: Argentina, Uruguay",
        "México",
        "España: Centro-Sur peninsular",
        "",
        "Andino-Pacífico: Colombia, Perú",
        "Rioplatense: Argentina, Uruguay",
        "",
        "",
    ]
    metadata_rows = [
        _metadata_row(audio_path, accent)
        for audio_path, accent in zip(cv_audio, accents, strict=True)
    ]
    metadata_rows[6]["locale"] = "pt"
    metadata_rows[7]["sentence"] = "Texto diferente"
    cv_root = tmp_path / "cv-root"
    cv_root.mkdir()
    _write_cv_metadata(cv_root / "validated.tsv", metadata_rows)
    _write_cv_metadata(cv_root / "train.tsv", metadata_rows[:3] + metadata_rows[5:])
    _write_cv_metadata(cv_root / "dev.tsv", [])
    _write_cv_metadata(cv_root / "test.tsv", metadata_rows[4:5])

    core_tsv = tmp_path / "rioplatense.tsv"
    with core_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audio", "source_split"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"audio": f"/core/{cv_audio[5].name}", "source_split": "test"})

    output_dir = tmp_path / "inventory"
    summary = audit_spanish_candidate_inventory(
        mls_tsv,
        cv_tsv,
        cv_root,
        core_tsv,
        output_dir,
        workers=2,
        progress_every=0,
    )

    assert summary["status"] == "warn"
    assert summary["mls"]["records"] == 1
    assert summary["mls"]["speaker_count_from_source_id"] == 1
    assert summary["common_voice"]["records"] == 8
    assert summary["common_voice"]["metadata_missing_records"] == 0
    assert summary["common_voice"]["metadata_text_mismatch_records"] == 1
    assert summary["common_voice"]["unexpected_locale_records"] == 1
    assert summary["common_voice"]["official_split_counts"] == {
        "test": 1,
        "train": 6,
        "unassigned_validated": 1,
    }
    assert summary["common_voice"]["core_overlap_counts"] == {"test": 1}
    assert summary["common_voice"]["training_pool_counts"] == {
        "candidate_unknown": 1,
        "exclude_core_overlap": 1,
        "exclude_metadata_text_mismatch": 1,
        "exclude_official_holdout": 1,
        "exclude_unexpected_locale": 1,
        "fallback_peninsular": 1,
        "priority_argentinian_rioplatense": 1,
        "priority_latin_american": 1,
    }
    assert summary["sealed_core_policy"][
        "rioplatense_overlap_is_never_eligible_for_auxiliary_training"
    ]

    joined = _read_tsv(output_dir / "common_voice_inventory.tsv")
    overlap = next(row for row in joined if row["source_id"] == "clip-5")
    assert overlap["core_overlap_split"] == "test"
    assert overlap["training_pool"] == "exclude_core_overlap"
    assert overlap["metadata_text_match"] == "true"

    for line in (output_dir / "sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        payload = (output_dir / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected


def test_candidate_inventory_rejects_duplicate_source_audio(tmp_path: Path) -> None:
    audio = tmp_path / "duplicate.wav"
    _write_wav(audio)
    source = tmp_path / "source.tsv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio", "text"), delimiter="\t")
        writer.writeheader()
        writer.writerow({"audio": str(audio), "text": "Uno"})
        writer.writerow({"audio": str(audio), "text": "Dos"})

    with pytest.raises(ValueError, match="duplicate audio"):
        cv_root = tmp_path / "cv-root"
        cv_root.mkdir()
        # Fail while reading the first source, before metadata content is inspected.
        audit_spanish_candidate_inventory(
            source,
            source,
            cv_root,
            source,
            tmp_path / "output",
        )
