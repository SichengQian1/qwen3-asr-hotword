from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.phonemes.manifest_dictionary import export_manifest_mfa_dictionary


def _row(sample_id: str, split: str, pronunciation: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "split": split,
        "language": "es-419",
        "word_pronunciations": [
            {
                "word": "hola",
                "mfa_pronunciation": pronunciation,
                "resolution": "exact",
            },
            {
                "word": "mundo",
                "mfa_pronunciation": "m u n d o",
                "resolution": "exact",
            },
        ],
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_export_manifest_dictionary_selects_majority_and_records_identity(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write(train, [_row("one", "train", "o l a"), _row("two", "train", "o l a")])
    _write(validation, [_row("three", "validation", "o ɫa")])

    output = tmp_path / "dictionary"
    summary = export_manifest_mfa_dictionary([train, validation], output, language="es")

    assert summary["status"] == "pass"
    assert summary["test_set_used"] is False
    assert summary["split_counts"] == {"train": 2, "validation": 1}
    assert summary["unique_words"] == 2
    assert summary["ambiguous_words"] == 1
    assert (output / "manifest_mfa_dictionary.dict").read_text(encoding="utf-8") == (
        "hola\to l a\nmundo\tm u n d o\n"
    )
    assert (output / "sha256.txt").is_file()


def test_export_manifest_dictionary_rejects_test_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "test.jsonl"
    _write(manifest, [_row("sealed", "test", "o l a")])

    with pytest.raises(ValueError, match="only train/validation"):
        export_manifest_mfa_dictionary([manifest], tmp_path / "output", language="es")
