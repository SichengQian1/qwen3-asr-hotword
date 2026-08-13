from pathlib import Path

import pytest

from qwen_hotword.training.data_audit import audit_training_tsv


def test_audit_resolves_unicode_relative_audio_paths(tmp_path: Path) -> None:
    audio_root = tmp_path / "noah_pt"
    first_audio = audio_root / "APY数据" / "data" / "category" / "first.wav"
    second_audio = audio_root / "APY数据" / "data" / "category" / "second.wav"
    first_audio.parent.mkdir(parents=True)
    first_audio.write_bytes(b"wav")
    second_audio.write_bytes(b"wav")
    tsv_path = tmp_path / "500小时巴西葡萄牙语.tsv"
    tsv_path.write_text(
        "audio\ttext\n"
        "APY数据/data/category/first.wav\tIsso aqui.\n"
        "APY数据/data/category/second.wav\tBom dia.\n",
        encoding="utf-8",
    )

    audit = audit_training_tsv(tsv_path, audio_root)

    assert audit.status == "pass"
    assert audit.rows_scanned == 2
    assert audit.resolved_audio_files == 2
    assert audit.samples[0].audio_path == str(first_audio)


def test_audit_reports_missing_relative_audio(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    tsv_path = tmp_path / "train.tsv"
    tsv_path.write_text("audio\ttext\nmissing.wav\tOlá.\n", encoding="utf-8")

    audit = audit_training_tsv(tsv_path, audio_root)

    assert audit.status == "fail"
    assert audit.missing_audio_files == 1
    assert audit.errors == ("1 relative audio paths did not resolve",)


def test_audit_rejects_unexpected_columns(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    tsv_path = tmp_path / "train.tsv"
    tsv_path.write_text("path\tsentence\na.wav\tOlá.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        audit_training_tsv(tsv_path, audio_root)


def test_audit_requires_explicit_opt_in_for_absolute_audio(tmp_path: Path) -> None:
    audio_root = tmp_path / "unused-root"
    audio_root.mkdir()
    audio_path = tmp_path / "external" / "sample.wav"
    audio_path.parent.mkdir()
    audio_path.write_bytes(b"wav")
    tsv_path = tmp_path / "external.tsv"
    tsv_path.write_text(f"audio\ttext\n{audio_path}\tHola.\n", encoding="utf-8")

    rejected = audit_training_tsv(tsv_path, audio_root)
    accepted = audit_training_tsv(
        tsv_path,
        audio_root,
        allow_absolute_audio=True,
    )

    assert rejected.status == "fail"
    assert rejected.absolute_audio_values == 1
    assert accepted.status == "pass"
    assert accepted.absolute_audio_values == 1
    assert accepted.resolved_audio_files == 1
