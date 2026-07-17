from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import cast

import pytest

from qwen_hotword.training.experiment_b import (
    SPLIT_NAMES,
    assign_split,
    build_experiment_b_manifests,
)


def _write_wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * int(16_000 * seconds))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    audio_root = tmp_path / "audio"
    rows: list[tuple[str, str]] = []
    for index in range(120):
        relative = f"sample_{index:03d}.wav"
        _write_wav(audio_root / relative)
        rows.append((relative, "bom dia"))

    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "audio\ttext\n" + "".join(f"{audio}\t{text}\n" for audio, text in rows),
        encoding="utf-8",
    )
    dictionary = tmp_path / "pt.dict"
    dictionary.write_text("bom\tb o m\ndia\td i a\n", encoding="utf-8")
    word_counts = tmp_path / "word_counts.tsv"
    word_counts.write_text("word\tcount\nbom\t1000\ndia\t1000\n", encoding="utf-8")
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", "a", "b", "d", "i", "m", "o"]}),
        encoding="utf-8",
    )
    return tsv, audio_root, dictionary, word_counts, vocab


def test_assign_split_is_stable_and_covers_all_splits() -> None:
    fractions = {"train": 0.8, "validation": 0.1, "test": 0.1}
    first = [assign_split(f"audio-{index}", seed=7, fractions=fractions) for index in range(100)]
    second = [
        assign_split(f"audio-{index}", seed=7, fractions=fractions) for index in range(100)
    ]

    assert first == second
    assert set(first) == set(SPLIT_NAMES)


def test_build_experiment_b_manifests_reaches_durations_without_overlap(
    tmp_path: Path,
) -> None:
    tsv, audio_root, dictionary, word_counts, vocab = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"
    excluded_manifest = tmp_path / "experiment_a.jsonl"
    excluded_audio = audio_root / "sample_000.wav"
    excluded_manifest.write_text(
        json.dumps({"audio_path": str(excluded_audio)}) + "\n",
        encoding="utf-8",
    )

    summary = build_experiment_b_manifests(
        tsv,
        audio_root,
        dictionary,
        word_counts,
        vocab,
        output_dir,
        train_hours=5 / 3600,
        validation_hours=5 / 3600,
        test_hours=5 / 3600,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        candidate_pool_size=100,
        seed=11,
        exclusion_manifest_paths=(excluded_manifest,),
    )

    assert summary.status == "pass"
    assert summary.cross_split_audio_overlaps == 0
    assert summary.speaker_disjoint is False
    assert summary.excluded_audio_paths == 1
    assert summary.rejection_counts["excluded_audio"] == 1
    audio_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        split_summary = summary.split_summaries[split]
        assert split_summary["status"] == "pass"
        assert cast(float, split_summary["selected_duration_seconds"]) >= 5.0
        rows = [
            json.loads(line)
            for line in (output_dir / f"experiment_b_{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert rows
        assert all(row["experiment"] == "B" and row["split"] == split for row in rows)
        audio_sets[split] = {row["audio_path"] for row in rows}
        assert str(excluded_audio) not in audio_sets[split]

    assert not audio_sets["train"] & audio_sets["validation"]
    assert not audio_sets["train"] & audio_sets["test"]
    assert not audio_sets["validation"] & audio_sets["test"]


def test_build_experiment_b_rejects_invalid_split_fractions(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, word_counts, vocab = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="sum to one"):
        build_experiment_b_manifests(
            tsv,
            audio_root,
            dictionary,
            word_counts,
            vocab,
            tmp_path / "output",
            train_fraction=0.7,
            validation_fraction=0.1,
            test_fraction=0.1,
        )
