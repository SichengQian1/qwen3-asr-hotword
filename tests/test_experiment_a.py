from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from qwen_hotword.training.experiment_a import (
    AudioMetadata,
    build_experiment_a_manifest,
    ctc_minimum_input_length,
    estimate_qwen_lengths,
)


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(seconds * sample_rate))


def _write_fixture_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    audio_root = tmp_path / "audio"
    rows = [
        ("a.wav", "Bom dia"),
        ("b.wav", "Casa azul"),
        ("c.wav", "Vida boa"),
        ("d.wav", "Mundo novo"),
        ("e.wav", "Boa casa"),
        ("h.wav", "Letra h"),
        ("connector.wav", "Bem-vindo"),
        ("missing.wav", "Palavra xpto"),
    ]
    for audio_name, _text in rows:
        _write_wav(audio_root / audio_name)

    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "audio\ttext\n" + "".join(f"{audio}\t{text}\n" for audio, text in rows),
        encoding="utf-8",
    )
    dictionary = tmp_path / "pt.dict"
    dictionary.write_text(
        "bom\tb o m\n"
        "dia\td i a\n"
        "casa\tk a z a\n"
        "azul\ta z u l\n"
        "vida\tv i d a\n"
        "boa\tb o a\n"
        "mundo\tm u n d o\n"
        "novo\tn o v o\n"
        "letra\tl e t r a\n"
        "palavra\tp a l a v r a\n",
        encoding="utf-8",
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps(
            {
                "tokens": [
                    "<blank>",
                    "<unk>",
                    "a",
                    "b",
                    "d",
                    "e",
                    "i",
                    "k",
                    "l",
                    "m",
                    "n",
                    "o",
                    "p",
                    "r",
                    "t",
                    "u",
                    "v",
                    "z",
                ]
            }
        ),
        encoding="utf-8",
    )
    return tsv, audio_root, dictionary, vocab


def test_ctc_minimum_length_accounts_for_repeated_labels() -> None:
    assert ctc_minimum_input_length([2, 3, 4]) == 3
    assert ctc_minimum_input_length([2, 2, 3, 3]) == 6


def test_estimate_qwen_lengths_matches_verified_probe_lengths() -> None:
    one_second = AudioMetadata(frames=16_000, sample_rate=16_000, duration_seconds=1.0)
    two_seconds = AudioMetadata(frames=32_000, sample_rate=16_000, duration_seconds=2.0)

    assert estimate_qwen_lengths(one_second) == (100, 13)
    assert estimate_qwen_lengths(two_seconds) == (200, 26)


def test_build_experiment_a_manifest_filters_unclean_labels(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture_files(tmp_path)
    output_dir = tmp_path / "output"

    summary = build_experiment_a_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        num_samples=4,
        seed=7,
        candidate_pool_size=16,
    )

    assert summary.status == "pass"
    assert summary.selected_samples == 4
    assert summary.rows_scanned == 8
    assert summary.lexically_clean_rows == 5
    assert summary.rejection_counts["standalone_h"] == 1
    assert summary.rejection_counts["unresolved_connector"] == 1
    assert summary.rejection_counts["dictionary_missing"] == 1

    records = [
        json.loads(line)
        for line in (output_dir / "experiment_a_train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 4
    assert all(record["label_length"] > 0 for record in records)
    assert all(
        record["estimated_ctc_input_length"]
        >= record["ctc_minimum_input_length"] + record["ctc_safety_margin"]
        for record in records
    )
    assert "phonemes:" in (output_dir / "experiment_a_review.txt").read_text(
        encoding="utf-8"
    )


def test_build_experiment_a_manifest_fails_when_clean_pool_is_too_small(
    tmp_path: Path,
) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture_files(tmp_path)

    summary = build_experiment_a_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        tmp_path / "output",
        num_samples=6,
        candidate_pool_size=8,
    )

    assert summary.status == "fail"
    assert summary.selected_samples == 5


def test_build_experiment_a_manifest_rejects_small_candidate_pool(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture_files(tmp_path)

    with pytest.raises(ValueError, match="candidate_pool_size"):
        build_experiment_a_manifest(
            tsv,
            audio_root,
            dictionary,
            vocab,
            tmp_path / "output",
            num_samples=4,
            candidate_pool_size=3,
        )
