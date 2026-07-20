from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from qwen_hotword.training.full_manifest import build_full_training_manifest


def _write_wav(path: Path, seconds: float = 1.0, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * max(1, int(seconds * sample_rate)))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    audio_root = tmp_path / "audio"
    for audio_name in ("ready.wav", "connector.wav", "h.wav", "missing_word.wav"):
        _write_wav(audio_root / audio_name)

    tsv = tmp_path / "train.tsv"
    tsv.write_text(
        "audio\ttext\n"
        "ready.wav\tBom dia\n"
        "connector.wav\tBem-vindo\n"
        "h.wav\tLetra h\n"
        "missing_word.wav\tPalavra xpto\n"
        "absent.wav\tBom dia\n",
        encoding="utf-8",
    )
    dictionary = tmp_path / "pt.dict"
    dictionary.write_text(
        "bom\tb o m\n"
        "dia\td i a\n"
        "bem-vindo\tb e m v i n d o\n"
        "letra\tl e t r a\n"
        "h\th\n"
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
                    "l",
                    "m",
                    "n",
                    "o",
                    "p",
                    "r",
                    "t",
                    "v",
                ]
            }
        ),
        encoding="utf-8",
    )
    return tsv, audio_root, dictionary, vocab


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_full_manifest_retains_every_source_row_and_resumes(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    output_dir = tmp_path / "output"

    summary = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        shard_size=2,
        workers=2,
    )

    assert summary.status == "pass"
    assert summary.source_records == 5
    assert summary.ready_records == 1
    assert summary.review_records == 4
    assert summary.valid_audio_records == 4
    assert summary.completed_shards == 3
    assert summary.resumed_shards == 0
    assert summary.issue_counts == {
        "dictionary_missing": 1,
        "missing_audio_file": 1,
        "oov_phone": 1,
        "standalone_h": 1,
        "unresolved_connector": 1,
    }

    ready = _read_jsonl(output_dir / "train_ready.jsonl")
    review = _read_jsonl(output_dir / "needs_review.jsonl")
    all_ids = {str(record["id"]) for record in ready + review}
    assert len(ready) == 1
    assert len(review) == 4
    assert len(all_ids) == 5
    assert all(record["label_status"] == "needs_review" for record in review)

    connector = next(record for record in review if record["text"] == "Bem-vindo")
    assert connector["phoneme_token_ids"]
    assert connector["audio_valid"] is True
    assert connector["training_ready"] is False

    resumed = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        shard_size=2,
        workers=2,
    )
    assert resumed.resumed_shards == 3
    assert resumed.source_records == 5


def test_full_manifest_handles_audio_shorter_than_one_feature_frame(
    tmp_path: Path,
) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    _write_wav(audio_root / "ready.wav", seconds=0.001)

    summary = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        tmp_path / "output",
        shard_size=5,
        workers=1,
    )

    assert summary.status == "pass"
    assert summary.issue_counts["ctc_length_infeasible"] == 1


def test_full_manifest_rejects_empty_tsv(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    tsv.write_text("audio\ttext\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        build_full_training_manifest(
            tsv,
            audio_root,
            dictionary,
            vocab,
            tmp_path / "output",
        )
