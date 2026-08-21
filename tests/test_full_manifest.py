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
    assert ready[0]["dataset"] == "noah_pt_full_500h"
    assert ready[0]["id"] == "noah_pt_row_2"
    assert ready[0]["split"] == "unsplit"
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


def test_full_manifest_supports_independent_train_corpus_identity(
    tmp_path: Path,
) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    output_dir = tmp_path / "finance-output"

    summary = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        dataset="noah_pt_finance_200h",
        id_prefix="noah_pt_finance_200h_row",
        split="train",
        shard_size=5,
        workers=1,
    )

    assert summary.dataset == "noah_pt_finance_200h"
    assert summary.id_prefix == "noah_pt_finance_200h_row"
    assert summary.split == "train"
    records = _read_jsonl(output_dir / "train_ready.jsonl")
    records.extend(_read_jsonl(output_dir / "needs_review.jsonl"))
    assert {record["dataset"] for record in records} == {"noah_pt_finance_200h"}
    assert {record["split"] for record in records} == {"train"}
    assert {record["id"] for record in records} == {
        f"noah_pt_finance_200h_row_{row_number}" for row_number in range(2, 7)
    }

    with pytest.raises(ValueError, match="different build configuration"):
        build_full_training_manifest(
            tsv,
            audio_root,
            dictionary,
            vocab,
            output_dir,
            dataset="another_dataset",
            id_prefix="noah_pt_finance_200h_row",
            split="train",
            shard_size=5,
            workers=1,
        )


def test_full_manifest_preserves_per_row_source_splits(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    tsv.write_text(
        "audio\ttext\tsource_split\n"
        "ready.wav\tBom dia\ttrain\n"
        "connector.wav\tBom dia\tvalidation\n"
        "h.wav\tBom dia\ttest\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "split-column-output"
    summary = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        split_column="source_split",
        shard_size=2,
        workers=1,
    )

    assert summary.split == "mixed"
    assert summary.split_column == "source_split"
    assert summary.split_counts == {"test": 1, "train": 1, "validation": 1}
    records = _read_jsonl(output_dir / "train_ready.jsonl")
    assert {str(record["audio_relative"]): record["split"] for record in records} == {
        "connector.wav": "validation",
        "h.wav": "test",
        "ready.wav": "train",
    }


def test_full_manifest_rejects_invalid_or_ambiguous_split_configuration(
    tmp_path: Path,
) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    tsv.write_text(
        "audio\ttext\tsource_split\nready.wav\tBom dia\tdev\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid split"):
        build_full_training_manifest(
            tsv,
            audio_root,
            dictionary,
            vocab,
            tmp_path / "invalid-split-output",
            split_column="source_split",
            workers=1,
        )

    with pytest.raises(ValueError, match="cannot both be set"):
        build_full_training_manifest(
            tsv,
            audio_root,
            dictionary,
            vocab,
            tmp_path / "ambiguous-split-output",
            split="train",
            split_column="source_split",
            workers=1,
        )


def test_full_manifest_can_allow_exact_dictionary_connectors(tmp_path: Path) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    output_dir = tmp_path / "connector-output"

    summary = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        shard_size=5,
        workers=1,
        allow_exact_dictionary_connectors=True,
    )

    assert summary.allow_exact_dictionary_connectors is True
    assert summary.ready_records == 2
    assert "unresolved_connector" not in summary.issue_counts
    ready = _read_jsonl(output_dir / "train_ready.jsonl")
    connector = next(record for record in ready if record["text"] == "Bem-vindo")
    assert connector["training_ready"] is True


def test_full_manifest_reviews_digit_fragments_instead_of_silently_dropping_them(
    tmp_path: Path,
) -> None:
    tsv, audio_root, dictionary, vocab = _write_fixture(tmp_path)
    tsv.write_text(
        "audio\ttext\nready.wav\tBom dia em 2026\n",
        encoding="utf-8",
    )
    dictionary.write_text(
        dictionary.read_text(encoding="utf-8") + "em\te m\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    summary = build_full_training_manifest(
        tsv,
        audio_root,
        dictionary,
        vocab,
        output_dir,
        shard_size=1,
        workers=1,
    )

    assert summary.ready_records == 0
    assert summary.review_records == 1
    assert summary.issue_counts == {"unresolved_digit": 1}
    record = _read_jsonl(output_dir / "needs_review.jsonl")[0]
    assert record["phoneme_token_ids"]
    assert record["issues"] == [{"detail": "2026", "reason": "unresolved_digit"}]


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
