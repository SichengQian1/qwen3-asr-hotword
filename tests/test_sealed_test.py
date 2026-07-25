from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    EpochMetrics,
    ExperimentRecord,
    save_ctc_head_checkpoint,
)
from qwen_hotword.training.sealed_test import evaluate_sealed_ctc_test


def _records(tmp_path: Path) -> list[ExperimentRecord]:
    records = []
    for index in range(2):
        audio = tmp_path / f"test-{index}.wav"
        audio.write_bytes(b"fake")
        records.append(
            ExperimentRecord(
                sample_id=f"test-{index}",
                audio_path=audio,
                text="a",
                language="pt-BR",
                token_ids=(1,),
                ctc_minimum_input_length=1,
            )
        )
    return records


def _fake_extractor(
    records: list[ExperimentRecord],
    _wrapper: object,
    **_kwargs: object,
) -> tuple[list[CachedSample], int, float]:
    torch = pytest.importorskip("torch")
    return (
        [
            CachedSample(
                sample_id=record.sample_id,
                hidden_states=torch.zeros((3, 1024), dtype=torch.bfloat16),
                token_ids=record.token_ids,
            )
            for record in records
        ],
        317_477_504,
        0.01,
    )


def test_sealed_test_evaluates_fixed_temporal_best_once(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from qwen_hotword.modeling.ctc_head import TemporalUpsampleCtcHead

    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps({"tokens": ["<blank>", "a", "b"]}) + "\n",
        encoding="utf-8",
    )
    vocab = load_phoneme_vocab(vocab_path)
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "full_ctc_test.jsonl"
    manifest.write_text("sealed test identity\n", encoding="utf-8")
    checkpoint = tmp_path / "ctc_head_best.pt"
    head = TemporalUpsampleCtcHead(
        1024,
        3,
        hidden_dimension=4,
        kernel_size=3,
        dropout=0.0,
        time_upsampling_factor=2,
    )
    for parameter in head.parameters():
        parameter.data.zero_()
    head.output_projection.bias.data[1] = 10.0
    save_ctc_head_checkpoint(
        checkpoint,
        head,
        vocab,
        EpochMetrics(
            epoch=24,
            loss=0.3,
            phoneme_error_rate=0.06,
            phoneme_errors=6,
            reference_phonemes=100,
        ),
        seed=7,
    )
    output = tmp_path / "sealed-report.json"

    report = evaluate_sealed_ctc_test(
        _records(tmp_path),
        SimpleNamespace(),
        checkpoint,
        vocab,
        output,
        test_manifest_path=manifest,
        vocab_path=vocab_path,
        model_path=model,
        device="cpu",
        encoder_batch_size=2,
        evaluation_batch_size=2,
        records_per_chunk=1,
        extractor=_fake_extractor,
    )

    assert report["test_set_used"] is True
    assert report["one_time_evaluation"] is True
    assert report["feature_cache_written"] is False
    assert report["metrics"]["sample_count"] == 2
    assert report["metrics"]["phoneme_error_rate"] == 0.0
    assert output.is_file()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        evaluate_sealed_ctc_test(
            _records(tmp_path),
            SimpleNamespace(),
            checkpoint,
            vocab,
            output,
            test_manifest_path=manifest,
            vocab_path=vocab_path,
            model_path=model,
            device="cpu",
            extractor=_fake_extractor,
        )


def test_sealed_test_rejects_latest_checkpoint_name(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(
        json.dumps({"tokens": ["<blank>", "a"]}) + "\n",
        encoding="utf-8",
    )
    vocab = load_phoneme_vocab(vocab_path)

    with pytest.raises(ValueError, match="ctc_head_best"):
        evaluate_sealed_ctc_test(
            _records(tmp_path),
            SimpleNamespace(),
            tmp_path / "ctc_head_latest.pt",
            vocab,
            tmp_path / "report.json",
            test_manifest_path=tmp_path / "test.jsonl",
            vocab_path=vocab_path,
            model_path=tmp_path / "Qwen3-ASR-1.7B",
            device="cpu",
            extractor=_fake_extractor,
        )
