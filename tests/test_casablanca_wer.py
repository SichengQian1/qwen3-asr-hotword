from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.evaluation.casablanca_wer import (
    evaluate,
    inspect_checkpoint,
    inspect_shards,
    normalize,
)


def test_checkpoint_locator_requires_complete_local_hf_export(tmp_path: Path) -> None:
    optimizer = tmp_path / "optimizer.pt"
    optimizer.write_bytes(b"this must not be opened as a model")
    with pytest.raises(ValueError, match="optimizer.pt is optimizer state"):
        inspect_checkpoint(optimizer)

    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"thinker.layer":"model-00001-of-00001.safetensors"}}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing or empty model weight"):
        inspect_checkpoint(optimizer)

    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
    for name in ("tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}', encoding="utf-8")
    checkpoint = inspect_checkpoint(optimizer)
    assert checkpoint["directory"] == str(tmp_path)
    assert checkpoint["optimizer_path_used_as_locator"]
    assert checkpoint["weights"] == [{"file": "model-00001-of-00001.safetensors", "bytes": 7}]
    assert checkpoint["weight_index_sha256"]

    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="merged HF export"):
        inspect_checkpoint(optimizer)


def test_normalization_preserves_arabic_letters_and_code_switching() -> None:
    assert normalize(" أَهْــلاً، C’EST 2026. ") == "أهلا c'est 2026"
    assert normalize("ا\u0654 إ ا ى ي ة ه") == "أ إ ا ى ي ة ه"


def test_batched_parquet_evaluation_is_corpus_wer(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    pytest.importorskip("jiwer")
    wav = io.BytesIO()
    sf.write(wav, np.zeros((800, 2), dtype=np.float32), 8000, format="WAV")
    audio = {"bytes": wav.getvalue(), "path": "not-needed.wav"}
    paths = [tmp_path / "00.parquet", tmp_path / "01.parquet"]
    for path, references in zip(paths, [["أهلاً!", "ب ج د ه و ز"], ["Bonjour, toi!"]], strict=True):
        pq.write_table(
            pa.table(
                {
                    "seg_id": [f"{path.stem}_{index}" for index in range(len(references))],
                    "audio": [audio] * len(references),
                    "transcription": references,
                }
            ),
            path,
        )

    class Model:
        def transcribe(
            self, *, audio: list[tuple[Any, int]], language: str | None, return_time_stamps: bool
        ) -> list[SimpleNamespace]:
            assert language is None and not return_time_stamps
            assert all(samples.shape == (800,) and rate == 8000 for samples, rate in audio)
            texts = ["", "ب ج د ه و ز"] if len(audio) == 2 else ["bonjour toi"]
            return [SimpleNamespace(text=text, language="Arabic") for text in texts]

    shards = inspect_shards(paths)
    assert [shard["rows"] for shard in shards] == [2, 1]
    result = evaluate(Model(), shards, tmp_path, 2, None)
    assert result["scored_samples"] == 3
    assert result["wer"]["normalized"]["wer"] == pytest.approx(1 / 9)
    assert result["wer"]["raw"]["wer"] == pytest.approx(3 / 9)
    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["seg_id"] for row in rows] == ["00_0", "00_1", "01_0"]
    assert rows[0]["reference"] == "أهلاً!" and rows[0]["prediction"] == ""
    assert rows[0]["edits"]["normalized"]["deletions"] == 1
    with pytest.raises(FileExistsError):
        evaluate(Model(), shards, tmp_path, 2, None)
