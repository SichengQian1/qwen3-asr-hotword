from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.evaluation.casablanca_wer import (
    attach_lora,
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

    (tmp_path / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing adapter_model"):
        inspect_checkpoint(optimizer)


def test_lora_preflight_uses_recorded_local_base_or_explicit_override(tmp_path: Path) -> None:
    base = tmp_path / "exact-base"
    base.mkdir()
    for name in ("tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
        (base / name).write_text("{}", encoding="utf-8")
    (base / "config.json").write_text('{"model_type":"qwen3_asr"}', encoding="utf-8")
    (base / "model.safetensors").write_bytes(b"fixture")

    adapter = tmp_path / "checkpoint-4900"
    adapter.mkdir()
    config = adapter / "adapter_config.json"
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    config.write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base)}))
    metadata = inspect_checkpoint(adapter / "optimizer.pt")
    assert metadata["format"] == "peft_lora"
    assert metadata["directory"] == str(adapter)
    assert metadata["base_model"]["directory"] == str(base)
    assert metadata["base_source"] == "adapter_config.json"

    config.write_text('{"peft_type":"LORA","base_model_name_or_path":"organization/unknown-base"}')
    with pytest.raises(ValueError, match=r"organization/unknown-base.*--base-model"):
        inspect_checkpoint(adapter)
    metadata = inspect_checkpoint(adapter, base)
    assert metadata["base_source"] == "--base-model"
    assert metadata["base_model"]["directory"] == str(base)
    assert metadata["recorded_base"] == "organization/unknown-base"
    with pytest.raises(ValueError, match="base-model path must be a full HF checkpoint"):
        inspect_checkpoint(adapter, adapter)


def test_attach_saved_lora_keeps_original_model_and_requires_all_tensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")

    def tiny_base() -> Any:
        model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False))
        torch.nn.init.zeros_(model[0].weight)
        return model

    trained = peft.get_peft_model(
        tiny_base(), peft.LoraConfig(r=1, lora_alpha=1, target_modules=["0"])
    )
    for parameter in trained.parameters():
        if parameter.requires_grad:
            torch.nn.init.ones_(parameter)
    trained.save_pretrained(tmp_path)

    model = tiny_base()
    inputs = torch.ones((1, 3))
    assert torch.count_nonzero(model(inputs)) == 0
    assert attach_lora(model, str(tmp_path)) == 2
    assert torch.equal(model(inputs), trained(inputs))
    assert not model.training

    weight_path = tmp_path / "adapter_model.safetensors"
    weights = safetensors.load_file(weight_path)
    weights.pop(next(iter(weights)))
    safetensors.save_file(weights, weight_path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="adapter tensors do not match.*missing"):
        attach_lora(tiny_base(), str(tmp_path))


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
