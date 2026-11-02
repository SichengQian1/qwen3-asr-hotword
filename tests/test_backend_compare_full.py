from __future__ import annotations

import copy
import json
import os
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.evaluation import backend_compare_full as full
from qwen_hotword.evaluation import backend_compare_run as runner
from qwen_hotword.evaluation import backend_compare_runtime as runtime
from qwen_hotword.evaluation.backend_comparison import digest, read, sha, write_once


def config() -> dict[str, Any]:
    return {
        "expected_audio_counts": {"en": 2, "es": 1, "pt": 1},
        "source_head_sha256": "head",
        "source_index_sha256": "index",
        "source_vocab_sha256": "vocab",
        "source_run": "source",
        "model": "/fixture/Qwen3-ASR-1.7B",
        "sample_rate": 16000,
        "pad_waveform_to_samples": 160,
        "max_model_len": 4096,
        "max_new_tokens": 512,
        "selection_policy": "full_saved_inventory",
    }


def inventory() -> list[dict[str, Any]]:
    result = []
    for lang, count in config()["expected_audio_counts"].items():
        for number in range(count):
            for condition in ("original", "noise10db"):
                identity = {
                    "tag": "new70",
                    "language": lang,
                    "group": lang,
                    "sample_id": str(number),
                    "condition": condition,
                    "checkpoint_sha256": "head",
                    "index_sha256": "index",
                    "vocab_sha256": "vocab",
                    "audio_sha256": "audio",
                }
                result.append(
                    {
                        "identity": identity,
                        "language": lang,
                        "group": lang,
                        "sample_id": str(number),
                        "condition": condition,
                        "reference_text": "a reference",
                        "top7": {"words": []},
                        "source_cell_sha256": digest(identity),
                    }
                )
    return result


def test_complete_inventory_retains_duplicate_content_and_counts_both_prompts() -> None:
    report = full.validate_inventory(inventory(), config())
    assert report["base_utterances"] == 4
    assert report["audio_conditions"] == 8
    assert report["backend_pairs"] == 16
    assert report["planned_backend_calls"] == 32


def test_missing_duplicate_unpaired_and_changed_identity_are_rejected() -> None:
    rows = inventory()
    with pytest.raises(ValueError, match="counts differ"):
        full.validate_inventory(rows[:-1], config())
    with pytest.raises(ValueError, match="duplicate"):
        full.validate_inventory([*rows, rows[0]], config())
    changed = copy.deepcopy(rows)
    changed[1]["sample_id"] = changed[1]["identity"]["sample_id"] = "other"
    with pytest.raises(ValueError, match="not paired"):
        full.validate_inventory(changed, config())
    changed = copy.deepcopy(rows)
    changed[0]["identity"]["index_sha256"] = "old"
    with pytest.raises(ValueError, match="identity differs"):
        full.validate_inventory(changed, config())
    changed = copy.deepcopy(rows)
    changed[1]["reference_text"] = "other"
    with pytest.raises(ValueError, match="references differ"):
        full.validate_inventory(changed, config())


@pytest.mark.parametrize("counts", [{"en": 1}, {"en": True, "es": 1, "pt": 1}])
def test_invalid_counts_are_rejected(counts: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="per-language counts"):
        full.counts({"expected_audio_counts": counts})


@pytest.fixture
def cpu_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Any]:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    rows = inventory()
    cfg = config()
    source = tmp_path / "source"
    write_once(
        source / "evaluation_inputs.json",
        {"checkpoints": {"new70": "head"}, "new_index_sha256": "index"},
    )
    write_once(source / "retrieval/report.json", {"status": "completed"})
    for number, row in enumerate(rows):
        audio = tmp_path / f"{number}.wav"
        # Keep a >30s input, exact duplicate content and a non-hop-aligned length.
        samples = 16000 * 33 + 17 if row["language"] == "pt" else 16017
        sf.write(audio, np.full(samples, 0.02, dtype=np.float32), 16000)
        row["audio_path"] = str(audio)
        row["identity"]["audio_sha256"] = sha(audio)
        row.pop("source_cell_sha256")
        write_once(source / f"retrieval/cells/{number:03}.json", {**row, "sha256": digest(row)})
    for name in ("qwen_asr", "qwen_asr.core", "qwen_asr.core.transformers_backend"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    class Processor:
        feature_extractor = SimpleNamespace(hop_length=160)
        calls = 0
        fail_at = 0

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
            return cls()

        def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
            return "shared template "

        def tokenizer(self, prompt: str) -> dict[str, list[int]]:
            return {"input_ids": [1, len(prompt)]}

        def __call__(self, **kwargs: Any) -> dict[str, Any]:
            type(self).calls += 1
            if type(self).calls == type(self).fail_at:
                raise RuntimeError("simulated CPU interruption")
            return {"input_ids": np.array([[1, len(kwargs["text"][0])]])}

    backend = sys.modules["qwen_asr.core.transformers_backend"]
    monkeypatch.setattr(backend, "Qwen3ASRProcessor", Processor, raising=False)
    monkeypatch.setitem(sys.modules, "librosa", ModuleType("librosa"))
    monkeypatch.setattr(runtime, "versions", lambda: {"test": "fixed"})
    return cfg, Processor


def test_full_cpu_preparation_retains_long_audio_duplicates_and_resumes(
    tmp_path: Path, cpu_preparation: tuple[dict[str, Any], Any]
) -> None:
    cfg, processor = cpu_preparation
    output = tmp_path / "out"
    output.mkdir()
    processor.fail_at = 3
    with pytest.raises(RuntimeError, match="simulated CPU interruption"):
        full.prepare_full(tmp_path, cfg, output)
    assert len(list((output / "preparation/cells").glob("*.json"))) == 1
    processor.fail_at = 0
    full.prepare_full(tmp_path, cfg, output)
    assert processor.calls == 17  # two successful calls were reused after the interruption
    plan = read(output / "requests.json")
    assert len(plan["requests"]) == 16
    summary = read(output / "preparation_summary.json")
    assert summary["duration_max_seconds"] > 30
    assert summary["duplicate_file_records_retained"]["en/original"] == 1
    assert all(r["shared_padded_samples"] % 160 == 0 for r in plan["requests"])
    full.prepare_full(tmp_path, cfg, output)
    assert processor.calls == 17
    plan["requests"][0]["prompt"] = "changed"
    (output / "requests.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="requests differ"):
        full.prepare_full(tmp_path, cfg, output)


def test_context_budget_failure_is_explicit_before_any_inference(
    tmp_path: Path, cpu_preparation: tuple[dict[str, Any], Any]
) -> None:
    cfg, _processor = cpu_preparation
    output = tmp_path / "out"
    output.mkdir()
    cfg["max_model_len"] = 513
    with pytest.raises(ValueError, match="context exceeds budget before inference"):
        full.prepare_full(tmp_path, cfg, output)
    assert not (output / "requests.json").exists()
    assert not (output / "transformers").exists()


def test_full_mode_never_installs_a_missing_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = read(Path(__file__).parents[1] / "configs/asr_backend_comparison_v023_full.workzone.json")
    monkeypatch.setattr(runner, "command", lambda *a, **kw: pytest.fail("must not run pip"))
    with pytest.raises(ValueError, match="already verified environment"):
        runner.environment(tmp_path, cfg, tmp_path / "out", {})
    assert not (tmp_path / ".venv").exists()


def test_full_config_has_fixed_inventory_and_reuses_the_successful_environment() -> None:
    root = Path(__file__).parents[1]
    cfg = read(root / "configs/asr_backend_comparison_v023_full.workzone.json")
    small = read(root / "configs/asr_backend_comparison_v023.workzone.json")
    assert cfg["expected_audio_counts"] == {"en": 6000, "es": 400, "pt": 400}
    assert "maximum_audio_seconds" not in cfg
    assert cfg["environment"] == small["environment"]
    assert cfg["output"] != small["output"]
    assert runner.requirements(cfg) == runner.requirements(small)
    for key in ("max_new_tokens", "max_model_len", "seed", "gpu_memory_utilization"):
        assert cfg[key] == small[key]


def test_compact_bundle_is_under_100kb_and_excludes_full_results(tmp_path: Path) -> None:
    write_once(tmp_path / "report.json", {"status": "completed"})
    write_once(tmp_path / "requests.json", {"not_for_return": "large"})
    write_once(tmp_path / "transformers/cells/a.json", {"not_for_return": "large"})
    write_once(
        tmp_path / "differences_preview.json",
        {
            "preview_only": True,
            "total_different_pairs": 500,
            "rows": [{"text": os.urandom(6000).hex()} for _ in range(24)],
        },
    )
    bundle = full.compact_package(tmp_path)
    assert bundle.stat().st_size < 100000
    with zipfile.ZipFile(bundle) as archive:
        assert "report.json" in archive.namelist()
        assert "requests.json" not in archive.namelist()
        assert not any("cells/" in n for n in archive.namelist())
        sample = json.loads(archive.read("differences_preview.json"))
        assert sample["total_different_pairs"] == 500
        assert 0 < len(sample["rows"]) < 24


def test_difference_preview_includes_languages_and_is_explicitly_partial() -> None:
    rows = [
        {
            "classification": "word_difference",
            "language": lang,
            "condition": "original",
            "mode": "new70_top7",
        }
        for lang in ["en"] * 10 + ["es", "pt"]
    ]
    value = full.preview(rows, limit=3)
    assert {r["language"] for r in value["rows"]} == {"en", "es", "pt"}
    assert value["total_different_pairs"] == 12 and value["preview_only"]


def test_full_summary_and_compact_return_repeat_without_inference(tmp_path: Path) -> None:
    request = {
        "request_id": "one",
        "language": "pt",
        "condition": "original",
        "mode": "new70_top7",
        "audio_path": "/fixture/sample.wav",
        "words": ["São Paulo"],
        "reference_text": "Vou para São Paulo",
    }
    cell = {
        "request_sha256": digest(request),
        "generation_limit_reached": False,
        "input_token_ids": [1, 2],
        "waveform_sha256": "same",
        "content_token_ids": [3],
        "text": "Vou para São Paulo.",
    }
    cfg = {"selection_policy": "full_saved_inventory", "compact_return_bundle": True}
    write_once(tmp_path / "run_contract.json", {"config": cfg})
    write_once(
        tmp_path / "requests.json",
        {"requests": [request], "full_inventory": {"base_utterances": 1, "backend_pairs": 1}},
    )
    for backend in ("transformers", "vllm"):
        write_once(tmp_path / backend / "cells/one.json", {**cell, "sha256": digest(cell)})
    report = runner.summarize(tmp_path)
    assert report["scope"] == "full_H200_final_ASR_backend_comparison"
    assert report["by_group"]["language/pt"]["exact"] == 1
    assert runner.summarize(tmp_path) == report
    with zipfile.ZipFile(runner.package(tmp_path)) as archive:
        assert "differences_preview.json" in archive.namelist()
        assert "details.json" not in archive.namelist()
