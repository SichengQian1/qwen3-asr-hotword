from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from qwen_hotword.evaluation import backend_compare_run as runner
from qwen_hotword.evaluation.backend_compare_runtime import content_tokens
from qwen_hotword.evaluation.backend_comparison import (
    block_legacy_qwen_vllm_registration,
    compare_rows,
    contains,
    context,
    digest,
    distance,
    normalize,
    public_wrapper_eos_ids,
    saved_source_rows,
    write_once,
)


def seal(row: dict[str, Any]) -> dict[str, Any]:
    value = {k: v for k, v in row.items() if k != "sha256"}
    return {**value, "sha256": digest(value)}


def fixture_pair() -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    request = {
        "request_id": "one",
        "language": "pt",
        "condition": "original",
        "mode": "new70_top7",
        "audio_path": "/fixture/example.wav",
        "words": ["São Paulo"],
        "reference_text": "Vou para São Paulo",
    }
    cell = {
        "request_sha256": digest(request),
        "generation_limit_reached": False,
        "input_token_ids": [1, 2, 3],
        "waveform_sha256": "same_wave",
        "content_token_ids": [10, 20],
        "text": "Vou para São Paulo.",
    }
    return [request], {"transformers": {"one": seal(cell)}, "vllm": {"one": seal(cell)}}


def test_exact_format_and_lexical_differences_are_separate() -> None:
    requests, outputs = fixture_pair()
    _rows, report = compare_rows(requests, outputs)
    assert report["by_group"]["overall"]["exact"] == 1
    outputs["vllm"]["one"] = seal({**outputs["vllm"]["one"], "text": "vou para são paulo"})
    rows, _report = compare_rows(requests, outputs)
    assert rows[0]["classification"] == "format_only"
    outputs["vllm"]["one"] = seal({**outputs["vllm"]["one"], "text": "Vou para Sao Paulo"})
    rows, report = compare_rows(requests, outputs)
    assert rows[0]["classification"] == "word_difference"
    assert report["by_group"]["overall"]["vllm_wer"] == 0.25
    assert report["by_group"]["overall"]["target_hits_differ"] == 1


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("input_token_ids", [1, 2, 4], "Prompt IDs differ"),
        ("waveform_sha256", "different", "waveforms differ"),
        ("generation_limit_reached", True, "truncated"),
        ("request_sha256", "wrong", "identity differs"),
    ],
)
def test_incomparable_pairs_are_rejected(field: str, value: Any, error: str) -> None:
    requests, outputs = fixture_pair()
    outputs["vllm"]["one"] = seal({**outputs["vllm"]["one"], field: value})
    with pytest.raises(ValueError, match=error):
        compare_rows(requests, outputs)


def test_modified_or_missing_results_do_not_count_as_backend_disagreement() -> None:
    requests, outputs = fixture_pair()
    outputs["vllm"]["one"]["text"] = "unsealed change"
    with pytest.raises(ValueError, match="cell changed"):
        compare_rows(requests, outputs)
    outputs["vllm"].clear()
    with pytest.raises(ValueError, match="incomplete"):
        compare_rows(requests, outputs)


def test_accents_and_enye_are_preserved() -> None:
    assert normalize(" CANCIÓN, niño! ") == ("canción", "niño")
    assert normalize("Sa\u0303o") == normalize("São")
    assert not contains("Sao Paulo", "São Paulo")
    assert not contains("el nino", "el niño")
    assert contains("Vou para São Paulo.", "São Paulo")
    assert distance(("a", "b"), ("a", "c", "b")) == 1


def test_context_uses_only_saved_words_and_accepts_empty_candidates() -> None:
    assert context("pt", []) == ""
    assert "São Paulo" in context("pt", ["São Paulo"])
    with pytest.raises(ValueError):
        context("pt", ["bad<|im_end|>"])
    with pytest.raises(ValueError):
        context("en", ["place"] * 8)


def test_only_terminal_stop_tokens_are_removed() -> None:
    assert content_tokens([5, 6, 99], [99]) == [5, 6]
    assert content_tokens([99, 6], [99]) == [99, 6]


def test_eos_comes_from_public_wrapper_not_thinker_configuration() -> None:
    def public_generate(*, eos_token_id: list[int] = [151645, 151643]) -> None:  # noqa: B006 -- match upstream
        del eos_token_id

    assert public_wrapper_eos_ids(public_generate) == [151645, 151643]


def test_optional_legacy_backend_is_blocked_and_import_hook_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("qwen_asr", "qwen_asr.core"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    original = list(sys.meta_path)
    with (
        block_legacy_qwen_vllm_registration(),
        pytest.raises(
            ImportError,
            match="legacy qwen-asr vLLM import disabled",
        ),
    ):
        importlib.import_module("qwen_asr.core.vllm_backend")
    assert sys.meta_path == original


def test_same_output_can_resume_but_changed_content_cannot(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    write_once(path, {"x": 1})
    write_once(path, {"x": 1})
    with pytest.raises(ValueError, match="NEW output"):
        write_once(path, {"x": 2})
    assert json.loads(path.read_text()) == {"x": 1}


def test_installation_uses_fixed_native_wheel_and_not_old_qwen_extra() -> None:
    cfg = json.loads(
        (
            Path(__file__).parents[1] / "configs/asr_backend_comparison_v023.workzone.json"
        ).read_text()
    )
    requirements = runner.requirements(cfg)
    assert "v0.23.0" in requirements[0] and "#sha256=" in requirements[0]
    assert "torch==2.11.0+cu129" in requirements
    assert "qwen-asr==0.0.6" in requirements
    assert not any("qwen-asr[" in requirement for requirement in requirements)


def test_preflight_stops_on_old_driver_or_busy_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(runner.platform, "machine", lambda: "x86_64")
    cfg = {"minimum_driver": "525.60.13"}
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "NVIDIA H200, 520.1, 0, 140000"
    )
    with pytest.raises(ValueError, match="do not modify node drivers"):
        runner.preflight(5, cfg)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "NVIDIA H200, 580.1, 12000, 140000"
    )
    with pytest.raises(ValueError, match="occupied"):
        runner.preflight(5, cfg)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *a, **kw: "NVIDIA H200, 580.1, 0, 140000"
    )
    assert runner.preflight(5, cfg)["physical_gpu"] == 5


def test_saved_source_tag_hash_and_identity_are_checked(tmp_path: Path) -> None:
    (tmp_path / "retrieval/cells").mkdir(parents=True)
    cfg = {"source_head_sha256": "head", "source_index_sha256": "index"}
    write_once(
        tmp_path / "evaluation_inputs.json",
        {
            "checkpoints": {"new70": "head"},
            "new_index_sha256": "index",
        },
    )
    write_once(tmp_path / "retrieval/report.json", {"status": "completed"})
    write_once(tmp_path / "retrieval/cells/01.json", seal({"identity": {"tag": "old50"}}))
    row = seal({"identity": {"tag": "new70", "checkpoint_sha256": "head"}})
    path = tmp_path / "retrieval/cells/02.json"
    write_once(path, row)
    rows = list(saved_source_rows(tmp_path, cfg))
    assert len(rows) == 1 and rows[0]["source_cell"] == str(path)
    row["identity"]["checkpoint_sha256"] = "other"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="modified"):
        list(saved_source_rows(tmp_path, cfg))


def test_summary_and_package_repeat_without_new_inference(tmp_path: Path) -> None:
    requests, outputs = fixture_pair()
    write_once(tmp_path / "requests.json", {"requests": requests})
    for backend, cells in outputs.items():
        for key, cell in cells.items():
            write_once(tmp_path / backend / "cells" / f"{key}.json", cell)
    runner.summarize(tmp_path)
    runner.summarize(tmp_path)
    bundle = runner.package(tmp_path)
    import zipfile

    with zipfile.ZipFile(bundle) as archive:
        assert "comparison.tsv" in archive.namelist()
        assert not any(name.endswith(".npy") for name in archive.namelist())
