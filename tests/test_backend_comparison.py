from __future__ import annotations

import importlib
import json
import platform
import subprocess
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


def memory_config() -> dict[str, Any]:
    return {
        "minimum_driver": "525.60.13",
        "gpu_memory_utilization": 0.15,
        "transformers_allocator_limit_gib": 24,
        "shared_memory_reserve_mib": 8192,
    }


def test_preflight_allows_user_shared_gpu_but_rejects_old_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    cfg = memory_config()
    monkeypatch.setattr(
        subprocess, "check_output", lambda *a, **kw: "NVIDIA H200, 520.1, 0, 143771, 143000"
    )
    with pytest.raises(ValueError, match="do not modify node drivers"):
        runner.preflight(4, cfg)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **kw: "NVIDIA H200, 580.1, 98055, 143771, 45000",
    )
    result = runner.preflight(4, cfg)
    assert result["physical_gpu"] == 4 and result["used_mib"] == 98055
    assert result["memory_budget"] == {
        "vllm_executor_budget_mib": 21566,
        "transformers_allocator_limit_mib": 24576,
        "reserve_mib": 8192,
        "required_free_mib": 32768,
    }


def test_preflight_uses_reported_free_memory_and_keeps_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    # total-used appears sufficient, but driver-reported free excludes reserved memory.
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *a, **kw: "NVIDIA H200, 580.1, 110000, 143771, 32000",
    )
    with pytest.raises(ValueError, match="32000 MiB free, needs 32768 MiB"):
        runner.preflight(4, memory_config())


@pytest.mark.parametrize(
    "key,value",
    [
        ("gpu_memory_utilization", 0),
        ("gpu_memory_utilization", 1.1),
        ("gpu_memory_utilization", float("nan")),
        ("transformers_allocator_limit_gib", -1),
        ("transformers_allocator_limit_gib", 200),
        ("shared_memory_reserve_mib", 0),
        ("shared_memory_reserve_mib", float("inf")),
    ],
)
def test_invalid_memory_budgets_are_rejected(key: str, value: float) -> None:
    with pytest.raises(ValueError, match="invalid shared-GPU"):
        runner.memory_budget(143771, {**memory_config(), key: value})


def test_vllm_budget_is_fraction_of_total_and_backends_are_sequential() -> None:
    budget = runner.memory_budget(143771, {**memory_config(), "gpu_memory_utilization": 0.3})
    assert budget["vllm_executor_budget_mib"] == 43132
    assert budget["required_free_mib"] == 43132 + 8192


def test_changing_free_memory_does_not_change_device_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "physical_gpu": 4,
        "name": "NVIDIA H200",
        "total_mib": 143771,
        "used_mib": 98055,
        "free_mib": 45000,
        "memory_budget": runner.memory_budget(143771, memory_config()),
    }
    monkeypatch.setattr(runner, "preflight", lambda *a: dict(snapshot))
    runner.record_preflight(4, memory_config(), tmp_path, "before_install")
    snapshot.update({"used_mib": 100000, "free_mib": 43000})
    runner.record_preflight(4, memory_config(), tmp_path, "vllm")
    identity = json.loads((tmp_path / "device.json").read_text())
    assert "used_mib" not in identity and "free_mib" not in identity
    checks = sorted((tmp_path / "memory_checks").glob("*.json"))
    assert len(checks) == 2
    assert {json.loads(p.read_text())["free_mib"] for p in checks} == {45000, 43000}


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
