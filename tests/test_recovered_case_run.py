from __future__ import annotations

import json
import struct
import wave
import zipfile
from collections import Counter
from pathlib import Path

import pytest

from qwen_hotword.evaluation import recovered_case_assets as assets
from qwen_hotword.evaluation.recovered_audio_export import sha
from qwen_hotword.evaluation.recovered_case_run import read_json, run_case_hunt


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def seal(directory):
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "sha256.txt")
    (directory / "sha256.txt").write_text(
        "".join(f"{sha(p)}  {p.relative_to(directory).as_posix()}\n" for p in files)
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root, source, baseline = tmp_path, tmp_path / "source", tmp_path / "baseline"
    table_path = tmp_path / "table/table.json"
    targets = {"en": "Alpha", "es": "Málaga", "pt": "São Paulo"}
    write_json(table_path, {"entries": [{"surface": s} for s in [*targets.values(), "Beta"]]})
    monkeypatch.setattr(assets, "TABLE_SHA256", sha(table_path))
    model = tmp_path / "models/Qwen3-ASR-1.7B"
    for name, value in {
        "config.json": {"architectures": ["Qwen3ASRForConditionalGeneration"]},
        "model.safetensors.index.json": {"weight_map": {}},
        "preprocessor_config.json": {},
    }.items():
        write_json(model / name, value)
    samples, groups, durations, manifest = [], {}, {}, {}
    contexts = {"RAG_top5_stream": {}, "RAG_top7_stream": {}}
    target_by_path = {}
    for index in range(55):
        language = list(targets)[index % 3]
        target = targets[language]
        sid, group, key = f"sample_{index}", f"group_{language}", f"directory@@{language}"
        path = root / "wav" / f"{sid}.wav"
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            handle.writeframes(struct.pack("<h", index + 1) * 16)
        target_by_path[str(path)] = target
        sample = {
            "key": key,
            "group": group,
            "language": language,
            "sample_id": sid,
            "audio_path": str(path),
            "audio_sha256": sha(path),
            "frames": 16,
            "sample_rate": 16000,
            "channels": 1,
            "duration_seconds": 0.001,
        }
        samples.append(sample)
        groups.setdefault(group, {"records": []})["records"].append(
            {
                "sample_id": sid,
                "reference_text": f"say {target} now",
            }
        )
        manifest.setdefault(key, {})[sid] = sample
        durations.setdefault(key, {})[sid] = 0.001
        for mode in contexts:
            contexts[mode].setdefault(key, {})[sid] = [
                [
                    {"word": target, "phoneme": "a", "score": 0.9},
                    {"word": "Beta", "phoneme": "b", "score": 0.8},
                ]
            ]
    write_json(
        baseline / "run_config.json",
        {
            "configuration": {"model": str(model)},
            "groups": groups,
        },
    )
    seal(baseline)
    source_plan = {
        "checkpoint": {"sha256": assets.HEAD_SHA},
        "source_configuration": {"table_sha256": sha(table_path), "baseline_run": str(baseline)},
        "samples": samples,
        "inputs": {str(p): {"sha256": sha(p)} for p in model.iterdir()},
    }
    write_json(source / "run_config.json", source_plan)
    monkeypatch.setattr(assets, "SOURCE_CONFIG_SHA", sha(source / "run_config.json"))
    write_json(
        source / "report.json",
        {
            "status": "completed",
            "effect": "same-step",
            "table_sha256": sha(table_path),
            "checkpoint_sha256": assets.HEAD_SHA,
        },
    )
    for name, value in {
        "stream_contexts": contexts,
        "durations": durations,
        "audio_manifest": manifest,
    }.items():
        write_json(source / "delivery" / f"{name}.json", value)
    seal(source)
    calls, loads = [], []
    behavior = {
        "interrupt_at": None,
        "no_gain": False,
        "already_correct": False,
        "truncate": None,
        "audio_mismatch": False,
    }

    class Engine:
        def __init__(self, model_path, cfg):
            loads.append(model_path)
            self.metadata = {"test_only": True}
            assert cfg["max_new_tokens"] == 1024

        def transcribe(self, request):
            assert set(request) == {"audio_path", "audio_sha256", "language", "context", "seed"}
            calls.append(request)
            if len(calls) == behavior["interrupt_at"]:
                raise KeyboardInterrupt
            target = target_by_path[request["audio_path"]]
            if request["context"]:
                # Reference-only selection must NOT remove the actual false-positive candidate.
                assert request["context"].endswith(f"{target}\nBeta")
            prediction = (
                f"say {target} now"
                if (request["context"] and not behavior["no_gain"]) or behavior["already_correct"]
                else "say something now"
            )
            return {
                "prediction": prediction,
                "actual_context": request["context"],
                "prompt_tokens_verified": True,
                "returned_language": request["language"],
                "generation_limit_reached": behavior["truncate"]
                == ("retrieved" if request["context"] else "baseline"),
                "normalized_audio_sha256": (
                    "f" * 64
                    if request["context"] and behavior["audio_mismatch"]
                    else request["audio_sha256"]
                ),
            }

    output = tmp_path / "new_output"

    def run(count=50):
        return run_case_hunt(
            root,
            source,
            table_path,
            output,
            count=count,
            runtime_factory=Engine,
            runtime_probe=lambda: {"test_only": True},
        )

    return {
        "run": run,
        "output": output,
        "calls": calls,
        "loads": loads,
        "behavior": behavior,
        "source": source,
        "samples": samples,
        "table": table_path,
        "model": model,
    }


def test_real_pool_to_50_original_wavs_md_zip_and_completed_resume(fixture):
    result = fixture["run"]()
    assert result["status"] == "completed" and result["exported_cases"] == 50
    assert set(result["by_language"]) == {"en", "es", "pt"}
    assert len(fixture["calls"]) == 100 and len(fixture["loads"]) == 1
    package = fixture["output"] / "package"
    cases = [json.loads(line) for line in (package / "cases.jsonl").read_text().splitlines()]
    assert len(list((package / "audio").glob("*.wav"))) == 50
    for case in cases:
        assert sha(package / case["exported_audio"]) == sha(Path(case["source_audio"]))
        assert case["injected_hotwords"][-1] == "Beta"
        assert case["recovered_hotwords"] != ["Beta"]
    with zipfile.ZipFile(result["return_bundle"]) as archive:
        assert len([n for n in archive.namelist() if n.endswith(".wav")]) == 50
        assert "不加热词" in archive.read("案例说明.md").decode()
    assert fixture["run"]() == result
    assert len(fixture["loads"]) == 1 and len(fixture["calls"]) == 100
    with Path(result["return_bundle"]).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="ZIP changed"):
        fixture["run"]()


def test_resume_after_first_baseline_never_repeats_saved_call(fixture):
    fixture["behavior"]["interrupt_at"] = 2
    with pytest.raises(KeyboardInterrupt):
        fixture["run"](2)
    progress = read_json(fixture["output"] / "progress.json")
    assert progress["status"] == "interrupted_or_failed"
    first = fixture["calls"][0]
    fixture["behavior"]["interrupt_at"] = None
    result = fixture["run"](2)
    assert result["exported_cases"] == 2
    assert len(fixture["calls"]) == 5  # includes the interrupted, unsaved prompted call
    assert fixture["calls"].count(first) == 1
    assert read_json(fixture["output"] / "progress.json")["reused_ASR_calls_this_invocation"] == 1


@pytest.mark.parametrize("mode,expected_calls", [("already_correct", 55), ("no_gain", 110)])
def test_no_text_recovery_means_honest_partial_package(fixture, mode, expected_calls):
    fixture["behavior"][mode] = True
    result = fixture["run"]()
    assert result["status"] == "insufficient_verified_cases" and result["exported_cases"] == 0
    assert result["shortfall"] == 50 and result["checked_audio"] == 55
    assert len(fixture["calls"]) == expected_calls
    assert not list((fixture["output"] / "package/audio").iterdir())


@pytest.mark.parametrize("mode,expected_calls", [("baseline", 55), ("retrieved", 110)])
def test_truncated_outputs_never_create_gains(fixture, mode, expected_calls):
    fixture["behavior"]["truncate"] = mode
    assert fixture["run"](1)["exported_cases"] == 0
    assert len(fixture["calls"]) == expected_calls


def test_changed_saved_cell_rejected_before_loading_again(fixture):
    fixture["behavior"]["interrupt_at"] = 2
    with pytest.raises(KeyboardInterrupt):
        fixture["run"](1)
    cell = next((fixture["output"] / "cells").rglob("baseline.json"))
    value = read_json(cell)
    value["result"]["prediction"] = "tampered"
    write_json(cell, value)
    with pytest.raises(ValueError, match="saved ASR cell changed"):
        fixture["run"](1)
    assert len(fixture["loads"]) == 1


@pytest.mark.parametrize("changed", ["table", "model", "audio", "source"])
def test_changed_input_rejected_without_asr(fixture, changed):
    plan, _ = assets.prepare_cases(
        fixture["source"].parent, fixture["source"], fixture["table"], 20261028
    )
    path = {
        "table": fixture["table"],
        "model": fixture["model"] / "config.json",
        "audio": Path(plan["pool"][0]["audio_path"]),
        "source": fixture["source"] / "delivery/stream_contexts.json",
    }[changed]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity differs|SHA256 mismatch|table SHA256"):
        fixture["run"]()
    assert not fixture["calls"] and not fixture["loads"]


def test_normalized_audio_mismatch_never_exported(fixture):
    fixture["behavior"]["audio_mismatch"] = True
    with pytest.raises(ValueError, match="different normalized audio"):
        fixture["run"](1)
    assert not (fixture["output"] / "package").exists()


def test_changed_plan_rejects_automatic_resume(fixture):
    fixture["run"](1)
    with pytest.raises(ValueError, match="refusing overwrite"):
        fixture["run"](2)
    assert len(fixture["loads"]) == 1


def test_duplicate_audio_and_conflicting_references_excluded(fixture):
    source_plan = read_json(fixture["source"] / "run_config.json")
    contexts = read_json(fixture["source"] / "delivery/stream_contexts.json")
    old = read_json(fixture["source"].parent / "baseline/run_config.json")
    references = {
        (g, r["sample_id"]): r["reference_text"]
        for g, data in old["groups"].items()
        for r in data["records"]
    }
    samples = source_plan["samples"]
    table = {
        assets.normalize_match_words(e["surface"]): e["surface"]
        for e in read_json(fixture["table"])["entries"]
    }
    # Same language/reference, identical bytes, different source IDs: only one may be packaged.
    samples[3]["audio_sha256"] = samples[0]["audio_sha256"]
    pool, reasons = assets.pool_rows(samples, contexts, references, table, 1)
    assert len(pool) == 54 and reasons["duplicate_audio_not_selected"] == 1
    references[(samples[3]["group"], samples[3]["sample_id"])] = "different reference"
    pool, reasons = assets.pool_rows(samples, contexts, references, table, 1)
    assert len(pool) == 53 and reasons["duplicate_audio_conflicting_reference_or_language"] == 2
    assert Counter(r["audio_sha256"] for r in pool).most_common(1)[0][1] == 1
