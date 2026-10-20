from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_english_best_noise import noisy_case as noisy_case
from test_english_head_preview import case as case
from test_reviewed_retrieval import read, retrieval, write

from qwen_hotword.evaluation.stream_context_assets import select_samples
from qwen_hotword.inference.stream_context_contract import (
    audio_prefixes,
    boundaries,
    contexts_for_steps,
    validate_contexts,
)
from qwen_hotword.inference.stream_context_run import run_stream_contexts
from qwen_hotword.training.spanish_capacity import _sha


@pytest.fixture
def stream_case(noisy_case):
    import numpy as np
    import soundfile as sf

    c = noisy_case
    sizes = iter((11200, 32000, 32001, 64000, 68800, 84800))
    for group in c.cfg["datasets"]:
        for name in ("negative", "same"):
            size = next(sizes)
            pcm = np.linspace(-0.2, 0.2, size, dtype=np.float32)
            sf.write(c.noisy_root / group / "en/wav" / f"{name}.wav", pcm, 16000, subtype="FLOAT")
    cfg = read(Path("configs/english_stream_contexts.workzone.json"))
    cfg.update(retrieval_config=str(c.config), smoke_samples=3)
    c.stream_config = c.root / "stream_config.json"
    write(c.stream_config, cfg)
    c.stream_cfg = cfg
    c.seen = []

    def factory(**kwargs):
        assert kwargs["checkpoint"] == c.head and kwargs["language"] == "English"
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        c.calls.append("load")
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        c.seen.append(waveform.copy())
        c.calls.append("prefix")
        # A prior candidate can disappear: no permanent union is permitted.
        return retrieval(detector.hotwords, negative=len(waveform) > 64000)

    c.stream_kw = {"detector_factory": factory, "retrieve_function": retrieve}
    c.stream_output = c.root / "stream_smoke"
    return c


def test_exact_sample_boundaries_and_no_future_before_resampling(tmp_path):
    import numpy as np
    import soundfile as sf

    assert boundaries(1, 16000) == (1,)
    assert boundaries(32000, 16000) == (32000,)
    assert boundaries(64000, 16000) == (32000, 64000)
    assert boundaries(64001, 16000) == (32000, 64000, 64001)
    common = np.linspace(-0.4, 0.4, 16000, dtype=np.float32)
    early = []
    for name, future in (("a", 0.7), ("b", -0.7)):
        audio = np.concatenate((common, np.full(8000, future, dtype=np.float32)))
        stereo = np.stack((audio, audio * 0.5), axis=1)
        path = tmp_path / f"{name}.wav"
        sf.write(path, stereo, 8000, subtype="FLOAT")
        result = list(
            audio_prefixes(str(path), {"frames": 24000, "sample_rate": 8000, "channels": 2})
        )
        assert [s["source_end_frame"] for s in result] == [16000, 24000]
        assert [s["model_input_samples"] for s in result] == [32000, 48000]
        early.append(result[0])
    np.testing.assert_array_equal(early[0]["waveform"], early[1]["waveform"])
    assert early[0]["input_pcm_sha256"] == early[1]["input_pcm_sha256"]


def test_effect_and_contract_validation():
    item = {"word": "New York", "phoneme": "n u j ɔ ɹ k", "score": 0.9}
    steps = [{"candidates": []}, {"candidates": [item]}, {"candidates": []}]
    assert contexts_for_steps(steps, "same-step", 5) == [[], [item], []]
    assert contexts_for_steps(steps, "next-step", 7) == [[], [], [item]]
    durations = {"enmx@@en": {"utt": 5.3}}
    valid = {
        m: {"enmx@@en": {"utt": contexts_for_steps(steps, "same-step", k)}}
        for m, k in (("RAG_top5_stream", 5), ("RAG_top7_stream", 7))
    }
    assert validate_contexts(valid, durations)["unique_utterances"] == 1
    for mutation in ("missing", "short", "null", "duplicate", "score"):
        changed = json.loads(json.dumps(valid))
        value = changed["RAG_top5_stream"]["enmx@@en"]["utt"]
        if mutation == "missing":
            changed["RAG_top5_stream"]["enmx@@en"].pop("utt")
        elif mutation == "short":
            value.pop()
        elif mutation == "null":
            value[1] = None
        elif mutation == "duplicate":
            value[1].append(item)
        else:
            value[1][0]["score"] = float("nan")
        with pytest.raises(ValueError):
            validate_contexts(changed, durations)


def test_stream_smoke_causal_delivery_vendor_validator_and_resume(stream_case):
    c = stream_case
    # Original clean waveforms and local noisy references never enter prefix retrieval.
    for path in (c.root / "audio").rglob("*.wav"):
        path.unlink()
    before = {p: _sha(p) for p in c.root.rglob("*") if p.is_file()}
    audit = run_stream_contexts(
        c.root,
        c.stream_config,
        c.stream_output,
        stage="smoke",
        effect="same-step",
        audit_only=True,
        **c.stream_kw,
    )
    assert audit["selected_samples"] == 3 and not audit["model_loaded"]
    assert not c.stream_output.exists() and not c.seen
    report = run_stream_contexts(
        c.root, c.stream_config, c.stream_output, stage="smoke", effect="same-step", **c.stream_kw
    )
    assert report["downstream_validator"]["returncode"] == 0
    assert report["schema_validation"]["unique_utterances"] == 3
    assert report["candidate_pool_surfaces"] == 11 and not report["qwen_decoder_used"]
    plan = read(c.stream_output / "run_config.json")
    expected_lengths = [end for s in plan["samples"] for end in boundaries(s["frames"], 16000)]
    assert [len(w) for w in c.seen] == expected_lengths
    manifest = read(c.stream_output / "delivery/audio_manifest.json")
    assert set(manifest) == {"encl@@en", "encol@@en", "enmx@@en"}
    assert all("reference_text" not in s for s in plan["samples"])
    contexts = read(c.stream_output / "delivery/stream_contexts.json")
    first = plan["samples"][0]
    assert len(contexts["RAG_top5_stream"][first["key"]][first["sample_id"]][0]) == 5
    assert len(contexts["RAG_top7_stream"][first["key"]][first["sample_id"]][0]) == 7
    for by_utt in contexts["RAG_top7_stream"].values():
        for steps in by_utt.values():
            assert all(len(items) <= 7 for items in steps)
    calls = len(c.seen)
    run_stream_contexts(
        c.root,
        c.stream_config,
        c.stream_output,
        stage="smoke",
        effect="same-step",
        resume=True,
        **c.stream_kw,
    )
    assert len(c.seen) == calls
    assert all(_sha(p) == sha for p, sha in before.items())
    with zipfile.ZipFile(report["return_file"]) as archive:
        assert "delivery/stream_contexts.json" in archive.namelist()
        assert not any(
            "prefix_shards" in name or name.endswith(".pt") for name in archive.namelist()
        )


def test_next_step_shifts_contexts_and_preserves_tail_count(stream_case):
    c = stream_case
    report = run_stream_contexts(
        c.root, c.stream_config, c.stream_output, stage="smoke", effect="next-step", **c.stream_kw
    )
    assert report["effect"] == "next-step"
    contexts = read(c.stream_output / "delivery/stream_contexts.json")
    for by_key in contexts.values():
        for by_utt in by_key.values():
            assert all(steps[0] == [] for steps in by_utt.values())
    rows = [
        json.loads(line)
        for line in (c.stream_output / "causality_audit.jsonl").read_text().splitlines()
    ]
    for row in rows:
        assert row["retrieval_input_end_frame"] <= row["available_audio_end_frame"]
        assert row["context_source_step"] == (
            row["effective_step"] - 1 if row["effective_step"] else None
        )


def test_prefix_interruption_and_input_drift(stream_case):
    c = stream_case
    retrieve = c.stream_kw["retrieve_function"]
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("interrupted")
        return retrieve(*args)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_stream_contexts(
            c.root,
            c.stream_config,
            c.stream_output,
            stage="smoke",
            effect="same-step",
            **{**c.stream_kw, "retrieve_function": interrupted},
        )
    assert len(list(c.stream_output.glob("prefix_shards/*/*/*.json"))) == 1
    shard = next(c.stream_output.glob("prefix_shards/*/*/*.json"))
    original = shard.read_bytes()
    broken = read(shard)
    broken["input_pcm_sha256"] = "wrong-prefix"
    write(shard, broken)
    with pytest.raises(ValueError, match="corrupted prefix shard"):
        run_stream_contexts(
            c.root,
            c.stream_config,
            c.stream_output,
            stage="smoke",
            effect="same-step",
            resume=True,
            **c.stream_kw,
        )
    shard.write_bytes(original)
    report = run_stream_contexts(
        c.root,
        c.stream_config,
        c.stream_output,
        stage="smoke",
        effect="same-step",
        resume=True,
        **c.stream_kw,
    )
    assert len(c.seen) == report["prefix_count"]
    plan = read(c.stream_output / "run_config.json")
    path = Path(plan["samples"][0]["audio_path"])
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="resume run_config differs"):
        run_stream_contexts(
            c.root,
            c.stream_config,
            c.stream_output,
            stage="smoke",
            effect="same-step",
            resume=True,
            **c.stream_kw,
        )


def test_full_requires_matching_smoke(stream_case):
    c = stream_case
    full = c.root / "stream_full"
    with pytest.raises(ValueError, match="requires --smoke-run"):
        run_stream_contexts(
            c.root, c.stream_config, full, stage="full", effect="same-step", **c.stream_kw
        )
    assert not c.seen and not full.exists()
    run_stream_contexts(
        c.root, c.stream_config, c.stream_output, stage="smoke", effect="same-step", **c.stream_kw
    )
    report = run_stream_contexts(
        c.root,
        c.stream_config,
        full,
        stage="full",
        effect="same-step",
        smoke_run=c.stream_output,
        **c.stream_kw,
    )
    assert report["selected_samples"] == 6 and report["scope"] == "full_inventory"
    with pytest.raises(ValueError, match="compatible smoke"):
        run_stream_contexts(
            c.root,
            c.stream_config,
            c.root / "wrong_effect",
            stage="full",
            effect="next-step",
            smoke_run=c.stream_output,
            **c.stream_kw,
        )


def test_vendor_samples_and_truncation_conflict(tmp_path):
    fixture = Path("tests/fixtures")
    for name, info in read(fixture / "stream_interface_v1_provenance.json")["members"].items():
        assert _sha(fixture / name) == info["sha256"]
    validator = fixture / "stream_interface_v1_validator.py.txt"
    command = [
        sys.executable,
        str(validator),
        "--ctx",
        str(fixture / "stream_interface_v1_sample.json"),
        "--durations",
        str(fixture / "stream_interface_v1_durations.json"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0 and "VALID" in result.stdout
    data = read(fixture / "stream_interface_v1_sample.json")
    data["RAG_top7_stream"]["enmx@@en"]["10168_quite_0022"][0] = [
        {"word": str(i), "phoneme": "a", "score": 0.9} for i in range(8)
    ]
    path = tmp_path / "over_k.json"
    write(path, data)
    command[3] = str(path)
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_selection_covers_groups_without_claiming_missing_buckets():
    rows = [
        {"group": g, "sample_id": str(i), "duration_bucket": "tail"}
        for g in ("en_chile", "en_colombia", "en_mexico")
        for i in range(3)
    ]
    selected = select_samples(rows, 3, "fixed")
    assert {r["group"] for r in selected} == {r["group"] for r in rows}
    assert selected == select_samples(rows, 3, "fixed")
