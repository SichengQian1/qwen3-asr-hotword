from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_english_best_noise import noisy_case as noisy_case
from test_english_head_preview import case as case
from test_reviewed_retrieval import read, retrieval, write

from qwen_hotword.inference import english_sample_selection as selection
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.english_best_noise import load_completed_best
from qwen_hotword.inference.english_head_preview import prepare_preview, run_preview
from qwen_hotword.training.spanish_capacity import _sha


@pytest.fixture
def selected_case(noisy_case):
    c = noisy_case
    c.cfg.pop("audio_root")
    c.cfg.update(
        checkpoint_sha256=_sha(c.head),
        samples_per_dataset=1,
        seed="selection-test-v1",
        expected_target_sample_pairs=9,
        output="selection",
    )
    write(c.config, c.cfg)
    c.output = c.root / "selection"

    def factory(**kwargs):
        assert kwargs["checkpoint"] == c.head
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        assert len(kwargs["hotwords"]) > 7
        c.calls.append("load")
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        assert waveform == b"positive"
        c.calls.append("sample")
        return retrieval(detector.hotwords)

    c.kw = {"detector_factory": factory, "retrieve_function": retrieve}
    return c


def test_uniform_pool_uses_true_distinct_targets_not_model_results(selected_case):
    c = selected_case
    plan, runtime = selection.prepare_selection(c.root, c.config)
    targets = [
        e
        for e in runtime["bundle"].entries
        if e.hotword_id in plan["groups"]["en_chile"]["target_ids"]
    ]
    targets = targets[:4]
    # Use actual target surfaces; repetitions count once, substrings don't count.
    assert len(targets) == 3
    targets.append(replace(targets[0], hotword_id="extra", surface="extra", normalized="extra"))
    texts = ["unrelated", targets[0].surface + " " + targets[0].surface]
    texts += [" ".join(e.surface for e in targets[:n]) for n in (2, 3, 4)]
    texts.append(targets[0].surface + "ing")
    records = [
        external.ExternalAudioRecord(str(i), "pool", Path(f"/{i}.wav"), text, "en")
        for i, text in enumerate(texts)
    ]
    samples, distribution = selection.choose_samples(records, targets, count=3, seed="fixed")
    assert distribution == {"0": 2, "1": 1, "2": 1, "3": 1, "4": 1}
    assert {s["sample_id"] for s in samples} == {"1", "2", "3"}
    again, _ = selection.choose_samples(list(reversed(records)), targets, count=3, seed="fixed")
    assert samples == again
    assert len(samples[0]["expected_surfaces"]) in {1, 2, 3}
    with pytest.raises(ValueError, match="only 3 eligible"):
        selection.choose_samples(records, targets, count=4, seed="fixed")


def test_prepare_then_top7_delivery_and_completed_resume(selected_case):
    c = selected_case
    before = {p: _sha(p) for p in c.root.rglob("*") if p.is_file()}
    prepared = selection.run_selection(c.root, c.config, c.output, prepare_only=True, **c.kw)
    assert prepared["sample_count"] == 3 and not c.calls
    assert not (c.output / "sample_selection.json").exists()
    plan = read(c.output / "run_config.json")
    hashed_wavs = [name for name in plan["inputs"] if name.endswith(".wav")]
    assert len(hashed_wavs) == 3 and all(name.endswith("same.wav") for name in hashed_wavs)
    result = selection.run_selection(c.root, c.config, c.output, resume=True, **c.kw)
    data = read(Path(result["delivery_file"]))
    assert data["sample_counts_by_source"] == dict.fromkeys(c.cfg["datasets"], 1)
    assert data["actual_group_counts"] == {"three_targets": 3}
    assert data["test_set_used"] and not data["qwen_decoder_used"]
    assert data["checkpoint"]["sha256"] == _sha(c.head)
    assert all(len(s["expected_hotword_ids"]) == 3 for s in data["samples"])
    assert all(len(s["selected_matches"]) == 7 for s in data["samples"])
    assert all(s["redundant_family_ids"] == [] for s in data["samples"])
    for sample in data["samples"]:
        assert set(sample["selected_matches"][0]) == {
            "hotword_id",
            "surface",
            "score",
            "edit_ratio",
            "posterior_confidence",
        }
        assert sample["case_type"] == "positive" and "noisy" not in sample["audio_path"]
    assert c.calls == ["load"] + ["sample"] * 3
    selection.run_selection(c.root, c.config, c.output, resume=True, **c.kw)
    assert c.calls.count("load") == 1 and c.calls.count("sample") == 3
    assert all(_sha(p) == digest for p, digest in before.items())
    with pytest.raises(FileExistsError):
        selection.run_selection(c.root, c.config, c.output, **c.kw)


def test_interruption_resume_and_corrupt_shard_rejection(selected_case):
    c = selected_case
    retrieve = c.kw["retrieve_function"]
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return retrieve(*args)

    with pytest.raises(RuntimeError, match="interrupted"):
        selection.run_selection(
            c.root, c.config, c.output, **{**c.kw, "retrieve_function": interrupted}
        )
    assert len(list((c.output / "sample_shards").glob("*.json"))) == 1
    selection.run_selection(c.root, c.config, c.output, resume=True, **c.kw)
    assert c.calls.count("sample") == 3
    shard = next((c.output / "sample_shards").glob("*.json"))
    envelope = read(shard)
    envelope["row"]["raw_ranked_matches"][0]["score"] = 0.1
    write(shard, envelope)
    with pytest.raises(ValueError, match="checksum differs"):
        selection.run_selection(c.root, c.config, c.output, resume=True, **c.kw)


def test_drift_and_insufficient_pool_fail_before_model_or_output(selected_case):
    c = selected_case
    for patch in (
        {"samples_per_dataset": 2},
        {"checkpoint_sha256": "wrong"},
        {"expected_target_sample_pairs": 10},
        {"expected_epoch": 21},
    ):
        write(c.config, {**c.cfg, **patch})
        with pytest.raises(ValueError):
            selection.run_selection(c.root, c.config, c.output, **c.kw)
        assert not c.output.exists() and not c.calls
    write(c.config, c.cfg)
    path = c.root / "audio/en_chile/wav/same.wav"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        selection.run_selection(c.root, c.config, c.output, **c.kw)
    assert not c.output.exists() and not c.calls


def test_output_cannot_overlap_source(selected_case):
    c = selected_case
    with pytest.raises(ValueError, match="overlaps"):
        selection.run_selection(c.root, c.config, c.root / "expanded_run/selection", **c.kw)
    assert not c.calls


def test_sealed_clean_epoch27_can_be_reused_without_gpu(selected_case):
    c = selected_case

    def prepare(root, config):
        return prepare_preview(root, config, checkpoint_loader=load_completed_best)

    def retrieve(detector, waveform, vocab):
        return retrieval(detector.hotwords, negative=waveform == b"negative")

    prior = c.root / "clean_final27"
    run_preview(
        c.root, c.config, prior, prepare_function=prepare, **{**c.kw, "retrieve_function": retrieve}
    )
    c.calls.clear()
    selection.run_selection(c.root, c.config, c.output, reuse_run=prior, **c.kw)
    assert not c.calls
    data = read(c.output / "sample_selection.json")
    assert len(data["samples"]) == 3 and len(data["samples"][0]["selected_matches"]) == 7
    plan = read(prior / "run_config.json")
    old_sha = _sha(prior / "run_config.json")
    plan["checkpoint"]["sha256"] = "epoch21"
    write(prior / "run_config.json", plan)
    checks = prior / "sha256.txt"
    checks.write_text(checks.read_text().replace(old_sha, _sha(prior / "run_config.json")))
    with pytest.raises(ValueError, match="reuse requires"):
        selection.run_selection(c.root, c.config, c.root / "reject_reuse", reuse_run=prior, **c.kw)


def test_empty_actual_candidates_are_retained_without_padding_or_reselection(selected_case):
    c = selected_case

    def retrieve(detector, waveform, vocab):
        c.calls.append("sample")
        return retrieval(detector.hotwords, negative=True)

    selection.run_selection(c.root, c.config, c.output, **{**c.kw, "retrieve_function": retrieve})
    samples = read(c.output / "sample_selection.json")["samples"]
    assert len(samples) == 3
    assert all(s["selected_matches"] == [] and len(s["expected_surfaces"]) == 3 for s in samples)


def test_gate_is_applied_before_top7_and_candidates_keep_actual_scores(selected_case):
    c = selected_case
    next_surface_scores = []

    def retrieve(detector, waveform, vocab):
        result = retrieval(detector.hotwords)
        result["raw_ranked_matches"][0]["edit_ratio"] = 0.4
        first = result["raw_ranked_matches"][0]["surface"].casefold()
        next_surface_scores.append(
            next(
                m["score"] for m in result["raw_ranked_matches"] if m["surface"].casefold() != first
            )
        )
        return result

    selection.run_selection(c.root, c.config, c.output, **{**c.kw, "retrieve_function": retrieve})
    samples = read(c.output / "sample_selection.json")["samples"]
    assert all(len(s["selected_matches"]) == 7 for s in samples)
    assert all(
        s["selected_matches"][0]["score"] == score
        for s, score in zip(samples, next_surface_scores, strict=True)
    )
    assert all(m["edit_ratio"] <= 0.35 for s in samples for m in s["selected_matches"])
