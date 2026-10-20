from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_english_best_noise import noisy_case as noisy_case
from test_english_head_preview import case as case
from test_reviewed_retrieval import read, retrieval, write
from test_stream_contexts import stream_case as stream_case

from qwen_hotword.evaluation.multilingual_stream_assets import prepare_multilingual_stream
from qwen_hotword.evaluation.stream_retrieval_metrics import (
    reference_matcher,
    score_stream_contexts,
)
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.multilingual_stream_run import run_multilingual_stream
from qwen_hotword.inference.stream_context_contract import MODES
from qwen_hotword.training.spanish_capacity import _sha


@pytest.fixture
def multilingual_case(stream_case):
    import numpy as np
    import soundfile as sf

    c = stream_case
    cfg = read(Path("configs/multilingual_stream_contexts.workzone.json"))
    cfg.update(
        english_stream_config=str(c.stream_config),
        wave_audio_root=str(c.root / "noisy_waves"),
        checkpoint_sha256=_sha(c.head),
    )
    for info in cfg["additional_datasets"].values():
        info["expected_samples"] = 2
        directory = Path(cfg["wave_audio_root"]) / info["wave"] / info["language"] / "wav"
        directory.mkdir(parents=True)
        for utt in ("same", "negative"):
            sf.write(
                directory / f"{utt}.wav", np.linspace(-0.1, 0.2, 48000), 16000, subtype="FLOAT"
            )
        (directory.parent / "transcripts.txt").write_text("ignored noisy reference")
    c.multi_config = c.root / "multilingual.json"
    c.multi_cfg = cfg
    write(c.multi_config, cfg)
    c.multi_output = c.root / "multilingual_stream"
    c.languages = []

    def factory(**kwargs):
        c.languages.append(kwargs["language"])
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        c.seen.append((detector.language, len(waveform)))
        return retrieval(detector.hotwords)

    c.multi_kw = {"detector_factory": factory, "retrieve_function": retrieve}
    return c


def test_three_languages_direct_full_metrics_and_small_return(multilingual_case):
    c = multilingual_case
    for path in (c.root / "audio").rglob("*.wav"):
        path.unlink()
    before = {p: _sha(p) for p in c.root.rglob("*") if p.is_file()}
    audit = run_multilingual_stream(
        c.root, c.multi_config, c.multi_output, effect="same-step", audit_only=True, **c.multi_kw
    )
    assert audit["sample_count"] == 22
    assert audit["by_language"] == {"en": 6, "es": 8, "pt": 8}
    assert not c.seen and not c.multi_output.exists()
    report = run_multilingual_stream(
        c.root, c.multi_config, c.multi_output, effect="same-step", **c.multi_kw
    )
    assert c.languages == ["English", "Spanish", "Portuguese"]
    assert len(c.seen) == report["prefix_count"]
    assert report["sample_count"] == 22 and report["schema_validation"]["unique_utterances"] == 22
    assert not report["qwen_decoder_used"] and not report["word_timestamps_available"]
    assert report["downstream_format_validator_passed"]
    assert (
        report["by_language"]["en"]["RAG_top7_stream"]["last_context"][
            "expected_target_sample_pairs"
        ]
        == 9
    )
    for modes in report["by_language"].values():
        for views in modes.values():
            assert views["ever_delivered"]["target_hits"] >= views["last_context"]["target_hits"]
    plan = read(c.multi_output / "run_config.json")
    assert sum(plan["selected_duration_buckets"].values()) == plan["selected_samples"]
    assert plan["inventory_duration_buckets"] == plan["selected_duration_buckets"]
    assert all("reference_text" not in s for s in plan["samples"])
    assert all(
        str(c.noisy_root) in s["audio_path"] for s in plan["samples"] if s["language"] == "en"
    )
    contexts = read(c.multi_output / "delivery/stream_contexts.json")
    assert "wave1@@es" in contexts["RAG_top5_stream"]
    assert "wave1@@pt" in contexts["RAG_top5_stream"]
    details = [
        json.loads(s) for s in (c.multi_output / "sample_metrics.jsonl").read_text().splitlines()
    ]
    assert len(details) == 22
    with zipfile.ZipFile(c.multi_output / "summary_bundle.zip") as archive:
        assert set(archive.namelist()) == {
            "report.json",
            "terminal_summary.json",
            "checkpoint_metadata.json",
            "sha256.txt",
        }
        assert not any("delivery" in n or "run_config" in n for n in archive.namelist())
        assert sum(i.file_size for i in archive.infolist()) < 100000
    run_multilingual_stream(
        c.root, c.multi_config, c.multi_output, effect="same-step", resume=True, **c.multi_kw
    )
    assert c.languages == ["English", "Spanish", "Portuguese"]
    assert all(_sha(p) == sha for p, sha in before.items())
    with pytest.raises(FileExistsError):
        run_multilingual_stream(
            c.root, c.multi_config, c.multi_output, effect="same-step", **c.multi_kw
        )


def test_original_denominators_are_kept_and_production_phrase_matching(multilingual_case):
    c = multilingual_case
    plan, runtime = prepare_multilingual_stream(c.root, c.multi_config)
    entries = runtime["bundle"].entries
    extra = tuple(
        replace(entries[0], hotword_id=f"phrase_{i}", surface=word)
        for i, word in enumerate(("warning", "warnings", "New York", "can't", "new-york"))
    )
    match = reference_matcher(extra)
    for text in ("warnings in NEW YORK!", "A warning; I can't leave", "warningless New Yorker"):
        assert match(text) == {e.hotword_id for e in extra if strict_phrase_match(text, e.surface)}
    info = runtime["evaluation_groups"]["en_chile"]
    utt = next(iter(info["references"]))
    target = next(e for e in entries if e.hotword_id in info["target_ids"])
    sample = {"group": "g", "key": "k", "sample_id": "u"}
    one = {"word": target.surface, "phoneme": target.pronunciation, "score": 0.9}
    evaluation = {
        lang: {
            "language": lang,
            "target_ids": [target.hotword_id],
            "references": {"u": target.surface},
            "expected_target_sample_pairs": 1,
        }
        for lang in ("en", "es", "pt")
    }
    samples = [{**sample, "group": lang, "key": lang} for lang in evaluation]
    data = {mode: {lang: {"u": [[one], []]} for lang in evaluation} for mode in MODES}
    metrics, _ = score_stream_contexts(data, samples, evaluation, entries)
    for views in metrics["by_language"]["en"].values():
        assert views["last_context"]["target_hits"] == 0
        assert views["ever_delivered"]["target_hits"] == 1
    assert utt and plan["selected_samples"] == 22


def test_resume_missing_prefixes_and_bad_inputs(multilingual_case):
    c = multilingual_case
    retrieve = c.multi_kw["retrieve_function"]
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("interrupted")
        return retrieve(*args)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_multilingual_stream(
            c.root,
            c.multi_config,
            c.multi_output,
            effect="same-step",
            **{**c.multi_kw, "retrieve_function": interrupted},
        )
    assert len(c.seen) == 2
    report = run_multilingual_stream(
        c.root, c.multi_config, c.multi_output, effect="same-step", resume=True, **c.multi_kw
    )
    assert len(c.seen) == report["prefix_count"]
    with pytest.raises(ValueError, match="resume run_config differs"):
        run_multilingual_stream(
            c.root, c.multi_config, c.multi_output, effect="next-step", resume=True, **c.multi_kw
        )
    write(c.multi_config, {**c.multi_cfg, "checkpoint_sha256": "bad"})
    with pytest.raises(ValueError, match="Head SHA"):
        prepare_multilingual_stream(c.root, c.multi_config)


def test_missing_wave_reference_or_retrieval_drift_rejected(multilingual_case, monkeypatch):
    c = multilingual_case
    audio = Path(c.multi_cfg["wave_audio_root"]) / "wave1/es/wav/same.wav"
    audio.rename(audio.with_name("wrong.wav"))
    with pytest.raises(ValueError, match="mismatch"):
        prepare_multilingual_stream(c.root, c.multi_config)
    audio.with_name("wrong.wav").rename(audio)
    original = external._run_config

    def changed(**kwargs):
        result = original(**kwargs)
        if kwargs["language"] == "es":
            result["gate"]["threshold"] = 0.7
        return result

    monkeypatch.setattr("qwen_hotword.inference.external_keyword_retrieval._run_config", changed)
    with pytest.raises(ValueError, match="retrieval settings differ"):
        prepare_multilingual_stream(c.root, c.multi_config)
    assert not c.languages and not c.multi_output.exists()


def test_next_step_is_scored_from_delivered_contexts_and_drift_is_blocked(multilingual_case):
    c = multilingual_case
    report = run_multilingual_stream(
        c.root, c.multi_config, c.multi_output, effect="next-step", **c.multi_kw
    )
    assert report["effect"] == "next-step"
    contexts = read(c.multi_output / "delivery/stream_contexts.json")
    durations = read(c.multi_output / "delivery/durations.json")
    for mode in MODES:
        for key, by_utt in contexts[mode].items():
            for utt, steps in by_utt.items():
                assert steps[0] == []
                if durations[key][utt] <= 2:
                    assert steps == [[]]
    assert report["by_language"]["en"]["RAG_top7_stream"]["last_context"]["target_hits"] < 9
    path = Path(c.multi_cfg["wave_audio_root"]) / "wave4/pt/wav/same.wav"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="resume run_config differs"):
        run_multilingual_stream(
            c.root, c.multi_config, c.multi_output, effect="next-step", resume=True, **c.multi_kw
        )
