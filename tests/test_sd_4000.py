from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation.sd_keywords import build_sd_table
from qwen_hotword.evaluation.wave_keywords import _entry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.sd_retrieval import run_sd_retrieval
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def fixture(root):
    write(
        root / "sd.json",
        {
            "language": "Portuguese",
            "keyword_sets": {"all_keywords": ["Dra. ", "alvo"]},
            "keyword_phonemes": {"Dra.": "a", "alvo": "a b"},
        },
    )
    vocab = load_phoneme_vocab(VOCAB)
    fillers = [_entry(f"word{i}", "a", "pt", vocab) for i in range(10)]
    fillers.append(_entry("alvo", "a", "pt", vocab))  # target pronunciation must win
    fillers[1]["language"] = "pt-BR"  # legacy alias must be accepted
    (root / "old.jsonl").write_text("\n".join(json.dumps(r) for r in fillers))
    write(root / "model/config.json", {})
    (root / "head.pt").write_bytes(b"mock head")
    sources = {}
    for name in ("mls_portuguese", "delivery_20260706_ptbr"):
        wav = root / name / "audio"
        wav.mkdir(parents=True)
        (wav / "same.wav").write_bytes(b"positive")
        (wav / "empty.flac").write_bytes(b"empty")
        (root / name / "transcripts.txt").write_text("same\talvo word0\nempty\tnothing\n")
        sources[name] = {
            "audio": f"{name}/audio",
            "transcripts": f"{name}/transcripts.txt",
            "expected_samples": 2,
        }
    cfg = {
        "primary": "sd.json",
        "expected_primary": 2,
        "fillers": "old.jsonl",
        "filler_sha256": _sha(root / "old.jsonl"),
        "target_size": 8,
        "expected_keywords": 8,
        "seed": "test",
        "tables": "tables",
        "output": "run",
        "vocab": str(VOCAB),
        "vocab_sha256": _sha(VOCAB),
        "model": "model",
        "checkpoint": "head.pt",
        "checkpoint_sha256": _sha(root / "head.pt"),
        "sources": sources,
    }
    write(root / "config.json", cfg)
    return root / "config.json", cfg


def test_builder_target_preservation_determinism_and_no_overwrite(tmp_path):
    config, cfg = fixture(tmp_path)
    output = tmp_path / "tables"
    report = build_sd_table(tmp_path, config, output)
    assert (report["total"], report["mandatory"], report["old_table_fillers"]) == (8, 2, 6)
    table = json.loads((output / "keyword_bias_phoneme.json").read_text())
    assert table["keyword_sets"]["all_keywords"][:2] == ["Dra.", "alvo"]
    assert table["keyword_phonemes"]["alvo"] == "a b"
    build_sd_table(tmp_path, config, tmp_path / "repeat")
    assert (output / "keyword_bias_phoneme.json").read_bytes() == (
        tmp_path / "repeat/keyword_bias_phoneme.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        build_sd_table(tmp_path, config, output)
    cfg["target_size"] = 99
    write(config, cfg)
    with pytest.raises(ValueError, match="insufficient"):
        build_sd_table(tmp_path, config, tmp_path / "short")
    assert not (tmp_path / "short").exists()


def test_corrupt_filler_or_oov_target_stops_before_publication(tmp_path):
    config, _ = fixture(tmp_path)
    (tmp_path / "old.jsonl").write_text("{}")
    output = tmp_path / "tables"
    with pytest.raises(ValueError, match="SHA256"):
        build_sd_table(tmp_path, config, output)
    assert not output.exists()
    write(
        tmp_path / "sd.json",
        {
            "keyword_sets": {"all_keywords": ["bad"]},
            "keyword_phonemes": {"bad": "☃"},
        },
    )
    with pytest.raises(ValueError, match="phoneme audit"):
        build_sd_table(tmp_path, config, output)
    assert not output.exists()


def mocks(monkeypatch):
    calls = {"loads": 0, "retrieved": 0}

    def factory(**kw):
        calls["loads"] += 1
        return SimpleNamespace(**kw)

    monkeypatch.setattr(
        external,
        "_load_audio",
        lambda path: (path.read_bytes(), {"duration_seconds": 1.0, "audio_load_seconds": 0.01}),
    )

    def retrieve(detector, audio, vocab):
        calls["retrieved"] += 1
        # Put the target at rank 7 to distinguish Top5 from Top7, and fillers from targets.
        entries = sorted(detector.hotwords, key=lambda e: e.surface == "alvo")
        entries.insert(6, entries.pop())
        raw = (
            []
            if audio == b"empty"
            else [
                {
                    "hotword_id": e.hotword_id,
                    "surface": e.surface,
                    "score": 0.99 - i * 0.01,
                    "edit_ratio": 0.0,
                    "posterior_confidence": 0.9,
                }
                for i, e in enumerate(entries)
            ]
        )
        return {
            "raw_ranked_matches": raw,
            "selected_matches": raw[:5],
            "shortlist_candidates": len(raw),
            "postings_visited": 8,
            "timing": {
                k: 0.01
                for k in (
                    "ctc_processor_seconds",
                    "ctc_encoder_seconds",
                    "ctc_head_seconds",
                    "ctc_decode_seconds",
                    "anchor_query_seconds",
                    "hotword_matching_seconds",
                    "hotword_sorting_seconds",
                    "hotword_selection_seconds",
                    "pure_retrieval_seconds",
                    "model_retrieval_seconds",
                )
            },
        }

    return calls, factory, retrieve


def test_two_sources_four_exports_resume_and_reject_changed_inputs(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path)
    build_sd_table(tmp_path, config, tmp_path / "tables")
    calls, factory, retrieve = mocks(monkeypatch)
    kw = {"detector_factory": factory, "retrieve_function": retrieve}
    output = tmp_path / "run"
    audit = run_sd_retrieval(tmp_path, config, output, audit_only=True, **kw)
    assert audit["status"] == "audit_pass" and not output.exists()
    assert all(v["expected_target_sample_pairs"] == 1 for v in audit["target_presence"].values())
    assert calls == {"loads": 0, "retrieved": 0}
    report = run_sd_retrieval(tmp_path, config, output, **kw)
    assert report["sample_count"] == 4 and report["delivery_file_count"] == 4
    assert calls == {"loads": 1, "retrieved": 4}
    for name, group in report["groups"].items():
        assert group["top5"]["expected_target_sample_pairs"] == 1
        assert group["top5"]["gated_target_hits"] == 0
        assert group["top7"]["gated_target_hits"] == 1
        for k in (5, 7):
            data = json.loads((output / "delivery" / f"{name}_top{k}.json").read_text())
            assert set(data) == {"same", "empty"} and data["empty"] == []
            assert len(data["same"]) == k
            assert all(set(r) == {"word", "phoneme"} for r in data["same"])
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        assert _sha(output / filename) == digest
    assert run_sd_retrieval(tmp_path, config, output, resume=True, **kw) == report
    assert calls == {"loads": 1, "retrieved": 4}
    with pytest.raises(FileExistsError):
        run_sd_retrieval(tmp_path, config, output, **kw)
    shard = next((output / "mls_portuguese/sample_shards").glob("*.json"))
    saved = shard.read_bytes()
    damaged = json.loads(saved)
    damaged["selected_hotword_ids"] = ["invented"]
    write(shard, damaged)
    with pytest.raises(ValueError, match="corrupted/changed sample"):
        run_sd_retrieval(tmp_path, config, output, resume=True, **kw)
    shard.write_bytes(saved)
    audio = tmp_path / "mls_portuguese/audio/same.wav"
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="resume input/config/code"):
        run_sd_retrieval(tmp_path, config, output, resume=True, **kw)
    assert calls["retrieved"] == 4


def test_wrong_head_and_source_mismatch_no_model_or_output(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path)
    build_sd_table(tmp_path, config, tmp_path / "tables")
    calls, factory, retrieve = mocks(monkeypatch)
    (tmp_path / "head.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256"):
        run_sd_retrieval(tmp_path, config, tmp_path / "run", detector_factory=factory)
    assert calls["loads"] == 0 and not (tmp_path / "run").exists()
    (tmp_path / "head.pt").write_bytes(b"mock head")
    (tmp_path / "mls_portuguese/audio/same.wav").unlink()
    with pytest.raises(ValueError, match="audio/transcript mismatch"):
        run_sd_retrieval(
            tmp_path, config, tmp_path / "run", detector_factory=factory, retrieve_function=retrieve
        )
    assert calls["loads"] == 0 and not (tmp_path / "run").exists()
