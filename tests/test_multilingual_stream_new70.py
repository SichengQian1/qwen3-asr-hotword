from __future__ import annotations

import json
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.external_keyword_retrieval import KeywordBundle
from qwen_hotword.inference.multilingual_stream_run import run_multilingual_stream
from qwen_hotword.inference.reviewed_joint import joint_bundle
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def test_new70_retrieves_actual_prefixes_and_keeps_inactive_targets(tmp_path: Path) -> None:
    old_path = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
    vocab_path = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json"
    old_vocab, vocab = load_phoneme_vocab(old_path), load_phoneme_vocab(vocab_path)
    old_entries = tuple(
        HotwordEntry(
            word,
            "en",
            word,
            word,
            (word,),
            phone,
            (phone,),
            (old_vocab.token_to_id[phone],),
            "fixture",
            1,
        )
        for word, phone in (("cat", "a"), ("dog", "b"))
    )
    bundle, old_variants, _ = joint_bundle(
        {
            "en": KeywordBundle(old_entries, {}, {}),
            "es": KeywordBundle((), {}, {}),
            "pt": KeywordBundle((), {}, {}),
        }
    )
    baseline = tmp_path / "baseline"
    model = tmp_path / "Qwen3-ASR-1.7B"
    metadata = [
        write(model / n, {})
        for n in ("config.json", "model.safetensors.index.json", "preprocessor_config.json")
    ]
    groups, samples, gates = {}, [], {}
    for lang in ("en", "es", "pt"):
        audio = tmp_path / f"{lang}.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            handle.writeframes(b"\x01\x00" * 48000)
        samples.append(
            {
                "group": lang,
                "key": f"old@@{lang}",
                "language": lang,
                "sample_id": "sample",
                "audio_path": str(audio),
                "audio_sha256": _sha(audio),
                "frames": 48000,
                "sample_rate": 16000,
                "channels": 1,
                "step_count": 2,
                "duration_seconds": 3.0,
            }
        )
        groups[lang] = {
            "language": lang,
            "target_ids": [e.hotword_id for e in bundle.entries],
            "records": [{"sample_id": "sample", "reference_text": "cat dog"}],
        }
        gates[lang] = {
            "threshold": 0.75,
            "maximum_edit_ratio": 0.35,
            "minimum_posterior_confidence": 0.5,
            "minimum_top1_margin": 0,
        }
    original = write(
        baseline / "run_config.json",
        {
            "configuration": {"vocab": str(old_path), "model": str(model)},
            "groups": groups,
        },
    )
    index = baseline / "indexed_pronunciations.jsonl"
    index.write_text("\n".join(json.dumps(e.to_dict()) for e in old_variants) + "\n")
    source = tmp_path / "old_stream"
    old = write(
        source / "run_config.json",
        {
            "effect": "same-step",
            "samples": samples,
            "gate": gates,
            "code_sha256": {},
            "inputs": {str(p): {"sha256": _sha(p)} for p in [original, index, *metadata]},
            "dataset_counts": {lang: 1 for lang in groups},
            "selected_samples": 3,
            "prefix_count": 6,
            "audio_pairing": {},
            "source_configuration": {
                "baseline_run": str(baseline),
                "vocab_sha256": _sha(old_path),
                "table_sha256": "old-table",
            },
        },
    )
    write(
        source / "report.json",
        {
            "status": "completed",
            "effect": "same-step",
            "sample_count": 3,
            "prefix_count": 6,
            "run_config_sha256": _sha(old),
        },
    )
    seal(source)
    new = tmp_path / "new70"
    head_report = write(new / "head/report.json", {"best_epoch": 23})
    head = new / "head/ctc_head_best.pt"
    head.write_bytes(b"fixture new checkpoint")
    write(
        new / "training_completed.json",
        {
            "status": "fresh_head_training_completed",
            "num_classes": 100,
            "head_initialization": "random",
            "checkpoint_sha256": _sha(head),
            "report_sha256": _sha(head_report),
            "best_epoch": 23,
        },
    )
    phone = vocab.tokens[90]
    variant = {
        **old_entries[0].to_dict(),
        "hotword_id": "v03_cat",
        "pronunciation": phone,
        "phoneme_tokens": [phone],
        "token_ids": [90],
        "validation_occurrences": 0,
    }
    new_index = new / "indexed_pronunciations.v03.jsonl"
    new_index.write_text(json.dumps(variant) + "\n")
    config = write(
        tmp_path / "new.json",
        {
            "source_stream": str(source),
            "new70_source": str(new),
            "source_run_config_sha256": _sha(old),
            "checkpoint_sha256": _sha(head),
            "index_sha256": _sha(new_index),
            "vocab": str(vocab_path),
            "vocab_sha256": _sha(vocab_path),
            "dataset_counts": {lang: 1 for lang in groups},
            "prefix_count": 6,
            "declared_surfaces": 2,
            "active_surfaces": 1,
            "pronunciations": 1,
            "full_run_authorization": "fixture",
        },
    )
    observed = []

    def factory(**kwargs):
        assert kwargs["checkpoint"] == head and len(kwargs["vocab"].tokens) == 100
        assert [e.hotword_id for e in kwargs["hotwords"]] == ["v03_cat"]
        return SimpleNamespace(language=kwargs["language"])

    def retrieve(detector, waveform, vocab):
        observed.append((detector.language, len(waveform)))
        matches = (
            [{"hotword_id": "v03_cat", "score": 0.9, "edit_ratio": 0, "posterior_confidence": 0.9}]
            if len(waveform) == 48000
            else []
        )
        return {
            "raw_ranked_matches": matches,
            "shortlist_candidates": len(matches),
            "timing": {"pure_retrieval_seconds": 0, "model_retrieval_seconds": 0},
        }

    output = tmp_path / "delivery_run"
    args = dict(effect="same-step", detector_factory=factory, retrieve_function=retrieve)
    report = run_multilingual_stream(tmp_path, config, output, **args)
    assert report["index_sha256"] == _sha(new_index) and report["prefix_count"] == 6
    assert [length for _, length in observed] == [32000, 48000] * 3
    for lang in groups:
        scored = report["by_language"][lang]["RAG_top7_stream"]["last_context"]
        assert scored["target_hits"] == 1 and scored["expected_target_sample_pairs"] == 2
    contexts = json.loads((output / "delivery/stream_contexts.json").read_text())
    assert contexts["RAG_top7_stream"]["old@@pt"]["sample"] == [
        [],
        [
            {"word": "cat", "phoneme": phone, "score": 0.9},
        ],
    ]
    assert run_multilingual_stream(tmp_path, config, output, resume=True, **args) == report
    assert len(observed) == 6
    with pytest.raises(ValueError, match="same-step timing"):
        run_multilingual_stream(
            tmp_path, config, tmp_path / "bad", effect="next-step", audit_only=True
        )
    Path(samples[0]["audio_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_multilingual_stream(
            tmp_path, config, tmp_path / "bad", effect="same-step", audit_only=True
        )
