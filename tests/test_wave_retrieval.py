from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.wave_retrieval import prepare_run, run_waves
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def sign(root):
    (root / "sha256.txt").write_text(
        "".join(
            f"{_sha(p)}  {p.relative_to(root).as_posix()}\n"
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "sha256.txt"
        )
    )


def fixture(root):
    tables = root / "tables"
    write(tables / "report.json", {"status": "completed", "tables_created": True})
    inventory = {"datasets": {}}
    for lang in ("es", "pt"):
        words = [f"word{i}" for i in range(8)]
        write(
            tables / lang / "keyword_bias_phoneme.json",
            {
                "language": lang,
                "keyword_sets": {"all_keywords": words},
                "keyword_phonemes": {w: "a" for w in words},
            },
        )
        write(tables / lang / "targets_by_wave.json", {f"wave{i}": ["word6"] for i in range(1, 5)})
        for i in range(1, 5):
            base = root / f"wave{i}" / lang
            (base / "wav").mkdir(parents=True)
            (base / "wav/same.wav").write_bytes(b"positive")
            (base / "wav/empty.wav").write_bytes(b"empty")
            refs = base / "transcripts.txt"
            refs.write_text("same\tword0 word6\nempty\tnothing\n")
            inventory["datasets"][f"wave{i}/{lang}"] = {"transcripts_sha256": _sha(refs)}
    sign(tables)
    write(root / "inventory/report.json", inventory)
    sign(root / "inventory")
    write(root / "Qwen3-ASR-1.7B/config.json", {})
    (root / "head.pt").write_bytes(b"mock only")
    config = root / "run.json"
    write(
        config,
        {
            "tables": "tables",
            "inventory": "inventory",
            "model": "Qwen3-ASR-1.7B",
            "checkpoint": "head.pt",
            "checkpoint_sha256": _sha(root / "head.pt"),
            "vocab": str(VOCAB),
            "vocab_sha256": _sha(VOCAB),
            "expected_keywords": 8,
            "expected_primary": {"es": 1, "pt": 1},
            "expected_samples_per_group": 2,
        },
    )
    return config


def mocks(monkeypatch):
    calls = {"loads": [], "retrievals": 0}

    def factory(**kwargs):
        calls["loads"].append(kwargs["language"])
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        external,
        "_load_audio",
        lambda path: (path.read_bytes(), {"duration_seconds": 1.0, "audio_load_seconds": 0.01}),
    )

    def retrieve(detector, audio, _vocab):
        calls["retrievals"] += 1
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
                for i, e in enumerate(detector.hotwords)
            ]
        )
        return {
            "raw_ranked_matches": raw,
            "selected_matches": raw[:5],
            "shortlist_candidates": len(raw),
            "postings_visited": 8,
            "timing": {
                name: 0.01
                for name in (
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


def test_eight_groups_sixteen_exports_exact_replay_and_resume(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    calls, factory, retrieve = mocks(monkeypatch)
    output = tmp_path / "run"
    report = run_waves(
        tmp_path, config, output, detector_factory=factory, retrieve_function=retrieve
    )
    assert calls == {"loads": ["Spanish", "Portuguese"], "retrievals": 16}
    assert report["delivery_file_count"] == 16
    assert report["sample_count"] == 16
    assert report["top5_reproduced"] and report["top7_exact_replay"]
    for group, metrics in report["groups"].items():
        assert metrics["top5"]["gated_target_recall"] == 0.0
        assert metrics["top7"]["gated_target_recall"] == 1.0
        assert metrics["top5"]["expected_target_sample_pairs"] == 1
        assert metrics["top5"]["full_4000_view"]["expected_hotwords"] == 2
        assert metrics["top5"]["selected_words_absent_from_transcript"] == 4
        assert metrics["top7"]["selected_words_absent_from_transcript"] == 5
        for k in (5, 7):
            delivery = json.loads((output / "delivery" / f"{group}_top{k}.json").read_text())
            assert set(delivery) == {"same", "empty"}  # stems can repeat between groups
            assert delivery["empty"] == []
            assert len(delivery["same"]) == k
            assert all(set(item) == {"word", "phoneme"} for item in delivery["same"])
        row = json.loads((output / group / "retrieval_details.jsonl").read_text().splitlines()[0])
        assert row["language"] == ("es" if group.endswith("es") else "pt-BR")
    checks = output / "sha256.txt"
    for line in checks.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        assert _sha(output / filename) == digest
    original = checks.read_bytes()
    assert (
        run_waves(
            tmp_path,
            config,
            output,
            resume=True,
            detector_factory=factory,
            retrieve_function=retrieve,
        )
        == report
    )
    assert calls["retrievals"] == 16 and len(calls["loads"]) == 2
    assert checks.read_bytes() == original
    with pytest.raises(FileExistsError):
        run_waves(tmp_path, config, output, detector_factory=factory, retrieve_function=retrieve)


def test_audit_only_and_all_group_preflight_before_model_loading(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    calls, factory, retrieve = mocks(monkeypatch)
    output = tmp_path / "run"
    report = run_waves(
        tmp_path,
        config,
        output,
        audit_only=True,
        detector_factory=factory,
        retrieve_function=retrieve,
    )
    assert report["status"] == "audit_pass" and not output.exists()
    assert calls["loads"] == []
    (tmp_path / "wave4/pt/wav/same.wav").unlink()
    with pytest.raises(ValueError, match="audio/transcript mismatch"):
        run_waves(tmp_path, config, output, detector_factory=factory, retrieve_function=retrieve)
    assert calls["loads"] == [] and not output.exists()


@pytest.mark.parametrize("kind", ["keywords", "checkpoint", "transcripts"])
def test_input_sha_drift_fails_before_inference(tmp_path, monkeypatch, kind):
    config = fixture(tmp_path)
    path = {
        "keywords": "tables/es/keyword_bias_phoneme.json",
        "checkpoint": "head.pt",
        "transcripts": "wave4/pt/transcripts.txt",
    }[kind]
    with (tmp_path / path).open("a") as f:
        f.write(" ")
    calls, factory, retrieve = mocks(monkeypatch)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_waves(
            tmp_path, config, tmp_path / "run", detector_factory=factory, retrieve_function=retrieve
        )
    assert not calls["loads"]


def test_interrupted_run_resumes_missing_samples_only(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    calls, factory, retrieve = mocks(monkeypatch)
    interrupted = False

    def flaky(detector, audio, vocab):
        nonlocal interrupted
        if calls["retrievals"] == 3 and not interrupted:
            interrupted = True
            raise RuntimeError("simulated interruption")
        return retrieve(detector, audio, vocab)

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="simulated"):
        run_waves(tmp_path, config, output, detector_factory=factory, retrieve_function=flaky)
    assert len(list(output.glob("*/sample_shards/*.json"))) == 3
    report = run_waves(
        tmp_path, config, output, resume=True, detector_factory=factory, retrieve_function=retrieve
    )
    assert report["delivery_file_count"] == 16 and calls["retrievals"] == 16


@pytest.mark.parametrize("kind", ["audio", "shard", "delivery"])
def test_resume_refuses_changed_inputs_or_results(tmp_path, monkeypatch, kind):
    config = fixture(tmp_path)
    calls, factory, retrieve = mocks(monkeypatch)
    output = tmp_path / "run"
    run_waves(tmp_path, config, output, detector_factory=factory, retrieve_function=retrieve)
    if kind == "audio":
        (tmp_path / "wave1/es/wav/same.wav").write_bytes(b"different audio")
    elif kind == "shard":
        path = next(output.glob("wave1_es/sample_shards/*.json"))
        row = json.loads(path.read_text())
        row["row_sha256"] = "wrong"
        write(path, row)
    else:
        write(output / "delivery/wave1_es_top5.json", {})
    with pytest.raises(ValueError):
        run_waves(
            tmp_path,
            config,
            output,
            resume=True,
            detector_factory=factory,
            retrieve_function=retrieve,
        )
    assert calls["retrievals"] == 16


def test_spanish_entries_and_language_contract(tmp_path):
    config = fixture(tmp_path)
    plan, runtime = prepare_run(tmp_path, config)
    bundle = runtime["bundles"]["es"]
    assert all(
        e.language == "es" and e.hotword_id.startswith("external_es_") for e in bundle.entries
    )
    assert plan["groups"]["wave1_es"]["config"]["language"] == "Spanish"
    assert plan["groups"]["wave1_es"]["config"]["retrieval"]["saved_raw_rank_depth"] == 64
    with pytest.raises(ValueError, match="language mismatch"):
        external.load_keyword_bias(
            tmp_path / "tables/es/keyword_bias_phoneme.json",
            vocab=runtime["vocab"],
            keyword_set="all_keywords",
            language="pt",
        )
