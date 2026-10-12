from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation.english_keywords import DATASETS, build_english_table
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.english_retrieval import run_english_retrieval
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def fixture(root):
    sources = {}
    for name, target in zip(DATASETS, ("Target", "Colombia", "Mexico"), strict=True):
        base = root / name / "en"
        write(
            base / "en_keyword_bias_phoneme.json",
            {
                "language": "English",
                "keyword_sets": {"all_test": ["SHARED", target]},
                "keyword_phonemes": {"SHARED": "/a b/", target: "/b a/"},
            },
        )
        (base / "transcripts.txt").write_text(
            f"same.wav\t{target} can't well-known ＡＰＰＬＥ.\n"
            f"empty\tOnly {name.removeprefix('en_')} visible\n",
            encoding="utf-8",
        )
        (base / "wav").mkdir()
        (base / "wav/same.wav").write_bytes(b"positive")
        (base / "wav/empty.wav").write_bytes(b"empty")
        sources[name] = {
            "primary": str(base / "en_keyword_bias_phoneme.json"),
            "transcripts": str(base / "transcripts.txt"),
            "audio": str(base / "wav"),
        }
    words = ["filler" + chr(97 + i) for i in range(16)]
    words += [
        "target",
        "apple",
        "can",
        "known",
        "colombia",
        "mexico",
        "chile",
        "shared",
        "missing",
        "ambiguous",
        "oov",
        "one-two",
        "x",
    ]
    train = root / "train.jsonl"
    train.write_text(
        json.dumps({"split": "train", "language": "en-US", "text": " ".join(words)}) + "\n"
    )
    dictionary = root / "mfa.dict"
    dictionary.write_text(
        "\n".join(w + " a b" for w in words if w != "missing") + "\nambiguous b a\noov ☃\n"
    )
    write(root / "model/config.json", {})
    (root / "head.pt").write_bytes(b"mock head")
    cfg = {
        "primary_keyword_set": "all_test",
        "target_size": 12,
        "seed": "fixture",
        "training_manifest": str(train),
        "train_sha256": _sha(train),
        "dictionary": str(dictionary),
        "dictionary_sha256": _sha(dictionary),
        "vocab": str(VOCAB),
        "vocab_sha256": _sha(VOCAB),
        "tables": "tables",
        "output": "run",
        "model": "model",
        "checkpoint": "head.pt",
        "checkpoint_sha256": _sha(root / "head.pt"),
        "sources": sources,
    }
    write(root / "config.json", cfg)
    return root / "config.json", cfg


def test_union_exclusion_all_three_transcripts_and_determinism(tmp_path):
    config, cfg = fixture(tmp_path)
    report = build_english_table(tmp_path, config, tmp_path / "tables")
    assert (report["mandatory_union"], report["fillers"], report["total"]) == (4, 8, 12)
    assert report["all_targets_retained"] and report["selected_filler_transcript_overlap"] == 0
    assert report["filler_decisions"]["present_in_any_test_transcript"] == 4
    assert report["filler_decisions"]["dictionary_missing_or_ambiguous"] == 3
    table = json.loads((tmp_path / "tables/keyword_bias_phoneme.json").read_text())
    assert table["keyword_phonemes"]["Target"] == "/b a/"
    assert set(table["keyword_sets"]["all_keywords"][:4]) == {
        "Target",
        "Colombia",
        "Mexico",
        "SHARED",
    }
    assert all(w.startswith("filler") for w in table["keyword_sets"]["all_keywords"][4:])
    build_english_table(tmp_path, config, tmp_path / "repeat")
    assert (tmp_path / "tables/keyword_bias_phoneme.json").read_bytes() == (
        tmp_path / "repeat/keyword_bias_phoneme.json"
    ).read_bytes()
    for line in (tmp_path / "tables/sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(tmp_path / "tables" / name) == digest
    with pytest.raises(FileExistsError):
        build_english_table(tmp_path, config, tmp_path / "tables")
    cfg["target_size"] = 100
    write(config, cfg)
    with pytest.raises(ValueError, match="insufficient"):
        build_english_table(tmp_path, config, tmp_path / "short")
    assert not (tmp_path / "short").exists()


@pytest.mark.parametrize("failure", ["conflict", "oov", "missing", "language", "duplicate_json"])
def test_bad_mandatory_labels_fail_before_publication(tmp_path, failure):
    config, cfg = fixture(tmp_path)
    primary = Path(cfg["sources"]["en_colombia"]["primary"])
    data = json.loads(primary.read_text())
    if failure == "conflict":
        data["keyword_phonemes"]["SHARED"] = "b b"
    elif failure == "oov":
        data["keyword_phonemes"]["Colombia"] = "☃"
    elif failure == "missing":
        del data["keyword_phonemes"]["Colombia"]
    elif failure == "language":
        data["language"] = "Spanish"
    write(primary, data)
    if failure == "duplicate_json":
        primary.write_text('{"language":"English","language":"English"}')
    with pytest.raises(ValueError):
        build_english_table(tmp_path, config, tmp_path / "tables")
    assert not (tmp_path / "tables").exists()


def test_same_target_alias_is_deduplicated_with_provenance(tmp_path):
    config, cfg = fixture(tmp_path)
    path = Path(cfg["sources"]["en_mexico"]["primary"])
    data = json.loads(path.read_text())
    data["keyword_sets"]["all_test"][0] = "Shared"
    data["keyword_phonemes"]["Shared"] = data["keyword_phonemes"].pop("SHARED")
    write(path, data)
    report = build_english_table(tmp_path, config, tmp_path / "tables")
    assert report["mandatory_union"] == 4
    assert report["normalization_aliases"] == [
        {"dataset": "en_mexico", "surface": "Shared", "canonical_surface": "SHARED"}
    ]


@pytest.mark.parametrize("change", ["sha", "split", "language", "empty", "target_overflow"])
def test_source_identity_and_train_scope(tmp_path, change):
    config, cfg = fixture(tmp_path)
    path = Path(cfg["training_manifest"])
    row = json.loads(path.read_text())
    if change == "split":
        row["split"] = "validation"
    if change == "language":
        row["language"] = "es"
    path.write_text("" if change == "empty" else json.dumps(row) + " \n")
    if change != "sha":
        cfg["train_sha256"] = _sha(path)
    if change == "target_overflow":
        cfg["target_size"] = 3
    write(config, cfg)
    with pytest.raises(ValueError):
        build_english_table(tmp_path, config, tmp_path / "tables")
    assert not (tmp_path / "tables").exists()


def mocks(monkeypatch):
    calls = {"loads": 0, "retrieved": 0}

    def factory(**kw):
        assert kw["language"] == "English" and kw["device"] == "cuda:0"
        assert len(kw["hotwords"]) == 12
        assert all(e.language == "en" for e in kw["hotwords"])
        calls["loads"] += 1
        return SimpleNamespace(**kw)

    monkeypatch.setattr(
        external,
        "_load_audio",
        lambda path: (path.read_bytes(), {"duration_seconds": 1.0, "audio_load_seconds": 0.01}),
    )

    def retrieve(detector, audio, vocab):
        calls["retrieved"] += 1
        entries = sorted(detector.hotwords, key=lambda e: e.surface == "Target")
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
                key: 0.01
                for key in (
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


def test_six_deliveries_exact_replay_no_hits_and_same_stems(tmp_path, monkeypatch):
    config, cfg = fixture(tmp_path)
    build_english_table(tmp_path, config, tmp_path / "tables")
    calls, factory, retrieve = mocks(monkeypatch)
    kw = {"detector_factory": factory, "retrieve_function": retrieve}
    output = tmp_path / "run"
    audit = run_english_retrieval(tmp_path, config, output, audit_only=True, **kw)
    assert audit["samples"] == {name: 2 for name in DATASETS}
    assert calls == {"loads": 0, "retrieved": 0} and not output.exists()
    report = run_english_retrieval(tmp_path, config, output, **kw)
    assert report["delivery_file_count"] == 6 and report["sample_count"] == 6
    assert calls == {"loads": 1, "retrieved": 6}
    assert report["groups"]["en_chile"]["top5"]["gated_target_hits"] == 0
    assert report["groups"]["en_chile"]["top7"]["gated_target_hits"] == 1
    for name in DATASETS:
        for k in (5, 7):
            table = json.loads((output / "delivery" / f"{name}_top{k}.json").read_text())
            assert set(table) == {"same", "empty"} and table["empty"] == []
            assert len(table["same"]) == k
            assert all(set(item) == {"word", "phoneme"} for item in table["same"])
            view = report["groups"][name][f"top{k}"]
            assert "full_3000_view" in view and "full_4000_view" not in view
        summary = json.loads((output / name / "evaluation_summary.json").read_text())
        assert summary["table_filler_exclusion_used_transcripts"]
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        assert _sha(output / filename) == digest
    assert run_english_retrieval(tmp_path, config, output, resume=True, **kw) == report
    assert calls == {"loads": 1, "retrieved": 6}
    with pytest.raises(FileExistsError):
        run_english_retrieval(tmp_path, config, output, **kw)
    shard = next((output / "en_chile/sample_shards").glob("*.json"))
    row = json.loads(shard.read_text())
    row["selected_hotword_ids"] = ["invented"]
    write(shard, row)
    with pytest.raises(ValueError, match="corrupted/changed sample"):
        run_english_retrieval(tmp_path, config, output, resume=True, **kw)


@pytest.mark.parametrize("kind", ["transcript", "primary", "head", "table", "audio"])
def test_resume_rejects_changed_inputs(tmp_path, monkeypatch, kind):
    config, cfg = fixture(tmp_path)
    build_english_table(tmp_path, config, tmp_path / "tables")
    _, factory, retrieve = mocks(monkeypatch)
    kw = {"detector_factory": factory, "retrieve_function": retrieve}
    output = tmp_path / "run"
    run_english_retrieval(tmp_path, config, output, **kw)
    paths = {
        "transcript": Path(cfg["sources"]["en_mexico"]["transcripts"]),
        "primary": Path(cfg["sources"]["en_mexico"]["primary"]),
        "head": tmp_path / "head.pt",
        "table": tmp_path / "tables/keyword_bias_phoneme.json",
        "audio": tmp_path / "en_mexico/en/wav/same.wav",
    }
    path = paths[kind]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256|identity differs"):
        run_english_retrieval(tmp_path, config, output, resume=True, **kw)


def test_mismatched_audio_transcript_blocks_before_model_load(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path)
    build_english_table(tmp_path, config, tmp_path / "tables")
    (tmp_path / "en_chile/en/wav/surprise.wav").write_bytes(b"unknown")
    calls, factory, _ = mocks(monkeypatch)
    with pytest.raises(ValueError, match="audio/transcript mismatch"):
        run_english_retrieval(tmp_path, config, tmp_path / "run", detector_factory=factory)
    assert calls["loads"] == 0 and not (tmp_path / "run").exists()


def test_interrupted_run_resumes_only_missing_samples(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path)
    build_english_table(tmp_path, config, tmp_path / "tables")
    calls, factory, retrieve = mocks(monkeypatch)

    def interrupt(detector, audio, vocab):
        if calls["retrieved"] == 1:
            raise RuntimeError("interrupted")
        return retrieve(detector, audio, vocab)

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted"):
        run_english_retrieval(
            tmp_path, config, output, detector_factory=factory, retrieve_function=interrupt
        )
    assert not (output / "report.json").exists()
    report = run_english_retrieval(
        tmp_path, config, output, resume=True, detector_factory=factory, retrieve_function=retrieve
    )
    assert calls["retrieved"] == 6 and report["delivery_file_count"] == 6


def test_build_exactly_3000_unique_entries(tmp_path):
    config, cfg = fixture(tmp_path)
    # Exercise the actual capacity without a model, audio decoding, or production data.
    words = [
        "word" + chr(97 + i // 676) + chr(97 + i // 26 % 26) + chr(97 + i % 26) for i in range(3100)
    ]
    train, dictionary = Path(cfg["training_manifest"]), Path(cfg["dictionary"])
    train.write_text(json.dumps({"split": "train", "language": "en", "text": " ".join(words)}))
    dictionary.write_text("\n".join(w + " a b" for w in words))
    cfg.update(target_size=3000, train_sha256=_sha(train), dictionary_sha256=_sha(dictionary))
    write(config, cfg)
    report = build_english_table(tmp_path, config, tmp_path / "tables")
    assert report["total"] == 3000 and report["fillers"] == 2996
    bundle = external.load_keyword_bias(
        tmp_path / "tables/keyword_bias_phoneme.json",
        vocab=load_phoneme_vocab(VOCAB),
        keyword_set="all_keywords",
        language="en",
    )
    assert len({e.normalized for e in bundle.entries}) == 3000


@pytest.mark.parametrize("language", ["English", "en", "en-US"])
def test_english_loader_aliases_keep_phonemes(tmp_path, language):
    path = tmp_path / "english.json"
    write(
        path,
        {
            "language": language,
            "keyword_sets": {"all_test": ["'Touch'", "Air Force"]},
            "keyword_phonemes": {"'Touch'": "/tʰ ɐ tʃ/", "Air Force": "/ɛ ɹ f ɒ ɹ s/"},
        },
    )
    bundle = external.load_keyword_bias(
        path, vocab=load_phoneme_vocab(VOCAB), keyword_set="all_test", language="en"
    )
    assert bundle.entries[0].hotword_id == "external_en_0000"
    assert bundle.entries[0].normalized == "touch"
    assert bundle.phonemes["'Touch'"] == "/tʰ ɐ tʃ/"


def noisy_fixture(tmp_path, monkeypatch):
    config, cfg = fixture(tmp_path)
    build_english_table(tmp_path, config, tmp_path / "tables")
    calls, factory, retrieve = mocks(monkeypatch)
    kw = {"detector_factory": factory, "retrieve_function": retrieve}
    clean = run_english_retrieval(tmp_path, config, tmp_path / "run", **kw)
    noisy = json.loads(json.dumps(cfg))
    noisy.update(
        audio_root=str(tmp_path / "noisy"),
        comparison_run_config="run/run_config.json",
        output="noise_run",
    )
    for name in DATASETS:
        audio = tmp_path / "noisy" / name / "en/wav"
        audio.mkdir(parents=True)
        for stem in ("same", "empty"):
            (audio / f"{stem}.wav").write_bytes(f"noise {name} {stem}".encode())
    path = tmp_path / "noise_config.json"
    write(path, noisy)
    return path, noisy, clean, calls, kw


def test_noise_reuses_table_reruns_audio_and_delivers_six_files(tmp_path, monkeypatch):
    config, cfg, clean, calls, kw = noisy_fixture(tmp_path, monkeypatch)
    clean_checks = (tmp_path / "run/sha256.txt").read_bytes()
    table_checks = (tmp_path / "tables/sha256.txt").read_bytes()
    output = tmp_path / cfg["output"]
    audit = run_english_retrieval(tmp_path, config, output, audit_only=True, **kw)
    assert not output.exists() and calls == {"loads": 1, "retrieved": 6}
    assert audit["audio_comparison"] == {
        name: {"changed_files": 2, "identical_files": 0} for name in DATASETS
    }
    assert audit["snr_verified"] is False
    report = run_english_retrieval(tmp_path, config, output, **kw)
    assert report["sample_count"] == 6 and report["delivery_file_count"] == 6
    assert report["checkpoint_sha256"] == clean["checkpoint_sha256"]
    assert report["table_sha256"] == clean["table_sha256"]
    assert report["audio_comparison"] == audit["audio_comparison"]
    assert calls == {"loads": 2, "retrieved": 12}
    for name in DATASETS:
        for k in (5, 7):
            delivered = json.loads((output / "delivery" / f"{name}_top{k}.json").read_text())
            assert set(delivered) == {"same", "empty"}
            # Clean 'empty' audio had []; noisy mock audio is retrieved afresh.
            assert len(delivered["empty"]) == k
            assert all(set(item) == {"word", "phoneme"} for item in delivered["empty"])
            assert report["groups"][name][f"top{k}"]["expected_target_sample_pairs"] == (
                clean["groups"][name][f"top{k}"]["expected_target_sample_pairs"]
            )
        for path in (output / name / "sample_shards").glob("*.json"):
            row = json.loads(path.read_text())
            assert Path(row["audio_path"]).is_relative_to(tmp_path / "noisy")
    assert (tmp_path / "run/sha256.txt").read_bytes() == clean_checks
    assert (tmp_path / "tables/sha256.txt").read_bytes() == table_checks
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(output / name) == digest
    assert run_english_retrieval(tmp_path, config, output, resume=True, **kw) == report
    assert calls == {"loads": 2, "retrieved": 12}
    audio = tmp_path / "noisy/en_mexico/en/wav/same.wav"
    audio.write_bytes(b"different noisy version")
    with pytest.raises(ValueError, match="resume input/config/code identity differs"):
        run_english_retrieval(tmp_path, config, output, resume=True, **kw)


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "identical", "missing_baseline", "tampered_baseline", "head", "output"],
)
def test_noise_preflight_rejects_wrong_pairing_or_baseline(tmp_path, monkeypatch, change):
    config, cfg, _, calls, kw = noisy_fixture(tmp_path, monkeypatch)
    audio = tmp_path / "noisy/en_chile/en/wav"
    if change == "missing":
        (audio / "empty.wav").rename(audio / "empty.bak")
    elif change == "extra":
        (audio / "extra.wav").write_bytes(b"unexpected")
    elif change == "identical":
        for p in audio.glob("*.wav"):
            p.write_bytes((tmp_path / "en_chile/en/wav" / p.name).read_bytes())
    elif change == "missing_baseline":
        cfg.pop("comparison_run_config")
    elif change == "tampered_baseline":
        path = tmp_path / "run/run_config.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "head":
        (tmp_path / "another.pt").write_bytes(b"another head")
        cfg.update(checkpoint="another.pt", checkpoint_sha256=_sha(tmp_path / "another.pt"))
    else:
        cfg["output"] = "run"
    write(config, cfg)
    with pytest.raises(ValueError):
        run_english_retrieval(tmp_path, config, tmp_path / "noise_run", audit_only=True, **kw)
    assert calls == {"loads": 1, "retrieved": 6}
    assert not (tmp_path / "noise_run").exists()


def test_noise_rejects_changed_gate_even_when_inputs_match(tmp_path, monkeypatch):
    config, _, _, calls, kw = noisy_fixture(tmp_path, monkeypatch)
    original = external._run_config

    def changed(**kwargs):
        result = original(**kwargs)
        result["gate"]["threshold"] = 0.1
        return result

    monkeypatch.setattr(external, "_run_config", changed)
    with pytest.raises(ValueError, match="comparison retrieval/data contract changed"):
        run_english_retrieval(tmp_path, config, tmp_path / "noise_run", audit_only=True, **kw)
    assert calls == {"loads": 1, "retrieved": 6}


def test_noise_workzone_config_changes_only_audio_and_output():
    clean = json.loads(Path("configs/english_3000.workzone.json").read_text())
    noisy = json.loads(Path("configs/english_3000_noise10db.workzone.json").read_text())
    assert noisy.pop("audio_root").endswith("/2026-10-09-带噪10dB")
    assert noisy.pop("comparison_run_config") == clean["output"] + "/run_config.json"
    assert noisy["output"] != clean["output"]
    noisy["output"] = clean["output"]
    assert noisy == clean
