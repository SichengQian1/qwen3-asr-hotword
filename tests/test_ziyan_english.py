from __future__ import annotations

import itertools
import json
import string
import subprocess
import sys
import wave
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation import ziyan_assets as assets
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import combine_variants
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.ziyan_retrieval import run
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


@pytest.fixture(autouse=True)
def audio_header_dependency(monkeypatch):
    # No SoundFile installation required locally; read real test WAV headers via stdlib.
    def info(path):
        with wave.open(path, "rb") as f:
            frames, rate, channels = f.getnframes(), f.getframerate(), f.getnchannels()
        return SimpleNamespace(
            frames=frames, samplerate=rate, channels=channels, duration=frames / rate
        )

    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(info=info))
    monkeypatch.setattr(external, "_git_commit", lambda: "test-code-identity")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def variant(word, language, role="target", pronunciation="t a"):
    phones = tokenize_ipa_to_vocab(pronunciation, load_phoneme_vocab(VOCAB))
    return {
        "surface": word,
        "normalized": word,
        "language": language,
        "decision": "retained",
        "role": role,
        "pronunciation": pronunciation,
        "phoneme_tokens": phones.tokens,
        "token_ids": phones.token_ids,
        "source_groups": [],
    }


def setup(root):
    targets = [
        "target" + "".join(p)
        for p in itertools.islice(itertools.product(string.ascii_lowercase, repeat=3), 300)
    ]
    target_path = root / "config/targets.txt"
    target_path.parent.mkdir(parents=True)
    target_path.write_text("\n".join(targets) + "\n")
    references = [" ".join(targets), " ".join(targets[:2]) + " " + targets[0], "shared filler"]
    records = []
    for n, reference in enumerate(references):
        path = root / f"audio/{['airport', 'spoken', 'hotel'][n]}_{n}.wav"
        path.parent.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(bytes([n, 0]) * 16000)
        records.append({"audios": [str(path)], "response": reference})
    dataset = root / "data/input.json"
    write(dataset, records)
    old_variants = [
        variant("obsolete", "en"),
        variant("shared", "en"),
        variant("shared", "es"),
        variant(targets[0], "en", "filler"),
        variant("espanol", "es"),
        variant("brasil", "pt"),
    ]
    source = root / "source"
    write(source / "reviewed_keyword_table.json", {"entries": combine_variants(old_variants)})
    digest = _sha(source / "reviewed_keyword_table.json")
    write(
        source / "report.json",
        {
            "status": "completed",
            "candidate_scope": "joint_multilingual",
            "review": {
                "reviewed_table_sha256": "original targets SHA",
                "expanded_table_sha256": digest,
                "active_unique_surfaces_per_audio": 5,
            },
        },
    )
    seal(source)
    model = root / "models/g2p/en.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"mock MFA zip, never executed")
    (root / "model").mkdir(exist_ok=True)
    (root / "model/head.pt").write_bytes(b"mock Head, never loaded")
    write(root / "model/config.json", {})
    cfg = {
        "dataset": str(dataset),
        "dataset_sha256": _sha(dataset),
        "expected_samples": 3,
        "targets": str(target_path),
        "targets_sha256": _sha(target_path),
        "expected_targets": 300,
        "source_run": "source",
        "source_table_sha256": digest,
        "source_surfaces": 5,
        "old_english_targets": 2,
        "g2p_model": "models/g2p/en.zip",
        "g2p_model_sha256": _sha(model),
        "vocab": str(VOCAB),
        "vocab_sha256": _sha(VOCAB),
        "model": str(root / "model"),
        "checkpoint": "model/head.pt",
        "checkpoint_sha256": _sha(root / "model/head.pt"),
        "assets": "assets",
        "output": "results",
    }
    config = root / "config/input.json"
    write(config, cfg)
    return config, targets, old_variants


def mock_mfa(command, **kwargs):
    assert command[1:6] == ["g2p", "--num_jobs", "2", "--num_pronunciations", "1"]
    words = Path(command[-3]).read_text().splitlines()
    Path(command[-1]).write_text("".join(w + "\tt a\n" for w in words))
    kwargs["stdout"].write("mock MFA only\n")


def test_build_preserves_fillers_and_foreign_variants_and_shared_surfaces(tmp_path, monkeypatch):
    config, targets, old = setup(tmp_path)
    before = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(assets.subprocess, "run", mock_mfa)
    report = assets.build(tmp_path, config, tmp_path / "assets", 2, "mfa")
    assert report["status"] == "completed" and report["table_ready"]
    assert report["selection"]["removed_english_target_variants"] == 2
    assert report["selection"]["retained_fillers"] == 1
    assert report["selection"]["new_targets_already_in_retained_table"] == [targets[0]]
    table = json.loads((tmp_path / "assets/keyword_table.json").read_text())
    actual = [v for e in table["entries"] for v in e["variants"]]
    assert all(v in actual for v in old if v["language"] != "en" or v["role"] == "filler")
    assert old[0] not in actual and old[1] not in actual
    assert report["retained_filler_reference_overlap"] == [targets[0]]
    assert report["target_presence"]["expected_target_sample_pairs"] == 302
    assert report["target_presence"]["topk_capacity_upper_bound"]["top7"] == 9 / 302
    assert all(_sha(p) == digest for p, digest in before.items())
    with pytest.raises(ValueError, match="new assets"):
        assets.build(tmp_path, config, tmp_path / "assets", 2, "mfa")


@pytest.mark.parametrize("fault", ["missing", "ambiguous", "oov", "process", "changed_input"])
def test_g2p_failure_never_publishes_partial_targets(tmp_path, monkeypatch, fault):
    config, _, _ = setup(tmp_path)

    def generate(command, **kwargs):
        mock_mfa(command, **kwargs)
        path = Path(command[-1])
        if fault == "missing":
            path.write_text("\n".join(path.read_text().splitlines()[1:]) + "\n")
        elif fault == "ambiguous":
            with path.open("a") as f:
                f.write("targetaaa\tm a\n")
        elif fault == "oov":
            path.write_text(path.read_text().replace("t a", "☃"))
        elif fault == "process":
            raise subprocess.CalledProcessError(1, command)
        else:
            (tmp_path / "models/g2p/en.zip").write_bytes(b"changed")

    monkeypatch.setattr(assets.subprocess, "run", generate)
    report = assets.build(tmp_path, config, tmp_path / "assets", 2, "mfa")
    assert report["status"] == "failed" and not report["table_ready"]
    assert not (tmp_path / "assets/keyword_table.json").exists()
    assert not (tmp_path / "assets/targets.json").exists()


@pytest.mark.parametrize("fault", ["duplicates", "multiple_audio", "relative", "empty", "prefix"])
def test_json_contract_rejects_ambiguous_records(tmp_path, fault):
    config, _, _ = setup(tmp_path)
    cfg = json.loads(config.read_text())
    p = Path(cfg["dataset"])
    data = json.loads(p.read_text())
    if fault == "duplicates":
        data[1] = deepcopy(data[0])
    elif fault == "multiple_audio":
        data[0]["audios"] *= 2
    elif fault == "relative":
        data[0]["audios"] = ["relative.wav"]
    elif fault == "empty":
        data[0]["response"] = " "
    else:
        data[0]["response"] = "language English<asr_text>test"
    write(p, data)
    with pytest.raises(ValueError):
        assets.read_dataset(p, 3)


def fake_retrieval(detector, waveform, vocab):
    if "airport" in waveform:
        chosen = [e for e in detector.hotwords if e.surface.startswith("target")][:8]
    elif "spoken" in waveform:
        chosen = [e for e in detector.hotwords if e.surface in {"targetaaa", "targetaab", "shared"}]
    else:
        chosen = []
    matches = [
        {
            "hotword_id": e.hotword_id,
            "surface": e.surface,
            "score": 0.99 - n * 0.01,
            "edit_ratio": 0.0,
            "posterior_confidence": 0.9,
        }
        for n, e in enumerate(chosen)
    ]
    return {
        "raw_ranked_matches": matches,
        "selected_matches": matches[:5],
        "shortlist_candidates": len(matches),
        "postings_visited": 2,
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


def test_inference_exports_exact_top5_top7_and_safe_resume(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(assets.subprocess, "run", mock_mfa)
    assets.build(tmp_path, config, tmp_path / "assets", 2, "mfa")
    monkeypatch.setattr(
        external,
        "_load_audio",
        lambda path: (path.name, {"audio_load_seconds": 0.01, "duration_seconds": 1.0}),
    )
    calls = []

    def factory(**kwargs):
        assert kwargs["language"] == "English" and kwargs["device"] == "cuda:0"
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        assert not any(e.surface == "obsolete" for e in kwargs["hotwords"])
        calls.append(1)
        return SimpleNamespace(**kwargs)

    kwargs = {"detector_factory": factory, "retrieve_function": fake_retrieval}
    output = tmp_path / "results"
    audit = run(tmp_path, config, tmp_path / "assets", output, audit_only=True, **kwargs)
    assert audit["status"] == "audit_pass" and not output.exists() and not calls
    result = run(tmp_path, config, tmp_path / "assets", output, **kwargs)
    assert result["metrics"]["top5"]["expected_target_sample_pairs"] == 302
    assert result["metrics"]["top5"]["gated_target_hits"] == 7
    assert result["metrics"]["top7"]["gated_target_hits"] == 9
    assert result["delivery_file_count"] == 2 and len(calls) == 1
    assert set(result["by_scenario"]) == {"airport", "spoken", "hotel"}
    for k in (5, 7):
        delivery = json.loads((output / "delivery" / f"{assets.NAME}_top{k}.json").read_text())
        assert len(delivery) == 3 and len(delivery["airport_0"]) == k
        assert set(delivery["airport_0"][0]) == {"word", "phoneme"}
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert _sha(output / name) == digest
    run(tmp_path, config, tmp_path / "assets", output, resume=True, **kwargs)
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        run(tmp_path, config, tmp_path / "assets", output, **kwargs)
    shard = next((output / assets.NAME / "sample_shards").glob("*.json"))
    data = json.loads(shard.read_text())
    data["reference_text"] = "changed"
    write(shard, data)
    with pytest.raises(ValueError, match="corrupt/changed"):
        run(tmp_path, config, tmp_path / "assets", output, resume=True, **kwargs)


def test_changed_audio_refuses_resume_and_input_identity_fails_before_mfa(tmp_path, monkeypatch):
    config, _, _ = setup(tmp_path)
    monkeypatch.setattr(assets.subprocess, "run", mock_mfa)
    assets.build(tmp_path, config, tmp_path / "assets", 2, "mfa")
    output = tmp_path / "results"
    from qwen_hotword.inference.ziyan_retrieval import prepare

    plan, _ = prepare(tmp_path, config, tmp_path / "assets")
    write(output / "run_config.json", plan)
    with (tmp_path / "audio/airport_0.wav").open("ab") as f:
        f.write(b"changed bytes")
    with pytest.raises(ValueError, match="resume audio/targets"):
        run(tmp_path, config, tmp_path / "assets", output, resume=True)
    (tmp_path / "models/g2p/en.zip").write_bytes(b"changed model")
    monkeypatch.setattr(assets.subprocess, "run", lambda *a, **k: pytest.fail("must not run"))
    with pytest.raises(ValueError, match="SHA256"):
        assets.build(tmp_path, config, tmp_path / "new_assets", 2, "mfa")


def test_frozen_actual_target_list_count_and_basic_exclusions():
    cfg = json.loads(Path("configs/ziyan_english.workzone.json").read_text())
    path = Path(cfg["targets"])
    words = path.read_text().splitlines()
    assert len(words) == cfg["expected_targets"] == 372 and words == sorted(set(words))
    assert _sha(path) == cfg["targets_sha256"]
    assert not {"a", "the", "me", "i", "please", "twenty", "im", "ive", "fourty"} & set(words)
