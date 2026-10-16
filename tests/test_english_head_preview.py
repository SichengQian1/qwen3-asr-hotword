from __future__ import annotations

import hashlib
import zipfile
from types import SimpleNamespace

import pytest
from test_joint_fillers import setup
from test_reviewed_retrieval import VOCAB, read, retrieval, write

from qwen_hotword.evaluation.joint_fillers import build_joint_fillers
from qwen_hotword.inference import english_head_preview as preview
from qwen_hotword.inference.reviewed_retrieval import run_reviewed_retrieval
from qwen_hotword.modeling.ctc_head import TemporalUpsampleCtcHead, ctc_head_config
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha


@pytest.fixture
def case(tmp_path, monkeypatch):
    import torch

    _, cfg, build, _ = setup(tmp_path, monkeypatch)
    build_joint_fillers(tmp_path, build, tmp_path / "expanded")
    cfg.update(expanded_table="expanded/keyword_table.json", reviewed_baseline="new_run")
    config = tmp_path / "expanded_config.json"
    write(config, cfg)

    def factory(**kwargs):
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        return retrieval(detector.hotwords, waveform == b"negative")

    old = tmp_path / "expanded_run"
    report = run_reviewed_retrieval(
        tmp_path, config, old, detector_factory=factory, retrieve_function=retrieve
    )
    snapshot = tmp_path / "snapshot/head.pt"
    snapshot.parent.mkdir()
    head = TemporalUpsampleCtcHead(1024, 90, hidden_dimension=8)
    torch.save(
        {
            "head_config": ctc_head_config(head),
            "state_dict": head.state_dict(),
            "vocab_tokens": load_phoneme_vocab(VOCAB).tokens,
            "epoch_metrics": {"epoch": 21, "phoneme_error_rate": 0.064029},
        },
        snapshot,
    )
    (snapshot.parent / "checkpoint.sha256").write_text(f"{_sha(snapshot)}  snapshot/head.pt\n")
    en = report["by_language"]["en"]["current"]
    cfg = {
        "baseline_run": "expanded_run",
        "checkpoint": "snapshot/head.pt",
        "checkpoint_checksum": "snapshot/checkpoint.sha256",
        "expected_epoch": 21,
        "output": "preview",
        "baseline_checkpoint_sha256": cfg["checkpoint_sha256"],
        "table_sha256": _sha(tmp_path / "expanded/keyword_table.json"),
        "vocab_sha256": _sha(VOCAB),
        "expected_surfaces": 11,
        "datasets": dict.fromkeys(("en_chile", "en_colombia", "en_mexico"), 2),
        "expected_baseline": {
            "sample_count": 6,
            "expected_target_sample_pairs": 9,
            "top5_hits": en["top5"]["gated_target_hits"],
            "top7_hits": en["top7"]["gated_target_hits"],
        },
    }
    path = tmp_path / "preview_config.json"
    write(path, cfg)
    calls = []

    def preview_factory(**kwargs):
        assert kwargs["language"] == "English"
        assert kwargs["checkpoint"] == snapshot
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        assert len({e.normalized for e in kwargs["hotwords"]}) == 11
        calls.append("load")
        return SimpleNamespace(**kwargs)

    def preview_retrieve(detector, waveform, vocab):
        calls.append("sample")
        return retrieve(detector, waveform, vocab)

    return SimpleNamespace(
        root=tmp_path,
        config=path,
        cfg=cfg,
        calls=calls,
        kw={"detector_factory": preview_factory, "retrieve_function": preview_retrieve},
        output=tmp_path / "preview",
    )


def test_frozen_head_only_english_full_index_metrics_bundle_and_resume(case):
    # ES/PT audio can be unavailable: neither their hashes nor their waveforms are read.
    for directory in (case.root / "audio").iterdir():
        if not directory.name.startswith("en_"):
            for audio in (directory / "wav").iterdir():
                audio.unlink()
    before = {p: _sha(p) for p in case.root.rglob("*") if p.is_file()}
    audit = preview.run_preview(case.root, case.config, case.output, audit_only=True, **case.kw)
    assert audit["status"] == "audit_pass" and not audit["model_loaded"]
    assert not case.calls and not case.output.exists()
    report = preview.run_preview(case.root, case.config, case.output, **case.kw)
    assert report["sample_count"] == 6
    assert len(report["delivery_files"]) == 6
    assert set(report["groups"]) == set(case.cfg["datasets"])
    assert not (case.output / "languages/es").exists()
    assert not (case.output / "languages/pt").exists()
    assert report["checkpoint"]["epoch_metrics"]["epoch"] == 21
    assert report["recall_delta_percentage_points"] == {"top5": 0, "top7": 0}
    assert report["current"]["top7"]["expected_target_sample_pairs"] == 9
    assert case.calls == ["load"] + ["sample"] * 6
    with zipfile.ZipFile(report["return_file"]) as archive:
        assert set(archive.namelist()) == {
            "report.json",
            "run_config.json",
            "checkpoint_metadata.json",
            "sha256.txt",
        }
        for line in archive.read("sha256.txt").decode().splitlines():
            digest, name = line.split()
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    preview.run_preview(case.root, case.config, case.output, resume=True, **case.kw)
    assert case.calls.count("sample") == 6
    assert all(_sha(p) == digest for p, digest in before.items())
    for line in (case.output / "sha256.txt").read_text().splitlines():
        digest, name = line.split()
        assert _sha(case.output / name) == digest
    with pytest.raises(FileExistsError):
        preview.run_preview(case.root, case.config, case.output, **case.kw)


def test_resume_after_interruption_only_computes_missing_samples(case):
    retrieve = case.kw["retrieve_function"]
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("interruption")
        return retrieve(*args)

    with pytest.raises(RuntimeError, match="interruption"):
        preview.run_preview(
            case.root,
            case.config,
            case.output,
            **{**case.kw, "retrieve_function": interrupted},
        )
    assert len(list(case.output.glob("languages/en/*/sample_shards/*.json"))) == 2
    preview.run_preview(case.root, case.config, case.output, resume=True, **case.kw)
    assert case.calls.count("sample") == 6
    shard = next(case.output.glob("languages/en/*/sample_shards/*.json"))
    data = read(shard)
    data["audio_sha256"] = "changed"
    write(shard, data)
    with pytest.raises(ValueError, match="corrupted sample shard"):
        preview.run_preview(case.root, case.config, case.output, resume=True, **case.kw)


def test_new_predictions_are_measured_against_fixed_old_hits(case):
    def all_missed(detector, waveform, vocab):
        return retrieval(detector.hotwords, negative=True)

    report = preview.run_preview(
        case.root,
        case.config,
        case.output,
        **{**case.kw, "retrieve_function": all_missed},
    )
    for k in ("top5", "top7"):
        assert report["current"][k]["gated_target_hits"] == 0
        assert report["current"][k]["expected_target_sample_pairs"] == 9
        baseline = report["baseline_same_retained_targets"][k]
        assert baseline["gated_target_hits"] == case.cfg["expected_baseline"][f"{k}_hits"]
        assert baseline["gated_target_hits"] > 0
        assert report["recall_delta_percentage_points"][k] == -100 * baseline["gated_target_recall"]


def test_drift_rejected_before_output_or_model_load(case):
    changes = {
        case.root / "snapshot/head.pt": b"changed",
        case.root / "expanded/keyword_table.json": b"changed",
        case.root / "audio/en_chile/wav/same.wav": b"changed",
        case.root / "audio/en_chile/transcripts.txt": b"changed",
        case.root / "expanded_run/indexed_pronunciations.jsonl": b"changed",
        case.root / "expanded_run/languages/en/en_chile/retrieval_details.jsonl": b"changed",
    }
    for path, content in changes.items():
        original = path.read_bytes()
        try:
            path.write_bytes(content)
            with pytest.raises(ValueError):
                preview.run_preview(case.root, case.config, case.output, **case.kw)
            assert not case.output.exists() and not case.calls, path
        finally:
            path.write_bytes(original)
    for key, value in (("expected_epoch", 20), ("expected_surfaces", 12)):
        changed = {**case.cfg, key: value}
        write(case.config, changed)
        with pytest.raises(ValueError):
            preview.run_preview(case.root, case.config, case.output, **case.kw)
        assert not case.output.exists() and not case.calls
    write(
        case.config,
        {
            **case.cfg,
            "expected_baseline": {
                **case.cfg["expected_baseline"],
                "top7_hits": 999,
            },
        },
    )
    with pytest.raises(ValueError, match="baseline hit count"):
        preview.run_preview(case.root, case.config, case.output, **case.kw)


def test_retrieval_settings_or_code_drift_cannot_be_hidden_in_source(case, monkeypatch):
    original = preview.external._run_config

    def changed(**kwargs):
        result = original(**kwargs)
        result["gate"] = {**result["gate"], "threshold": 0.7}
        return result

    monkeypatch.setattr(preview.external, "_run_config", changed)
    with pytest.raises(ValueError, match="settings differ beyond Head"):
        preview.run_preview(case.root, case.config, case.output, **case.kw)
    assert not case.calls and not case.output.exists()
    monkeypatch.setattr(preview.external, "_run_config", original)
    original_sha = preview._sha

    def changed_sha(path):
        return "changed" if path.name == "scoring.py" else original_sha(path)

    monkeypatch.setattr(preview, "_sha", changed_sha)
    with pytest.raises(ValueError, match="baseline retrieval code changed"):
        preview.run_preview(case.root, case.config, case.output, **case.kw)


def test_output_must_be_separate(case):
    for output in (case.root, case.root / "expanded_run/new", case.root / "snapshot/new"):
        with pytest.raises(ValueError, match="output must be separate"):
            preview.run_preview(case.root, case.config, output, **case.kw)
    assert not case.calls
