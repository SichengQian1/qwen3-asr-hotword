from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_english_head_preview import case as case
from test_reviewed_retrieval import read, retrieval, write

from qwen_hotword.inference.english_best_noise import prepare_noise
from qwen_hotword.inference.english_head_preview import run_preview
from qwen_hotword.training.spanish_capacity import _sha


@pytest.fixture
def noisy_case(case):
    import torch

    training = case.root / "final_training"
    head = training / "head/ctc_head_best.pt"
    head.parent.mkdir(parents=True)
    payload = torch.load(case.root / case.cfg["checkpoint"], weights_only=True)
    payload["epoch_metrics"] = {
        "epoch": 27,
        "loss": 0.25,
        "phoneme_error_rate": 0.1,
        "phoneme_errors": 6,
        "reference_phonemes": 60,
    }
    torch.save(payload, head)
    report = {
        "status": "completed",
        "test_set_used": False,
        "cache_sha256_verified": True,
        "best_checkpoint_path": str(head),
        "best_epoch": 27,
        "epochs_completed": 27,
        "epochs_requested": 40,
        "early_stopped": True,
        "checkpoint_selection_metric": "validation_macro_per",
        "validation_groups": ["en", "es", "pt"],
        "head_config": payload["head_config"],
        "best_validation_loss": 0.25,
        "best_validation_phoneme_error_rate": 0.1,
        "best_validation_macro_phoneme_error_rate": 0.1,
        "best_validation_by_group": {
            lang: {
                "phoneme_error_rate": 0.1,
                "phoneme_errors": 2,
                "reference_phonemes": 20,
                "sample_count": 2,
            }
            for lang in ("en", "es", "pt")
        },
    }
    write(training / "head/report.json", report)
    write(training / "plan.json", {"epochs": 40, "test_set_used": False})
    write(
        training / "training_completed.json",
        {
            "report_sha256": _sha(training / "head/report.json"),
            "plan_sha256": _sha(training / "plan.json"),
            "checkpoint_sha256": _sha(head),
            "test_set_used": False,
        },
    )
    noisy_root = case.root / "noisy"
    for name in case.cfg["datasets"]:
        directory = noisy_root / name / "en/wav"
        directory.mkdir(parents=True)
        (directory / "same.wav").write_bytes(b"noisy_positive")
        (directory / "negative.wav").write_bytes(b"noisy_negative")
        # Deliberately invalid local metadata must NOT replace the frozen clean reference/table.
        (directory.parent / "transcripts.txt").write_text("wrong reference")
        (directory.parent / "en_keyword_bias_phoneme.json").write_text("invalid")
    case.cfg.update(
        training_run="final_training",
        checkpoint=str(head),
        expected_epoch=27,
        expected_epochs_completed=27,
        expected_epochs_requested=40,
        audio_root=str(noisy_root),
        output="noise_result",
    )
    case.cfg.pop("checkpoint_checksum")
    write(case.config, case.cfg)
    case.output = case.root / "noise_result"
    case.training = training
    case.head = head
    case.report = report
    case.noisy_root = noisy_root

    def factory(**kwargs):
        assert kwargs["checkpoint"] == head
        assert kwargs["language"] == "English"
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        assert len({e.normalized for e in kwargs["hotwords"]}) == 11
        case.calls.append("load")
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        assert waveform in {b"noisy_positive", b"noisy_negative"}
        case.calls.append("sample")
        return retrieval(detector.hotwords, waveform == b"noisy_negative")

    case.kw = {
        "detector_factory": factory,
        "retrieve_function": retrieve,
        "prepare_function": prepare_noise,
    }
    return case


def test_final_best_noise_uses_original_refs_full_index_without_clean_audio(noisy_case):
    c = noisy_case
    for audio in (c.root / "audio").rglob("*.wav"):
        audio.unlink()
    before = {p: _sha(p) for p in c.root.rglob("*") if p.is_file()}
    audit = run_preview(c.root, c.config, c.output, audit_only=True, **c.kw)
    assert audit["checkpoint"]["epoch_metrics"]["epoch"] == 27
    assert not audit["model_loaded"] and not c.output.exists() and not c.calls
    assert not audit["audio_pairing"]["snr_verified"]
    report = run_preview(c.root, c.config, c.output, **c.kw)
    assert report["sample_count"] == 6 and len(report["delivery_files"]) == 6
    assert report["current"]["top7"]["expected_target_sample_pairs"] == 9
    assert "recall_delta_percentage_points" not in report
    assert report["comparison_kind"] == "old_head_clean_reference_vs_final_head_noisy_audio"
    assert "reference_clean_old_head" in report and not report["qwen_decoder_used"]
    assert report["checkpoint"]["sha256"] == _sha(c.head)
    assert report["checkpoint"]["best_validation_by_group"] == c.report["best_validation_by_group"]
    for p in report["audio_pairing"]["groups"].values():
        assert p == {"sample_count": 2, "changed_files": 2, "identical_files": 0}
    plan = read(c.output / "run_config.json")
    for name, group in plan["groups"].items():
        assert group["config"]["sources"][0]["audio_dir"] == str(c.noisy_root / name / "en/wav")
        assert all(str(c.noisy_root) in r["audio_path"] for r in group["records"])
        assert "Fire" in group["records"][-1]["reference_text"]
    assert c.calls == ["load"] + ["sample"] * 6
    run_preview(c.root, c.config, c.output, resume=True, **c.kw)
    assert c.calls.count("sample") == 6
    assert all(_sha(p) == digest for p, digest in before.items())
    for line in (c.output / "sha256.txt").read_text().splitlines():
        digest, name = line.split()
        assert _sha(c.output / name) == digest


def test_noisy_samples_and_final_training_identity_checked_before_loading(noisy_case):
    c = noisy_case
    path = c.noisy_root / "en_chile/en/wav/same.wav"
    path.rename(path.with_name("wrong_id.wav"))
    with pytest.raises(ValueError):
        run_preview(c.root, c.config, c.output, **c.kw)
    path.with_name("wrong_id.wav").rename(path)
    for relative in ("head/report.json", "plan.json", "head/ctc_head_best.pt"):
        asset = c.training / relative
        original = asset.read_bytes()
        try:
            asset.write_bytes(original + b"changed")
            with pytest.raises(ValueError, match="SHA256"):
                run_preview(c.root, c.config, c.output, **c.kw)
        finally:
            asset.write_bytes(original)
    assert not c.calls and not c.output.exists()
    write(c.config, {**c.cfg, "expected_epoch": 21})
    with pytest.raises(ValueError, match="completed formal training best identity"):
        run_preview(c.root, c.config, c.output, **c.kw)


def test_all_identical_audio_rejected_partial_identical_reported(noisy_case):
    c = noisy_case
    for name in ("same.wav", "negative.wav"):
        (c.noisy_root / "en_chile/en/wav" / name).write_bytes(
            (c.root / "audio/en_chile/wav" / name).read_bytes()
        )
    with pytest.raises(ValueError, match="all noisy files are identical"):
        run_preview(c.root, c.config, c.output, **c.kw)
    assert not c.calls and not c.output.exists()
    (c.noisy_root / "en_chile/en/wav/negative.wav").write_bytes(b"noisy_negative")
    audit = run_preview(c.root, c.config, c.output, audit_only=True, **c.kw)
    assert audit["audio_pairing"]["groups"]["en_chile"]["changed_files"] == 1
    assert audit["audio_pairing"]["groups"]["en_chile"]["identical_files"] == 1


def test_noisy_interruption_resume_and_head_drift(noisy_case):
    c = noisy_case
    retrieve = c.kw["retrieve_function"]
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("interruption")
        return retrieve(*args)

    with pytest.raises(RuntimeError, match="interruption"):
        run_preview(c.root, c.config, c.output, **{**c.kw, "retrieve_function": interrupted})
    assert len(list(c.output.glob("languages/en/*/sample_shards/*.json"))) == 2
    run_preview(c.root, c.config, c.output, resume=True, **c.kw)
    assert c.calls.count("sample") == 6
    original = c.head.read_bytes()
    c.head.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        run_preview(c.root, c.config, c.output, resume=True, **c.kw)


def test_resealed_wrong_checkpoint_epoch_does_not_pass(noisy_case):
    import torch

    c = noisy_case
    payload = torch.load(c.head, weights_only=True)
    payload["epoch_metrics"]["epoch"] = 21
    torch.save(payload, c.head)
    marker_path = c.training / "training_completed.json"
    marker = read(marker_path)
    marker["checkpoint_sha256"] = _sha(c.head)
    write(marker_path, marker)
    with pytest.raises(ValueError, match="best checkpoint metrics differ"):
        run_preview(c.root, c.config, c.output, **c.kw)
    assert not c.calls and not c.output.exists()
