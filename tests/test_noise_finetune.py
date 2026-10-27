from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_overfit import (
    CachedSample,
    EpochMetrics,
    ExperimentRecord,
    save_ctc_head_checkpoint,
)
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.noise_audio import audit_noise, mix_10db
from qwen_hotword.training.noise_cache import build_noise_cache, compose_views
from qwen_hotword.training.noise_evaluation import aggregate, paired_delta
from qwen_hotword.training.noise_prepare import nested_selection, prepare, verify_plan
from qwen_hotword.training.noise_run import bundle, train
from qwen_hotword.training.sharded_ctc import load_feature_shard, train_sharded_ctc_head
from qwen_hotword.training.spanish_capacity import _sha

ROOT = Path(__file__).resolve().parents[1]


def test_nested_selection_uses_duration_and_is_order_independent() -> None:
    items = [(f"id-{i}", float(i % 7 + 1), f"{i % 3}") for i in range(300)]
    a, b = nested_selection(items, 13)
    assert a < b
    assert (a, b) == nested_selection(list(reversed(items)), 13)
    for ids, fraction in ((a, 0.3), (b, 0.5)):
        assert (
            abs(sum(s for i, s, _ in items if i in ids) / sum(s for _, s, _ in items) - fraction)
            < 0.02
        )


def test_active_snr_and_common_gain_preserve_duration_and_prevent_clipping() -> None:
    np = pytest.importorskip("numpy")
    speech = np.zeros(16000)
    speech[8000:] = 0.95 * np.sin(np.arange(8000) * 0.1)
    noise = np.random.default_rng(123).normal(size=4000)
    mixed, audit = mix_10db(speech, noise, offset=170)
    assert len(mixed) == len(speech)
    assert audit["snr_db"] == pytest.approx(10)
    assert audit["common_gain"] < 1
    assert audit["peak"] <= 0.990001
    assert audit["noise_looped"] is True
    assert audit["active_samples"] <= 8320
    assert np.array_equal(mixed, mix_10db(speech, noise, offset=170)[0])
    with pytest.raises(ValueError, match="silent source"):
        mix_10db(np.zeros(16000), noise, offset=0)
    with pytest.raises(ValueError, match="insufficient"):
        mix_10db(speech, np.zeros(100), offset=0)


def test_noise_audit_groups_pcm_duplicates_before_splitting(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    source = tmp_path / "free-sound"
    source.mkdir()
    for i in range(5):
        sf.write(source / f"{i}.wav", np.random.default_rng(i).normal(0, 0.1, 1600), 16000)
    (source / "copy.wav").write_bytes((source / "0.wav").read_bytes())
    report = audit_noise(tmp_path, 42)
    assert len(report["entries"]) == 5
    assert len(report["duplicates"]) == 1
    assert {e["split"] for e in report["entries"]} == {"train", "validation"}


def test_silent_noise_segment_is_realigned_deterministically(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from qwen_hotword.training.noise_cache import NoiseExtractor

    speech, noise = tmp_path / "speech.wav", tmp_path / "noise.wav"
    sf.write(speech, np.full(1600, 0.2), 16000)
    background = np.zeros(16000)
    background[12000:12010] = 0.3
    sf.write(noise, background, 16000)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "a",
                "noise10db": {
                    "path": str(noise),
                    "sha256": _sha(noise),
                    "offset": 0,
                },
            }
        )
        + "\n"
    )
    extractor = NoiseExtractor(manifest, tmp_path / "audit.json")
    record = ExperimentRecord("a", speech, "a", "en", (2,), 1, 2)
    mixed = extractor.waveform(record)
    assert len(mixed) == 1600
    assert extractor.audit[0]["silent_segment_realigned"] is True
    assert extractor.audit[0]["snr_db"] == pytest.approx(10)
    assert np.array_equal(mixed, NoiseExtractor(manifest, tmp_path / "other.json").waveform(record))


def fixture(tmp_path: Path) -> tuple[dict, dict]:
    """Synthetic cache/metadata fixtures, never evidence of acoustic/H200 quality."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    from qwen_hotword.modeling.ctc_head import build_ctc_head, ctc_head_config

    torch.set_num_threads(1)
    vocab = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    for name in ("config.json", "model.safetensors.index.json"):
        (model / name).write_text("{}")
    source = tmp_path / "source"
    source.mkdir()
    audio = tmp_path / "audio"
    audio.mkdir()
    caches = {}
    manifests = {}
    for split, count in (("train", 60), ("validation", 6)):
        rows, records = [], []
        for i in range(count):
            lang = ("en", "es", "pt")[i % 3]
            path = audio / f"{split}-{i}.wav"
            sf.write(path, np.sin(np.arange(1600) * 0.03) * 0.2, 16000)
            sample_id = f"{split}-{i}"
            rows.append(
                {
                    "id": sample_id,
                    "audio_path": str(path),
                    "text": "a",
                    "language": lang,
                    "balanced_language_bucket": lang,
                    "source": f"source_{lang}",
                    "duration_seconds": 86400.0 if split == "train" else 0.1,
                    "phoneme_token_ids": [2],
                    "label_length": 1,
                    "ctc_minimum_input_length": 1,
                    "ctc_time_upsampling_factor": 2,
                    "experiment": "full-ctc-v1",
                    "split": split,
                }
            )
            records.append(ExperimentRecord(sample_id, path, "a", lang, (2,), 1, 2))
        manifest = source / f"full_ctc_{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifests[split] = str(manifest)

        def extractor(batch: list[ExperimentRecord], wrapper: object, **kwargs: object) -> tuple:
            return (
                [CachedSample(r.sample_id, torch.zeros(3, 1024), r.token_ids) for r in batch],
                317477504,
                0.01,
            )

        folder = source / "cache" / split
        cache_feature_split(
            records,
            None,
            folder,
            split=split,
            source_manifest_path=manifest,
            model_path=model,
            model_dtype="bfloat16",
            vocab_path=vocab,
            samples_per_shard=6,
            extractor=extractor,
        )
        caches[split] = str(folder)
    head = build_ctc_head(
        head_type="temporal_upsample",
        input_dimension=1024,
        num_classes=90,
        hidden_dimension=4,
        kernel_size=3,
        dropout=0.1,
    )
    head_folder = source / "head"
    head_folder.mkdir()
    head_path = head_folder / "ctc_head_best.pt"
    save_ctc_head_checkpoint(
        head_path, head, load_phoneme_vocab(vocab), EpochMetrics(27, 1, 1, 1, 1), 11
    )
    report = {
        "status": "completed",
        "best_epoch": 27,
        "train_sample_count": 60,
        "validation_sample_count": 6,
        "head_config": ctc_head_config(head),
    }
    (head_folder / "report.json").write_text(json.dumps(report))
    old_plan = {
        "vocab": str(vocab),
        "train_manifest": manifests["train"],
        "train_cache": caches["train"],
        "model": {"path": str(model), "dtype": "bfloat16"},
        "encoder_batch_size": 2,
        "workzone_config": "unused",
        "input_sha256": {
            str(ROOT / "src/qwen_hotword/training/sharded_ctc.py"): "historical-code-identity"
        },
        "manifest_sha256": {},
        "audio_isolation_policy": "fixture_path_only",
    }
    (source / "plan.json").write_text(json.dumps(old_plan))
    noise = tmp_path / "musan"
    for category in ("free-sound", "sound-bible"):
        folder = noise / category
        folder.mkdir(parents=True)
        for i in range(4):
            sf.write(
                folder / f"{i}.wav",
                np.random.default_rng(i + (100 if category == "sound-bible" else 0)).normal(
                    0, 0.1, 800
                ),
                16000,
            )
    cfg = {
        "source_run": str(source),
        "source_plan_sha256": _sha(source / "plan.json"),
        "source_report_sha256": _sha(head_folder / "report.json"),
        "initial_head_sha256": _sha(head_path),
        "vocab": str(vocab),
        "vocab_sha256": _sha(vocab),
        "noise_root": str(noise),
        "seed": 123,
        "train_counts": {"en": 20, "es": 20, "pt": 20},
        "training": {"epochs": 1, "batch_size": 6, "learning_rate": 3e-5},
    }
    return cfg, {"path": model, "dtype": "bfloat16", "device": "cuda:0", "local_files_only": True}


def test_real_prepare_cache_views_warmstart_and_idempotent_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from qwen_hotword.training import noise_cache

    cfg, model_config = fixture(tmp_path)
    output = tmp_path / "noise_run"
    plan = prepare(ROOT, cfg, output)
    assert plan["noisy_hours"]["30"] == {"en": 144.0, "es": 144.0, "pt": 144.0}
    assert plan["noisy_hours"]["50"] == {"en": 240.0, "es": 240.0, "pt": 240.0}
    assert verify_plan(output, cfg) == plan
    monkeypatch.setattr(
        noise_cache,
        "load_workzone_config",
        lambda *a, **k: SimpleNamespace(model=SimpleNamespace(**model_config)),
    )
    monkeypatch.setattr("qwen_hotword.modeling.qwen_backbone.load_asr_model", lambda *a: None)

    def noisy_extract(
        records: list, wrapper: object, *, waveform_loader: object, **kwargs: object
    ) -> tuple:
        samples = []
        for record in records:
            wave = waveform_loader(record)
            samples.append(
                CachedSample(
                    record.sample_id,
                    torch.full((3, 1024), float(wave.mean()) + 1),
                    record.token_ids,
                )
            )
        return samples, 317477504, 0.01

    monkeypatch.setattr(noise_cache, "extract_frozen_features", noisy_extract)
    build_noise_cache(output, plan)
    # Completed cache verification never loads a model or recomputes features.
    monkeypatch.setattr(
        noise_cache, "extract_frozen_features", lambda *a, **k: pytest.fail("recache")
    )
    build_noise_cache(output, plan)
    a, va, ga = compose_views(output, plan, 30)
    b, vb, gb = compose_views(output, plan, 50)
    assert a.sample_count == b.sample_count == 60
    assert a.fingerprint != b.fingerprint
    assert va.fingerprint == vb.fingerprint
    assert ga == gb and len(ga) == 12
    assert len(set(ga.values())) == 6
    a_noisy = b_noisy = 0
    for sa, sb in zip(a.shards, b.shards, strict=True):
        left = load_feature_shard(sa, num_classes=90)
        right = load_feature_shard(sb, num_classes=90)
        assert [s.sample_id for s in left] == [s.sample_id for s in right]
        for x, y in zip(left, right, strict=True):
            if bool(x.hidden_states.any()):
                a_noisy += 1
                assert torch.equal(x.hidden_states, y.hidden_states)
            b_noisy += int(bool(y.hidden_states.any()))
    assert (a_noisy, b_noisy) == (18, 30)
    baseline = torch.load(plan["initial_head"], weights_only=True)
    for ratio in (30, 50):
        result = train(output, plan, ratio, resume=False, device="cpu")
        assert result["train_sample_count"] == 60 and result["epochs_completed"] == 1
        initial = json.loads((output / f"head_{ratio}/initial_head.json").read_text())
        assert initial["sha256"] == cfg["initial_head_sha256"]
        assert result["initial_validation_by_group"].keys() == set(ga.values())
        again = train(output, plan, ratio, resume=True, device="cpu")
        assert again == result
    assert all(
        torch.equal(value, torch.load(plan["initial_head"], weights_only=True)["state_dict"][key])
        for key, value in baseline["state_dict"].items()
    )
    assert bundle(output).is_file()
    part = next(p for p in plan["parts"] if p["split"] == "train")
    manifest = output / part["manifest"]
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_plan(output, cfg)


def test_paired_metrics_use_audio_target_counts_and_negative_coverage() -> None:
    def row(i: int, hits: int, expected: int, selected: int) -> dict:
        return {
            "group": "g",
            "sample_id": str(i),
            "pure_ms": 2,
            **{
                f"top{k}": {
                    "expected": expected,
                    "hits": hits,
                    "selected": selected,
                    "outside_reference": selected - hits,
                    "no_reference_hotwords": expected == 0,
                }
                for k in (5, 7)
            },
        }

    base = [row(1, 1, 2, 1), row(2, 0, 0, 0)]
    new = [row(1, 2, 2, 2), row(2, 0, 0, 1)]
    assert aggregate(new)["top7"]["recall"] == 1
    assert aggregate(new)["top7"]["negative_audio_false_positive_rate"] == 1
    assert paired_delta(base, new, 7)["delta_recall_pp"] == 50
    with pytest.raises(ValueError, match="sample sets differ"):
        paired_delta(base, new[:1], 7)


def test_replacement_rejects_label_drift_and_warmstart_rejects_wrong_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = fixture(tmp_path)
    plan = prepare(ROOT, cfg, tmp_path / "output")
    from qwen_hotword.training.noise_prepare import source_cache

    train_cache = source_cache(plan, "train", verify=True)
    val_cache = source_cache(plan, "validation", verify=True)
    a = train_cache.shards[0]
    invalid = replace(a, replacement=train_cache.shards[1], replacement_ids=a.sample_ids)
    with pytest.raises(ValueError, match="missing selected"):
        load_feature_shard(invalid, num_classes=90)
    torch = pytest.importorskip("torch")
    payload = torch.load(a.feature_path, weights_only=True)
    payload["token_ids"] += 1
    wrong_labels = tmp_path / "wrong_labels.pt"
    torch.save(payload, wrong_labels)
    invalid = replace(
        a, replacement=replace(a, feature_path=wrong_labels), replacement_ids=a.sample_ids
    )
    with pytest.raises(ValueError, match="changes labels"):
        load_feature_shard(invalid, num_classes=90)
    vocab = load_phoneme_vocab(plan["vocab"])
    wrong = replace(vocab, tokens=vocab.tokens[:2] + tuple(reversed(vocab.tokens[2:])))
    with pytest.raises(ValueError, match="vocabulary tokens"):
        train_sharded_ctc_head(
            train_cache,
            val_cache,
            wrong,
            tmp_path / "bad_head",
            vocab_path=plan["vocab"],
            device="cpu",
            epochs=1,
            minimum_epochs=1,
            head_type="temporal_upsample",
            head_hidden_dimension=4,
            head_kernel_size=3,
            initial_head_checkpoint=plan["initial_head"],
        )


def test_prepare_rejects_changed_source_manifest_before_creating_output(tmp_path: Path) -> None:
    cfg, _ = fixture(tmp_path)
    source = Path(cfg["source_run"]) / "full_ctc_train.jsonl"
    source.write_text(source.read_text() + "\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="source manifest"):
        prepare(ROOT, cfg, output)
    assert not output.exists()


def test_warmstart_has_fresh_optimizer_and_resume_preserves_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from qwen_hotword.training import sharded_ctc
    from qwen_hotword.training.noise_prepare import source_cache

    cfg, _ = fixture(tmp_path)
    plan = prepare(ROOT, cfg, tmp_path / "prepared")
    cache = source_cache(plan, "train", verify=True)
    validation = source_cache(plan, "validation", verify=True)
    baseline = torch.load(plan["initial_head"], weights_only=True)["state_dict"]
    original_evaluate = sharded_ctc._evaluate_cache
    original_train = sharded_ctc._train_cache_epoch
    optimizer_states = []

    def spy_evaluate(head: object, *args: object, **kwargs: object) -> object:
        if kwargs["epoch"] == 0:
            assert all(
                torch.equal(value.cpu(), baseline[key]) for key, value in head.state_dict().items()
            )
        return original_evaluate(head, *args, **kwargs)

    def spy_train(head: object, optimizer: object, *args: object, **kwargs: object) -> object:
        optimizer_states.append((kwargs["epoch"], bool(optimizer.state)))
        return original_train(head, optimizer, *args, **kwargs)

    monkeypatch.setattr(sharded_ctc, "_evaluate_cache", spy_evaluate)
    monkeypatch.setattr(sharded_ctc, "_train_cache_epoch", spy_train)
    kwargs = dict(
        vocab_path=plan["vocab"],
        device="cpu",
        minimum_epochs=1,
        learning_rate=3e-5,
        train_batch_size=6,
        head_type="temporal_upsample",
        head_hidden_dimension=4,
        head_kernel_size=3,
        initial_head_checkpoint=plan["initial_head"],
    )
    vocab = load_phoneme_vocab(plan["vocab"])
    first = tmp_path / "interrupted"
    train_sharded_ctc_head(cache, validation, vocab, first, epochs=1, **kwargs)
    resumed = train_sharded_ctc_head(
        cache, validation, vocab, first, epochs=2, resume=True, **kwargs
    )
    assert resumed.resumed_from_epoch == 1
    assert optimizer_states[:2] == [(1, False), (2, True)]
    whole = tmp_path / "whole"
    train_sharded_ctc_head(cache, validation, vocab, whole, epochs=2, **kwargs)
    a = torch.load(first / "training_state_latest.pt", weights_only=True)
    b = torch.load(whole / "training_state_latest.pt", weights_only=True)
    assert all(
        torch.equal(value, b["head_state_dict"][key]) for key, value in a["head_state_dict"].items()
    )


def test_three_head_evaluation_pairs_original_noise_and_resumes_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    from qwen_hotword.training import noise_evaluation as evaluation

    output = tmp_path / "run"
    output.mkdir()
    source = tmp_path / "baseline"
    source.mkdir()
    original = {"groups": {}}
    samples, inputs, groups, gates = [], {}, {}, {}
    gate = {
        "threshold": 0.75,
        "maximum_edit_ratio": 0.35,
        "minimum_posterior_confidence": 0.5,
        "minimum_top1_margin": 0.0,
    }
    for lang in ("en", "es", "pt"):
        clean = tmp_path / f"{lang}-clean.wav"
        noisy = tmp_path / f"{lang}-noise.wav"
        clean.write_bytes(b"original")
        noisy.write_bytes(b"noise")
        sample_id = "audio"
        original["groups"][lang] = {
            "records": [
                {"sample_id": sample_id, "reference_text": "target", "audio_path": str(clean)}
            ]
        }
        samples.append(
            {
                "group": lang,
                "sample_id": sample_id,
                "language": lang,
                "audio_path": str(noisy),
                "audio_sha256": _sha(noisy),
                "clean_sha256": _sha(clean),
            }
        )
        for path in (clean, noisy):
            inputs[str(path)] = {"sha256": _sha(path)}
        groups[lang] = {"target_ids": ["target"]}
        gates[lang] = gate
    (source / "run_config.json").write_text(json.dumps(original))
    baseline = tmp_path / "head.pt"
    baseline.write_bytes(b"head")
    (output / "plan.json").write_text("{}")
    for ratio in (30, 50):
        folder = output / f"head_{ratio}"
        folder.mkdir()
        (folder / "report.json").write_text("{}")
        (folder / "ctc_head_best.pt").write_bytes(str(ratio).encode())
        (folder / "completed.json").write_text(
            json.dumps(
                {
                    "report_sha256": _sha(folder / "report.json"),
                    "checkpoint_sha256": _sha(folder / "ctc_head_best.pt"),
                }
            )
        )
    inventory = {
        "samples": samples,
        "inputs": inputs,
        "gate": gates,
        "source_configuration": {"baseline_run": str(source), "table_sha256": "table"},
    }
    runtime = {
        "bundle": SimpleNamespace(entries=[]),
        "evaluation_groups": groups,
        "model": tmp_path,
        "vocab": None,
        "variants": (),
        "by_variant": {},
    }
    monkeypatch.setattr(evaluation, "prepare_multilingual_stream", lambda *a: (inventory, runtime))
    monkeypatch.setattr(evaluation, "reference_matcher", lambda *a: lambda text: {"target"})
    monkeypatch.setattr(evaluation, "collapse_variants", lambda retrieval, *a: retrieval)
    count = []

    def detector(**kwargs: object) -> object:
        count.append(kwargs["checkpoint"])
        return SimpleNamespace(language="English")

    monkeypatch.setattr(evaluation.external, "_load_detector", detector)
    monkeypatch.setattr(evaluation.external, "_load_audio", lambda *a: ([], {}))
    monkeypatch.setattr(
        evaluation.external,
        "_retrieve_waveform",
        lambda *a, **k: {
            "raw_ranked_matches": [
                {"hotword_id": "target", "score": 0.9, "edit_ratio": 0, "posterior_confidence": 0.9}
            ],
            "timing": {"pure_retrieval_seconds": 0.001},
        },
    )
    plan = {"initial_head": str(baseline), "configuration": {"evaluation_config": "unused"}}
    report = evaluation.evaluate(tmp_path, output, plan, resume=False)
    assert report["rows"] == 18 and len(count) == 3
    assert len(report["by_language"]) == 18
    assert all(v["top7"]["recall"] == 1 for v in report["by_language"].values())
    monkeypatch.setattr(evaluation.external, "_load_detector", lambda **k: pytest.fail("reload"))
    assert evaluation.evaluate(tmp_path, output, plan, resume=True) == report
    sample = next((output / "evaluation/samples").glob("*.json"))
    row = json.loads(sample.read_text())
    row["top7"]["hits"] = 0
    sample.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="corruption mismatch"):
        evaluation.evaluate(tmp_path, output, plan, resume=True)
