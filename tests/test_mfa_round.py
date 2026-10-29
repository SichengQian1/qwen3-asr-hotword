from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation import mfa_round as evaluation
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.phonemes.mfa_v03_review import rows, write_rows
from qwen_hotword.training import mfa_round_cache as cache
from qwen_hotword.training import mfa_round_common as common
from qwen_hotword.training import mfa_round_labels as labels
from qwen_hotword.training import mfa_round_run as run
from qwen_hotword.training.ctc_overfit import CachedSample, load_experiment_records
from qwen_hotword.training.feature_cache import cache_feature_split

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
NEW = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json"


def fake_predictions(word_sets):
    return {
        lang: {
            word: {
                "status": "ok",
                "candidates": [{"usable": True, "oov_units": [], "token_ids": [24, 2, 99, 45]}],
            }
            for word in values
        }
        for lang, values in word_sets.items()
    }


def test_relabel_never_reuses_old_ids_and_rejects_partial_words():
    old = {"text": "carro", "phoneme_token_ids": [24, 2, 51, 45]}
    generated = fake_predictions({"pt": {"carro"}})["pt"]
    row, issues = labels.relabel(old, generated, 6)
    assert not issues and row["phoneme_token_ids"] == [24, 2, 99, 45]
    assert row["legacy_label_sha256"] == common.digest(old["phoneme_token_ids"])
    rejected, issues = labels.relabel({**old, "text": "carro unknown"}, generated, 6)
    assert rejected is None and "g2p_unresolved:unknown" in issues
    assert labels.relabel(old, generated, 1)[0] is None
    assert labels.relabel({**old, "text": "carro 123"}, generated, 6)[0] is None


def test_noise_is_seventy_percent_of_records_per_language_not_added_copies():
    items = [
        {"id": f"{lang}-{i}", "language": lang} for lang in ("en", "es", "pt") for i in range(10)
    ]
    old = {r["id"] for r in items if int(r["id"].split("-")[-1]) < 5}
    selected = common.select_noise(items, old, 7)
    assert len(selected) == 21 and old <= selected
    assert Counter(i.split("-")[0] for i in selected) == {"en": 7, "es": 7, "pt": 7}
    assert common.select_noise(list(reversed(items)), old, 7) == selected


def test_round_contract_fixes_random_head_noise_and_rejects_changed_configuration(tmp_path):
    config_path = ROOT / "configs/mfa_v03_noise70.workzone.json"
    cfg = json.loads(config_path.read_text())
    cfg["config_path"] = str(config_path)
    output = tmp_path / "run"
    common.contract(ROOT, cfg, output)
    common.contract(ROOT, cfg, output)
    with pytest.raises(ValueError, match="NEW output"):
        common.contract(ROOT, {**cfg, "training_seed": 1}, output)
    with pytest.raises(ValueError, match="fresh 100-class"):
        common.contract(ROOT, {**cfg, "head_initialization": "warm_start"}, tmp_path / "other")


def test_status_is_metadata_only_and_does_not_restart_work(tmp_path):
    write_json(tmp_path / "report.json", {"status": "running", "active_stage": "prepare"})
    (tmp_path / "head").mkdir()
    (tmp_path / "head/metrics.jsonl").write_text('{"epoch": 1}\n{"epoch":')
    value = run.status(tmp_path)
    assert value["metadata_only"] and not value["gpu_used"]
    assert value["run"]["active_stage"] == "prepare"
    assert value["g2p_word_parts"]["completed"] == 0
    assert value["last_completed_epoch_metrics"]["epoch"] == 1
    assert value["partial_metrics_line_ignored"]


@pytest.mark.parametrize("text", ["A&B", "€", "70%", "BR-101", "a\x00b"])
def test_no_silent_surface_symbol_or_number_loss(text):
    assert common.words(text)[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    import torch

    model, source, previous = (tmp_path / name for name in ("model", "source", "previous"))
    for p in (model, source / "head", previous):
        p.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text("{}")
    head = {
        "head_type": "temporal_upsample",
        "hidden_dimension": 8,
        "kernel_size": 3,
        "dropout": 0.0,
        "time_upsampling_factor": 2,
    }
    old = {
        "vocab": str(OLD),
        "parts": [],
        "model": {"path": str(model), "dtype": "bfloat16"},
        "encoder_batch_size": 4,
        "workzone_config": "mock-zone",
        "test_set_used": False,
        "input_sha256": {},
        "head_config": head,
    }
    original_rows = {}

    def extraction(records, wrapper, **kwargs):
        values = []
        for record in records:
            noisy = bool(wrapper)
            # One historical noisy sample has one fewer frame, matching the real boundary case.
            frames = 5 if noisy and record.sample_id == "train-en-0" else 6
            hidden = torch.full((frames, 1024), 2.0 if noisy else 1.0, dtype=torch.bfloat16)
            values.append(CachedSample(record.sample_id, hidden, record.token_ids))
        return values, 317477504, 0.0

    for split, per_language in (("train", 10), ("validation", 1)):
        values = []
        for lang, word in (("en", "cat"), ("es", "casa"), ("pt", "carro")):
            for i in range(per_language):
                audio = tmp_path / f"{split}-{lang}-{i}.wav"
                audio.write_bytes(b"synthetic fixture; no Qwen inference")
                values.append(
                    {
                        "id": f"{split}-{lang}-{i}",
                        "audio_path": str(audio),
                        "text": word,
                        "language": lang,
                        "split": split,
                        "duration_seconds": 1.0,
                        "experiment": "full-ctc-v1",
                        "phoneme_token_ids": [24, 2, 45],
                        "label_length": 3,
                        "ctc_minimum_input_length": 3,
                        "ctc_time_upsampling_factor": 2,
                    }
                )
        manifest = source / f"{split}.jsonl"
        write_rows(manifest, values)
        original_rows[split] = values
        old[f"{split}_manifest"], old[f"{split}_cache"] = (
            str(manifest),
            str(source / "cache" / split),
        )
        records = load_experiment_records(
            manifest,
            num_classes=90,
            expected_split=split,
            expected_experiment="full-ctc-v1",
            prevalidated_audio_paths=True,
        )
        cache_feature_split(
            records,
            None,
            old[f"{split}_cache"],
            split=split,
            source_manifest_path=manifest,
            model_path=model,
            model_dtype="bfloat16",
            vocab_path=OLD,
            samples_per_shard=len(records),
            extractor=extraction,
        )
        old["input_sha256"][str(manifest)] = sha(manifest)
        for name in ("cache_config.json", "cache_index.json", "cache_summary.json"):
            path = Path(old[f"{split}_cache"]) / name
            old["input_sha256"][str(path)] = sha(path)
    write_json(
        source / "plan.json",
        {"train_manifest": old["train_manifest"], "train_cache": old["train_cache"]},
    )
    write_json(source / "head/report.json", {"head_config": head, "status": "completed"})
    noise_root = tmp_path / "noise"
    noise_root.mkdir()
    inventory = {"entries": []}
    for split in ("train", "validation"):
        wav = noise_root / f"{split}.wav"
        wav.write_bytes(split.encode())
        entry = {
            "path": str(wav),
            "sha256": sha(wav),
            "pcm_sha256": split,
            "samples_16k": 16000,
            "split": split,
        }
        inventory["entries"].append(entry)
        values = []
        for row in original_rows[split]:
            if split == "train" and int(row["id"].split("-")[-1]) >= 5:
                continue
            row = {
                **row,
                "noise10db": {
                    "path": str(wav),
                    "sha256": sha(wav),
                    "offset": int(common.rank(7, "offset:" + row["id"]), 16) % 16000,
                },
            }
            if split == "validation":
                row["id"] = "noise10db::" + row["id"]
            values.append(row)
        manifest = previous / split / "manifest.jsonl"
        manifest.parent.mkdir()
        write_rows(manifest, values)
        records = load_experiment_records(
            manifest,
            num_classes=90,
            expected_split=split,
            expected_experiment="full-ctc-v1",
            prevalidated_audio_paths=True,
        )
        cache_feature_split(
            records,
            True,
            manifest.parent / "features",
            split=split,
            source_manifest_path=manifest,
            model_path=model,
            model_dtype="bfloat16",
            vocab_path=OLD,
            samples_per_shard=len(records),
            extractor=extraction,
        )
        old["parts"].append(
            {
                "split": split,
                "base_shard": 0,
                "manifest": str(manifest.relative_to(previous)),
                "sha256": sha(manifest),
                "records": len(values),
            }
        )
    write_json(previous / "noise_inventory.json", inventory)
    old["noise_inventory_sha256"] = sha(previous / "noise_inventory.json")
    old["configuration"] = {
        "seed": 7,
        "noise_root": str(noise_root),
        "source_plan_sha256": sha(source / "plan.json"),
    }
    write_json(previous / "plan.json", old)
    write_json(previous / "prepared.json", {"plan_sha256": sha(previous / "plan.json")})
    write_json(previous / "cache_completed.json", {"status": "noise_cache_completed", "parts": 2})
    table = tmp_path / "table.json"
    entries = []
    for i in range(4000):
        word = (
            ("cat", "casa", "carro")[i]
            if i < 3
            else "word" + chr(97 + i // 676) + chr(97 + i // 26 % 26) + chr(97 + i % 26)
        )
        lang = ("en", "es", "pt")[i] if i < 3 else "en"
        entries.append(
            {"normalized": word, "surface": word, "variants": [{"language": lang, "surface": word}]}
        )
    write_json(table, {"entries": entries})
    foundation = {"hotword_table": str(table), "hotword_table_sha256": sha(table)}
    foundation_cfg = tmp_path / "foundation.json"
    write_json(foundation_cfg, foundation)
    foundation_out = tmp_path / "foundation"
    write_json(
        foundation_out / "report.json",
        {"status": "prepared_pending_human_review", "stages": {"review": {"audio_copied": 200}}},
    )
    write_json(foundation_out / "foundation_contract.json", {"configuration": foundation})
    cfg = {
        "source_run": str(source),
        "model": str(model),
        "source_plan_sha256": sha(source / "plan.json"),
        "source_report_sha256": sha(source / "head/report.json"),
        "noise_source": str(previous),
        "noise_root": str(noise_root),
        "seed": 7,
        "vocab": str(NEW),
        "vocab_sha256": sha(NEW),
        "old_vocab_sha256": sha(OLD),
        "foundation_config": str(foundation_cfg),
        "foundation_config_sha256": sha(foundation_cfg),
        "foundation_output": str(foundation_out),
        "assets_dir": str(tmp_path / "assets"),
        "train_counts": {"en": 10, "es": 10, "pt": 10},
        "maximum_quarantine_fraction": 0.05,
        "training": {"epochs": 5, "batch_size": 30, "learning_rate": 0.003},
        "training_seed": 7,
    }
    output = tmp_path / "new"
    output.mkdir()
    monkeypatch.setattr(labels, "validate_inventory", lambda *a: None)
    monkeypatch.setattr(
        labels, "generate", lambda root, cfg, output, word_sets: fake_predictions(word_sets)
    )
    monkeypatch.setattr(
        "qwen_hotword.config.load_workzone_config",
        lambda *a, **k: SimpleNamespace(
            model=SimpleNamespace(
                path=model, dtype="bfloat16", device="cuda:0", local_files_only=True
            )
        ),
    )
    monkeypatch.setattr("qwen_hotword.modeling.qwen_backbone.load_asr_model", lambda *a: object())

    class NewNoise:
        def __init__(self, manifest, audit):
            self.audit = audit

        def __call__(self, records, wrapper, **kwargs):
            write_json(self.audit, [])
            return extraction(records, True, **kwargs)

    monkeypatch.setattr(cache, "NoiseExtractor", NewNoise)
    return cfg, output, source, previous


def test_full_new_labels_cache_and_random_head_train_resume(experiment):
    import torch

    cfg, output, source, previous = experiment
    before = {
        str(p): sha(p) for folder in (source, previous) for p in folder.rglob("*") if p.is_file()
    }
    prepared = labels.prepare(ROOT, cfg, output)
    assert {v["records"] for v in prepared["noise"].values()} == {7}
    assert {v["total_records"] for v in prepared["noise"].values()} == {10}
    new_rows = [
        r for p in prepared["parts"] if p["split"] == "train" for r in rows(output / p["manifest"])
    ]
    assert len(new_rows) == 30 and sum(r["noise_selected"] for r in new_rows) == 21
    assert all(r["phoneme_token_ids"] == [24, 2, 99, 45] for r in new_rows)
    assert labels.prepare(ROOT, cfg, output) == prepared
    report = cache.build(ROOT, cfg, output, prepared)
    assert report["backends"] == {
        "original_hidden_reused": 12,
        "identical_old_noise_hidden_reused": 18,
        "new_noise_hidden_extracted": 6,
    }
    assert cache.build(ROOT, cfg, output, prepared) == report
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        trained = run.train(ROOT, cfg, output, prepared, device="cpu")
        assert trained["num_classes"] == 100 and not trained["old_head_loaded"]
        assert run.train(ROOT, cfg, output, prepared, device="cpu") == trained
        head = torch.load(output / "head/ctc_head_best.pt", weights_only=True)
        assert head["head_config"]["num_classes"] == 100
    finally:
        torch.set_num_threads(threads)
    assert not (output / "head/initial_head.json").exists()
    assert before == {
        str(p): sha(p) for folder in (source, previous) for p in folder.rglob("*") if p.is_file()
    }
    bundle = run.bundle(output)
    import zipfile

    with zipfile.ZipFile(bundle) as z:
        assert not any(n.endswith((".pt", ".wav", ".fst")) for n in z.namelist())


def test_relabel_preparation_rejects_source_cache_identity_drift(experiment):
    cfg, output, source, _ = experiment
    (source / "cache/train/cache_config.json").write_text("{}")
    with pytest.raises(ValueError, match="changed pinned"):
        labels.prepare(ROOT, cfg, output)


def test_saved_evaluation_cells_bind_requests_and_detect_tampering(tmp_path):
    path = tmp_path / "cell.json"
    evaluation.save(path, {"checkpoint": "new", "audio": "a"}, {"prediction": "abc"})
    assert evaluation.saved(path, {"checkpoint": "new", "audio": "a"})
    with pytest.raises(ValueError, match="changed"):
        evaluation.saved(path, {"checkpoint": "old", "audio": "a"})
    value = json.loads(path.read_text())
    value["prediction"] = "def"
    write_json(path, value)
    with pytest.raises(ValueError, match="changed"):
        evaluation.saved(path, value["identity"])


def test_wer_normalization_keeps_portuguese_accents_and_numbers():
    assert evaluation.wer_text("AVÓ, avô! BR-101") == "avó avô br 101"


def test_retrieval_keeps_missing_new_pronunciations_in_same_denominator(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "indexed_pronunciations.v03.jsonl").write_text("")
    checkpoints = {}
    for tag in ("v1", "old50", "new70"):
        path = tmp_path / tag
        path.write_bytes(tag.encode())
        checkpoints[tag] = path
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    samples = [
        {
            "group": lang,
            "sample_id": lang,
            "language": lang,
            "audio_path": str(audio),
            "audio_sha256": sha(audio),
            "clean_sha256": sha(audio),
        }
        for lang in ("en", "es", "pt")
    ]
    data = {
        "checkpoints": checkpoints,
        "old_index": (None, (), {}),
        "new_index": (None, (), {}),
        "bundle": SimpleNamespace(entries=[]),
        "targets": {lang: {"carro"} for lang in ("en", "es", "pt")},
        "references": {
            (lang, lang): {"reference_text": "carro", "audio_path": str(audio)}
            for lang in ("en", "es", "pt")
        },
        "inventory": {"samples": samples, "gate": {lang: {} for lang in ("en", "es", "pt")}},
    }
    monkeypatch.setattr(evaluation, "assets", lambda *args: data)
    monkeypatch.setattr(evaluation, "reference_matcher", lambda e: lambda text: {"carro"})
    loads = []

    def detector(**kwargs):
        loads.append(kwargs)
        return SimpleNamespace(tag=kwargs["checkpoint"].name)

    def retrieve(detector, *args, **kwargs):
        return {
            "raw_ranked_matches": []
            if detector.tag == "new70"
            else [{"hotword_id": "carro", "surface": "carro"}],
            "timing": {"pure_retrieval_seconds": 0.001},
        }

    monkeypatch.setattr(evaluation.external, "_load_detector", detector)
    monkeypatch.setattr(evaluation.external, "_load_audio", lambda *args: (None, 16000))
    monkeypatch.setattr(evaluation.external, "_retrieve_waveform", retrieve)
    monkeypatch.setattr(evaluation, "collapse_variants", lambda value, *args: value)
    monkeypatch.setattr(evaluation, "_select", lambda values, **kwargs: values)
    cfg = {
        "model": str(tmp_path),
        "vocab_sha256": "new",
        "old_vocab_sha256": "old",
        "hotword_table_sha256": "table",
    }
    report = evaluation.evaluate(ROOT, cfg, output)
    assert report["rows"] == 18 and len(loads) == 3
    assert report["by_language"]["new70/original/pt"]["top7"]["target_pairs"] == 1
    assert report["by_language"]["new70/original/pt"]["top7"]["target_hits"] == 0
    assert (
        report["paired_comparisons"]["new70_vs_old50/original/pt"]["top7"]["delta_recall_pp"]
        == -100
    )
    assert evaluation.evaluate(ROOT, cfg, output) == report and len(loads) == 3


def test_asr_runs_all_modes_without_sending_reference_and_reuses_identical_contexts(
    tmp_path, monkeypatch
):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    output = tmp_path / "run"
    write_json(output / "retrieval/report.json", {"status": "completed"})
    write_json(output / "contract.json", {"fixture": True})
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors.index.json").write_text("{}")
    bundle = SimpleNamespace(entries=[SimpleNamespace(hotword_id="carro", surface="carro")])
    monkeypatch.setattr(
        evaluation,
        "reference_matcher",
        lambda entries: lambda t: {"carro"} if "carro" in t else set(),
    )
    monkeypatch.setattr(
        evaluation,
        "assets",
        lambda *a: {
            "bundle": bundle,
            "inventory": {"samples": [{"group": "pt", "sample_id": "sample", "language": "pt"}]},
        },
    )
    for tag in ("old50", "new70"):
        for condition in ("original", "noise10db"):
            identity = {
                "tag": tag,
                "condition": condition,
                "group": "pt",
                "sample_id": "sample",
                "audio_sha256": sha(audio),
            }
            evaluation.save(
                output / "retrieval/cells" / f"{tag}-{condition}.json",
                identity,
                {"audio_path": str(audio), "reference_text": "carro", "top7": {"words": ["carro"]}},
            )
    calls = []

    class Engine:
        metadata = {}

        def __init__(self, *args):
            pass

        def transcribe(self, request):
            assert set(request) == {"audio_path", "audio_sha256", "language", "context", "seed"}
            calls.append(request)
            return {
                "prediction": "carro" if request["context"] else "caro",
                "actual_context": request["context"],
                "returned_language": request["language"],
                "prompt_tokens_verified": True,
                "generation_limit_reached": False,
                "normalized_audio_sha256": "a" * 64,
            }

    monkeypatch.setattr(evaluation, "RecoveredCaseRuntime", Engine)
    cfg = {"model": str(model), "asr": {"seed": 7, "max_new_tokens": 512}}
    report = evaluation.asr(ROOT, cfg, output)
    assert report["rows"] == 6 and len(calls) == 2
    assert report["by_mode_condition_language"]["new70/original/pt"]["wer"] == 0
    assert report["by_mode_condition_language"]["no_hotwords/original/pt"]["wer"] == 1
    resumed = evaluation.asr(ROOT, cfg, output)
    assert resumed["by_mode_condition_language"] == report["by_mode_condition_language"]
    assert resumed["new_decoder_calls_this_invocation"] == 0 and len(calls) == 2
