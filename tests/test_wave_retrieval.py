from __future__ import annotations

import json
import shutil
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


def trained_fixture(root):
    original = fixture(root)
    baseline, _ = prepare_run(root, original)
    write(root / "baseline/run_config.json", baseline)
    cfg = json.loads(original.read_text())
    (root / "new_head.pt").write_bytes(b"new trained head mock")
    write(
        root / "training/run_plan.json",
        {
            "config": {"vocab_sha256": cfg["vocab_sha256"]},
            "train": {"records": 100},
            "validation": {"records": 20},
        },
    )
    write(
        root / "training/report.json",
        {
            "status": "completed",
            "test_set_used": False,
            "cache_sha256_verified": True,
            "best_epoch": 25,
            "best_checkpoint_path": str(root / "new_head.pt"),
            "train_sample_count": 100,
            "validation_sample_count": 20,
        },
    )
    cfg.pop("checkpoint_sha256")
    cfg.update(
        checkpoint="new_head.pt",
        comparison_run_config="baseline/run_config.json",
        training_provenance={
            "run_plan": "training/run_plan.json",
            "run_plan_sha256": _sha(root / "training/run_plan.json"),
            "report": "training/report.json",
            "best_epoch": 25,
        },
    )
    path = root / "new_config.json"
    write(path, cfg)
    return path


def test_new_head_preserves_baseline_inputs_and_sixteen_delivery_schema(tmp_path, monkeypatch):
    config = trained_fixture(tmp_path)
    calls, factory, retrieve = mocks(monkeypatch)
    output = tmp_path / "new_run"
    report = run_waves(
        tmp_path, config, output, detector_factory=factory, retrieve_function=retrieve
    )
    assert report["checkpoint_sha256"] == _sha(tmp_path / "new_head.pt")
    assert report["delivery_file_count"] == 16
    assert report["top5_reproduced"] and report["top7_exact_replay"]
    for path in (output / "delivery").glob("*.json"):
        data = json.loads(path.read_text())
        assert data["empty"] == [] and set(data) == {"empty", "same"}
        assert all(set(x) == {"word", "phoneme"} for x in data["same"])
    run_waves(
        tmp_path, config, output, resume=True, detector_factory=factory, retrieve_function=retrieve
    )
    assert calls["retrievals"] == 16
    (tmp_path / "new_head.pt").write_bytes(b"changed head")
    with pytest.raises(ValueError, match="resume input/config/code"):
        run_waves(tmp_path, config, output, resume=True, detector_factory=factory)


@pytest.mark.parametrize("kind", ["epoch", "path", "plan", "audio", "gate", "tables"])
def test_new_head_rejects_provenance_or_comparison_drift(tmp_path, kind):
    config = trained_fixture(tmp_path)
    report = tmp_path / "training/report.json"
    if kind in {"epoch", "path"}:
        data = json.loads(report.read_text())
        data["best_epoch" if kind == "epoch" else "best_checkpoint_path"] = (
            26 if kind == "epoch" else str(tmp_path / "head.pt")
        )
        write(report, data)
    elif kind == "plan":
        (tmp_path / "training/run_plan.json").write_text("{}")
    elif kind == "audio":
        (tmp_path / "wave1/es/wav/same.wav").write_bytes(b"new audio")
    elif kind == "gate":
        p = tmp_path / "baseline/run_config.json"
        data = json.loads(p.read_text())
        data["groups"]["wave1_es"]["config"]["gate"]["threshold"] = 0.9
        write(p, data)
    else:
        p = tmp_path / "tables/es/keyword_bias_phoneme.json"
        data = json.loads(p.read_text())
        data["keyword_phonemes"]["word0"] = "e"
        write(p, data)
        sign(tmp_path / "tables")
    with pytest.raises(ValueError):
        prepare_run(tmp_path, config)


def test_checked_in_480h_config_changes_only_head_provenance_and_output():
    old = json.loads(Path("configs/wave_retrieval.workzone.json").read_text())
    new = json.loads(Path("configs/wave_retrieval_480h.workzone.json").read_text())
    for key in set(old) - {"checkpoint", "checkpoint_sha256", "output"}:
        assert old[key] == new[key]
    assert old["output"] != new["output"]
    assert new["checkpoint"].endswith("multilingual_480h_run_v1/head/ctc_head_best.pt")


def table_comparison_fixture(root):
    trained = trained_fixture(root)
    near_plan, _ = prepare_run(root, trained)
    write(root / "near/run_config.json", near_plan)
    shutil.copytree(root / "tables", root / "old_only_tables")
    report = {"status": "completed", "tables_created": True, "languages": {}}
    for lang in ("es", "pt"):
        p = root / "old_only_tables" / lang / "keyword_bias_phoneme.json"
        data = json.loads(p.read_text())
        words = data["keyword_sets"]["all_keywords"]
        words.remove("word0")
        words.append("zzz filler")
        del data["keyword_phonemes"]["word0"]
        data["keyword_phonemes"]["zzz filler"] = "a"
        write(p, data)
        report["languages"][lang] = {
            "filler_policy": "old_only_exclude_supplied",
            "selected_optional_supplied_neighbor_overlap": 0,
            "selected_by_source": {"primary": 1, "old_table": 7},
        }
    write(root / "old_only_tables/report.json", report)
    sign(root / "old_only_tables")
    cfg = json.loads(trained.read_text())
    cfg.update(
        tables="old_only_tables",
        comparison_change="keyword_table",
        comparison_run_config="near/run_config.json",
        expected_filler_policy="old_only_exclude_supplied",
    )
    config = root / "old_only_config.json"
    write(config, cfg)
    return config, near_plan


def test_table_comparison_reindexes_targets_and_preserves_delivery_schema(tmp_path, monkeypatch):
    cfg, previous = table_comparison_fixture(tmp_path)
    plan, _ = prepare_run(tmp_path, cfg)
    assert plan["groups"]["wave1_es"]["target_ids"] != previous["groups"]["wave1_es"]["target_ids"]
    calls, factory, retrieve = mocks(monkeypatch)
    output = tmp_path / "new_run"
    report = run_waves(tmp_path, cfg, output, detector_factory=factory, retrieve_function=retrieve)
    assert report["delivery_file_count"] == 16 and report["top7_exact_replay"]
    for path in (output / "delivery").glob("*.json"):
        data = json.loads(path.read_text())
        assert set(data) == {"same", "empty"} and data["empty"] == []
        assert all(set(r) == {"word", "phoneme"} for r in data["same"])


@pytest.mark.parametrize("kind", ["head", "phones", "targets", "overlap", "gate"])
def test_table_comparison_rejects_other_changes(tmp_path, kind):
    cfg, _ = table_comparison_fixture(tmp_path)
    tables = tmp_path / "old_only_tables"
    if kind == "head":
        (tmp_path / "new_head.pt").write_bytes(b"different head")
    elif kind == "phones":
        p = tables / "es/keyword_bias_phoneme.json"
        data = json.loads(p.read_text())
        data["keyword_phonemes"]["word6"] = "e"
        write(p, data)
        sign(tables)
    elif kind == "targets":
        p = tables / "es/targets_by_wave.json"
        write(p, {f"wave{i}": ["word5"] for i in range(1, 5)})
        sign(tables)
    elif kind == "overlap":
        p = tables / "report.json"
        data = json.loads(p.read_text())
        data["languages"]["es"]["selected_optional_supplied_neighbor_overlap"] = 1
        write(p, data)
        sign(tables)
    else:
        p = tmp_path / "near/run_config.json"
        data = json.loads(p.read_text())
        data["groups"]["wave1_es"]["config"]["gate"]["threshold"] = 0.9
        write(p, data)
    with pytest.raises(ValueError):
        prepare_run(tmp_path, cfg)
