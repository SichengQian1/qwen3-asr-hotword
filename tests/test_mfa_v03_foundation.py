from __future__ import annotations

import json
import shutil
import subprocess
import wave
import zipfile
from pathlib import Path

import pytest

from qwen_hotword.phonemes import mfa_v03_assets as assets
from qwen_hotword.phonemes import mfa_v03_foundation as foundation
from qwen_hotword.phonemes import mfa_v03_review as review
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.phonemes.mfa_v03_runtime import input_profile, missing_graphemes

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
NEW = ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json"


def test_new_units_and_old_legacy_behavior_are_separate() -> None:
    old, new = load_phoneme_vocab(OLD), load_phoneme_vocab(NEW)
    assert new.tokens[:90] == old.tokens
    assert len(new.tokens) == 100
    assert tokenize_ipa_to_vocab("kʷ a", old).tokens == ["k", "w", "a"]
    assert tokenize_ipa_to_vocab("kʷ a", new).tokens == ["kʷ", "a"]
    assert tokenize_ipa_to_vocab("k w a", new).tokens == ["k", "w", "a"]
    assert tokenize_ipa_to_vocab("ã ɾ̃ ɾʲ ʁ ɡʷ", new).tokens == ["ã", "ɾ̃", "ɾʲ", "ʁ", "ɡʷ"]
    assert tokenize_ipa_to_vocab("ẽ t͡ʃ g", new).tokens == ["ẽ", "tʃ", "ɡ"]
    assert tokenize_ipa_to_vocab("aɪ", old).tokens == ["aj"]
    assert tokenize_ipa_to_vocab("aɪ", new).oov_units == ["aɪ"]


@pytest.mark.parametrize("text,missing", [("k q a", ["q"]), ("ˈk a", ["ˈk"]), ("k | a", ["|"])])
def test_new_tokenizer_rejects_partial_results_and_unencoded_annotations(text, missing) -> None:
    result = tokenize_ipa_to_vocab(text, load_phoneme_vocab(NEW))
    assert result.oov_units == missing
    assert result.tokens == result.token_ids == []


def test_new_config_is_pinned_and_union_has_no_discarded_old_units() -> None:
    value = json.loads(NEW.read_text())
    cfg = json.loads((ROOT / "configs/mfa_v03_foundation.workzone.json").read_text())
    assert cfg["vocab_sha256"] == assets.sha(NEW)
    assert cfg["old_vocab_sha256"] == assets.sha(OLD)
    union = {p for model in cfg["models"].values() for p in model["phones"]}
    assert set(value["tokens"][2:]) == union
    assert value["metadata"]["training_ready"] is False
    assert set(value["changes_from_v0_2"]["pt_additions_to_shared_vocab"]) == {"ʁ", "kʷ", "ɡʷ"}
    assert cfg["environment"]["name"] != cfg["environment"]["base"]
    for model in cfg["models"].values():
        assert len(model["revision"]) == 40
        assert all(not p.startswith("acoustic/") for p in model["files"])
        assert all(len(v["sha256"]) == 64 for v in model["files"].values())


def test_download_reuses_exact_files_and_refuses_corruption(tmp_path) -> None:
    calls = []
    payload = tmp_path / "payload"
    payload.write_bytes(b"model")
    cfg = {
        "models": {
            lang: {
                "repo": "owner/model",
                "revision": "a" * 40,
                "files": {"g2p/model.fst": {"bytes": 5, "sha256": assets.sha(payload)}},
            }
            for lang in ("en", "es", "pt")
        }
    }

    def fetch(url, target):
        calls.append(url)
        shutil.copyfile(payload, target)

    folder = tmp_path / "assets"
    first = assets.download(cfg, folder, fetch=fetch)
    assert len(calls) == 3 and len(first["files"]) == 3
    assert assets.download(cfg, folder, fetch=fetch) == first
    assert len(calls) == 3
    (folder / "pt/g2p/model.fst").write_bytes(b"bad")
    with pytest.raises(ValueError, match="changed pinned"):
        assets.download(cfg, folder, fetch=fetch)
    assert (folder / "pt/g2p/model.fst").read_bytes() == b"bad"


def test_failed_partial_download_is_never_published(tmp_path) -> None:
    cfg = {
        "models": {
            lang: {
                "repo": "owner/model",
                "revision": "a" * 40,
                "files": {"model.fst": {"bytes": 1, "sha256": "0" * 64}},
            }
            for lang in ("en", "es", "pt")
        }
    }

    def fetch(url, target):
        target.write_bytes(b"partial")
        raise OSError("interrupted")

    with pytest.raises(OSError, match="interrupted"):
        assets.download(cfg, tmp_path / "assets", fetch=fetch)
    assert not (tmp_path / "assets/en/model.fst").exists()
    assert list((tmp_path / "assets/en").glob("*.part_*"))


def test_offline_archive_import_checks_bytes_without_network(tmp_path) -> None:
    payload = tmp_path / "payload"
    payload.write_bytes(b"model")
    cfg = {
        "models": {
            lang: {
                "repo": "owner/model",
                "revision": "a" * 40,
                "files": {"g2p/model.fst": {"bytes": 5, "sha256": assets.sha(payload)}},
            }
            for lang in ("en", "es", "pt")
        }
    }
    archive = tmp_path / "offline.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for lang in ("en", "es", "pt"):
            stream.write(payload, lang + "/g2p/model.fst")

    def no_network(*args):
        raise AssertionError("offline import must not use network")

    result = assets.download(cfg, tmp_path / "models", fetch=no_network, archive=archive)
    assert result["status"] == "verified" and len(result["files"]) == 3


def test_unowned_or_changed_asset_directories_are_not_repurposed(tmp_path) -> None:
    folder = tmp_path / "assets"
    folder.mkdir()
    (folder / "old.zip").write_text("keep")
    with pytest.raises(ValueError, match="unowned"):
        assets.download({"models": {k: {"files": {}} for k in ("en", "es", "pt")}}, folder)
    assert (folder / "old.zip").read_text() == "keep"


def test_strict_unicode_guard_does_not_filter_unsupported_letters() -> None:
    meta = {"unicode_decomposition": False, "graphemes": list("são")}
    assert missing_graphemes("são", meta, set("são")) == []
    assert missing_graphemes("sãoq", meta, set("são")) == ["q"]
    # Published Spanish decomposition can yield a combining mark unsupported by its table.
    spanish = {"unicode_decomposition": True, "graphemes": list("año")}
    assert missing_graphemes("año", spanish, set("año"))
    exact, original_missing = input_profile("año", spanish, set("año"), allow_exact_nfc=True)
    assert not exact["unicode_decomposition"] and original_missing
    assert missing_graphemes("año", exact, set("año")) == []
    rejected, _ = input_profile("añoq", spanish, set("año"), allow_exact_nfc=True)
    assert rejected == spanish


def test_runtime_never_upgrades_an_unowned_existing_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CONDA_EXE", "conda")
    commands = []

    def fake(args, **kwargs):
        commands.append(args)
        stdout = '{"envs":["/envs/aligner", "/envs/new_aligner"]}' if "env" in args else "3.4.0\n"
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(assets.subprocess, "run", fake)
    cfg = {"environment": {"name": "new_aligner", "base": "aligner", "mfa_version": "3.4.3"}}
    with pytest.raises(ValueError, match="refusing to modify"):
        assets.environment(cfg, tmp_path, tmp_path / "logs")
    assert all("install" not in c and "create" not in c for c in commands)


def test_probability_columns_are_not_phonemes_and_variants_not_auto_selected(tmp_path) -> None:
    path = tmp_path / "lexicon.dict"
    path.write_text("carro 1.0 0.2 0.4 0.5 k a ʁ u\ncarro 0.1 k a h u\n", encoding="utf-8")
    result = review.lexicon(path, {"carro"})
    assert [r["ipa"] for r in result["carro"]] == ["k a ʁ u", "k a h u"]
    assert result["carro"][0]["mfa_parameters"] == [1.0, 0.2, 0.4, 0.5]


def test_numeric_and_symbol_surface_fragments_are_not_full_pronunciations() -> None:
    assert review.input_issues("BR-101") == ["numbers_require_explicit_verbalization"]
    assert review.input_issues("A&B") == ["symbols_require_explicit_verbalization"]
    assert review.input_issues("São Paulo") == []


@pytest.fixture
def staged(tmp_path):
    source = tmp_path / "baseline"
    source.mkdir()
    training = []
    for i, lang in enumerate(["en", "es"] + ["pt"] * 8):
        audio = tmp_path / f"{i}.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\0\0" * 1600)
        training.append(
            {
                "id": f"sample-{i}",
                "audio_path": str(audio),
                "text": "carro banho",
                "language": lang,
                "split": "train",
                "duration_seconds": 0.1,
                "source_corpus": "noah" if i % 2 else "mls",
                "phoneme_token_ids": [24, 2, 51, 45],
            }
        )
    train = source / "train.jsonl"
    review.write_rows(train, training)
    plan = {"train_manifest": str(train), "input_sha256": {str(train): assets.sha(train)}}
    assets.write_json(source / "plan.json", plan)
    table = tmp_path / "table.json"
    assets.write_json(
        table,
        {
            "entries": [
                {
                    "variants": [
                        {
                            "language": "pt",
                            "surface": "carro",
                            "pronunciation": "k a x u",
                            "token_ids": [24, 2, 51, 45],
                        }
                    ]
                }
            ]
        },
    )
    asset_dir = tmp_path / "assets"
    (asset_dir / "pt/dictionary").mkdir(parents=True)
    (asset_dir / "pt/dictionary/pt.dict").write_text("carro 1.0 k a ʁ u\nbanho 1.0 b ɐ̃ j̃ u\n")
    assets.write_json(asset_dir / "verified.json", {"status": "mock_for_unit_test"})
    cfg = {
        "source_run": str(source),
        "source_plan_sha256": assets.sha(source / "plan.json"),
        "hotword_table": str(table),
        "hotword_table_sha256": assets.sha(table),
        "max_hotword_entries": 1,
        "review_samples": 4,
        "max_review_words": 100,
        "seed": 4,
        "expected_train_counts": {"en": 1, "es": 1, "pt": 8},
        "vocab": str(NEW),
        "vocab_sha256": assets.sha(NEW),
        "old_vocab": str(OLD),
        "old_vocab_sha256": assets.sha(OLD),
        "models": {"pt": {"dialect": "pt"}},
        "smoke_words": {"en": ["cat"], "es": ["casa"], "pt": ["carro"]},
    }
    config_path = tmp_path / "config.json"
    assets.write_json(config_path, cfg)
    cfg["_config_path"] = str(config_path)
    return tmp_path, cfg, asset_dir, train


def test_source_audit_is_deterministic_and_does_not_read_audio_or_heldout(
    staged, monkeypatch
) -> None:
    root, cfg, _, train = staged
    output = root / "out"
    output.mkdir()
    before = assets.sha(train)
    original = Path.open

    def no_audio(path, *args, **kwargs):
        assert path.suffix != ".wav" and "validation" not in path.name and "test" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_audio)
    targets = review.portuguese_targets(root, cfg)
    selected = review.scan_sources(root, cfg, output, targets)
    assert len(selected) == 4 and all(r["language"] == "pt" for r in selected)
    assert review.scan_sources(root, cfg, output, targets) == selected
    assert assets.sha(train) == before
    report = json.loads((output / "source_audit.json").read_text())
    assert set(report["pt_sources"]) == {"mls", "noah"}
    assert not report["dialect_verified"] and not report["speaker_disjoint_verified"]
    assert report["pt_missing_metadata_records"]["speaker_id"] == 8


def test_full_preparation_mock_backend_resume_and_human_status(staged, monkeypatch) -> None:
    tmp, cfg, asset_dir, train = staged
    output = tmp / "out"
    calls = []
    monkeypatch.setattr(
        foundation, "download", lambda *args, **kwargs: {"status": "verified", "files": {}}
    )
    monkeypatch.setattr(foundation, "validate_inventory", lambda *args: {"classes": 100})

    def fake_command(args, log, label):
        calls.append(args)
        request = Path(args[args.index("--request") + 1])
        target = Path(args[args.index("--worker-output") + 1])
        data, counts = [], {}
        for row in review.rows(request):
            data.append(
                {
                    **row,
                    "status": "ok",
                    "candidates": [
                        {
                            "ipa": "a",
                            "tokens": ["a"],
                            "token_ids": [2],
                            "usable": True,
                        }
                    ],
                }
            )
            key = row["language"] + ":ok"
            counts[key] = counts.get(key, 0) + 1
        review.write_rows(target, data)
        assets.write_json(target.with_suffix(target.suffix + ".report.json"), {"counts": counts})

    monkeypatch.setattr(assets, "command", fake_command)
    before = assets.sha(train)
    report = foundation.run(ROOT, cfg, output, asset_dir, "all", mfa_python="mock-python")
    assert report["status"] == "prepared_pending_human_review", report
    assert report["stages"]["review"]["audio_copied"] == 4
    assert not report["training_started"] and not report["encoder_loaded"]
    assert len(calls) == 2
    again = foundation.run(ROOT, cfg, output, asset_dir, "all", mfa_python="mock-python")
    assert again["status"] == "prepared_pending_human_review" and len(calls) == 2
    assert assets.sha(train) == before
    with zipfile.ZipFile(output / "review_audio.zip") as archive:
        assert len([n for n in archive.namelist() if n.endswith(".wav")]) == 4
        assert "review.html" in archive.namelist()
    with zipfile.ZipFile(output / "return_bundle.zip") as archive:
        assert not any(n.endswith((".wav", ".fst", ".pt")) for n in archive.namelist())
    for case in review.rows(output / "review_package/review.jsonl"):
        assert assets.sha(Path(case["original_audio_path"])) == case["audio_sha256"]
        assert case["human_review"] == "pending"
    changed = dict(cfg, review_samples=3)
    with pytest.raises(ValueError, match="config/code changed"):
        foundation.run(ROOT, changed, output, asset_dir, "audit", mfa_python="mock-python")


def test_source_identity_drift_is_rejected(staged) -> None:
    root, cfg, _, train = staged
    with train.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed pinned"):
        review.source_manifest(root, cfg)


def test_output_cannot_be_nested_in_baseline(staged) -> None:
    _, cfg, asset_dir, _ = staged
    with pytest.raises(ValueError, match="separate"):
        foundation.prepare_contract(ROOT, cfg, Path(cfg["source_run"]) / "new", asset_dir)
