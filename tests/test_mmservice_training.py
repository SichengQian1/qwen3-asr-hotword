from __future__ import annotations

import base64
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes import mmservice_training as training
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.phonemes.mmservice_smoke import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def audit(tmp_path):
    """Synthetic source dictionaries and C++ mock; no native FST/model execution."""
    vocab = load_phoneme_vocab(VOCAB)
    cpp_pronunciations = {
        "en": {"cat": "k æ t", "fire": ""},
        "es": {"mañana": "m ã ɲ a n a"},
        "pt": {"são": "s ɐ̃ w̃"},
    }
    source_specs = [
        ("en", "first", {"cat": "k æ t", "fire": "f aj ɹ"}),
        ("en", "second", {"cat": "k ə t"}),
        ("es", "spanish", {"mañana": "m a ɲ a n a"}),
        ("pt", "portuguese", {"são": "s ɐ̃ w̃"}),
    ]
    builds = {lang: [] for lang in ("en", "es", "pt")}
    tsves = {}
    for lang, name, words in source_specs:
        folder = tmp_path / name
        folder.mkdir()
        dictionary = folder / "original.dict"
        dictionary.write_text("".join(f"{w}\t{ipa}\n" for w, ipa in words.items()))
        tsv = folder / "source.tsv"  # deliberately absent: raw source text must not be reopened
        tsves[name] = str(tsv)
        build = folder / "build_config.json"
        save(
            build,
            {
                "dictionary": {"path": str(dictionary)},
                "tsv": {"path": str(tsv)},
                "vocab": {"path": str(VOCAB)},
            },
        )
        builds[lang].append(str(build))
    examples = [
        ("en", "first", "cat cat", "k æ t k æ t"),
        ("en", "second", "cat", "k ə t"),
        ("en", "first", "fire", "f aj ɹ"),
        ("es", "spanish", "mañana", "m a ɲ a n a"),
        ("pt", "portuguese", "são", "s ɐ̃ w̃"),
    ]
    manifest = tmp_path / "frozen/actual_train.jsonl"
    manifest.parent.mkdir()
    records = [
        {
            "id": str(i),
            "text": text,
            "language": lang,
            "split": "train",
            "source_tsv": tsves[source],
            "phoneme_token_ids": tokenize_ipa_to_vocab(ipa, vocab).token_ids,
            "audio_path": "/nonexistent/audio/never_read.wav",
            "duration_seconds": 2.0,
        }
        for i, (lang, source, text, ipa) in enumerate(examples)
    ]
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    binary = tmp_path / "build/mmservice_g2p"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + (62).to_bytes(2, "little"))
    binary.chmod(0o700)
    source_cpp = binary.parent / "source.cpp"
    source_cpp.write_text("fixture, not compiled")
    cpp_cfg = tmp_path / "configs/cpp.json"
    save(
        cpp_cfg,
        {
            "vocab": str(VOCAB),
            "vocab_sha256": _sha(VOCAB),
            "source_sha256": {source_cpp.name: _sha(source_cpp)},
        },
    )
    models = {}
    for lang in ("en", "es", "pt"):
        model = tmp_path / f"models/{lang}.zip"
        model.parent.mkdir(exist_ok=True)
        model.write_bytes(b"fixture, not a model")
        models[lang] = {"path": str(model), "sha256": _sha(model)}
    table_cfg = tmp_path / "configs/table.json"
    save(table_cfg, {"models": models})
    full_cfg = tmp_path / "configs/full.json"
    save(
        full_cfg,
        {
            "binary": str(binary),
            "binary_sha256": _sha(binary),
            "cpp_config": str(cpp_cfg),
            "cpp_config_sha256": _sha(cpp_cfg),
            "table_config": str(table_cfg),
            "table_config_sha256": _sha(table_cfg),
        },
    )
    config = tmp_path / "configs/config.json"
    save(
        config,
        {
            "manifest": str(manifest),
            "manifest_sha256": _sha(manifest),
            "expected_counts": {"en": 3, "es": 1, "pt": 1},
            "sources": builds,
            "full_comparison_config": str(full_cfg),
        },
    )
    script = tmp_path / "scripts/compare_mmservice_g2p_training.py"
    script.parent.mkdir()
    script.write_text("fixture script identity")
    for name in ("g2p_prep", "spanish_mfa_repair"):
        path = tmp_path / f"src/qwen_hotword/training/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture identity")
    calls = []

    def inspector(model, directory):
        directory.mkdir()
        (directory / "model.fst").write_text("not read by mock")
        (directory / "phones.sym").write_text("not read by mock")
        return {"fixture": True}

    def runner(command, stdout, stderr, **kwargs):
        calls.append(command)
        lang = stdout.parent.name
        words = Path(command[1]).read_text().splitlines()
        assert command[0] == str(binary) and len(command) == 4
        result = [
            {
                "word": word,
                "ipa": cpp_pronunciations[lang][word],
                "ok": bool(cpp_pronunciations[lang][word]),
            }
            for word in words
        ]
        stdout.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
        stderr.write_text("")
        return {"returncode": 0, "timed_out": False}

    return SimpleNamespace(
        root=tmp_path,
        config=config,
        manifest=manifest,
        binary=binary,
        records=records,
        calls=calls,
        output=tmp_path / "audit",
        runner=runner,
        inspector=inspector,
        builds=builds,
    )


def execute(audit):
    return training.run(
        audit.root, audit.config, audit.output, runner=audit.runner, inspector=audit.inspector
    )


def test_actual_training_all_words_source_labels_and_occurrence_denominators(audit):
    before = {p: _sha(p) for p in audit.root.rglob("*") if p.is_file()}
    result = execute(audit)
    assert result["status"] == "completed" and not result["official_g2p_rerun"]
    assert len(audit.calls) == 3 and not result["replacement_verified"]
    en = result["by_language"]["en"]
    assert en["samples"] == 3 and en["word_tokens"] == 4 and en["unique_words"] == 2
    native = en["word_label_comparison"]["native"]
    assert native["source_word_pairs"] == {"equal": 1, "different": 1, "cpp_failed": 1}
    assert native["word_occurrences"] == {"equal": 2, "different": 1, "cpp_failed": 1}
    assert native["word_occurrences_non_equal_rate"] == 0.5
    assert en["baseline_reconstructed_equal"] == 3
    assert en["sentence_label_comparison"]["native"]["different"] == 1
    es = result["by_language"]["es"]["sentence_label_comparison"]
    assert es["native"]["cpp_unusable"] == 1 and es["existing_es_cleanup"]["equal"] == 1
    pt = result["by_language"]["pt"]["sentence_label_comparison"]
    assert pt["native"]["equal"] == pt["existing_es_cleanup"]["equal"] == 1
    words = [json.loads(s) for s in (audit.output / "words.jsonl").read_text().splitlines()]
    cats = [r for r in words if r["word"] == "cat"]
    assert len(cats) == 2 and cats[0]["saved_reference"] != cats[1]["saved_reference"]
    assert all(_sha(path) == digest for path, digest in before.items())
    with zipfile.ZipFile(result["return_file"]) as handle:
        assert "words.jsonl" not in handle.namelist() and "samples.jsonl" not in handle.namelist()
        for line in handle.read("sha256.txt").decode().splitlines():
            digest, name = line.split(maxsplit=1)
            assert hashlib.sha256(handle.read(name)).hexdigest() == digest
    payload = "".join(
        Path(part["path"]).read_text().splitlines()[1] for part in result["text_parts"]
    )
    assert base64.b64decode(payload, validate=True) == Path(result["return_file"]).read_bytes()
    assert all(part["bytes"] <= 32768 for part in result["text_parts"])
    with pytest.raises(FileExistsError):
        execute(audit)


def test_source_dictionary_not_matching_saved_ids_is_reported_not_attributed_to_cpp(audit):
    build = json.loads(Path(audit.builds["en"][0]).read_text())
    dictionary = Path(build["dictionary"]["path"])
    dictionary.write_text("cat\tk ə t\nfire\tf aj ɹ\n")
    result = execute(audit)
    assert result["status"] == "completed"
    en = result["by_language"]["en"]
    assert en["baseline_not_verified"] == 1
    assert en["sentence_label_comparison"]["native"]["baseline_not_verified"] == 1
    # C++ matches the ACTUAL first sentence; the changed dictionary is a separate issue.
    assert en["sentence_label_comparison"]["native"]["equal"] == 1


@pytest.mark.parametrize("kind", ["missing_build", "binary_hash", "test_split", "unknown_source"])
def test_training_audit_rejects_bad_inputs_without_audio_or_mfa(audit, kind):
    if kind == "missing_build":
        Path(audit.builds["pt"][0]).unlink()
    elif kind == "binary_hash":
        audit.binary.write_bytes(b"changed")
    else:
        if kind == "test_split":
            audit.records[0]["split"] = "test"
        else:
            audit.records[0]["source_tsv"] = "/unknown/source.tsv"
        audit.manifest.write_text("".join(json.dumps(r) + "\n" for r in audit.records))
        config = json.loads(audit.config.read_text())
        config["manifest_sha256"] = _sha(audit.manifest)
        save(audit.config, config)
    if kind in {"missing_build", "binary_hash"}:
        with pytest.raises((OSError, ValueError)):
            execute(audit)
        assert not audit.output.exists()
    else:
        result = execute(audit)
        assert result["status"] == "failed" and not audit.calls


def test_failed_cpp_stops_with_small_failure_summary(audit):
    def fail(command, stdout, stderr, **kwargs):
        stdout.write_text("")
        stderr.write_text("fixture failure")
        return {"returncode": 1}

    audit.runner = fail
    result = execute(audit)
    assert result["status"] == "failed"
    assert Path(result["return_file"]).is_file()
