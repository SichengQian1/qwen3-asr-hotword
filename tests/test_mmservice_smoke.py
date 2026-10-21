from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes import mmservice_smoke as smoke
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.g2p_prep import extract_word_tokens

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def setup(tmp_path):
    cfg = json.loads(Path("configs/mmservice_g2p_smoke.workzone.json").read_text())
    sources = tmp_path / "compiled"
    sources.mkdir()
    cfg["source_sha256"] = {}
    for name in ("Makefile", "mmservice_g2p_main.cpp"):
        path = sources / name
        path.write_bytes(b"fixture source, not executed")
        cfg["source_sha256"][name] = smoke._sha(path)
    binary = sources / "mmservice_g2p"
    binary.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + (62).to_bytes(2, "little"))
    binary.chmod(0o700)
    models = {}
    for lang in ("en", "es", "pt"):
        path = tmp_path / f"{lang}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{lang}/model.fst", b"raw unchanged FST fixture")
            archive.writestr(f"{lang}/phones.sym", "<eps>\t0\na\t1\nb\t2\n")
            archive.writestr(
                f"{lang}/meta.json",
                json.dumps(
                    {
                        "architecture": "pynini" if lang == "pt" else "phonetisaurus",
                        "unicode_decomposition": lang == "es",
                        "grapheme_order": 2,
                    }
                ),
            )
            if lang != "pt":
                archive.writestr(f"{lang}/graphemes.sym", "a\t1\na|b\t2\n")
        models[lang] = {"path": str(path), "sha256": smoke._sha(path)}
    official = tmp_path / "official.json"
    write(official, {"models": models})
    cfg.update(official_config=str(official), vocab=str(VOCAB))
    config = tmp_path / "config.json"
    write(config, cfg)
    calls = []

    def run(command, stdout, log, **kwargs):
        calls.append(command)
        log.write_text("fixture log; no real G2P inference")
        if command[0] == "mfa":
            words = Path(command[-3]).read_text().splitlines()
            Path(command[-1]).write_text("".join(f"{w}\ta b\n" for w in words))
            stdout.write_text("MFA fixture")
        else:
            cases = Path(command[1]).read_text().splitlines()
            stdout.write_text(
                "".join(
                    json.dumps(
                        {
                            "word": surface,
                            "ipa": " ".join("a b" for _ in extract_word_tokens(surface)),
                            "ok": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    for surface in cases
                )
            )
        return {"returncode": 0, "timed_out": False, "seconds": 0.1}

    return SimpleNamespace(
        root=tmp_path,
        config=config,
        cfg=cfg,
        binary=binary,
        calls=calls,
        runner=run,
        output=tmp_path / "output",
    )


@pytest.fixture
def inspector(monkeypatch):
    fst = SimpleNamespace(
        input_symbols=lambda: None,
        fst_type=lambda: "vector",
        arc_type=lambda: "standard",
        num_states=lambda: 1,
    )
    # Only mock the native FST reader; exercise the real ZIP copying/inventory logic.
    module = SimpleNamespace(Fst=SimpleNamespace(read=lambda path: fst))
    original = smoke.importlib.import_module
    monkeypatch.setattr(
        smoke.importlib,
        "import_module",
        lambda name: module if name == "pynini" else original(name),
    )
    return smoke.inspect_model


def test_three_languages_raw_final_comparison_and_small_bundle(setup, inspector):
    c = setup
    before = {p: smoke._sha(p) for p in c.root.rglob("*") if p.is_file()}
    result = smoke.run(
        c.root, c.config, c.binary, c.output, mfa="mfa", runner=c.runner, inspector=inspector
    )
    assert result["status"] == "completed" and not result["replacement_verified"]
    assert result["totals"]["cases"] == 27
    assert result["totals"]["input_policy_blocked"] == 1
    assert result["totals"]["token_ids_equal"] == 26
    assert len(c.calls) == 9
    report = json.loads((c.output / "report.json").read_text())
    assert not report["gpu_used"] and not report["full_corpus_compared"]
    for lang in ("en", "es", "pt"):
        model = report["by_language"][lang]["model"]
        assert not model["input_symbols_embedded"] and not model["fst_modified"]
        assert model["external_graphemes_present"] == (lang != "pt")
        assert (c.output / lang / "model/model.fst").read_bytes() == b"raw unchanged FST fixture"
    with zipfile.ZipFile(result["return_file"]) as archive:
        assert not any("model.fst" in name or "mfa_work" in name for name in archive.namelist())
        for line in archive.read("sha256.txt").decode().splitlines():
            sha, name = line.split(maxsplit=1)
            import hashlib

            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    assert all(smoke._sha(p) == sha for p, sha in before.items())
    with pytest.raises(FileExistsError):
        smoke.run(c.root, c.config, c.binary, c.output, mfa="mfa", runner=c.runner)


def test_both_failures_oov_numeric_and_changed_dictionary_keys_are_not_equal(tmp_path):
    cases = ["missing", "known", "GPT4", "café"]
    dictionary = tmp_path / "raw.dict"
    dictionary.write_text("known\ta b\ncafe\ta b\n")
    rows = [{"word": word, "ipa": "", "ok": False} for word in cases]
    rows[1] = {"word": "known", "ipa": "q", "ok": True}
    rows[2] = {"word": "GPT4", "ipa": "a b", "ok": True}
    result = smoke.compare_cases(cases, dictionary, rows, rows, load_phoneme_vocab(VOCAB))
    assert not any(r["token_ids_equal"] for r in result)
    assert result[0]["mfa_missing_words"] == ["missing"]
    assert result[1]["cpp_normalized"]["oov_units"] == ["q"]
    assert result[2]["numeric_or_empty_input"]
    assert result[3]["mfa_missing_words"] == ["café"]


def test_cpp_process_failure_is_explicit_and_other_languages_continue(setup, inspector):
    c = setup

    def runner(command, stdout, log, **kwargs):
        if command[0] != "mfa" and "en" in str(stdout.parent.name):
            stdout.write_text("")
            log.write_text("native model read failed")
            return {"returncode": 1, "timed_out": False, "seconds": 0.1}
        return c.runner(command, stdout, log, **kwargs)

    result = smoke.run(
        c.root, c.config, c.binary, c.output, mfa="mfa", runner=runner, inspector=inspector
    )
    assert result["status"] == "completed"
    report = json.loads((c.output / "report.json").read_text())
    assert report["by_language"]["en"]["counts"]["cpp_ok"] == 0
    assert report["by_language"]["pt"]["counts"]["cpp_ok"] == 8


def test_model_source_and_wrong_cpu_rejected_before_any_process(setup):
    c = setup
    path = c.binary.parent / "Makefile"
    original = path.read_bytes()
    path.write_bytes(b"different source")
    with pytest.raises(ValueError, match="SHA256"):
        smoke.run(c.root, c.config, c.binary, c.output, mfa="mfa", runner=c.runner)
    assert not c.calls and not c.output.exists()
    path.write_bytes(original)
    model = c.root / "en.zip"
    original_model = model.read_bytes()
    model.write_bytes(b"changed official asset")
    with pytest.raises(ValueError, match="SHA256"):
        smoke.run(c.root, c.config, c.binary, c.output, mfa="mfa", runner=c.runner)
    assert not c.calls and not c.output.exists()
    model.write_bytes(original_model)
    c.binary.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + (183).to_bytes(2, "little"))
    with pytest.raises(ValueError, match="x86-64"):
        smoke.run(c.root, c.config, c.binary, c.output, mfa="mfa", runner=c.runner)
    assert not c.calls and not c.output.exists()


def test_output_order_and_timeout_guard(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"word":"other","ipa":"a","ok":true}\n')
    with pytest.raises(ValueError, match="count/order"):
        smoke._cpp_rows(path, ["requested"], {"returncode": 0})
    result = smoke._command(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        tmp_path / "stdout",
        tmp_path / "stderr",
        timeout=0.05,
        env=dict(os.environ),
    )
    assert result["timed_out"] and result["returncode"] is None


def test_official_failure_returns_error_and_log_bundle(setup, inspector):
    c = setup

    def runner(command, stdout, log, **kwargs):
        stdout.write_text("")
        log.write_text("official baseline fixture failed")
        return {"returncode": 1, "timed_out": False, "seconds": 0.1}

    result = smoke.run(
        c.root, c.config, c.binary, c.output, mfa="mfa", runner=runner, inspector=inspector
    )
    assert result["status"] == "failed" and "official MFA baseline failed" in result["error"]
    with zipfile.ZipFile(result["return_file"]) as archive:
        assert b"fixture failed" in archive.read("en/mfa.stderr.log")
