from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.phonemes import mmservice_pipeline as pipeline
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.g2p_prep import extract_word_tokens

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def fixture_run(tmp_path):
    """Explicit synthetic evidence; no native binary, model or MFA is executed."""
    source = tmp_path / "outputs/source"
    source.mkdir(parents=True)
    cases = {
        "en": ["cat", "ﬁre", "GPT4"],
        "es": ["casa", "mañana", "San José", "niño"],
        "pt": ["casa", "São Paulo"],
    }
    dictionaries = {
        "en": "cat\tcʰ æ t\nfire\tf aj ɹ\n",
        "es": "casa\tk a s a\nsan\ts ã n\n",
        "pt": "casa\tk a z ɐ\nsão\ts ɐ̃ w̃\npaulo\tp a w l u\n",
    }
    cpp_ipa = {
        "cat": "cʰ æ t",
        "ﬁre": "",
        "GPT4": "dʒ iː pʰ iː tʲ iː",
        "casa": "k a s a",
        "mañana": "m a ɲ a n a",
        "San José": "s ã n x o s e",
        "niño": "n i ɲ o",
        "São Paulo": "s ɐ̃ w̃ p a w l u",
    }
    save(source / "input_cases.json", cases)
    (source / "comparison.jsonl").write_text("fixture; original summary unused\n")
    for lang, surfaces in cases.items():
        directory = source / lang
        directory.mkdir()
        words = sorted(
            {w for s in surfaces if not any(c.isdigit() for c in s) for w in extract_word_tokens(s)}
        )
        (directory / "words.txt").write_text("\n".join(words) + "\n")
        (directory / "mfa_raw.dict").write_text(dictionaries[lang], encoding="utf-8")
        rows = [
            {
                "word": s,
                "ipa": "k a z ɐ" if lang == "pt" and s == "casa" else cpp_ipa[s],
                "ok": s != "ﬁre",
            }
            for s in surfaces
        ]
        for mode in ("raw", "normalized"):
            (directory / f"cpp_{mode}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
            )
    (source / "es/model").mkdir()
    (source / "es/model/model.fst").write_bytes(b"fixture only, not a real FST")
    (source / "es/model/meta.json").write_text('{"architecture":"phonetisaurus"}')
    inputs = {}
    for name in (
        "src/qwen_hotword/phonemes/mmservice_pipeline.py",
        "scripts/compare_mmservice_g2p_pipeline.py",
        "src/qwen_hotword/training/spanish_mfa_repair.py",
        "configs/joint_4000_mfa.workzone.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture code identity, never executed")
        inputs[str(path)] = {"sha256": pipeline._sha(path)}
    vocab = tmp_path / VOCAB
    vocab.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VOCAB, vocab)
    model_path = tmp_path / "es.zip"
    model_path.write_bytes(b"official ZIP fixture, not a real model")
    official = tmp_path / "configs/joint_4000_mfa.workzone.json"
    save(official, {"models": {"es": {"path": "es.zip"}}})
    for path in (official, vocab, model_path, source / "es/model/model.fst"):
        inputs[str(path)] = {"sha256": pipeline._sha(path)}
    save(
        source / "report.json",
        {
            "status": "completed",
            "scope": "three_language_small_list_unmodified_cpp_smoke",
            "output_dir": str(source),
            "inputs": inputs,
            "by_language": {lang: {} for lang in cases},
            "totals": {"cases": 9},
        },
    )
    members = sorted(p for p in source.rglob("*") if p.is_file() and "model" not in p.parts)
    (source / "sha256.txt").write_text(
        "".join(f"{pipeline._sha(p)}  {p.relative_to(source)}\n" for p in members)
    )
    # Unlisted files must not enter the new evidence package.
    (source / "unlisted-model.bin").write_bytes(b"do not copy this")
    config = tmp_path / "config.json"
    save(
        config,
        {
            "source_run": "outputs/source",
            "source_report_sha256": pipeline._sha(source / "report.json"),
            "source_manifest_sha256": pipeline._sha(source / "sha256.txt"),
        },
    )
    calls = []

    def runner(command, stdout, log, **kwargs):
        calls.append(command)
        stdout.write_text("fixture subprocess; no actual model")
        log.write_text("fixture log")
        if command[0] == "mfa":
            assert Path(command[-3]).read_text().splitlines() == ["jose", "maniana", "ninio"]
            Path(command[-1]).write_text(
                "jose\tx o s e\nmaniana\tm a ɲ j a n a\nninio\tn i ɲ j o\n", encoding="utf-8"
            )
        else:
            assert "--probe-worker" in command
            words = (source / "es/words.txt").read_text().splitlines()
            base = {"casa": "k a s a", "san": "s ã n"}
            save(
                Path(command[-1]),
                {
                    "status": "completed",
                    "symbol_tables_equal": True,
                    "rows": [
                        {
                            "word": w,
                            "unicode_decomposition": flag,
                            "pronunciations": [base[w]] if w in base else ([] if flag else ["a"]),
                            "status": "ok" if w in base or not flag else "error",
                        }
                        for flag in (True, False)
                        for w in words
                    ],
                },
            )
        return {"returncode": 0, "timed_out": False, "seconds": 0.1}

    return SimpleNamespace(
        root=tmp_path,
        source=source,
        config=config,
        output=tmp_path / "new",
        calls=calls,
        runner=runner,
    )


def test_pipeline_reuses_base_and_native_outputs_keeps_adapter_separate(fixture_run):
    c = fixture_run
    before = {p: pipeline._sha(p) for p in c.root.rglob("*") if p.is_file()}
    result = pipeline.run(c.root, c.config, c.output, mfa="mfa", runner=c.runner)
    assert result["status"] == "completed"
    assert len(c.calls) == 2  # Only new ES proxies and official flag probe.
    assert result["probe_default_reproduces_saved_mfa"]
    assert result["by_language"]["es"]["native_token_ids_equal"] == 3
    assert result["by_language"]["es"]["after_existing_es_cleanup_token_ids_equal"] == 4
    assert result["by_language"]["en"]["cpp_accepts_policy_blocked_input"] == 1
    rows = [json.loads(s) for s in (c.output / "comparison.jsonl").read_text().splitlines()]
    san = next(r for r in rows if r["surface"] == "San José")
    assert san["cpp_native"]["oov_units"] == ["̃"]
    assert san["diagnostic_cleanup"] == {"combining_tilde_removed": 1}
    assert san["word_provenance"][1]["proxy"] == "jose"
    assert san["word_provenance"][0]["resolution"] == "base_exact"
    pt = next(r for r in rows if r["surface"] == "São Paulo")
    assert "ɐ̃" in pt["reference"]["raw_ipa"] and not pt["diagnostic_cleanup"]
    report = json.loads((c.output / "report.json").read_text())
    assert not report["replacement_verified"] and not report["cpp_rerun"] and not report["gpu_used"]
    assert not report["full_corpus_compared"]
    assert all(pipeline._sha(p) == digest for p, digest in before.items())
    with zipfile.ZipFile(result["return_file"]) as archive:
        assert not any(
            "model.fst" in n or "unlisted-model" in n or "mfa_work" in n for n in archive.namelist()
        )
        for line in archive.read("sha256.txt").decode().splitlines():
            digest, name = line.split(maxsplit=1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        pipeline.run(c.root, c.config, c.output, mfa="mfa", runner=c.runner)


@pytest.mark.parametrize("target", ["en/mfa_raw.dict", "../es.zip", "report.json", "sha256.txt"])
def test_source_identity_drift_rejected_before_new_output(fixture_run, target):
    c = fixture_run
    path = c.root / "es.zip" if target == "../es.zip" else c.source / target
    path.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="SHA256"):
        pipeline.run(c.root, c.config, c.output, mfa="mfa", runner=c.runner)
    assert not c.calls and not c.output.exists()


@pytest.mark.parametrize("where", ["mfa", "probe", "probe_incomplete", "drift_after_execution"])
def test_failed_subprocess_or_probe_preserves_diagnostic_bundle(fixture_run, where):
    c = fixture_run

    def runner(command, stdout, log, **kwargs):
        if (where == "mfa" and command[0] == "mfa") or (where == "probe" and command[0] != "mfa"):
            stdout.write_text("")
            log.write_text("fixture failure")
            return {"returncode": 1, "timed_out": False, "seconds": 0.1}
        result = c.runner(command, stdout, log, **kwargs)
        if command[0] != "mfa":
            if where == "probe_incomplete":
                save(Path(command[-1]), {"status": "completed", "rows": []})
            if where == "drift_after_execution":
                (c.source / "en/mfa_raw.dict").write_text("changed after execution")
        return result

    result = pipeline.run(c.root, c.config, c.output, mfa="mfa", runner=runner)
    assert result["status"] == "failed" and not result["replacement_verified"]
    assert Path(result["return_file"]).is_file()
    with zipfile.ZipFile(result["return_file"]) as archive:
        report = json.loads(archive.read("report.json"))
        assert report["status"] == "failed" and report["error"]


def test_raw_ipa_equality_does_not_hide_extra_cpp_normalization(tmp_path):
    dictionary = tmp_path / "raw.dict"
    dictionary.write_text("cat\ttʃ\n")
    cpp = [{"word": "cat", "ipa": "ˈt͡ʃ", "ok": True}]
    rows = pipeline.compare_pipeline(["cat"], dictionary, cpp, load_phoneme_vocab(VOCAB), "en")
    assert rows[0]["comparison"]["native"]["token_ids_equal"]
    assert not rows[0]["comparison"]["native"]["ipa_equal"]


def test_ambiguous_missing_both_failed_and_numeric_not_agreement(tmp_path):
    path = tmp_path / "raw.dict"
    path.write_text("known\ta\nknown\tb\n")
    cases = ["known", "missing", "GPT4"]
    cpp = [{"word": s, "ipa": "", "ok": False} for s in cases]
    rows = pipeline.compare_pipeline(cases, path, cpp, load_phoneme_vocab(VOCAB), "en")
    assert rows[0]["ambiguous_words"] == ["known"]
    assert "unresolved_digit" in rows[2]["reference_policy_reasons"]
    assert pipeline.summarize(rows)["native_token_ids_equal"] == 0


def test_official_rewriter_probe_holds_tables_fixed_and_varies_only_flag(tmp_path, monkeypatch):
    class RewriteError(Exception):
        pass

    model = tmp_path / "model"
    model.mkdir()
    save(
        model / "meta.json",
        {
            "architecture": "phonetisaurus",
            "unicode_decomposition": True,
            "graphemes": list("mañ"),
            "grapheme_order": 2,
        },
    )
    (model / "model.fst").write_bytes(b"fixture not native FST")
    (model / "graphemes.sym").write_text("<eps> 0\nm 1\na 2\nn 3\ñ 4\n")
    (model / "phones.sym").write_text("<eps> 0\na 1\n")
    words = tmp_path / "words.txt"
    words.write_text("mañana\n")
    code = tmp_path / "official_generator.py"
    code.write_text("official generator fixture; not executed")
    symbols = SimpleNamespace(
        find=lambda s: {"<eps>": 0, "m": 1, "a": 2, "n": 3, "̃": 4}[s], num_symbols=lambda: 5
    )
    fst = SimpleNamespace(
        input_symbols=lambda: symbols,
        set_input_symbols=lambda s: None,
        set_output_symbols=lambda s: None,
    )
    calls = []

    class Rewriter:
        def __init__(self, fst, inp, out, **kwargs):
            calls.append((inp, out, kwargs))
            self.decompose = kwargs["unicode_decomposition"]

        def __call__(self, word):
            if self.decompose:
                raise RewriteError("fixture no path for decomposed input")
            return [SimpleNamespace(pronunciation="m a ɲ a n a")]

    generator = SimpleNamespace(
        __file__=str(code),
        PhonetisaurusRewriter=Rewriter,
        rewrite=SimpleNamespace(Error=RewriteError),
        clean_up_word=lambda word, graphemes, **kw: (word, set()),
    )
    modules = {
        "montreal_forced_aligner.g2p.generator": generator,
        "pynini": SimpleNamespace(Fst=SimpleNamespace(read=lambda p: fst)),
        "pywrapfst": SimpleNamespace(SymbolTable=SimpleNamespace(read_text=lambda p: symbols)),
    }
    original_import = pipeline.importlib.import_module
    monkeypatch.setattr(
        pipeline.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    monkeypatch.setattr(pipeline.importlib.metadata, "version", lambda name: "3.4.0")
    before = {p: pipeline._sha(p) for p in model.iterdir()}
    result = pipeline.probe_spanish(model, words, tmp_path / "probe.json")
    assert result["symbol_tables_equal"]
    assert result["rows"][0]["status"] == "error"
    assert result["rows"][0]["rewrite_input"] == unicodedata.normalize("NFKD", "mañana")
    assert result["rows"][1]["status"] == "ok"
    assert result["rows"][1]["rewrite_input"] == "mañana"
    assert calls[0][0] is calls[1][0] and calls[0][1] is calls[1][1]
    a, b = calls[0][2].copy(), calls[1][2].copy()
    assert a.pop("unicode_decomposition") is True and b.pop("unicode_decomposition") is False
    assert a == b and all(pipeline._sha(p) == digest for p, digest in before.items())

    monkeypatch.setattr(pipeline.importlib.metadata, "version", lambda name: "3.5.0")
    with pytest.raises(ValueError, match="requires MFA 3.4.0"):
        pipeline.probe_spanish(model, words, tmp_path / "other.json")
