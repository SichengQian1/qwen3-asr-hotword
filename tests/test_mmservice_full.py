from __future__ import annotations

import json
import os
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation.wave_keywords import _key
from qwen_hotword.phonemes import mmservice_full as full
from qwen_hotword.phonemes import mmservice_process as process_module
from qwen_hotword.phonemes import mmservice_rewriter as worker
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.phonemes.mmservice_process import execute
from qwen_hotword.training.g2p_prep import extract_word_tokens

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


@pytest.fixture
def setup(tmp_path):
    """Synthetic labels and native-process mocks; never execute actual models or C++."""
    vocab = load_phoneme_vocab(VOCAB)

    def variant(surface, lang, ipa, role="target"):
        return {
            "surface": surface,
            "normalized": _key(surface),
            "language": lang,
            "decision": "retained",
            "role": role,
            "pronunciation": ipa,
            "token_ids": tokenize_ipa_to_vocab(ipa, vocab).token_ids,
        }

    groups = [
        [
            variant("casa", "en", "k æ s ə", "filler"),
            variant("casa", "es", "k a s a"),
            variant("casa", "pt-BR", "k a z ɐ"),
        ],
        [variant("alegría", "es", "a l e ɣ ɾ j a"), variant("alegría", "es", "a l e ɣ ɾ i a")],
        [variant("São Paulo", "pt-BR", "s ɐ̃ w̃ p a w l u")],
        [variant("New York", "en", "ɲ ʉː j ɒ ɹ k")],
        [variant("ﬁre", "en", "f aj ɹ")],
        [variant("U2", "en", "j ʉː t ʉː")],
        [variant("San José", "es", "s a n x o s e")],
    ]
    table = {
        "entries": [
            {"surface": g[0]["surface"], "normalized": g[0]["normalized"], "variants": g}
            for g in groups
        ]
    }
    source = tmp_path / "outputs/table"
    save(source / "reviewed_keyword_table.json", table)
    digest = full._sha(source / "reviewed_keyword_table.json")
    save(
        source / "report.json",
        {
            "status": "completed",
            "candidate_scope": "joint_multilingual",
            "review": {"expanded_table_sha256": digest, "active_unique_surfaces_per_audio": 7},
        },
    )
    (source / "sha256.txt").write_text(
        "".join(f"{full._sha(p)}  {p.name}\n" for p in sorted(source.iterdir()))
    )
    models = {}
    for lang in full.LANGUAGES:
        model = tmp_path / "models" / f"{lang}.zip"
        model.parent.mkdir(exist_ok=True)
        model.write_bytes(b"fixture, not a model")
        models[lang] = {"path": str(model), "sha256": full._sha(model)}
    table_cfg = tmp_path / "configs/table.json"
    save(
        table_cfg,
        {
            "source_run": str(source),
            "table_sha256": digest,
            "expected_surfaces": 7,
            "models": models,
        },
    )
    binary = tmp_path / "build/mmservice_g2p"
    binary.parent.mkdir()
    binary.write_bytes(b"\x7fELF\x02\x01" + b"\0" * 12 + (62).to_bytes(2, "little"))
    binary.chmod(0o700)
    cpp_source = binary.parent / "Makefile"
    cpp_source.write_text("fixture not executed")
    cpp_cfg = tmp_path / "configs/cpp.json"
    save(
        cpp_cfg,
        {
            "vocab": str(VOCAB.resolve()),
            "vocab_sha256": full._sha(VOCAB),
            "source_sha256": {"Makefile": full._sha(cpp_source)},
        },
    )
    config = tmp_path / "configs/full.json"
    save(
        config,
        {
            "table_config": str(table_cfg),
            "cpp_config": str(cpp_cfg),
            "table_config_sha256": full._sha(table_cfg),
            "cpp_config_sha256": full._sha(cpp_cfg),
            "binary": str(binary),
            "binary_sha256": full._sha(binary),
            "expected_variants": 10,
        },
    )
    for name in (
        "mmservice_full",
        "mmservice_rewriter",
        "mmservice_process",
        "mmservice_pipeline",
        "mmservice_smoke",
        "coverage",
    ):
        path = tmp_path / f"src/qwen_hotword/phonemes/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture code identity, not executed")
    for name in (
        "g2p_prep",
        "mfa_audit",
        "spanish_mfa_repair",
        "spanish_capacity",
        "stage_progress",
    ):
        path = tmp_path / f"src/qwen_hotword/training/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture identity")
    script = tmp_path / "scripts/compare_mmservice_g2p_full.py"
    script.parent.mkdir()
    script.write_text("fixture identity")
    prompt = tmp_path / "src/qwen_hotword/inference/hotword_prompt.py"
    prompt.parent.mkdir()
    prompt.write_text("fixture identity")
    pronunciations = {
        "en": {"casa": "k æ s ə", "fire": "f aj ɹ", "new": "ɲ ʉː", "york": "j ɒ ɹ k"},
        "es": {"casa": "k a s a", "alegría": "a l e ɣ ɾ i a", "san": "s ã n", "josé": "x o s e"},
        "pt": {"casa": "k a z ɐ", "são": "s ɐ̃ w̃", "paulo": "p a w l u"},
    }
    calls = []

    def inspector(archive, directory):
        directory.mkdir()
        for name in ["model.fst", "phones.sym", "meta.json"]:
            (directory / name).write_text("fixture, not a native model")
        return {"member_sha256": {p.name: full._sha(p) for p in directory.iterdir()}}

    def runner(command, stdout, stderr, **kwargs):
        calls.append(command)
        stderr.write_text("fixture subprocess log")
        lang = stdout.parent.name
        if command[0] == str(binary):
            surfaces = Path(command[1]).read_text().splitlines()
            rows = []
            for surface in surfaces:
                if surface == "ﬁre":
                    ipa = ""
                elif surface == "U2":
                    ipa = "j ʉː"
                else:
                    ipa = " ".join(pronunciations[lang][w] for w in extract_word_tokens(surface))
                rows.append({"word": surface, "ipa": ipa, "ok": bool(ipa)})
            write_rows(stdout, rows)
        elif command[0] == "mfa":
            if "proxy" in stdout.name:
                Path(command[-1]).write_text("alegria\ta l e ɣ ɾ j a\njose\tx o s e\n")
            else:
                words = Path(command[-3]).read_text().splitlines()
                Path(command[-1]).write_text(
                    "".join(
                        f"{w}\t{pronunciations[lang][w]}\n"
                        for w in words
                        if not (lang == "es" and w in {"alegría", "josé"})
                    )
                )
            stdout.write_text("fixture MFA")
        else:
            assert "--rewriter-worker" in command
            out = Path(command[-1])
            out.mkdir()
            words = Path(command[command.index("--words") + 1]).read_text().splitlines()
            write_rows(
                out / "words.jsonl",
                [
                    {
                        "word": w,
                        "input": w,
                        "ipa": pronunciations[lang][w],
                        "ok": True,
                        "status": "ok",
                    }
                    for w in words
                ],
            )
            (out / "raw.dict").write_text(
                "".join(f"{w}\t{pronunciations[lang][w]}\n" for w in words)
            )
            save(
                out / "report.json",
                {
                    "status": "completed",
                    "words": len(words),
                    "same_embedded_input_symbols_as_cpp": True,
                },
            )
            stdout.write_text("fixture exact-input worker progress")
        return {"returncode": 0, "timed_out": False, "stdout": str(stdout)}

    return SimpleNamespace(
        root=tmp_path,
        source=source,
        config=config,
        table=table,
        vocab=vocab,
        output=tmp_path / "new",
        binary=binary,
        calls=calls,
        inspector=inspector,
        runner=runner,
    )


def test_full_table_preserves_all_variants_cross_language_and_any_variant_counts(setup):
    c = setup
    before = {p: full._sha(p) for p in c.root.rglob("*") if p.is_file()}
    result = full.run(c.root, c.config, c.output, mfa="mfa", runner=c.runner, inspector=c.inspector)
    assert result["status"] == "completed" and not result["replacement_verified"]
    assert len(c.calls) == 16
    assert result["totals"]["saved_variant_rows"] == 10
    assert result["totals"]["raw_surface_inputs"] == 9
    assert result["totals"]["language_normalized_surfaces"] == 9
    assert result["totals"]["saved_variant_native_ids_equal"] == 6
    assert result["totals"]["all_input_forms_match_some_saved_variant"] == 6
    es = result["by_language"]["es"]
    assert es["saved_variant_rows"] == 4 and es["raw_surface_inputs"] == 3
    assert es["saved_variant_native_ids_equal"] == 2  # casa and one alegría pronunciation.
    assert es["all_input_forms_match_some_saved_variant"] == 2
    assert es["all_input_forms_match_some_saved_variant_after_es_cleanup"] == 3
    assert es["project_rules_native_ids_equal"] == 1
    assert es["project_rules_after_es_cleanup_ids_equal"] == 2
    rows = [json.loads(s) for s in (c.output / "case_comparison.jsonl").read_text().splitlines()]
    fire = next(r for r in rows if r["surface"] == "ﬁre")
    assert "canonical_word_engine_agrees_but_surface_path_diff" in fire["difference_flags"]
    pt = next(r for r in rows if r["surface"] == "São Paulo")
    assert not pt["diagnostic_cleanup"] and "ɐ̃" in pt["cpp_native"]["raw_ipa"]
    alegria = next(r for r in rows if r["surface"] == "alegría")
    assert alegria["word_provenance"][0]["proxy"] == "alegria"
    assert all(full._sha(p) == digest for p, digest in before.items())
    with zipfile.ZipFile(result["return_file"]) as archive:
        assert not any("model.fst" in n or "mfa_work" in n for n in archive.namelist())
        for line in archive.read("sha256.txt").decode().splitlines():
            digest, name = line.split(maxsplit=1)
            import hashlib

            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    for process in json.loads((c.output / "report.json").read_text())["processes"]:
        assert Path(process["stdout"]).is_file()
    with pytest.raises(FileExistsError):
        full.run(c.root, c.config, c.output, mfa="mfa", runner=c.runner)


@pytest.mark.parametrize("kind", ["table", "binary", "source"])
def test_identity_drift_rejected_before_execution(setup, kind):
    c = setup
    path = {
        "table": c.source / "reviewed_keyword_table.json",
        "binary": c.binary,
        "source": c.binary.parent / "Makefile",
    }[kind]
    path.write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        full.run(c.root, c.config, c.output, mfa="mfa", runner=c.runner)
    assert not c.output.exists() and not c.calls


def test_inventory_rejects_count_and_label_drift_instead_of_discarding_variants(setup):
    c = setup
    with pytest.raises(ValueError, match="expected 9"):
        full.inventory(c.table, c.vocab, 7, 9)
    table = deepcopy(c.table)
    table["entries"][0]["variants"][0]["token_ids"] = [2]
    with pytest.raises(ValueError, match="inconsistency"):
        full.inventory(table, c.vocab, 7, 10)


@pytest.mark.parametrize("failure", ["cpp", "worker", "mfa", "drift", "interrupt"])
def test_failures_keep_bundle_and_never_report_complete_coverage(setup, failure):
    c = setup

    def runner(command, stdout, stderr, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        bad = (
            (failure == "cpp" and command[0] == str(c.binary))
            or (failure == "worker" and "--rewriter-worker" in command)
            or (failure == "mfa" and command[0] == "mfa")
        )
        if bad:
            stdout.write_text("")
            stderr.write_text("fixture error")
            return {"returncode": 1, "timed_out": True}
        result = c.runner(command, stdout, stderr, **kwargs)
        if failure == "drift" and stdout.parent.name == "pt":
            c.binary.write_bytes(b"changed after executing")
        return result

    result = full.run(c.root, c.config, c.output, mfa="mfa", runner=runner, inspector=c.inspector)
    assert result["status"] == "failed" and result["totals"] is None
    assert Path(result["return_file"]).is_file()


def test_both_word_failures_and_oov_are_not_valid_agreement(tmp_path):
    d = tmp_path
    (d / "exact").mkdir()
    (d / "words.txt").write_text("absent\n")
    (d / "surfaces.txt").write_text("absent\n")
    (d / "mfa_default.dict").write_text("")
    (d / "exact/raw.dict").write_text("")
    write_rows(
        d / "exact/words.jsonl", [{"word": "absent", "ipa": "", "ok": False, "status": "error"}]
    )
    for name in ["cpp_words", "cpp_raw", "cpp_final"]:
        write_rows(d / f"{name}.jsonl", [{"word": "absent", "ipa": "", "ok": False}])
    saved = [
        {
            "language": "en",
            "surface": "absent",
            "normalized": "absent",
            "variant_index": 0,
            "saved_token_ids": [2],
            "role": "target",
        }
    ]
    cases, words = full.compare_language(
        "en", saved, d, d / "mfa_default.dict", load_phoneme_vocab(VOCAB)
    )
    result = full.summarize(cases, words, saved)
    assert result["word_engine"]["both_failed"] == 1
    assert result["word_engine"]["raw_equal"] == result["word_engine"]["ids_equal"] == 0
    assert result["saved_variant_native_ids_equal"] == 0


def test_process_timeout_keeps_logs_and_counts_real_written_rows(tmp_path):
    result = execute(
        [sys.executable, "-c", "import time; print('row',flush=True); time.sleep(5)"],
        tmp_path / "out",
        tmp_path / "err",
        timeout=0.1,
        env=dict(os.environ),
        label="fixture rows written",
        rows_total=1,
    )
    assert result["timed_out"] and result["returncode"] != 0
    assert result["written_rows"] == 1


def test_process_interrupt_stops_child_group(tmp_path, monkeypatch):
    state = {"stopped": False}

    def wait(**kwargs):
        if not state["stopped"]:
            raise KeyboardInterrupt
        return 0

    process = SimpleNamespace(pid=999999, wait=wait, poll=lambda: 0 if state["stopped"] else None)
    monkeypatch.setattr(process_module.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(process_module.os, "killpg", lambda *args: state.update(stopped=True))
    with pytest.raises(KeyboardInterrupt):
        execute(
            ["fixture"],
            tmp_path / "out",
            tmp_path / "err",
            label="fixture interrupt",
            timeout=10,
            env={},
        )
    assert state["stopped"]


@pytest.mark.parametrize("architecture", ["phonetisaurus", "pynini"])
def test_exact_worker_uses_same_tokens_and_symbols_no_metadata_filter(
    tmp_path, monkeypatch, architecture
):
    class RewriteError(Exception):
        pass

    model = tmp_path / "model"
    model.mkdir()
    save(
        model / "meta.json",
        {"architecture": architecture, "grapheme_order": 2, "unicode_decomposition": True},
    )
    (model / "model.fst").write_bytes(b"not real")
    (model / "phones.sym").write_text("<eps> 0\na 1\n")
    (model / "graphemes.sym").write_text("<eps> 0\na 1\n")
    symbols = SimpleNamespace(find=lambda s: {"<eps>": 0, "a": 1}[s], num_symbols=lambda: 2)
    embedded = symbols if architecture == "phonetisaurus" else None
    fst = SimpleNamespace(input_symbols=lambda: embedded, set_output_symbols=lambda s: None)
    calls = []

    class Rewriter:
        def __init__(self, fst, inp, out, **kwargs):
            calls.append((inp, kwargs))

        def __call__(self, word):
            if word == "missing":
                raise RewriteError("fixture compose failure")
            return [SimpleNamespace(pronunciation="a")]

    code = tmp_path / "generator.py"
    code.write_text("fixture")
    generator = SimpleNamespace(
        __file__=str(code),
        PhonetisaurusRewriter=Rewriter,
        Rewriter=Rewriter,
        rewrite=SimpleNamespace(Error=RewriteError),
    )
    modules = {
        "montreal_forced_aligner.g2p.generator": generator,
        "pynini": SimpleNamespace(Fst=SimpleNamespace(read=lambda p: fst)),
        "pywrapfst": SimpleNamespace(SymbolTable=SimpleNamespace(read_text=lambda p: symbols)),
    }
    original = worker.importlib.import_module
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else original(name),
    )
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: "3.4.0")
    words_file = tmp_path / "words.txt"
    words_file.write_text("á\nmissing\n")
    before = {p: full._sha(p) for p in model.iterdir()}
    result = worker.rewrite_exact_words(model, words_file, tmp_path / "out")
    assert result["ok"] == 1 and result["words"] == 2
    assert calls[0][1]["graphemes"] is None
    assert calls[0][1]["unicode_decomposition"] is False
    assert calls[0][0] is symbols if architecture == "phonetisaurus" else calls[0][0] == "utf8"
    rows = [json.loads(s) for s in (tmp_path / "out/words.jsonl").read_text().splitlines()]
    assert rows[0]["input"] == "á" and rows[1]["status"] == "error"
    assert all(full._sha(p) == digest for p, digest in before.items())
