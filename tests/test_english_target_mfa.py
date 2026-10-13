from __future__ import annotations

import json
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

import pytest

from qwen_hotword.evaluation import english_target_mfa as audit
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _key
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def variant(surface, pronunciation, language="en"):
    phones = tokenize_ipa_to_vocab(pronunciation, load_phoneme_vocab(VOCAB))
    assert not phones.oov_units
    return {
        "surface": surface,
        "normalized": _key(surface),
        "language": language,
        "pronunciation": pronunciation,
        "phoneme_tokens": phones.tokens,
        "token_ids": phones.token_ids,
        "decision": "retained",
        "source_groups": ["test"],
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def setup(root):
    pairs = [
        ("Lie", "/ʎ i/"),
        ("Family", "/f æ m ə ʎ i/"),
        ("Twin Peaks", "/t w ɪ n p iː k s/"),
        ("Twin Twin", "/t w ɪ n t w ɪ n/"),
        ("el álbum", "/ɛ ɫ æ ɫ b ə m/"),
        ("U2", "/j ʉː/"),
        ("O’Brien", "/ow b ɹ aj ə n/"),
    ]
    entries = [{"id": str(i), "variants": [variant(*p)]} for i, p in enumerate(pairs)]
    entries.append({"id": "spanish", "variants": [variant("hola", "o l a", "es")]})
    source = root / "source"
    table = source / "reviewed_keyword_table.json"
    write(table, {"entries": entries})
    write(
        source / "report.json",
        {
            "status": "completed",
            "candidate_scope": "joint_multilingual",
            "review": {
                "filler_entries_added": 0,
                "reviewed_table_sha256": _sha(table),
                "keywords_by_language": {"en": len(pairs)},
            },
        },
    )
    coverage = Path(audit.__file__).parents[1] / "phonemes/coverage.py"
    write(source / "run_config.json", {"code_sha256": {"phonemes/coverage.py": _sha(coverage)}})
    seal(source)
    model = root / "assets/english_us_mfa.zip"
    model.parent.mkdir()
    model.write_bytes(b"mock model, never executed")
    training = root / "assets/training.dict"
    training.write_text(
        "lie\tɫ aj\nfamily\tf æ m ə ʎ i\ntwin\tt w ɪ n\npeaks\tp iː k s\no'brien\tow b ɹ aj ə n\n",
        encoding="utf-8",
    )
    cfg = {
        "source_run": "source",
        "table_sha256": _sha(table),
        "expected_english_targets": len(pairs),
        "vocab": str(VOCAB),
        "vocab_sha256": _sha(VOCAB),
        "model": str(model),
        "model_sha256": _sha(model),
        "training_dictionary": str(training),
        "training_dictionary_sha256": _sha(training),
    }
    config = root / "config/input.json"
    write(config, cfg)
    return config


def mock_mfa(command, **kwargs):
    assert command[1:6] == ["g2p", "--num_jobs", "2", "--num_pronunciations", "1"]
    words = Path(command[-3]).read_text().splitlines()
    assert words == sorted(set(words)) and "u" not in words
    assert "álbum" in words and "o'brien" in words
    assert all(" " not in w for w in words)
    Path(command[-1]).write_text(
        "lie\tˈɫ aj\nfamily\tf æ m ə ʎ i\ntwin\tt w ɪ n\npeaks\tp iː k s\n"
        "el\tɛ ɫ\nálbum\tæ ɫ b ə m\no'brien\tow b ɹ aj ə n\n",
        encoding="utf-8",
    )
    kwargs["stdout"].write("simulated only\n")
    work = Path(kwargs["env"]["MFA_ROOT_DIR"])
    work.mkdir(parents=True)
    (work / "scratch").write_text("not a return asset")


def test_fresh_training_comparison_preserves_phrase_order_and_originals(tmp_path, monkeypatch):
    config = setup(tmp_path)
    before = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(audit.subprocess, "run", mock_mfa)
    output = tmp_path / "new"
    report = audit.run_audit(tmp_path, config, output, 2, "mfa")
    assert report["status"] == "completed_with_unresolved"
    assert report["target_entries"] == 7 and not report["fillers_audited"]
    rows = {
        r["target"]["surface"]: r for r in json.loads((output / "comparisons.json").read_text())
    }
    assert rows["Lie"]["current_vs_fresh"]["status"] == "ctc_different"
    assert rows["Lie"]["fresh_vs_training"]["status"] == "ctc_equal"
    assert rows["Family"]["current_vs_fresh"]["status"] == "ctc_equal"
    assert not rows["Family"]["current_vs_fresh"]["raw_equal"]
    assert rows["Twin Twin"]["fresh_mfa"]["raw_pronunciation"] == "t w ɪ n t w ɪ n"
    assert rows["el álbum"]["current_vs_training"]["status"] == "unresolved"
    assert rows["U2"]["fresh_mfa"]["raw_pronunciation"] is None
    assert all(_sha(p) == h for p, h in before.items())
    with zipfile.ZipFile(output / "return_bundle.zip") as z:
        for line in z.read("sha256.txt").decode().splitlines():
            h, name = line.split(maxsplit=1)
            import hashlib

            assert hashlib.sha256(z.read(name)).hexdigest() == h
        assert all("mfa_work" not in p for p in z.namelist())
    with pytest.raises(FileExistsError):
        audit.run_audit(tmp_path, config, output, 2, "mfa")


def test_missing_ambiguous_oov_are_not_partial_labels():
    vocab = load_phoneme_vocab(VOCAB)
    for dictionary in [
        {"twin": ["t w ɪ n"]},
        {"twin": ["t w ɪ n"], "peaks": ["p i k s", "p ɛ k s"]},
    ]:
        label = audit.phrase_label(["twin", "peaks"], dictionary, vocab)
        assert label["status"] == "unresolved" and label["raw_pronunciation"] is None
    label = audit.phrase_label(["bad"], {"bad": ["🙂"]}, vocab)
    assert label["status"] == "unresolved" and label["oov_units"]
    a = variant("Night", "n aj t")
    b = audit.phrase_label(["night"], {"night": ["n a j t̪"]}, vocab)
    assert audit.compare_label(a, b)["status"] == "ctc_different"


@pytest.mark.parametrize(
    "field",
    [
        "model_sha256",
        "table_sha256",
        "vocab_sha256",
        "training_dictionary_sha256",
        "expected_english_targets",
    ],
)
def test_identity_and_scope_fail_before_execution(tmp_path, monkeypatch, field):
    config = setup(tmp_path)
    cfg = json.loads(config.read_text())
    cfg[field] = 999 if field.startswith("expected") else "0" * 64
    write(config, cfg)
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: pytest.fail("MFA must not run"))
    with pytest.raises(ValueError):
        audit.run_audit(tmp_path, config, tmp_path / "new", 2, "mfa")
    assert not (tmp_path / "new").exists()


def test_reject_filler_and_duplicate_target():
    vocab = load_phoneme_vocab(VOCAB)
    v = variant("Night", "n aj t")
    table = {"entries": [{"id": "night", "variants": [v]}]}
    assert len(audit.english_targets(table, vocab)) == 1
    duplicate = deepcopy(table)
    duplicate["entries"].append(duplicate["entries"][0])
    with pytest.raises(ValueError, match="duplicate"):
        audit.english_targets(duplicate, vocab)
    v["role"] = "filler"
    with pytest.raises(ValueError, match="target-only"):
        audit.english_targets(table, vocab)


def test_actual_training_word_normalization_and_dictionary_keys(tmp_path):
    vocab = load_phoneme_vocab(VOCAB)
    v = variant("O’Brien Twin—Peaks", "ow b ɹ aj ə n t w ɪ n p iː k s")
    table = {"entries": [{"id": "compound", "variants": [v]}]}
    assert audit.english_targets(table, vocab)[0]["words"] == ["o'brien", "twin-peaks"]
    dictionary = tmp_path / "raw.dict"
    dictionary.write_text("O’Brien\tow b ɹ aj ə n\no'brien\tow b ɹ aj ə n\n")
    assert audit._dictionary(dictionary) == {"o'brien": ["ow b ɹ aj ə n"]}


def test_process_failure_and_source_mutation_leave_failed_sealed_report(tmp_path, monkeypatch):
    config = setup(tmp_path)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(audit.subprocess, "run", fail)
    report = audit.run_audit(tmp_path, config, tmp_path / "failed", 2, "mfa")
    assert report["status"] == "failed"
    assert (tmp_path / "failed/return_bundle.zip").is_file()

    def mutate(command, **kwargs):
        mock_mfa(command, **kwargs)
        (tmp_path / "assets/training.dict").write_text("lie\tl aj\n")

    monkeypatch.setattr(audit.subprocess, "run", mutate)
    report = audit.run_audit(tmp_path, config, tmp_path / "changed", 2, "mfa")
    assert report["status"] == "failed" and "changed during audit" in report["error"]
    assert not (tmp_path / "changed/comparisons.json").exists()
