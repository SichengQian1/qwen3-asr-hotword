from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from qwen_hotword.evaluation import joint_mfa_dict as audit
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _key
from qwen_hotword.training.spanish_capacity import _sha


def variant(surface, language, role="target"):
    return {
        "surface": surface,
        "normalized": _key(surface),
        "language": language,
        "decision": "retained",
        "role": role,
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def setup(root):
    groups = [
        [variant("Twin Twin", "en")],
        [variant("el álbum", "en"), variant("el álbum", "es")],
        [variant("Adrián", "es")],
        [variant("São Paulo", "pt-BR")],
        [variant("filler", "en", "filler")],
        [variant("U2", "en")],
    ]
    table = {"entries": [
        {"surface": g[0]["surface"], "normalized": g[0]["normalized"], "variants": g}
        for g in groups
    ]}
    source = root / "source"
    write(source / "reviewed_keyword_table.json", table)
    digest = _sha(source / "reviewed_keyword_table.json")
    write(source / "report.json", {
        "status": "completed", "candidate_scope": "joint_multilingual",
        "review": {"reviewed_table_sha256": digest, "active_unique_surfaces_per_audio": 6},
    })
    seal(source)
    models = {}
    for lang in audit.LANGUAGES:
        path = root / "models" / f"{lang}.zip"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"mock only: {lang}".encode())
        models[lang] = {"path": str(path), "sha256": _sha(path)}
    config = root / "config" / "input.json"
    write(config, {
        "source_run": "source", "table_sha256": digest,
        "expected_surfaces": 6, "models": models,
    })
    return config, table


def mock_mfa(command, **kwargs):
    assert command[1:6] == ["g2p", "--num_jobs", "2", "--num_pronunciations", "1"]
    lang = Path(command[-2]).stem
    words = Path(command[-3]).read_text().splitlines()
    assert words == sorted(set(words)) and "u" not in words
    dictionaries = {
        # Simulate equal record count but altered accented word keys.
        "en": "twin\tt w ɪ n\nfiller\tf ɪ l\nel\tɛ ɫ\nlbum\tb\n",
        "es": "adrián\ta d ɾ i a n\nel\te l\nálbum\ta l β u m\n",
        "pt": "são\ts ɐ̃ w\npaulo\tp a w l u\n",
    }
    Path(command[-1]).write_text(dictionaries[lang], encoding="utf-8")
    kwargs["stdout"].write("mock only\n")
    work = Path(kwargs["env"]["MFA_ROOT_DIR"])
    work.mkdir(parents=True)
    (work / "scratch").write_text("excluded from checksum delivery")


def test_all_terms_one_dict_preserves_languages_phrases_and_missing(tmp_path, monkeypatch):
    config, _ = setup(tmp_path)
    before = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs["env"]["MFA_ROOT_DIR"]))
        mock_mfa(command, **kwargs)

    monkeypatch.setattr(audit.subprocess, "run", execute)
    output = tmp_path / "new"
    report = audit.run(tmp_path, config, output, 2, "mfa")
    assert report["status"] == "completed_with_unresolved"
    assert report["unique_surfaces"] == 6 and report["language_surface_rows"] == 7
    assert len(calls) == len({work for _, work in calls}) == 3
    assert [Path(c[-2]).stem for c, _ in calls] == ["en", "es", "pt"]
    dictionary = (output / "joint_4000_mfa.dict").read_text()
    assert "en\tTwin Twin\tt w ɪ n t w ɪ n\n" in dictionary
    assert "en\tfiller\tf ɪ l\n" in dictionary
    assert "es\tel álbum\te l a l β u m\n" in dictionary
    assert "pt\tSão Paulo\ts ɐ̃ w p a w l u\n" in dictionary
    assert "# UNRESOLVED\ten\tel álbum\tmissing_words=álbum\n" in dictionary
    assert "# UNRESOLVED\ten\tU2\tnumeric_or_empty_lexical_input;missing_words=u\n" in dictionary
    assert report["by_language"]["en"]["missing_input_words"] == ["álbum"]
    assert report["by_language"]["en"]["unexpected_output_words"] == ["lbum"]
    assert all(_sha(p) == digest for p, digest in before.items())
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert "mfa_work" not in name and _sha(output / name) == digest
    with pytest.raises(FileExistsError):
        audit.run(tmp_path, config, output, 2, "mfa")
    assert len(calls) == 3


def test_phrase_order_apostrophe_duplicate_variants_and_aliases(tmp_path):
    _, table = setup(tmp_path)
    table["entries"][0]["variants"].append(deepcopy(table["entries"][0]["variants"][0]))
    terms = audit.table_terms(table, 6)
    assert len(terms) == 7
    term = {**terms[0], "surface": "O’Brien O’Brien", "words": ["o'brien", "o'brien"]}
    dictionary = tmp_path / "raw.dict"
    dictionary.write_text("o'brien\tow b ɹ aj ə n\n")
    lines, stats = audit.reconstruct([term], dictionary)
    assert "O’Brien O’Brien\tow b ɹ aj ə n ow b ɹ aj ə n" in lines[0]
    assert stats["generated"] == 1
    dictionary.write_text("o'brien\tow\no'brien\taw\n")
    lines, stats = audit.reconstruct([term], dictionary)
    assert lines[0].startswith("# UNRESOLVED") and "ambiguous_words=o'brien" in lines[0]
    assert stats["unresolved"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "language", "surface", "removed", "empty"])
def test_invalid_table_rejected(tmp_path, fault):
    _, table = setup(tmp_path)
    if fault == "duplicate":
        table["entries"].append(deepcopy(table["entries"][0]))
    elif fault == "empty":
        table["entries"][0]["variants"] = []
    else:
        v = table["entries"][0]["variants"][0]
        v[{"language": "language", "surface": "surface", "removed": "decision"}[fault]] = {
            "language": "pt-PT", "surface": "Twin\tTwin", "removed": "removed_basic"
        }[fault]
    with pytest.raises(ValueError):
        audit.table_terms(table, 6)


@pytest.mark.parametrize("fault", ["model", "table", "count", "output", "jobs"])
def test_preflight_without_mfa_execution(tmp_path, monkeypatch, fault):
    config, _ = setup(tmp_path)
    output = tmp_path / "new"
    jobs = 2
    if fault == "model":
        (tmp_path / "models/pt.zip").write_bytes(b"different")
    elif fault == "table":
        (tmp_path / "source/reviewed_keyword_table.json").write_text("{}")
    elif fault == "count":
        cfg = json.loads(config.read_text())
        cfg["expected_surfaces"] = 4000
        write(config, cfg)
    elif fault == "output":
        output = tmp_path / "source/nested"
    else:
        jobs = 0
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: pytest.fail("must not run MFA"))
    with pytest.raises(ValueError):
        audit.run(tmp_path, config, output, jobs, "mfa")
    assert not output.exists()


@pytest.mark.parametrize("fault", ["mfa_failure", "changed_input"])
def test_failure_retains_logs_but_does_not_publish_dictionary(tmp_path, monkeypatch, fault):
    config, _ = setup(tmp_path)

    def execute(command, **kwargs):
        mock_mfa(command, **kwargs)
        if fault == "mfa_failure":
            raise subprocess.CalledProcessError(1, command)
        (tmp_path / "models/en.zip").write_bytes(b"changed during run")

    monkeypatch.setattr(audit.subprocess, "run", execute)
    output = tmp_path / "failed"
    report = audit.run(tmp_path, config, output, 2, "mfa")
    assert report["status"] == "failed" and report["error"]
    assert not (output / "joint_4000_mfa.dict").exists()
    assert (output / "work/en/mfa.log").exists()
    assert (output / "report.json").exists() and (output / "sha256.txt").exists()
