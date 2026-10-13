from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("mismatch_mfa", "scripts/run_mismatch_mfa.py")
assert SPEC and SPEC.loader
mfa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mfa)


def test_phrase_roundtrip_preserves_order_accents_and_missing_words(tmp_path):
    source = tmp_path / "phrases.txt"
    source.write_text("\ufeffEl nin\u0303o\n\nfather s day\nel niño el\n", encoding="utf-8")
    rows = mfa.read_terms(source)
    assert rows[0]["words"] == ["el", "niño"]
    assert rows[0]["original"] == "El nin\u0303o"
    assert rows[1]["input_line"] == 3
    dictionary = tmp_path / "raw.dict"
    dictionary.write_text("el\te l\nniño\tn i ɲ o\nfather\tf a\nday\td e\n")
    results = mfa.reconstruct(rows, dictionary)
    assert results[0]["raw_phrase_pronunciation"] == "e l n i ɲ o"
    assert results[1]["raw_phrase_pronunciation"] is None
    assert results[1]["missing_words"] == ["s"]
    assert results[2]["raw_phrase_pronunciation"] == "e l n i ɲ o e l"
    dictionary.write_text(dictionary.read_text() + "el\ta l\n")
    assert mfa.reconstruct(rows, dictionary)[0]["ambiguous_words"] == ["el"]


def inputs(root):
    for lang, filename in mfa.MODELS.items():
        (root / f"mismatch_words_{lang}.txt").write_text("Twin Peaks\nTwin Peaks\nunknown\n")
        model = root / "models/mfa/g2p" / filename
        model.parent.mkdir(parents=True, exist_ok=True)
        model.write_bytes(b"fake model for subprocess mock only")


def test_all_languages_isolated_outputs_missing_report_and_no_overwrite(tmp_path, monkeypatch):
    inputs(tmp_path)
    before = {p: mfa.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    calls = []

    def mock_run(command, **kwargs):
        calls.append((command, kwargs["env"]["MFA_ROOT_DIR"]))
        assert Path(command[-3]).read_text() == "peaks\ntwin\nunknown\n"
        Path(command[-1]).write_text("twin\tt w ɪ n\npeaks\tp i k s\n")
        kwargs["stdout"].write("mock success\n")

    monkeypatch.setattr(mfa.subprocess, "run", mock_run)
    output = tmp_path / "new_output"
    report = mfa.run(tmp_path, output, 2, "mfa")
    assert report["status"] == "completed_with_unresolved"
    assert len(calls) == len({directory for _, directory in calls}) == 3
    assert [Path(c[0][-2]).name for c in calls] == list(mfa.MODELS.values())
    for lang in mfa.MODELS:
        assert report["by_language"][lang]["generated_terms"] == 2
        assert report["by_language"][lang]["missing_words"] == ["unknown"]
        result = json.loads((output / lang / "phrases.json").read_text())
        assert result[0]["original"] == "Twin Peaks"
        assert result[0]["raw_phrase_pronunciation"] == "t w ɪ n p i k s"
    assert all(mfa.sha(p) == digest for p, digest in before.items())
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert mfa.sha(output / name) == digest
    with pytest.raises(FileExistsError):
        mfa.run(tmp_path, output, 2, "mfa")
    assert len(calls) == 3


def test_preflight_and_mfa_failure_are_explicit(tmp_path, monkeypatch):
    inputs(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        mfa.run(tmp_path, tmp_path / "bad", 0, "mfa")
    assert not (tmp_path / "bad").exists()

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(mfa.subprocess, "run", fail)
    report = mfa.run(tmp_path, tmp_path / "failed", 1, "mfa")
    assert report["status"] == "failed" and "error" in report
    assert not (tmp_path / "failed/en/phrases.json").exists()
    (tmp_path / "models/mfa/g2p/portuguese_brazil_mfa.zip").unlink()
    with pytest.raises(FileNotFoundError, match="missing local model"):
        mfa.run(tmp_path, tmp_path / "not_created", 1, "mfa")
    assert not (tmp_path / "not_created").exists()
