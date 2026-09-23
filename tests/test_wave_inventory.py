from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.evaluation.wave_inventory import inspect_waves, preview
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")


def make_inputs(root):
    for wave in range(1, 5):
        for lang in ("es", "pt"):
            base = root / f"wave{wave}" / lang
            (base / "wav").mkdir(parents=True)
            (base / "wav/shared.wav").write_bytes(b"not decoded by inventory")
            (base / "transcripts.txt").write_text("shared\ttexto\n")
            (base / f"{lang}_keyword_bias_phoneme.json").write_text(
                json.dumps(
                    {
                        "keyword_sets": {"baseline": [], "all_keywords": ["Casa"]},
                        "keyword_phonemes": {"Casa": "k a s a"},
                    }
                )
            )
            (base / f"{lang}_phonetic_neighbors_phoneme.json").write_text(
                json.dumps({"metadata": {}, "items": [{"word": "Casa", "neighbors": ["caza"]}]})
            )


def test_inventory_unions_all_waves_and_does_not_decode_or_modify(tmp_path):
    make_inputs(tmp_path)
    before = {str(p): _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = inspect_waves(tmp_path, VOCAB)
    assert report["issues"] == []
    assert not report["model_loaded"] and not report["audio_bytes_read"]
    assert not report["tables_created"]
    assert len(report["datasets"]) == 8
    for lang in ("es", "pt"):
        union = report["primary_unions"][lang]
        assert union["normalized_union_count"] == 1
        assert union["complete_primary_files"] == 4
        assert union["remaining_to_4000"] == 3999
        assert union["cross_wave_shared_stem_count"] == 1
    assert before == {str(p): _sha(p) for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("problem", ["conflict", "oov", "schema", "missing_audio"])
def test_inventory_reports_issues_without_silently_releasing_tables(tmp_path, problem):
    make_inputs(tmp_path)
    base = tmp_path / "wave4/pt"
    path = base / "pt_keyword_bias_phoneme.json"
    raw = json.loads(path.read_text())
    if problem == "conflict":
        raw["keyword_phonemes"]["Casa"] = "k a z a"
    elif problem == "oov":
        raw["keyword_phonemes"]["Casa"] = "☃"
    elif problem == "schema":
        raw = {"different": []}
    else:
        (base / "transcripts.txt").write_text("other\ttexto\n")
    path.write_text(json.dumps(raw))
    report = inspect_waves(tmp_path, VOCAB)
    assert report["issues"] and not report["tables_created"]
    if problem == "conflict":
        assert report["primary_unions"]["pt"]["phoneme_conflict_count"] == 1
    elif problem in {"oov", "schema"}:
        assert report["primary_unions"]["pt"]["counts_are_partial"]


def test_duplicate_json_keys_rejected_and_preview_bounded(tmp_path):
    make_inputs(tmp_path)
    path = tmp_path / "wave1/es/es_keyword_bias_phoneme.json"
    path.write_text('{"keyword_sets":{},"keyword_sets":{}}')
    report = inspect_waves(tmp_path, VOCAB)
    assert "duplicate JSON key" in report["issues"][0]["error"]
    assert len(json.dumps(preview(["x" * 10000] * 10000))) < 1000
