import hashlib
import json
import runpy
from pathlib import Path

PROBE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/inspect_zh_en_delivery_inputs.py")
)


def test_surface_coverage_requires_real_mapped_wav(tmp_path):
    src = "/old/f.wav"
    dest = hashlib.md5(src.encode()).hexdigest()[:16] + ".wav"
    (tmp_path / dest).touch()
    (tmp_path / "missing-from-index.wav").touch()
    path = tmp_path / "index.jsonl"
    path.write_text(
        json.dumps({"audio_path": src, "entities": [{"surface": "A"}, {"surface": "B"}]}) + "\n"
    )
    result = PROBE["english"](path, tmp_path, hashlib.sha256(b"A\nB").hexdigest())
    assert result["mapping"]["basename"]["unique_wavs_joined"] == 0
    assert result["mapping"]["md5_source_path"] == dict(
        unique_wavs_joined=1,
        directory_wavs_unmatched=1,
        distinct_target_words=2,
        exact_583_surface_set=True,
    )


def test_home_counts_val_only_and_flags_negative_labels(tmp_path):
    wav = tmp_path / "a.wav"
    wav.touch()
    row = dict(id="val_n1_zh_00001", wav=str(wav), keywords=["unexpected"], status="ok")
    path = tmp_path / "meta.jsonl"
    path.write_text("\n".join(json.dumps(v) for v in [row, dict(row, id="train_zh_00001")]))
    result = PROBE["home"](path)
    assert result["groups"] == {"zh/val_n1": 1}
    assert result["issues"] == {"unexpected_keyword_field_or_polarity": 1}
