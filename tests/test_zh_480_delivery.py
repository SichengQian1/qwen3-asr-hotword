import importlib
import json
import zipfile
from pathlib import Path

import pytest


def test_chinese_only_preserves_timelines_and_shared_aishell_id(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    app = importlib.import_module("export_zh_480_delivery")
    counts = dict.fromkeys(app.COUNTS, 1)
    keys = [*counts, "en-tts-583@@en"]
    durations = {k: {"u": 3.0} for k in keys}
    manifest = {
        k: {"u": dict(key=k, language=k.split("@@")[1], sample_id="u", duration_seconds=3.0)}
        for k in keys
    }
    contexts = {
        m: {k: {"u": [[], [dict(word="English", phoneme="e", score=1.0)]]} for k in keys}
        for m in app.MODES
    }
    source = tmp_path / "original.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for name, value in [
            ("durations", durations),
            ("audio_manifest", manifest),
            ("stream_contexts", contexts),
        ]:
            archive.writestr(f"delivery/{name}.json", json.dumps(value))
    with zipfile.ZipFile(app.export(source, tmp_path, counts)) as archive:
        for name, value in [("durations", durations), ("audio_manifest", manifest)]:
            assert json.loads(archive.read(f"delivery/{name}.json")) == {
                k: value[k] for k in counts
            }
        assert json.loads(archive.read("delivery/stream_contexts.json")) == {
            m: {k: contexts[m][k] for k in counts} for m in app.MODES
        }
    with pytest.raises(ValueError, match="counts"):
        app.export(source, tmp_path, dict.fromkeys(counts, 2))
