import importlib
import io
import json
import zipfile
from pathlib import Path

from qwen_hotword.evaluation.english_group_recall import Label
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.phonemes.mfa_v03_assets import write_json
from qwen_hotword.training.zh_en_common import digest


def test_split_packages_reproduce_timelines_without_gold_leakage(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    app = importlib.import_module("export_zh_en_group_delivery")
    monkeypatch.setattr(app, "ROOT", tmp_path)
    src = tmp_path / "outputs" / "source"
    labels = {
        g: Label(g, g, "洪崖洞" if g == "cross" else "", g == "cross")
        for g in ("regular", "cross", "navigation", "review")
    }
    results = {}
    samples = []
    for i, (key, uid, targets) in enumerate(
        [
            ("home@@en", "1", ["cross", "regular"]),
            ("home@@en", "2", ["review"]),
            ("tts@@en", "1", ["navigation"]),
            ("neg@@en", "1", []),
        ]
    ):
        s = dict(
            key=key,
            group=key.split("@@")[0],
            language="en",
            sample_id=uid,
            audio_sha256=str(i) * 64,
            targets=targets,
            step_count=2,
            duration_seconds=3.5,
            audio_path=f"{i}.wav",
        )
        samples.append(s)
        steps = [[], [dict(word=f"other{i}{j}", phoneme="a", score=1 - j * 0.05) for j in range(7)]]
        results[key, uid] = steps
        value = dict(steps=steps, raw_last=[])
        write_json(
            app.cached(src, s),
            dict(
                run=app.RUN,
                audio=s["audio_sha256"],
                language="en",
                digest=digest(value),
                result=value,
            ),
        )
    monkeypatch.setattr(app, "grouped_inputs", lambda: (src, labels, samples, {}, {}))
    monkeypatch.setattr(app, "EXPECTED", dict.fromkeys(app.GROUPS, (1, 1)))
    app.main()
    with zipfile.ZipFile(
        next((tmp_path / "outputs").glob("zh_en480_group_delivery_*/grouped_delivery.zip"))
    ) as outer:
        for g in app.GROUPS:
            line = json.loads(outer.read(g + "/evaluation_targets.jsonl"))
            assert [x["surface"] for x in line["targets"]] == ["cross" if g == "multi" else g]
            with zipfile.ZipFile(io.BytesIO(outer.read(g + "/delivery.zip"))) as inner:
                ctx, dur, manifest = (
                    json.loads(inner.read("delivery/" + name + ".json"))
                    for name in ("stream_contexts", "durations", "audio_manifest")
                )
                key, uid = line["key"], line["sample_id"]
                assert validate_contexts(ctx, dur)["unique_utterances"] == 1
                assert "targets" not in manifest[key][uid]
                for m, k in MODES.items():
                    assert ctx[m][key][uid] == [step[:k] for step in results[key, uid]]
