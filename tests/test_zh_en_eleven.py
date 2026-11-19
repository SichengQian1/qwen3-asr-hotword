from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.evaluation import zh_en_eleven as sets
from qwen_hotword.inference.stream_context_contract import validate_contexts
from qwen_hotword.phonemes.mfa_v03_assets import write_json
from qwen_hotword.training.zh_en_common import digest

ROOT = Path(__file__).resolve().parents[1]


def test_aishell_entities_respect_spans_and_count_pairs(tmp_path: Path) -> None:
    tagged, ref = tmp_path / "gold", tmp_path / "ref"
    tagged.write_text("x (北京)与(北京)和[张三]还有北京大学\ny <中>\n")
    ref.write_text("y 中\nx 北京 与 北京 和 张三 还有 北京大学\n")
    gold, mentions = sets.full_gold(tagged, ref, {"北京", "张三", "京大"})
    assert gold == {"x": ["北京", "张三"], "y": []} and mentions == 3
    ref.write_text("x 别的话\ny 中\n")
    with pytest.raises(ValueError, match="text/markers"):
        sets.full_gold(tagged, ref, set())


def test_probe_uses_joined_audio_annotations_and_audits_missing(tmp_path: Path) -> None:
    source = "/old/whole/path.wav"
    uid = hashlib.md5(source.encode()).hexdigest()[:16]
    (tmp_path / (uid + ".wav")).touch()
    row = {
        "audio_path": source,
        "entities": [{"surface": "First"}, {"surface": "SECOND"}, {"surface": "outside"}],
    }
    index = tmp_path / "index.jsonl"
    index.write_text("\n".join(json.dumps(r) for r in [row, row, {**row, "audio_path": "missing"}]))
    gold, audit = sets.probe_gold(index, tmp_path, {"First", "Second", "Third"})
    assert gold == {uid: ["first", "second"]}
    assert audit["missing_words"] == ["Third"] and audit["covered_words"] == 2
    assert audit["duplicate_index_rows"] == audit["index_rows_without_directory_wav"] == 1
    index.write_text(index.read_text() + "\n" + json.dumps({**row, "entities": []}))
    with pytest.raises(ValueError, match="conflicting"):
        sets.probe_gold(index, tmp_path, {"First"})


def test_eleven_dataset_delivery_and_negative_rates(tmp_path: Path, monkeypatch: Any) -> None:
    spec = importlib.util.spec_from_file_location(
        "eleven_cli", ROOT / "scripts/run_zh_en_480_eleven.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # Full test and subset reuse audio, with different target annotations.
    common = []
    extras = []
    for g in sets.DATASETS:
        uid = "shared" if g.startswith("zh-aishell") else g
        s = dict(
            group=g,
            key=f"{g}@@{g[:2]}",
            language=g[:2],
            sample_id=uid,
            audio_sha256=hashlib.sha256(uid.encode()).hexdigest(),
            duration_seconds=1.0,
            step_count=1,
            targets=[] if "val_n" in g else ["target"],
        )
        (extras if g in (sets.FULL, sets.PROBE) else common).append(s)
    monkeypatch.setattr(
        cli,
        "identity",
        lambda *_: (
            {
                "checkpoint_epoch": 25,
                "model_path": "x",
                "checkpoint_path": "y",
                "checkpoint_sha256": "fake",
            },
            None,
            None,
            None,
        ),
    )
    monkeypatch.setattr(cli.base, "registry", lambda *_: ([], {}))
    monkeypatch.setattr(cli.base, "inventory", lambda *_: common)
    monkeypatch.setattr(sets, "official_file", lambda *_: Path("gold"))
    monkeypatch.setattr(sets, "extra_inventory", lambda *_: (extras, {}))
    monkeypatch.setattr(cli, "check_gpu", lambda *_: None)
    monkeypatch.setattr(cli.external, "_load_detector", lambda **_: object())
    calls = []

    def infer(*args: Any) -> dict[str, Any]:
        calls.append(args[-1]["sample_id"])
        return {
            "steps": [[{"word": "target", "score": 1.0, "phoneme": "t"}]],
            "raw_last": ["target"],
        }

    monkeypatch.setattr(cli.base, "retrieve", infer)
    cli.run(tmp_path, 5, None)
    cli.run(tmp_path, 5, None)
    assert len(calls) == 10
    report = json.loads((tmp_path / "report.json").read_text())
    metrics = report["by_dataset"]
    assert set(metrics) == set(sets.DATASETS)
    assert metrics["en-smarthome-val_n2"]["gated_top7_last"]["target_recall"] is None
    assert metrics["en-smarthome-val_n2"]["gated_top7_last"]["negative_prompt_rate"] == 1
    assert report["en_tts_macro_word_recall"]["raw_top5_last"] == 1
    assert len((tmp_path / "by_dataset.tsv").read_text().splitlines()) == 12
    with zipfile.ZipFile(tmp_path / "delivery.zip") as z:
        durations = json.loads(z.read("delivery/durations.json"))
        assert (
            validate_contexts(json.loads(z.read("delivery/stream_contexts.json")), durations)[
                "unique_utterances"
            ]
            == 11
        )
        assert "targets" not in z.read("delivery/audio_manifest.json").decode()
    result_path = cli.cached(tmp_path, common[0])
    saved = json.loads(result_path.read_text())
    saved["digest"] = digest([])
    write_json(result_path, saved)
    with pytest.raises(ValueError, match="incompatible"):
        cli.run(tmp_path, 5, None)
