from __future__ import annotations

import importlib
import json
import zipfile
from pathlib import Path

from qwen_hotword.evaluation.english_group_recall import (
    VIEWS,
    Label,
    Word,
    policy,
    summary,
    views,
    worst,
)
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.zh_en_common import digest, rows

ROOT = Path(__file__).resolve().parents[1]


def test_registered_policy_and_topk_before_alias_credit():
    labels = policy(
        ROOT / "configs/zh_en_english_eval_groups.v1.tsv",
        list(rows(ROOT / "configs/zh_en_joint2351.v1.jsonl")),
    )
    assert len(labels) == 690 and labels["hongyadong"] == Label(
        "Hongyadong", "cross", "洪崖洞", False
    )
    assert labels["call back"].group == "regular"
    assert labels["al-malaz 503"].multi
    data = views(
        dict(
            raw_last=["a", "b", "c", "d", "e", "洪崖洞", "Hongyadong"],
            steps=[[{"word": "Hongyadong"}], [{"word": "洪崖洞"}]],
        )
    )
    w = Word()
    w.add("hongyadong", "洪崖洞", data)
    assert [w.strict[v] for v in VIEWS] == [0, 0, 1, 0]
    assert [w.entity[v] for v in VIEWS] == [0, 1, 1, 1]
    w2 = Word()
    for _ in range(3):
        w2.add("x", "", [[], [], [], []])
    assert summary([w, w2], "entity", True)[-1] == 0.5
    assert summary([w, w2], "entity", False)[-1] == 0.25
    assert worst({"hongyadong": w, "li xin": w2, "call back": w2}, labels) == [
        "li xin",
        "hongyadong",
    ]


def test_workzone_entry_packages_checked_audio(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    app = importlib.import_module("report_zh_en_480_groups")
    monkeypatch.setattr(app, "ROOT", tmp_path)
    src = tmp_path / "outputs/zh_en183_480_joint2351_eleven_v1"
    cfg = tmp_path / "configs"
    cfg.mkdir()
    table = cfg / "zh_en_joint2351.v1.jsonl"
    table.write_text((ROOT / "configs" / table.name).read_text())
    (cfg / "zh_en_english_eval_groups.v1.tsv").write_bytes(
        (ROOT / "configs/zh_en_english_eval_groups.v1.tsv").read_bytes()
    )
    meta = tmp_path / "meta"
    home = "en-smarthome-val_v1"
    uid = "val_v1_en_007"
    meta.write_text(json.dumps(dict(id=uid, text="Please find Hongyadong.")))
    index = tmp_path / "index"
    index.write_text("fixture")
    write_json(
        cfg / "zh_en_stream2351_probe.workzone.json",
        dict(home_meta=str(meta), work=str(tmp_path), en_index=index.name, en_wav=str(tmp_path)),
    )
    samples = {}
    cases = [(home, uid, "hongyadong"), (app.PROBE, "001", "al-malaz 503")]
    for group, sample, _word in cases:
        wav = tmp_path / (sample + ".wav")
        wav.write_bytes(b"fixture-" + sample.encode())
        samples[group] = dict(
            group=group,
            key=group + "@@en",
            sample_id=sample,
            language="en",
            audio_path=str(wav),
            audio_sha256=sha(wav),
        )
    audit = {"index_sha256": sha(index)}
    plan = dict(
        audit={app.PROBE: audit},
        table_sha256=sha(table),
        common_samples_sha256=digest([{**samples[home], "targets": ["hongyadong"]}]),
    )
    monkeypatch.setattr(app, "RUN", digest(plan))
    write_json(src / "run_config.json", plan)
    monkeypatch.setattr(app, "probe_gold", lambda *args: ({"001": ["al-malaz 503"]}, audit))
    monkeypatch.setattr(
        app.base,
        "inputs",
        lambda config: [(home, uid, Path(samples[home]["audio_path"]), ["Hongyadong"])],
    )
    metrics = {}
    for group, _sample, word in cases:
        s = samples[group]
        result = dict(raw_last=[word], steps=[[dict(word="洪崖洞")]])
        write_json(
            app.cached(src, s),
            dict(
                audio=s["audio_sha256"],
                language="en",
                run=app.RUN,
                digest=digest(result),
                result=result,
            ),
        )
        metrics[group] = {
            v: dict(
                target_pairs=1,
                target_hits=int(v.startswith("raw")),
                negative_with_candidate=0,
                negative_utterances=0,
            )
            for v in VIEWS
        }
    write_json(
        src / "report.json", dict(plan_sha256=app.RUN, status="completed", by_dataset=metrics)
    )
    with zipfile.ZipFile(src / "delivery.zip", "w") as z:
        z.writestr(
            "delivery/audio_manifest.json",
            json.dumps({s["key"]: {s["sample_id"]: s} for s in samples.values()}),
        )
    app.main()
    with zipfile.ZipFile(
        next((tmp_path / "outputs").glob("zh_en480_english_groups_*/listening.zip"))
    ) as z:
        assert z.read(f"audio/01_{uid}.wav") == Path(samples[home]["audio_path"]).read_bytes()
        assert "Hongyadong　词面 0/1　实体 1/1" in z.read("listen.md").decode()
        assert (
            "Hongyadong／洪崖洞\t实体\t逐目标\t1\t1\t100.00%\t100.00%\t100.00%\t100.00%"
            in z.read("groups.tsv").decode()
        )
