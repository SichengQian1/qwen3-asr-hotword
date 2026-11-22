"""Ensure mic continuity, acoustic identity and bounded verified-only cache reuse."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.demo_v2 import runtime
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.zh_en_common import digest, save_rows


def studio():
    d = runtime.Studio.__new__(runtime.Studio)
    d.extras = {}
    d.omit = ""
    d.entries = ()
    d.clips = []
    d.checked = set()
    d.init_studio()

    def configure(w, extras):
        d.extras = dict(extras)
        d.omit = w

    d.configure = configure
    d.start = lambda: setattr(d, "pcm", bytearray())

    def step(data, final):
        d.pcm.extend(data)
        return dict(
            left="甲", right="乙" if d.extras else "甲", candidates=[{"word": w} for w in d.extras]
        )

    d.step = step
    return d


def test_replay_and_mode_isolation():
    d = studio()
    w = "乙"
    d.clips = [{"word": "丙"}, {"word": "丁"}]
    d.pcm_for = lambda _: b"sample"
    d.begin_pass("mic")
    d.record_step(b"original", True)
    d.extras[w] = SimpleNamespace(pronunciation="i")
    d.begin_pass("replay")
    d.record_step(b"original", True)
    d.select(0)
    assert not d.extras and not d.history
    d.extras["丙"] = object()
    d.select(1)
    assert not d.extras
    d.select(-1)
    assert list(d.extras) == [w] and len(d.history) == 2
    d.remove(w)
    d.begin_pass("replay")
    with pytest.raises(ValueError, match="不一致"):
        d.record_step(b"different", True)
    assert len(d.history) == 2
    d.begin_pass("replay")
    d.record_step(b"original", True)
    assert len({r["audio"] for r in d.history}) == 1
    assert [r["right"] for r in d.history] == ["甲", "乙", "甲"]
    assert d.history[-1]["removed"] == [w]
    d.begin_pass("mic")
    d.record_step(b"new", True)
    assert len(d.history) == 1 and d.history[0]["removed"] == []


def test_hunt_resumes_and_caps_at_ten(tmp_path, monkeypatch):
    d = studio()
    d.root = d.out = tmp_path
    d.kw = {}
    d.checkpoint = "m"
    monkeypatch.setenv("DEMO_DECODER_PY", sys.executable)
    import subprocess

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: "vllm torch")
    monkeypatch.setattr(runtime, "version", lambda x: x)
    train = tmp_path / "outputs/zh_en_183_noise70_dual_v2"
    for name in ("model_identity.json", "contract.json"):
        write_json(train / name, {})
    cache = tmp_path / "outputs/zh_en183_480_joint2351_eleven_v1"
    write_json(
        cache / "report.json", dict(status="completed", checkpoint_sha256="m", plan_sha256="p")
    )
    source = dict(previous_subset="ais.jsonl", home_meta=str(tmp_path / "home.jsonl"))
    write_json(tmp_path / "configs/zh_en_stream2351_probe.workzone.json", source)
    write_json(
        tmp_path / "configs/aishell_review_1054.v1.json",
        dict(candidate_surfaces=[], excluded_target_pairs=[]),
    )
    save_rows(tmp_path / "ais.jsonl", [])
    home = []
    for i in range(14):
        w = "姓名" + chr(0x4E00 + i)
        p = tmp_path / f"{i}.wav"
        p.write_bytes(w.encode())
        home.append(
            dict(id=f"val_v1_zh_{i}", status="ok", dur_sec=4, keywords=[w], text=w, wav=str(p))
        )
        a = sha(p)
        body = dict(steps=[[dict(word=w)]])
        write_json(
            cache / "audio_results/zh" / a[:2] / (a + ".json"),
            dict(run="p", audio=a, language="zh", result=body, digest=digest(body)),
        )
    save_rows(tmp_path / "home.jsonl", home)
    d.pcm_for = lambda r: Path(r["audio_path"]).read_bytes()
    d.add = lambda w: d.extras.update({w: True})

    def infer(pcm):
        w = pcm.decode()
        text = w if d.extras or w == home[0]["text"] else "错"
        return dict(left="错" if w != home[0]["text"] else w, right=text, seconds=4), list(d.extras)

    d.run_clip = infer
    d.hunt(4)
    assert len(d.clips) == 3
    d.checked = set()
    d.clips = []
    d.hunt(20)
    assert len(d.clips) == 10 and len(d.checked) == 11
    d.clips = []
    d.checked = set()
    d.run_clip = lambda _: pytest.fail("cache hit should not decode")
    d.hunt(20)
    assert len(d.clips) == 10
