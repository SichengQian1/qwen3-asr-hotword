import json
from types import SimpleNamespace

import pytest

from qwen_hotword.demo_v2.session import SharedStudio


def test_shared_words_actual_index_and_historical_snapshots(tmp_path, monkeypatch):
    from qwen_hotword.demo import runtime
    from qwen_hotword.hotwords.registry import HotwordEntry

    def entry(word, id):
        return HotwordEntry(id, "zh", word, word, (word,), "a", ("a",), (2,), "test", 0)

    d = SharedStudio.__new__(SharedStudio)
    d.out = tmp_path
    d.base = (entry("甲", "1"), entry("乙", "2"))
    d.extras = {}
    d.base_owners = {e.hotword_id: e for e in d.base}
    d.det = SimpleNamespace(anchor_index=SimpleNamespace(config=None))
    d.clips = [
        dict(word=w, sample_id=str(i), demo_source="样例", after={"seconds": 1})
        for i, w in enumerate(("甲", "乙"))
    ]
    d.init_studio()
    d.select(0)

    def add(w):
        d.configure(d.omit, {**d.extras, w: entry(w, "new" + w)})
        return {"word": w}

    d.add = add

    def step(self, data, final):
        self.pcm = bytearray(data)
        return dict(left="甲", right="甲乙", candidates=[dict(word="乙")])

    monkeypatch.setattr(runtime.Demo, "step", step)
    d.expected = ""
    d.ever = set()
    d.register("甲")
    d.select(1)
    assert "new甲" in d.det.hotword_by_id and "2" not in d.det.hotword_by_id
    d.register("乙")
    d.record_step(b"abc", True)
    assert set(d.snapshot()["extras"]) == {"甲", "乙"}
    d.select(-1)
    d.remove("甲")
    d.record_step(b"def", True)
    assert d.snapshot()["overrides"] == ["乙"] and "1" in d.det.hotword_by_id
    events = d.history
    assert [e["kind"] for e in events] == ["add", "add", "test", "remove", "test"]
    assert events[2]["table"]["extras"] == {"甲": "a", "乙": "a"}
    assert events[2]["sample"] == "1" and events[2]["prompted"] == ["乙"]
    assert events[-1]["source"] == "自由麦克风" and events[-1]["table"]["extras"] == {"乙": "a"}
    with pytest.raises(ValueError):
        d.register("乙")
    d.select(0)
    assert set(d.extras) == {"乙"} and len(d.history) == 5
    assert [
        json.loads(line) for line in (tmp_path / "session_history.jsonl").read_text().splitlines()
    ] == events
