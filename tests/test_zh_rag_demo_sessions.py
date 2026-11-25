"""Session isolation and concurrent admission with shared, serialized GPU work."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from qwen_hotword.demo import runtime
from qwen_hotword.demo_v2.session import SharedStudio
from qwen_hotword.demo_v2.sessions import SessionError, Sessions
from qwen_hotword.hotwords.anchor_index import PhonemeAnchorIndex
from qwen_hotword.hotwords.registry import HotwordEntry


def entry(word, token):
    return HotwordEntry(word, "zh", word, word, (word,), word, (word,), (token,), "test", 0)


@pytest.fixture
def pool(tmp_path, monkeypatch):
    d = SharedStudio.__new__(SharedStudio)
    d.out = tmp_path
    d.base = [entry("北京", 1)]
    d.base_owners = {d.base[0].hotword_id: d.base[0]}
    d.entries = d.base
    d.clips = []
    d.extras = d.kw = {}
    d.omit = ""
    d.vocab = None
    d.det = SimpleNamespace(anchor_index=PhonemeAnchorIndex(d.base), wrapper=object())
    d.init_studio()

    class Backend:
        def init_streaming_state(self, **kwargs):
            return SimpleNamespace(text="", values=[])

        def streaming_transcribe(self, wave, state):
            state.values.append(round(float(wave[0]) * 32768))
            state.text = str(state.values)

        def finish_streaming_transcribe(self, state):
            pass

    d.backend = Backend()
    monkeypatch.setattr(runtime.external, "_retrieve_waveform", lambda *a, **k: [])
    monkeypatch.setattr(runtime, "collapse_variants", lambda *a: {"raw_ranked_matches": []})
    monkeypatch.setattr(runtime, "_select", lambda *a, **k: [])
    monkeypatch.setattr(runtime, "refresh_streaming_prompt", lambda *a, **k: None)
    p = Sessions(d, max_active=2, max_sessions=3)
    yield p
    p.close()


def call(pool, key, op, params=None, data=b"", **headers):
    return pool.dispatch(key, op, params or {}, data, headers)


def test_separate_indices_streams_files_and_history_share_weights(pool):
    a, b = pool.create(), pool.create()
    da, db = pool.clients[a].studio, pool.clients[b].studio
    assert da.det is not db.det
    assert da.det.wrapper is db.det.wrapper is pool.model.det.wrapper
    assert da.backend is db.backend is pool.model.backend
    da.configure("", {"上海": entry("上海", 2)})
    assert call(pool, a, "/vocabulary")["extras"] == ["上海"]
    assert call(pool, b, "/vocabulary")["extras"] == []
    assert "上海" not in db.det.hotword_by_id
    sa = call(pool, a, "/start", {"kind": "mic"})["session"]
    sb = call(pool, b, "/start", {"kind": "mic"})["session"]
    assert pool.capacity()["active_users"] == 2
    for i in range(2):
        for key, sid, sample in [(a, sa, b"\x01\x00"), (b, sb, b"\x02\x00")]:
            result = call(
                pool,
                key,
                "/step",
                data=sample * 32000,
                **{"X-Session": sid, "X-Seq": str(i), "X-Final": str(i)},
            )
            assert result["left"] == str([int.from_bytes(sample, "little")] * (i + 1))
    assert pool.capacity()["active_users"] == 0
    assert da._mic_path != db._mic_path
    assert call(pool, a, "/clip", {"offset": 64000}) == b"\x01\x00" * 32000
    assert call(pool, b, "/clip", {"offset": 64000}) == b"\x02\x00" * 32000
    assert len(da.history) == len(db.history) == 1
    assert da.history[0]["audio"] != db.history[0]["audio"]
    assert da.history[0]["table"]["extras"] == {"上海": "上海"}
    assert db.history[0]["table"]["extras"] == {}
    assert (da.out / "session_history.jsonl").exists()
    assert not pool.model.history
    assert json.loads((pool.model.out / "concurrency_summary.json").read_text())["total_steps"] == 4


def test_capacity_foreign_recording_sequence_and_cleanup(pool):
    a, b, c = (pool.create() for _ in range(3))
    with pytest.raises(SessionError, match="网页会话已满"):
        pool.create()
    sa = call(pool, a, "/start", {"kind": "mic"})["session"]
    call(pool, b, "/start", {"kind": "mic"})
    with pytest.raises(SessionError, match="同时录音人数"):
        call(pool, c, "/start", {"kind": "mic"})
    with pytest.raises(SessionError, match="会话已失效"):
        call(pool, b, "/abort", **{"X-Session": sa})
    with pytest.raises(SessionError, match="顺序"):
        call(pool, a, "/step", data=bytes(runtime.CHUNK), **{"X-Session": sa, "X-Seq": "1"})
    assert pool.clients[a].studio.seq == 0
    call(pool, a, "/abort", **{"X-Session": sa})
    call(pool, c, "/start", {"kind": "mic"})
    call(pool, c, "/leave")
    with pytest.raises(SessionError, match="网页会话已过期"):
        call(pool, c, "/state")
    assert pool.create() != c
    pool.clients[b].seen -= 121
    assert call(pool, a, "/state")["active_users"] == 0
    pool.clients[b].seen -= 1800
    with pytest.raises(SessionError, match="网页会话已过期"):
        call(pool, b, "/state")


def test_gpu_steps_serialize_but_another_page_can_get_state(pool, monkeypatch):
    a, b = pool.create(), pool.create()
    sa = call(pool, a, "/start", {"kind": "mic"})["session"]
    sb = call(pool, b, "/start", {"kind": "mic"})["session"]
    entered, release = threading.Event(), threading.Event()
    original = SharedStudio.record_step
    active, peak = 0, 0

    def record(self, data, final):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if self is pool.clients[a].studio:
            entered.set()
            assert release.wait(5)
        try:
            return original(self, data, final)
        finally:
            active -= 1

    monkeypatch.setattr(SharedStudio, "record_step", record)
    with ThreadPoolExecutor(max_workers=2) as executor:
        fa = executor.submit(
            call,
            pool,
            a,
            "/step",
            data=bytes(runtime.CHUNK),
            **{"X-Session": sa, "X-Seq": "0", "X-Final": "1"},
        )
        assert entered.wait(5)
        try:
            assert call(pool, b, "/state")["active_users"] == 2
            fb = executor.submit(
                call,
                pool,
                b,
                "/step",
                data=bytes(runtime.CHUNK),
                **{"X-Session": sb, "X-Seq": "0", "X-Final": "1"},
            )
        finally:
            release.set()
        assert fa.result(timeout=5)["final"]
        assert fb.result(timeout=5)["final"]
    assert peak == 1


def test_bad_audio_releases_only_its_own_slot(pool):
    a, b = pool.create(), pool.create()
    sa = call(pool, a, "/start", {"kind": "mic"})["session"]
    sb = call(pool, b, "/start", {"kind": "mic"})["session"]
    with pytest.raises(ValueError, match="音频块"):
        call(pool, a, "/step", data=b"x", **{"X-Session": sa, "X-Seq": "0"})
    assert not pool.clients[a].recording
    assert pool.clients[b].recording == sb
    assert pool.clients[a].studio._record_file is None
    result = call(
        pool, b, "/step", data=b"\x01\x00", **{"X-Session": sb, "X-Seq": "0", "X-Final": "1"}
    )
    assert result["final"]
