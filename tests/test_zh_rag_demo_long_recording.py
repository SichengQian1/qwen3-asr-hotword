"""Exercise real chunk validation with a fake decoder, across multiple segments."""

import hashlib
from types import SimpleNamespace

import pytest

from qwen_hotword.demo import runtime
from qwen_hotword.demo_v2.session import SharedStudio


@pytest.fixture
def studio(tmp_path, monkeypatch):
    d = SharedStudio.__new__(SharedStudio)
    d.out = tmp_path
    d.base = d.entries = d.clips = []
    d.extras = d.kw = {}
    d.omit = ""
    d.det = d.vocab = d.owners = None
    d.init_studio()
    states = []

    class Backend:
        def init_streaming_state(self, **kwargs):
            state = SimpleNamespace(text="", samples=0, finished=False, index=len(states) // 2)
            states.append(state)
            return state

        def streaming_transcribe(self, wave, state):
            assert not state.finished
            state.samples += len(wave)
            assert state.samples <= 480000
            state.text = f"segment{state.index}:{state.samples}"

        def finish_streaming_transcribe(self, state):
            assert not state.finished
            state.finished = True

    d.backend = Backend()
    monkeypatch.setattr(runtime.external, "_retrieve_waveform", lambda *a, **k: [])
    monkeypatch.setattr(runtime, "collapse_variants", lambda *a: {"raw_ranked_matches": []})
    monkeypatch.setattr(runtime, "_select", lambda *a, **k: [])
    monkeypatch.setattr(runtime, "refresh_streaming_prompt", lambda *a, **k: None)
    yield d, states
    d.abort_pass()


def test_long_recording_disk_replay_and_global_sequence(studio):
    d, states = studio
    data = b"\x01\x00" * (16000 * 63)
    d.begin_pass("mic")
    for seq, offset in enumerate(range(0, len(data), runtime.CHUNK)):
        block = data[offset : offset + runtime.CHUNK]
        final = offset + len(block) == len(data)
        assert d.seq == seq
        result = d.record_step(block, final)
        assert result["final"] is final
        assert len(d.pcm) <= runtime.CHUNK * 15
    assert result["seconds"] == 63
    assert result["completed_segments"] == 3
    assert result["left"].splitlines() == ["segment0:480000", "segment1:480000", "segment2:48000"]
    assert all(s.finished for s in states)
    assert d._mic_path.read_bytes() == data
    assert d._record_file is None
    assert d.mic_pcm == b""  # The whole recording is not retained in host RAM.
    assert d.replay_size() == len(data)
    assert d.history[-1]["audio"] == hashlib.sha256(data).hexdigest()
    d.begin_pass("replay")
    for offset in range(0, len(data), runtime.CHUNK):
        block = d.replay_chunk(offset)
        assert block == data[offset : offset + runtime.CHUNK]
        replay = d.record_step(block, offset + len(block) == len(data))
    assert replay["seconds"] == 63
    assert len(replay["right"].splitlines()) == 3
    assert len({r["audio"] for r in d.history}) == 1
    assert len(d.history) == 2


@pytest.mark.parametrize("chunks", [15, 30])
def test_manual_stop_on_exact_boundary_does_not_double_finalize(studio, chunks):
    d, states = studio
    d.begin_pass("mic")
    for _ in range(chunks):
        before = d.record_step(bytes(runtime.CHUNK), False)
        assert not before["final"]
    result = d.record_step(b"", True)
    assert result["left"] == before["left"]
    assert result["seconds"] == chunks * 2
    assert d.seq == chunks + 1
    assert len(states) == chunks // 15 * 2
    assert all(s.finished for s in states)


def test_abort_preserves_last_finished_recording_and_replay_rejects_change(studio):
    d, _ = studio
    d.begin_pass("mic")
    d.record_step(b"\x01\x00", True)
    original = d._mic_path
    d.begin_pass("mic")
    d.record_step(bytes(runtime.CHUNK), False)
    stream = d._record_file
    d.abort_pass()
    assert stream.closed
    assert d._mic_path == original
    assert d.replay_chunk(0) == b"\x01\x00"
    assert len(d.history) == 1
    d.begin_pass("replay")
    with pytest.raises(ValueError, match="不一致"):
        d.record_step(b"\x02\x00", True)
    assert len(d.history) == 1
    with pytest.raises(ValueError, match="偏移"):
        d.replay_chunk(1)
