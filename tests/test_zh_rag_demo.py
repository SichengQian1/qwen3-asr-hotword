"""No model weights: use the installed upstream streaming state machine when available."""

from types import SimpleNamespace

import pytest

from qwen_hotword.demo import runtime


def test_verified_case_requires_retrieval_and_both_controls():
    miss = {"left": "原音", "right": "原音"}
    hit = {"left": "原音", "right": "元音"}
    assert runtime.recovered("元音", miss, hit, ["元音"])
    assert not runtime.recovered("元音", miss, hit, [])
    assert not runtime.recovered("元音", hit, hit, ["元音"])
    assert not runtime.recovered("元音", miss, {**hit, "left": "不同"}, ["元音"])


def test_real_sdk_paired_audio_context_and_tail(monkeypatch):
    np = pytest.importorskip("numpy")
    sdk = pytest.importorskip("qwen_asr")
    calls = []

    class Recorder:
        def generate(self, inputs, **kwargs):
            req = inputs[0]
            wave = req["multi_modal_data"]["audio"][0]
            calls.append((req["prompt"], wave.copy()))
            return [SimpleNamespace(outputs=[SimpleNamespace(text="重庆")])]

    processor = SimpleNamespace(
        apply_chat_template=lambda msgs, **kw: msgs[0]["content"] + "<audio>",
        tokenizer=SimpleNamespace(
            encode=lambda s: list(map(ord, s)), decode=lambda ids: "".join(map(chr, ids))
        ),
    )
    d = runtime.Demo.__new__(runtime.Demo)
    d.backend = sdk.Qwen3ASRModel(backend="vllm", model=Recorder(), processor=processor)
    d.kw = dict(language="Chinese", unfixed_token_num=5, unfixed_chunk_num=2, chunk_size_sec=2.0)
    d.det = d.vocab = d.entries = d.owners = None
    monkeypatch.setattr(runtime.external, "_retrieve_waveform", lambda *a, **kw: {})
    monkeypatch.setattr(runtime, "collapse_variants", lambda *a: {"raw_ranked_matches": []})
    monkeypatch.setattr(runtime, "_select", lambda *a, **kw: [])
    monkeypatch.setattr(runtime, "candidates", lambda m: [{"word": "重庆"}])
    d.start()
    d.step(b"\x01\x00" * 32000, False)
    out = d.step(b"\x02\x00" * 5000, True)
    assert out["final"] and out["seconds"] == 37000 / 16000
    assert len(calls) == 4
    for i, n in [(0, 32000), (2, 37000)]:
        assert "仅供转写参考" not in calls[i][0] and "仅供转写参考：重庆" in calls[i + 1][0]
        assert len(calls[i][1]) == n and np.array_equal(calls[i][1], calls[i + 1][1])
    assert calls[2][1][-1] == 2 / 32768
    d.start()
    with pytest.raises(ValueError):
        d.step(b"", True)


def test_decoder_starts_without_callable_rpc(tmp_path, monkeypatch):
    import json
    import os
    import sys
    from contextlib import ExitStack
    from multiprocessing import Pipe
    from threading import Thread

    from qwen_hotword.demo import decoder

    class NativeEngine:
        def __init__(self, *, model_impl, **kwargs):
            assert model_impl == "vllm"

        # Deliberately no apply_model: this worker accepts data requests only.
        def generate(self, request, **kwargs):
            assert request["prompt_token_ids"] == [42]
            audio, rate = request["multi_modal_data"]["audio"]
            assert rate == 16000 and audio.tolist() == [0]
            return [SimpleNamespace(outputs=[SimpleNamespace(text="中文")])]

    monkeypatch.setattr(decoder, "version", {"vllm": "0.23.0", "qwen-asr": "0.0.6"}.__getitem__)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=NativeEngine, SamplingParams=dict))
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3ASRForConditionalGeneration"]})
    )

    with ExitStack() as stack:
        a, b = Pipe()
        stack.enter_context(a)
        stack.enter_context(b)
        worker = Thread(target=decoder.serve, args=(os.dup(b.fileno()), str(tmp_path)), daemon=True)
        worker.start()
        assert a.poll(10) and a.recv() == {"ok": True}
        a.send({"ids": [42], "wave": bytes(4)})
        assert a.poll(10) and a.recv() == {"text": "中文"}
    worker.join(5)
    assert not worker.is_alive()
