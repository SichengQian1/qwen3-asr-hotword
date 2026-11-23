from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.evaluation.egy_streaming_e2e import (
    MODES,
    PrefixReplay,
    aggregate,
    score_row,
    smoke_rows,
    summary,
)
from qwen_hotword.inference.streaming_core import StreamingSample, run_streaming_sample
from scripts import run_egy_streaming_e2e as runner

TABLE = [
    {"id": "cairo", "surface": "القاهرة", "pronunciation": {"status": "active"}},
    {"id": "egypt", "surface": "مصر", "pronunciation": {"status": "inactive"}},
    {"id": "bank", "surface": "بنك مصر", "pronunciation": {"status": "active"}},
    {"id": "wrong", "surface": "باريس", "pronunciation": {"status": "active"}},
]


def candidate(identifier: str, surface: str = "القاهرة") -> dict[str, Any]:
    return {
        "hotword_id": identifier,
        "surface": surface,
        "score": 0.9,
        "edit_ratio": 0.1,
        "posterior_confidence": 0.9,
    }


def test_replay_rejects_future_wrong_table_and_incomplete_stream() -> None:
    replay = PrefixReplay(
        [{"end_sample": 32000, "selected_matches": [candidate("cairo")]}], ("cairo",), 5
    )
    with pytest.raises(ValueError, match="boundary"):
        replay([0] * 64000, ("cairo",))
    with pytest.raises(ValueError, match="table"):
        replay([0] * 32000, ("wrong",))
    with pytest.raises(ValueError, match="every"):
        replay.finish()
    assert replay([0] * 32000, ("cairo",))[0].hotword_id == "cairo"
    replay.finish()
    with pytest.raises(ValueError):
        replay([0] * 32000, ("cairo",))


def test_top5_top7_share_order_without_filling_rejected_candidates() -> None:
    ids = tuple(str(n) for n in range(9))
    prefixes = [
        {"end_sample": 3, "selected_matches": [candidate(i) for i in ids[:7]]},
        {"end_sample": 4, "selected_matches": []},
    ]
    five, seven = PrefixReplay(prefixes, ids, 5), PrefixReplay(prefixes, ids, 7)
    assert [x.hotword_id for x in five([0] * 3, ids)] == list(ids[:5])
    assert [x.hotword_id for x in seven([0] * 3, ids)] == list(ids[:7])
    assert five([0] * 4, ids) == seven([0] * 4, ids) == ()


def test_inactive_gold_and_nested_forms_remain_correctly_scored() -> None:
    # Inactive dictionary entry can still be transcribed by the Decoder.
    metrics = score_row("مصر القاهرة", "مصر", TABLE, {"cairo", "wrong"})
    assert metrics["target_pairs"] == 2 and metrics["target_hits"] == 1
    assert metrics["injected_target_pairs"] == 1
    assert metrics["injected_targets_in_final_text"] == 0
    assert metrics["wrong_injected_pairs"] == 1
    nested = score_row("بنك مصر", "بنك مصر", TABLE, {"egypt", "bank"})
    assert nested["expected_ids"] == ["bank"]
    assert nested["wrong_injected_pairs"] == 0


def test_corpus_errors_include_negative_samples_and_arabic_normalization() -> None:
    good = score_row("في القَاهِرَة،", "في القاهرة", TABLE, set())
    bad = score_row("مرحبا", "مرحبا باريس", TABLE, {"wrong"})
    assert good["word_errors"] == 0
    assert bad["negative_with_unreferenced_table_word"] == 1
    assert bad["wrong_injected_in_final_text"] == 1
    result = aggregate([{"metrics": good}, {"metrics": bad}])
    assert result["corpus_WER"] == pytest.approx(1 / 3)
    assert result["target_recall"] == 1.0
    assert result["negative_false_word_rate"] == 1.0
    assert aggregate([])["target_recall"] is None


def test_paired_gains_losses_and_missing_mode_rejected() -> None:
    cells = [
        {"id": "one", "mode": mode, "metrics": score_row("مصر القاهرة", pred, TABLE, set())}
        for mode, pred in zip(MODES, ("مصر", "القاهرة", "مصر القاهرة"), strict=True)
    ]
    report = summary(cells, {"one"})
    assert report["top5"]["paired_vs_no_rag"] == {
        "target_gains": 1,
        "target_losses": 1,
        "net_target_gain": 0,
    }
    assert report["top7"]["paired_vs_no_rag"]["net_target_gain"] == 1
    with pytest.raises(ValueError, match="paired"):
        summary(cells[:-1], {"one"})
    with pytest.raises(ValueError, match="duplicate"):
        summary(cells + cells[:1], {"one"})


def test_smoke_is_reproducible_covers_source_and_polarity() -> None:
    rows = [
        {"id": str(n), "source_group": str(n % 3), "text": "القاهرة" if n % 2 else "مرحبا"}
        for n in range(40)
    ]
    first = smoke_rows(rows, TABLE)
    assert first == smoke_rows(list(reversed(rows)), TABLE)
    assert len(first) == 20
    assert {(r["source_group"], r["text"]) for r in first} == {
        (str(n), text) for n in range(3) for text in ("القاهرة", "مرحبا")
    }


class Tokenizer:
    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: Any, **_: Any) -> str:
        return "".join(chr(i) for i in ids)


class Backend:
    tokenizer = Tokenizer()

    def __init__(self) -> None:
        self.received: list[tuple[int, str]] = []
        self.finish_count = 0

    def init_streaming_state(self, *, context: str, language: str, **_: Any) -> Any:
        return SimpleNamespace(
            prompt_raw="SYSTEM:" + context,
            context=context,
            force_language=language,
            text="",
            _raw_decoded="",
        )

    def streaming_transcribe(self, wave: Any, state: Any) -> None:
        self.received.append((len(wave), state.context))
        state.text = "القاهرة"
        state._raw_decoded = state.text

    def finish_streaming_transcribe(self, state: Any) -> None:
        self.finish_count += 1


@pytest.mark.parametrize("total", [64000, 65000])
def test_streaming_uses_current_prefix_and_new_chunks_only(total: int) -> None:
    backend = Backend()
    ends = [32000, 64000] + ([65000] if total > 64000 else [])
    replay = PrefixReplay(
        [
            {"end_sample": end, "selected_matches": [] if n == 0 else [candidate("cairo")]}
            for n, end in enumerate(ends)
        ],
        ("cairo",),
        7,
    )
    sample = StreamingSample("s", "s", "", "ar", (), (), ("cairo",))
    result, timeline = run_streaming_sample(
        backend=backend,
        waveform=[0] * total,
        sample=sample,
        group="D",
        hotword_surfaces={"cairo": "القاهرة"},
        ctc_detector=replay,
        prompt_template="Reference: {hotwords}",
        asr_language="Arabic",
    )
    replay.finish()
    assert backend.received[0] == (32000, "")
    assert backend.received[1] == (32000, "Reference: القاهرة")
    assert sum(size for size, _ in backend.received) == total
    assert backend.finish_count == 1
    assert result["prediction"] == "القاهرة"
    assert all(step["expected_hotword_status"] == {} for step in timeline)


def test_prompt_probe_rejects_missing_refresh_fields() -> None:
    config = {"chunk_seconds": 2, "unfixed_chunk_num": 2, "unfixed_token_num": 5}
    runner.prompt_probe(Backend(), config)
    broken = SimpleNamespace(init_streaming_state=lambda **_: SimpleNamespace(prompt_raw="fixed"))
    with pytest.raises(RuntimeError, match="cannot refresh"):
        runner.prompt_probe(broken, config)


def test_cell_checksum_and_identity_prevent_bad_resume(tmp_path: Path) -> None:
    path = tmp_path / "cell.json"
    runner.save_cell(path, {"identity": "abc", "prediction": "القاهرة"})
    assert runner.load_cell(path, "abc")["prediction"] == "القاهرة"
    with pytest.raises(ValueError):
        runner.load_cell(path, "other")
    contents = json.loads(path.read_text())
    contents["payload"]["prediction"] = "changed"
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        runner.load_cell(path, "abc")


def test_failed_smoke_does_not_start_full_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *_: {})
    monkeypatch.setattr(runner, "collect", lambda *_: pytest.fail("failed smoke must not collect"))

    def child(args: list[str], **_: Any) -> None:
        calls.append(args)
        if args[args.index("--stage") + 1] == "asr":
            raise RuntimeError("injected smoke failure")

    monkeypatch.setattr(runner.subprocess, "run", child)
    with pytest.raises(RuntimeError, match="smoke failure"):
        runner.run(tmp_path / "config", tmp_path / "run", 4, False)
    assert len(calls) == 2
    assert all(args[args.index("--scope") + 1] == "smoke" for args in calls)


def test_successful_smoke_precedes_full_and_both_use_same_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []
    output = tmp_path / "run"
    monkeypatch.setattr(runner, "prepare", lambda *_: {})
    monkeypatch.setattr(runner, "inputs_unchanged", lambda *_: None)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda args, **kw: events.append(
            (args[args.index("--scope") + 1], args[args.index("--stage") + 1])
        ),
    )

    def collect(out: Path, scope: str) -> dict[str, Any]:
        assert out == output
        events.append((scope, "collect"))
        for name in (
            "report.json",
            "smoke_report.json",
            "contract.json",
            "runtime_main.json",
            "runtime_decoder.json",
        ):
            (out / name).write_text("{}")
        return {"status": "completed"}

    monkeypatch.setattr(runner, "collect", collect)
    assert runner.run(tmp_path / "cfg", output, 5, False)["status"] == "completed"
    assert events == [
        (scope, stage) for scope in ("smoke", "full") for stage in ("prefix", "asr", "collect")
    ]
    assert (output / "return_summary.zip").is_file()


def test_prefix_worker_reuses_verified_cells_and_never_passes_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from qwen_hotword.inference import external_keyword_retrieval as retrieval
    from qwen_hotword.phonemes import coverage

    output = tmp_path
    (output / "contract.json").write_text("{}")
    row = {
        "id": "audio",
        "audio_sha256": "hash",
        "audio_path": "/unused",
        "text": "SECRET_REFERENCE",
    }
    contract = {
        "vocab": "/vocab",
        "model": "/model",
        "checkpoint": "/head",
        "config": {"chunk_seconds": 2},
    }
    monkeypatch.setattr(runner, "selection", lambda *_: (contract, [row], TABLE))
    monkeypatch.setattr(runner, "inputs_unchanged", lambda *_: None)
    monkeypatch.setattr(runner, "audio", lambda *_: [0] * 65000)
    monkeypatch.setattr(runner, "pin", lambda *_: None)
    monkeypatch.setattr(coverage, "load_phoneme_vocab", lambda *_: object())
    loads, prefixes = [], []
    monkeypatch.setattr(runner, "load_detector", lambda *args: loads.append(args) or "detector")

    def retrieve(detector: Any, wave: Any, vocab: Any, **kw: Any) -> dict[str, Any]:
        assert detector == "detector"
        prefixes.append(len(wave))
        return {"selected_matches": [], "raw_ranked_matches": [], "timing": {}}

    monkeypatch.setattr(retrieval, "_retrieve_waveform", retrieve)
    runner.prefix_worker(output, "full")
    runner.prefix_worker(output, "full")
    assert prefixes == [32000, 64000, 65000]
    assert len(loads) == 1
    assert "SECRET_REFERENCE" not in repr(loads)


def test_bridge_verifies_actual_prompt_and_receipted_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    channel = SimpleNamespace(send=lambda req: requests.append(req))
    wave = SimpleNamespace(astype=lambda *a, **kw: SimpleNamespace(tobytes=lambda: b"audio"))

    class Wave:
        def __len__(self) -> int:
            return 32000

        astype = wave.astype

    def receive(*_: Any) -> dict[str, Any]:
        return {
            "text": "القاهرة",
            "finish_reason": "stop",
            "audio_samples": 32000,
            "prompt_ids_sha256": runner.fingerprint(requests[-1]["ids"]),
        }

    monkeypatch.setattr(runner, "receive", receive)
    bridge = runner.DecoderBridge(channel, object(), Tokenizer())  # type: ignore[arg-type]
    bridge.expected_prompt = "SYSTEM:القاهرة"
    result = bridge.generate(
        [{"prompt": "SYSTEM:القاهرة prefix", "multi_modal_data": {"audio": [Wave()]}}]
    )
    assert result[0].outputs[0].text == "القاهرة"
    assert bridge.calls[0]["prompt"] == "SYSTEM:القاهرة prefix"
    with pytest.raises(ValueError, match="refreshed prompt"):
        bridge.generate([{"prompt": "OLD_PROMPT", "multi_modal_data": {"audio": [Wave()]}}])


@pytest.mark.parametrize("free_gib,total_gib", [(30.27, 139.81), (120, 139.81), (5, 139.81)])
def test_dynamic_budget_uses_free_not_total_and_keeps_headroom(
    free_gib: float,
    total_gib: float,
) -> None:
    free, total = int(free_gib * 1024**3), int(total_gib * 1024**3)
    result = runner.memory_budget(free, total, "auto", 0.15)
    assert result["reserved_free_bytes"] >= max(2 * 1024**3, free * 0.1)
    expected = free - max(2 * 1024**3, free * 0.1)
    assert expected - total / 10000 - 2 <= result["requested_bytes"] <= expected
    assert result["effective_gpu_memory_utilization"] < free / total
    assert result["original_config_utilization"] == 0.15


@pytest.mark.parametrize(
    "free,total,mode,fixed",
    [
        (0, 100, "auto", 0.15),
        (101, 100, "auto", 0.15),
        (2 * 1024**3, 140 * 1024**3, "auto", 0.15),
        (30 * 1024**3, 140 * 1024**3, "fixed", 0.5),
        (30 * 1024**3, 140 * 1024**3, "invalid", 0.15),
    ],
)
def test_dynamic_budget_rejects_invalid_or_insufficient_memory(
    free: int,
    total: int,
    mode: str,
    fixed: float,
) -> None:
    with pytest.raises(ValueError):
        runner.memory_budget(free, total, mode, fixed)


def test_fixed_memory_option_preserves_configured_budget() -> None:
    result = runner.memory_budget(30 * 1024**3, 140 * 1024**3, "fixed", 0.15)
    assert result["effective_gpu_memory_utilization"] == 0.15
    assert result["requested_bytes"] == 21 * 1024**3


def contract_upgrade_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    model = tmp_path / "model"
    model.write_text("unchanged model")
    old = {
        "inputs": {
            str(Path(runner.__file__)): runner.LEGACY_MEMORY_SCRIPT_SHA256,
            str(model): runner.sha(model),
        },
        "config": {"gpu_memory_utilization": 0.15},
    }
    path = tmp_path / "contract.json"
    runner.sealed(path, old)
    current = json.loads(json.dumps(old))
    current["inputs"][str(Path(runner.__file__))] = runner.sha(Path(runner.__file__))
    return path, old, current


def test_memory_upgrade_keeps_original_contract_and_completed_cells(tmp_path: Path) -> None:
    path, old, current = contract_upgrade_fixture(tmp_path)
    before = path.read_bytes()
    row = {"id": "sample", "audio_sha256": "audio"}
    identity = runner.row_identity(tmp_path, row)
    cell = tmp_path / "completed.json"
    runner.save_cell(cell, {"identity": identity, "prediction": "القاهرة"})
    assert runner.seal_stream_contract(path, current) == old
    assert path.read_bytes() == before
    assert runner.row_identity(tmp_path, row) == identity
    assert runner.load_cell(cell, identity)["prediction"] == "القاهرة"
    runner.inputs_unchanged(old)
    assert runner.seal_stream_contract(path, current) == old  # repeated restart
    audit = json.loads((tmp_path / "memory_runtime_upgrade.json").read_text())
    assert audit["original_contract_sha256"] == runner.sha(path)
    assert audit["reviewed_script_sha256"] == runner.sha(Path(runner.__file__))
    (tmp_path / "model").write_text("changed model")
    with pytest.raises(ValueError):
        runner.inputs_unchanged(old)


@pytest.mark.parametrize("change", ["model", "config", "unknown_script"])
def test_memory_upgrade_does_not_allow_other_identity_changes(tmp_path: Path, change: str) -> None:
    path, old, current = contract_upgrade_fixture(tmp_path)
    if change == "model":
        current["inputs"][str(tmp_path / "model")] = "different"
    elif change == "config":
        current["config"]["seed"] = 99
    else:
        old["inputs"][str(Path(runner.__file__))] = "unknown historical code"
        path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="sealed identity changed"):
        runner.seal_stream_contract(path, current)
    assert not (tmp_path / "memory_runtime_upgrade.json").exists()


@pytest.mark.parametrize("mode", ["auto", "fixed"])
def test_native_engine_receives_effective_memory_and_records_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    import sys

    free, total = 30 * 1024**3, 140 * 1024**3
    calls, messages = [], []
    monkeypatch.setenv("EGY_DECODER_MEMORY", mode)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda device: (free, total))),
    )
    fake_vllm = SimpleNamespace(LLM=lambda **kw: calls.append(kw), SamplingParams=lambda **kw: kw)
    monkeypatch.setattr(runner, "import_module", lambda _: fake_vllm)

    class Channel:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def send(self, message: Any) -> None:
            messages.append(message)

        def recv(self) -> Any:
            raise EOFError

    monkeypatch.setattr(runner, "Connection", lambda _: Channel())
    runner.write_json(
        tmp_path / "contract.json",
        {
            "model": "/merged-r15",
            "config": {
                "gpu_memory_utilization": 0.15,
                "max_model_len": 8192,
                "max_new_tokens": 512,
                "seed": 20261123,
            },
        },
    )
    runner.decoder_server(0, tmp_path)
    expected = runner.memory_budget(free, total, mode, 0.15)
    assert calls[0]["gpu_memory_utilization"] == expected["effective_gpu_memory_utilization"]
    assert calls[0]["max_num_seqs"] == 3
    assert calls[0]["model"] == "/merged-r15"
    assert messages[0]["ready"] is True
    launch = runner.read(next((tmp_path / "decoder_launches").glob("*.json")))
    assert launch["status"] == "ready"
    assert launch["physical_gpu"] == "4"
    assert launch["launch_id"] == messages[0]["decoder_launch_id"]
    assert launch["requested_bytes"] == expected["requested_bytes"]


@pytest.mark.parametrize("remaining_modes", [1, 2, 3])
def test_broker_batches_remaining_modes_without_crossing_their_results(
    monkeypatch: pytest.MonkeyPatch,
    remaining_modes: int,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    packets: list[dict[str, Any]] = []
    channel = SimpleNamespace(send=lambda value: packets.append(value))
    barrier = threading.Barrier(remaining_modes)

    def receive(*_: Any) -> dict[str, Any]:
        return {"results": [{"text": str(request["ids"])} for request in packets[-1]["requests"]]}

    monkeypatch.setattr(runner, "receive", receive)
    broker = runner.BatchBroker(channel, object(), wait_seconds=0.1)  # type: ignore[arg-type]
    try:

        def stream(mode: int) -> list[str]:
            replies = []
            for step in range(3):
                barrier.wait(timeout=3)
                replies.append(broker.request({"ids": [mode, step], "wave": b"pcm"})["text"])
            return replies

        with ThreadPoolExecutor(max_workers=remaining_modes) as pool:
            results = list(pool.map(stream, range(remaining_modes)))
        assert len(packets) == 3
        assert all(len(packet["requests"]) == remaining_modes for packet in packets)
        for mode, replies in enumerate(results):
            assert replies == [str([mode, step]) for step in range(3)]
    finally:
        broker.close()
    assert not broker.thread.is_alive()


def test_broker_failure_releases_all_waiters_and_rejects_later_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    def receive(*_: Any) -> Any:
        raise RuntimeError("native engine failed")

    monkeypatch.setattr(runner, "receive", receive)
    channel = SimpleNamespace(send=lambda _: None)
    broker = runner.BatchBroker(channel, object(), wait_seconds=0.1)  # type: ignore[arg-type]
    barrier = threading.Barrier(3)
    try:

        def stream(n: int) -> Any:
            barrier.wait(timeout=3)
            return broker.request({"ids": [n], "wave": b""})

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(stream, n) for n in range(3)]
            for future in futures:
                with pytest.raises(RuntimeError, match="native engine failed"):
                    future.result(timeout=3)
        with pytest.raises(RuntimeError, match="native engine failed"):
            broker.request({"ids": [4], "wave": b""})
    finally:
        broker.close()


def test_native_decoder_submits_batch_and_preserves_request_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sys

    import numpy as np

    sent, batches = [], []
    requests = [
        {"ids": [i, 10 + i], "wave": np.zeros(32000 + i, dtype="<f4").tobytes()} for i in range(3)
    ]
    incoming = iter([{"requests": requests}])

    class Channel:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def send(self, value: Any) -> None:
            sent.append(value)

        def recv(self) -> Any:
            try:
                return next(incoming)
            except StopIteration:
                raise EOFError from None

    class Engine:
        def generate(self, inputs: Any, **kw: Any) -> Any:
            batches.append(inputs)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(text=str(item["prompt_token_ids"]), finish_reason="stop")
                    ]
                )
                for item in inputs
            ]

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _: (30 * 1024**3, 140 * 1024**3))),
    )
    monkeypatch.setattr(runner, "Connection", lambda _: Channel())
    monkeypatch.setattr(
        runner,
        "import_module",
        lambda _: SimpleNamespace(LLM=lambda **kw: Engine(), SamplingParams=lambda **kw: kw),
    )
    runner.write_json(
        tmp_path / "contract.json",
        {
            "model": "/r15",
            "config": {
                "gpu_memory_utilization": 0.15,
                "max_model_len": 8192,
                "max_new_tokens": 512,
                "seed": 1,
            },
        },
    )
    runner.decoder_server(0, tmp_path)
    assert len(batches) == 1 and len(batches[0]) == 3
    assert [item["prompt_token_ids"] for item in batches[0]] == [r["ids"] for r in requests]
    replies = sent[1]["results"]
    for n, reply in enumerate(replies):
        assert reply["text"] == str(requests[n]["ids"])
        assert reply["audio_samples"] == 32000 + n
        assert reply["prompt_ids_sha256"] == runner.fingerprint(requests[n]["ids"])
        assert reply["batch_size"] == 3 and reply["batch_id"] == 0
