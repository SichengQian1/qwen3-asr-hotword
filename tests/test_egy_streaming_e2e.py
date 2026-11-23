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
