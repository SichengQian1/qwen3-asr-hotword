#!/usr/bin/env python3
"""One command: engineering smoke -> causal prefix retrieval -> three-way streaming ASR."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from importlib import import_module
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.evaluation.backend_comparison import block_legacy_qwen_vllm_registration
from qwen_hotword.evaluation.casablanca_wer import NORMALIZATION
from qwen_hotword.evaluation.egy_corpus_hotwords import (
    assert_disjoint,
    expected_ids,
    order_key,
    select_audio,
)
from qwen_hotword.evaluation.egy_hotword_review import verify_run
from qwen_hotword.evaluation.egy_streaming_e2e import (
    MODES,
    PrefixReplay,
    aggregate,
    score_row,
    smoke_rows,
    summary,
)
from qwen_hotword.inference.streaming_core import (
    StreamingSample,
    refresh_streaming_prompt,
    run_streaming_sample,
    schedule_stream_chunks,
)
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.stage_progress import stage_progress
from scripts.evaluate_egy_corpus_hotwords import fingerprint, load_detector, read, sealed
from scripts.label_arabic_native import exclusive_job


def save_cell(path: Path, value: dict[str, Any]) -> None:
    write_json(path, {"payload": value, "sha256": fingerprint(value)})


def load_cell(path: Path, identity: str) -> Any:
    if not path.exists():
        return None
    saved = read(path)
    value = saved["payload"]
    if saved["sha256"] != fingerprint(value) or value["identity"] != identity:
        raise ValueError("completed result changed: " + str(path))
    return value


def runtime_info(decoder: bool) -> dict[str, Any]:
    packages: tuple[str, ...] = (
        "qwen-asr",
        "torch",
        "transformers",
        "numpy",
        "librosa",
        "soundfile",
    )
    if decoder:
        packages += ("vllm",)
    versions = {name: importlib.metadata.version(name) for name in packages}
    if versions["qwen-asr"] != "0.0.6":
        raise ValueError("streaming state requires installed qwen-asr 0.0.6")
    if decoder and versions["vllm"].split("+")[0] != "0.23.0":
        raise ValueError("requires previously verified native vLLM 0.23.0 environment")
    sources = [("qwen-asr", "qwen_asr/inference/qwen3_asr.py")]
    if decoder:
        sources.append(("vllm", "vllm/model_executor/models/qwen3_asr.py"))
        # Import native engine in its own process, without importing qwen_asr.
        import_module("vllm")
    else:
        import librosa  # noqa: F401
        import soundfile  # noqa: F401

        with block_legacy_qwen_vllm_registration():
            from qwen_asr import Qwen3ASRModel
        for name in ("init_streaming_state", "streaming_transcribe", "finish_streaming_transcribe"):
            if not hasattr(Qwen3ASRModel, name):
                raise ValueError("official streaming method missing: " + name)
    return {
        "python": sys.executable,
        "versions": versions,
        "sources": {
            rel: sha(Path(str(importlib.metadata.distribution(pkg).locate_file(rel))))
            for pkg, rel in sources
        },
    }


def prepare(config_path: Path, output: Path) -> dict[str, Any]:
    cfg = read(config_path)
    source_dir = ROOT / cfg["source_output"]
    original = read(ROOT / cfg["source_config"])
    review = read(ROOT / original["source_config"])
    report = read(source_dir / "report.json")
    if report["status"] != "completed":
        raise ValueError("whole-audio source evaluation is incomplete")
    verified = verify_run(ROOT, review)
    pin(source_dir / "table.json", cfg["expected_table_sha256"])
    rows, table = read(source_dir / "selected_audio.json"), read(source_dir / "table.json")
    test = ROOT / original["test_manifest"]
    val = ROOT / original["validation_manifest"]
    pin(test, original["test_sha256"])
    pin(val, original["validation_sha256"])
    selected, selection = select_audio(list(iter_jsonl(test)))
    if rows != selected or len(rows) != cfg["expected_samples"]:
        raise ValueError("evaluation audio differs from frozen whole-audio test selection")
    targets = sum(len(expected_ids(row["text"], table)) for row in rows)
    if targets != cfg["expected_target_pairs"]:
        raise ValueError("gold target denominator changed")
    assert_disjoint(rows, iter_jsonl(val))
    assert_disjoint(rows, iter_jsonl(Path(verified["train_manifest"])))
    plan = read(ROOT / original["encoder_plan"])
    model = Path(plan["model"]["source"])
    model_inputs = {
        str(model / name): value["sha256"] for name, value in plan["model"]["files"].items()
    }
    model_inputs[str(model / "model.safetensors")] = review["expected"]["model_weight_sha256"]
    for name, digest in model_inputs.items():
        pin(Path(name), digest)
    if read(model / "config.json")["architectures"] != ["Qwen3ASRForConditionalGeneration"]:
        raise ValueError("wrong Qwen architecture")
    decoder = ROOT / cfg["decoder_python"]
    if not decoder.is_file():
        raise ValueError("existing vLLM environment not found: " + str(decoder))
    runtime_files = []
    for kind, python in (("main", sys.executable), ("decoder", str(decoder))):
        dest = output / ("runtime_" + kind + ".json")
        subprocess.run(
            [python, "-u", "-B", str(Path(__file__)), "--probe", kind, "--output-dir", str(output)],
            check=True,
            cwd=ROOT,
        )
        runtime_files.append(dest)
    paths = [
        config_path,
        ROOT / cfg["source_config"],
        ROOT / original["source_config"],
        ROOT / original["encoder_plan"],
        test,
        val,
        Path(__file__),
        *runtime_files,
    ]
    paths += [
        source_dir / name
        for name in (
            "report.json",
            "contract.json",
            "table.json",
            "selected_audio.json",
            "selection.json",
        )
    ]
    paths += [
        ROOT / ("src/qwen_hotword/" + name)
        for name in (
            "evaluation/egy_streaming_e2e.py",
            "evaluation/egy_corpus_hotwords.py",
            "evaluation/casablanca_wer.py",
            "inference/streaming_core.py",
            "inference/streaming_backends.py",
            "inference/hotword_prompt.py",
            "inference/external_keyword_retrieval.py",
            "modeling/ctc_head.py",
            "modeling/audio_encoder.py",
            "hotwords/scoring.py",
            "hotwords/anchor_index.py",
        )
    ]
    paths += [ROOT / "scripts/evaluate_egy_corpus_hotwords.py"]
    contract = {
        "schema_version": 1,
        "config": cfg,
        "model": str(model),
        "checkpoint": str(ROOT / review["source_run"] / "head/ctc_head_best.pt"),
        "vocab": str(ROOT / review["vocab"]),
        "expected": review["expected"],
        "inputs": {**verified["files"], **model_inputs, **{str(p): sha(p) for p in paths}},
        "selection": selection,
        "target_pairs": targets,
        "smoke_ids": [r["id"] for r in smoke_rows(rows, table)],
        "audio": "original_no_added_noise",
        "effect": "same_step_state_refresh",
        "inference_reference_access": False,
        "evaluation_scope": "corpus_conditioned_frozen_table_streaming_simulation",
    }
    sealed(output / "contract.json", contract)
    return contract


def inputs_unchanged(contract: dict[str, Any]) -> None:
    for path, digest in contract["inputs"].items():
        pin(Path(path), digest)


def selection(output: Path, scope: str) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    contract = read(output / "contract.json")
    source = ROOT / contract["config"]["source_output"]
    rows = read(source / "selected_audio.json")
    if scope == "smoke":
        chosen = set(contract["smoke_ids"])
        rows = [r for r in rows if r["id"] in chosen]
    return contract, rows, read(source / "table.json")


def audio(row: dict[str, Any]) -> Any:
    from qwen_hotword.inference.external_keyword_retrieval import _load_audio

    pin(Path(row["audio_path"]), row["audio_sha256"])
    wave, metadata = _load_audio(Path(row["audio_path"]))
    if abs(metadata["duration_seconds"] - row["duration_seconds"]) > 0.02:
        raise ValueError("audio duration changed: " + row["audio_path"])
    return wave


def row_identity(output: Path, row: dict[str, Any]) -> str:
    return fingerprint([sha(output / "contract.json"), row["id"], row["audio_sha256"]])


def prefix_worker(output: Path, scope: str) -> None:
    from qwen_hotword.inference.external_keyword_retrieval import _retrieve_waveform
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab

    contract, rows, table = selection(output, scope)
    inputs_unchanged(contract)
    vocab = load_phoneme_vocab(Path(contract["vocab"]))
    detector = None
    with stage_progress(
        "Causal 2s prefixes: r15 Encoder + epoch29 x3 CTC", total=len(rows), every=1
    ) as tick:
        for n, row in enumerate(rows, 1):
            path = output / "prefixes" / (order_key(row["id"]) + ".json")
            identity = row_identity(output, row)
            if load_cell(path, identity) is None:
                if detector is None:
                    with (
                        stage_progress("Load frozen r15 Encoder and x3 CTC"),
                        block_legacy_qwen_vllm_registration(),
                    ):
                        detector = load_detector(
                            Path(contract["model"]), Path(contract["checkpoint"]), table, vocab
                        )
                wave = audio(row)
                prefixes = []
                for chunk in schedule_stream_chunks(
                    len(wave), chunk_size_sec=contract["config"]["chunk_seconds"]
                ):
                    pred = _retrieve_waveform(
                        detector, wave[: chunk.end_sample], vocab, saved_raw_rank_depth=64
                    )
                    prefixes.append(
                        {
                            "end_sample": chunk.end_sample,
                            "selected_matches": pred["selected_matches"],
                            "raw_ranked_matches": pred["raw_ranked_matches"],
                            "timing": pred["timing"],
                        }
                    )
                pin(Path(row["audio_path"]), row["audio_sha256"])
                save_cell(path, {"identity": identity, "prefixes": prefixes})
            tick(n)


def receive(channel: Connection, child: subprocess.Popen[Any], timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if channel.poll(1):
            value = channel.recv()
            if "error" in value:
                raise RuntimeError(value["error"])
            return value
        if child.poll() is not None:
            raise RuntimeError("native decoder exited: " + str(child.returncode))
    raise TimeoutError("native decoder response timed out; existing cells retained")


def decoder_server(fd: int, output: Path) -> None:
    """Native vLLM subprocess deliberately never imports qwen_asr."""
    import numpy as np

    vllm = import_module("vllm")
    LLM, SamplingParams = vllm.LLM, vllm.SamplingParams

    contract = read(output / "contract.json")
    cfg = contract["config"]
    with Connection(fd) as channel:
        try:
            engine = LLM(
                model=contract["model"],
                model_impl="vllm",
                dtype="float16",
                enforce_eager=True,
                max_num_seqs=1,
                max_model_len=cfg["max_model_len"],
                gpu_memory_utilization=cfg["gpu_memory_utilization"],
                seed=cfg["seed"],
                generation_config="vllm",
                enable_prefix_caching=False,
            )
            sampling = SamplingParams(
                temperature=0, max_tokens=cfg["max_new_tokens"], seed=cfg["seed"]
            )
            channel.send({"ready": True})
            while True:
                request = channel.recv()
                wave = np.frombuffer(request["wave"], dtype="<f4")
                prediction = engine.generate(
                    {
                        "prompt_token_ids": request["ids"],
                        "multi_modal_data": {"audio": (wave, 16000)},
                    },
                    sampling_params=sampling,
                    use_tqdm=False,
                )[0].outputs[0]
                if prediction.finish_reason == "length":
                    raise RuntimeError("decoder token limit reached; refusing truncated evaluation")
                channel.send(
                    {
                        "text": prediction.text,
                        "finish_reason": prediction.finish_reason,
                        "prompt_ids_sha256": fingerprint(request["ids"]),
                        "audio_samples": len(wave),
                    }
                )
        except EOFError:
            pass
        except Exception as exc:
            channel.send({"error": str(exc)})
            raise


class DecoderBridge:
    def __init__(self, channel: Connection, child: subprocess.Popen[Any], tokenizer: Any):
        self.channel, self.child, self.tokenizer = channel, child, tokenizer
        self.expected_prompt = ""
        self.calls: list[dict[str, Any]] = []

    def generate(self, inputs: list[dict[str, Any]], **kwargs: Any) -> list[Any]:
        (request,) = inputs
        (wave,) = request["multi_modal_data"]["audio"]
        prompt = request["prompt"]
        if not self.expected_prompt or not prompt.startswith(self.expected_prompt):
            raise ValueError("official streaming request did not use refreshed prompt")
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        self.channel.send({"ids": ids, "wave": wave.astype("<f4", copy=False).tobytes()})
        reply = receive(self.channel, self.child, 600)
        if reply["prompt_ids_sha256"] != fingerprint(ids) or reply["audio_samples"] != len(wave):
            raise ValueError("native decoder did not acknowledge the exact causal request")
        self.calls.append(
            {
                "prompt": prompt,
                "audio_samples": len(wave),
                "finish_reason": reply["finish_reason"],
                "prompt_ids_sha256": reply["prompt_ids_sha256"],
            }
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(text=reply["text"])])]


class AuditedBackend:
    def __init__(self, wrapper: Any, bridge: DecoderBridge):
        self.wrapper, self.bridge = wrapper, bridge
        self.tokenizer = wrapper.processor.tokenizer

    def init_streaming_state(self, **kwargs: Any) -> Any:
        return self.wrapper.init_streaming_state(**kwargs)

    def streaming_transcribe(self, wave: Any, state: Any) -> None:
        self.bridge.expected_prompt = state.prompt_raw
        self.wrapper.streaming_transcribe(wave, state)

    def finish_streaming_transcribe(self, state: Any) -> None:
        self.bridge.expected_prompt = state.prompt_raw
        self.wrapper.finish_streaming_transcribe(state)


def prompt_probe(backend: Any, cfg: dict[str, Any]) -> None:
    args = dict(
        language="Arabic",
        chunk_size_sec=cfg["chunk_seconds"],
        unfixed_chunk_num=cfg["unfixed_chunk_num"],
        unfixed_token_num=cfg["unfixed_token_num"],
    )
    state = backend.init_streaming_state(context="", **args)
    before = state.prompt_raw
    refresh_streaming_prompt(backend, state, prompt="القاهرة", **args)
    if (
        state.context != "القاهرة"
        or state.prompt_raw == before
        or "القاهرة" not in state.prompt_raw
    ):
        raise ValueError("dynamic context refresh is not supported by this installed runtime")


def asr_worker(output: Path, scope: str) -> None:
    contract, rows, table = selection(output, scope)
    inputs_unchanged(contract)
    cfg = contract["config"]
    pending = []
    for row in rows:
        prefix_path = output / "prefixes" / (order_key(row["id"]) + ".json")
        if load_cell(prefix_path, row_identity(output, row)) is None:
            raise ValueError("missing causal prefixes: " + row["id"])
        for mode in MODES:
            identity = fingerprint([row_identity(output, row), sha(prefix_path), mode])
            path = output / "asr" / mode / (order_key(row["id"]) + ".json")
            if load_cell(path, identity) is None:
                pending.append((row, mode, path, identity))
    if not pending:
        print("All streaming cells verified; no GPU reload needed", flush=True)
        return
    with block_legacy_qwen_vllm_registration():
        from qwen_asr import Qwen3ASRModel

        Qwen3ASRProcessor = import_module("qwen_asr.core.transformers_backend").Qwen3ASRProcessor
    processor = Qwen3ASRProcessor.from_pretrained(
        contract["model"], local_files_only=True, fix_mistral_regex=True
    )
    left, right = socket.socketpair()
    channel = Connection(left.detach())
    try:
        child = subprocess.Popen(
            [
                str(ROOT / cfg["decoder_python"]),
                "-u",
                "-B",
                str(Path(__file__)),
                "--stage",
                "decoder",
                "--fd",
                str(right.fileno()),
                "--output-dir",
                str(output),
            ],
            pass_fds=(right.fileno(),),
            cwd=ROOT,
        )
    finally:
        right.close()
    try:
        with stage_progress("Load native vLLM r15 Decoder; Encoder extraction process has exited"):
            receive(channel, child, 1200)
        bridge = DecoderBridge(channel, child, processor.tokenizer)
        wrapper = Qwen3ASRModel(backend="vllm", model=bridge, processor=processor)
        backend = AuditedBackend(wrapper, bridge)
        prompt_probe(backend, cfg)
        surfaces = {t["id"]: t["surface"] for t in table}
        active = tuple(t["id"] for t in table if t["pronunciation"]["status"] == "active")
        with stage_progress(
            "Streaming final ASR: no-RAG / Top5 / Top7", total=len(pending), every=1
        ) as tick:
            previous_id, wave = "", None
            for n, (row, mode, path, identity) in enumerate(pending, 1):
                if row["id"] != previous_id:
                    wave, previous_id = audio(row), row["id"]
                prefixes = load_cell(
                    output / "prefixes" / (order_key(row["id"]) + ".json"),
                    row_identity(output, row),
                )["prefixes"]
                replay = None if mode == "no_rag" else PrefixReplay(prefixes, active, int(mode[3:]))
                # Reference and expected targets are withheld from ALL model/prompt calls.
                sample = StreamingSample(
                    case_id=row["id"],
                    sample_id=row["id"],
                    reference_text="",
                    language="ar",
                    expected_hotword_ids=(),
                    expected_surfaces=(),
                    active_hotword_ids=active,
                )
                bridge.calls.clear()
                result, timeline = run_streaming_sample(
                    backend=backend,
                    waveform=wave,
                    sample=sample,
                    group="C" if replay is None else "D",
                    hotword_surfaces=surfaces,
                    ctc_detector=replay,
                    prompt_template=cfg["prompt_template"],
                    chunk_size_sec=cfg["chunk_seconds"],
                    unfixed_chunk_num=cfg["unfixed_chunk_num"],
                    unfixed_token_num=cfg["unfixed_token_num"],
                    asr_language="Arabic",
                )
                if replay is not None:
                    replay.finish()
                if not bridge.calls:
                    raise ValueError("no actual decoder generation was performed")
                injected = {
                    str(i)
                    for step in timeline
                    for i in cast(list[str], step["injected_hotword_ids"])
                }
                prediction = str(result["prediction"])
                pin(Path(row["audio_path"]), row["audio_sha256"])
                save_cell(
                    path,
                    {
                        "identity": identity,
                        "id": row["id"],
                        "mode": mode,
                        "source_group": row["source_group"],
                        "prediction": prediction,
                        "reference": row["text"],
                        "timeline": timeline,
                        "actual_decoder_requests": list(bridge.calls),
                        "inference_seconds": result["inference_seconds"],
                        "metrics": score_row(row["text"], prediction, table, injected),
                    },
                )
                tick(n)
    finally:
        channel.close()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def collect(output: Path, scope: str) -> dict[str, Any]:
    contract, rows, table = selection(output, scope)
    cells = []
    for row in rows:
        prefix = output / "prefixes" / (order_key(row["id"]) + ".json")
        load_cell(prefix, row_identity(output, row))
        for mode in MODES:
            identity = fingerprint([row_identity(output, row), sha(prefix), mode])
            cell = load_cell(output / "asr" / mode / (order_key(row["id"]) + ".json"), identity)
            if cell is None:
                raise ValueError("incomplete evaluation; no final report will be emitted")
            cells.append(cell)
    report = {
        "status": "completed",
        "scope": scope,
        "output_dir": str(output),
        "contract_sha256": sha(output / "contract.json"),
        "samples": len(rows),
        "table_entries": len(table),
        "active_table_entries": sum(t["pronunciation"]["status"] == "active" for t in table),
        "prompt_audit": {
            mode: {
                "decoder_requests": sum(
                    len(c["actual_decoder_requests"]) for c in cells if c["mode"] == mode
                ),
                "nonempty_prompt_steps": sum(
                    bool(step["injected_hotword_ids"])
                    for c in cells
                    if c["mode"] == mode
                    for step in c["timeline"]
                ),
            }
            for mode in MODES
        },
        "by_mode": summary(cells, {r["id"] for r in rows}),
        "by_source": {
            source: {
                mode: aggregate(
                    [c for c in cells if c["mode"] == mode and c["source_group"] == source]
                )
                for mode in MODES
            }
            for source in sorted({r["source_group"] for r in rows})
        },
        "model": contract["model"],
        "checkpoint": contract["checkpoint"],
        "expected": contract["expected"],
        "selection": contract["selection"],
        "table_sha256": contract["config"]["expected_table_sha256"],
        "WER_CER_normalization": NORMALIZATION,
        "notes": [
            "Final transcript recall, not retrieval candidate recall.",
            "Same-step causal prefix prompts; original audio, no added noise.",
            "Offline streaming simulation, not concurrent service latency.",
            "Corpus-conditioned table; not unseen-business-vocabulary evaluation.",
            "Prompt adoption is co-occurrence, not a causal attribution.",
            "False words are measured against reference text, not human audio review.",
        ],
    }
    inputs_unchanged(contract)
    write_json(output / ("smoke_report.json" if scope == "smoke" else "report.json"), report)
    return report


def run(config_path: Path, output: Path, gpu: int, check: bool) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_job(output):
        with stage_progress("Verify frozen 2078 audio, table, epoch29 and installed runtimes"):
            contract = prepare(config_path, output)
        if check:
            return {"status": "checked", "output_dir": str(output)}
        for scope in ("smoke", "full"):
            for stage in ("prefix", "asr"):
                with stage_progress(f"{scope} {stage}; see child sample/chunk progress"):
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            "-B",
                            str(Path(__file__)),
                            "--stage",
                            stage,
                            "--scope",
                            scope,
                            "--output-dir",
                            str(output),
                        ],
                        check=True,
                        cwd=ROOT,
                    )
            report = collect(output, scope)
            if scope == "smoke":
                print(
                    "20-sample three-way engineering smoke passed; continue full frozen set",
                    flush=True,
                )
        with stage_progress("Recheck input identities and package small report"):
            inputs_unchanged(contract)
            with zipfile.ZipFile(
                output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for name in (
                    "report.json",
                    "smoke_report.json",
                    "contract.json",
                    "runtime_main.json",
                    "runtime_decoder.json",
                ):
                    archive.write(output / name, name)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/egy_streaming_e2e.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stage", choices=("prefix", "asr", "decoder"))
    parser.add_argument("--scope", choices=("smoke", "full"), default="full")
    parser.add_argument("--probe", choices=("main", "decoder"))
    parser.add_argument("--fd", type=int)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / read(args.config)["output"]).resolve()
    try:
        if args.probe:
            sealed(
                output / ("runtime_" + args.probe + ".json"), runtime_info(args.probe == "decoder")
            )
        elif args.stage == "decoder":
            if args.fd is None:
                raise ValueError("internal decoder requires socket")
            decoder_server(args.fd, output)
        elif args.stage == "prefix":
            prefix_worker(output, args.scope)
        elif args.stage == "asr":
            asr_worker(output, args.scope)
        else:
            print(
                json.dumps(
                    run(args.config.resolve(), output, args.gpu, args.check),
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error": str(exc), "output_dir": str(output)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
