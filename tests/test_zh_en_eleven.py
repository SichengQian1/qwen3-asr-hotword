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


def _fixture_retrieve(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": [[{"word": sample["language"], "score": 1.0, "phoneme": "t"}]],
        "raw_last": [sample["language"]],
    }


def test_parallel_reuses_serial_identity_and_spawn_results(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from concurrent.futures import ProcessPoolExecutor

    from qwen_hotword.evaluation import zh_en_parallel as par

    spec = importlib.util.spec_from_file_location(
        "eleven_parallel_cli", ROOT / "scripts/run_zh_en_480_eleven.py"
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    old = {"runner_sha256": cli.SERIAL_RUNNER_SHA, "head": "original", "audio_set": "unchanged"}
    write_json(tmp_path / "run_config.json", old)
    new = {**old, "runner_sha256": "later"}
    rid = cli.resume_id(tmp_path, new)
    assert rid == digest(old) and json.loads((tmp_path / "run_config.json").read_text()) == old
    with pytest.raises(ValueError, match="inputs or scoring"):
        cli.resume_id(tmp_path, {**new, "head": "different"})
    examples = [
        dict(sample_id=f"a{i}", language="en", audio_sha256=str(i) * 64, step_count=1)
        for i in range(4)
    ]
    saved = cli.cached(tmp_path, examples[0])
    first = _fixture_retrieve(examples[0])
    write_json(
        saved,
        {
            "run": rid,
            "audio": examples[0]["audio_sha256"],
            "language": "en",
            "digest": digest(first),
            "result": first,
        },
    )
    before = saved.stat().st_mtime_ns
    monkeypatch.setattr(par, "worker_limit", lambda *_: 2)
    monkeypatch.setattr(par, "_infer", _fixture_retrieve)
    for key in (
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(key, "1")
    # Real spawn pool on CPU; substitute just the per-utterance model operation.
    monkeypatch.setattr(
        par,
        "ProcessPoolExecutor",
        lambda max_workers, mp_context, **_: ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context
        ),
    )
    groups = {"first": examples, "overlap": examples[:2]}
    par.prefill(tmp_path, groups, rid, 2, 5, (), cli.cached, cli.load_result)
    assert before == saved.stat().st_mtime_ns
    assert len(list((tmp_path / "audio_results").rglob("*.json"))) == 4
    for sample in examples:
        assert cli.load_result(cli.cached(tmp_path, sample), rid, sample) == _fixture_retrieve(
            sample
        )
    monkeypatch.setattr(par, "worker_limit", lambda *_: pytest.fail("no GPU required on resume"))
    par.prefill(tmp_path, groups, rid, 2, 5, (), cli.cached, cli.load_result)


def test_parallel_capacity_retry_and_replay_mismatch(tmp_path: Path, monkeypatch: Any) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from qwen_hotword.evaluation import zh_en_parallel as par

    examples = [
        dict(sample_id=str(i), audio_sha256=str(i), language="zh", step_count=1) for i in range(3)
    ]
    attempts = []

    def pool(max_workers: int, **_: Any) -> ThreadPoolExecutor:
        attempts.append(max_workers)
        return ThreadPoolExecutor(max_workers=max_workers)

    def infer(sample: dict[str, Any]) -> dict[str, Any]:
        if attempts == [2] and sample == examples[1]:
            raise par.WorkerCapacityError("synthetic OOM")
        return _fixture_retrieve(sample)

    monkeypatch.setattr(par, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(par, "_infer", infer)
    monkeypatch.setattr(par, "worker_limit", lambda *_: 2)
    for key in (
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(key, "1")

    def path_for(out: Path, s: dict[str, Any]) -> Path:
        return out / (str(s["sample_id"]) + ".json")

    def load(p: Path, run: str, s: dict[str, Any]) -> dict[str, Any]:
        return dict(json.loads(p.read_text())["result"])

    par.prefill(tmp_path, {"test": examples}, "test", 2, 5, (), path_for, load)
    assert attempts == [2, 1] and len(list(tmp_path.glob("*.json"))) == 3
    new = {**examples[0], "sample_id": "new"}
    monkeypatch.setattr(par, "_infer", lambda s: {"steps": [[]], "raw_last": []})
    with pytest.raises(ValueError, match="timeline differs"):
        par.prefill(tmp_path, {"test": examples + [new]}, "test", 2, 5, (), path_for, load)
    assert not path_for(tmp_path, new).exists()
