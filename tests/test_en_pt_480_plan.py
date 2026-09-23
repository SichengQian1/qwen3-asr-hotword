from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.en_pt_480_plan import compact_report, prepare_plan
from qwen_hotword.training.spanish_capacity import _sha


def dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_row(root: Path, name: str, language: str, recovery: bool) -> dict[str, Any]:
    tokens = [1, 2, 3] * (2 if recovery else 1)
    return {
        "id": name,
        "split": "train",
        "language": language,
        "audio_path": str(root / f"{name}.wav"),
        "source_corpus": "conversation" if recovery else "read",
        "release_source": "temporal_2x_recovered" if recovery else "original_ready",
        "speaker_id": None if recovery else "speaker",
        "duration_seconds": 4.0,
        "estimated_ctc_input_length": 5,
        "ctc_time_upsampling_factor": 2,
        "effective_ctc_input_length": 10,
        "phoneme_token_ids": tokens,
        "ctc_minimum_input_length": len(tokens),
        "label_length": len(tokens),
        "effective_ctc_target_ratio": len(tokens) / 10,
    }


def fixture(root: Path) -> dict[str, Any]:
    config: dict[str, Any] = {"target_hours": 40 / 3600, "seed": 23, "pools": {}}
    for language, tag in (("en", "en-US"), ("pt", "pt-BR")):
        pool = root / language
        pool.mkdir()
        train = pool / "full_ctc_train.jsonl"
        data = [make_row(root, f"{language}{i}", tag, bool(i % 2)) for i in range(20)]
        write_rows(train, data)
        hashes = {"train": _sha(train), "validation": "v" * 64, "test": "t" * 64}
        summary = {
            "status": "pass",
            "test_set_used": False,
            "test_set_sealed": True,
            "manifest_paths": {"train": str(train)},
            "manifest_sha256": hashes,
            "split_records": {"train": 20, "validation": 2, "test": 2},
            "split_audio_hours": {"train": 80 / 3600, "validation": 1, "test": 1},
            "corpus_metrics": {
                s: {"split_train": {"records": 10, "hours": 40 / 3600}}
                for s in ("conversation", "read")
            },
        }
        dump(pool / "split_summary.json", summary)
        config["pools"][language] = {
            "pool": str(pool),
            "manifest_sha256": hashes,
            "records": 20,
            "hours": 80 / 3600,
            "sources": ["read", "conversation"],
        }
    es = root / "frozen_es.jsonl"
    write_rows(es, [make_row(root, "es0", "es", False)])
    config["frozen_es"] = {
        "train_manifest": str(es),
        "sha256": _sha(es),
        "records": 1,
        "hours": 4 / 3600,
    }
    return config


def repin(config: dict[str, Any], language: str) -> None:
    spec = config["pools"][language]
    pool = Path(spec["pool"])
    spec["manifest_sha256"]["train"] = _sha(pool / "full_ctc_train.jsonl")
    path = pool / "split_summary.json"
    summary = json.loads(path.read_text())
    summary["manifest_sha256"] = spec["manifest_sha256"]
    dump(path, summary)


def test_proportional_plan_deterministic_preserves_es_and_never_opens_holdout(
    tmp_path: Path,
) -> None:
    config = fixture(tmp_path)
    es = Path(config["frozen_es"]["train_manifest"])
    original = es.read_bytes()
    report = prepare_plan(config, tmp_path / "out")
    assert not report["training_ready"]
    assert not report["audio_read"]
    assert not report["test_manifest_content_read"]
    # Audio/validation/test files do not exist in this fixture.
    assert es.read_bytes() == original
    assert not list((tmp_path / "out").glob("full_ctc_train*"))
    for lang in ("en", "pt"):
        data = report["languages"][lang]
        assert data["selected"]["hours"] == pytest.approx(40 / 3600)
        assert data["selected"]["records"] == 10
        assert data["selected_speakers"]["missing_speaker_records"] == 5
        assert all(
            v["hours"] == pytest.approx(20 / 3600)
            for v in data["selected"]["by_dimension"]["release"].values()
        )
    prepare_plan(config, tmp_path / "again")
    for lang in ("en", "pt"):
        name = f"proposed_ids_{lang}.jsonl"
        assert (tmp_path / "out" / name).read_bytes() == (tmp_path / "again" / name).read_bytes()
    assert "stratum" not in compact_report(report)["languages"]["pt"]["selected"]["by_dimension"]
    assert "stratum" in report["languages"]["pt"]["selected"]["by_dimension"]
    for line in (tmp_path / "out/sha256.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert digest == _sha(tmp_path / "out" / name)


@pytest.mark.parametrize("mutation", ["sha", "tags", "duplicate", "ratio", "length", "minimum"])
def test_bad_train_identity_or_contract_blocks(tmp_path: Path, mutation: str) -> None:
    cfg = fixture(tmp_path)
    train = Path(cfg["pools"]["en"]["pool"]) / "full_ctc_train.jsonl"
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    if mutation == "sha":
        rows[0]["untracked_change"] = True
    elif mutation == "tags":
        rows[0]["language"] = "en-GB"
    elif mutation == "duplicate":
        rows[1]["audio_path"] = rows[0]["audio_path"]
    elif mutation == "ratio":
        rows[0]["effective_ctc_target_ratio"] = 0.6
    elif mutation == "length":
        rows[0]["label_length"] = 99
    else:
        rows[0]["phoneme_token_ids"] = [1, 1, 3]
    write_rows(train, rows)
    if mutation != "sha":
        repin(cfg, "en")
    with pytest.raises(ValueError):
        prepare_plan(cfg, tmp_path / "out")
    assert (tmp_path / "out/FAILED.txt").exists()
    assert not (tmp_path / "out/report.json").exists()


@pytest.mark.parametrize(
    "mutation", ["es_sha", "es_overlap", "source_hours", "capacity", "holdout"]
)
def test_fixed_es_summary_and_capacity_guards(tmp_path: Path, mutation: str) -> None:
    cfg = fixture(tmp_path)
    if mutation == "es_sha":
        Path(cfg["frozen_es"]["train_manifest"]).write_text("changed")
    elif mutation == "es_overlap":
        chosen = prepare_plan(cfg, tmp_path / "probe")
        assert chosen["status"] == "plan_completed"
        first = json.loads((tmp_path / "probe/proposed_ids_en.jsonl").read_text().splitlines()[0])
        es = Path(cfg["frozen_es"]["train_manifest"])
        write_rows(
            es, [dict(make_row(tmp_path, "es0", "es", False), audio_path=first["audio_path"])]
        )
        cfg["frozen_es"]["sha256"] = _sha(es)
    elif mutation == "capacity":
        cfg["target_hours"] = 500
    else:
        path = Path(cfg["pools"]["en"]["pool"]) / "split_summary.json"
        data = json.loads(path.read_text())
        if mutation == "source_hours":
            data["corpus_metrics"]["read"]["split_train"]["hours"] = 50
        else:
            data["manifest_sha256"]["test"] = "changed"
        dump(path, data)
    with pytest.raises(ValueError):
        prepare_plan(cfg, tmp_path / "out")
    assert not (tmp_path / "out/report.json").exists()


def test_no_overwrite_even_empty_directory(tmp_path: Path) -> None:
    cfg = fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(FileExistsError):
        prepare_plan(cfg, out)
    assert not list(out.iterdir())
