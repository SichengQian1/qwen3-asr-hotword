from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.spanish_480_plan import candidate, prepare_plan, stratified_select
from qwen_hotword.training.spanish_capacity import _sha


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value))


def rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in values))


def checks(root: Path) -> None:
    (root / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(root.iterdir()) if p.name != "sha256.txt")
    )


def raw(root: Path, name: str, split: str = "unsplit") -> dict[str, Any]:
    return {
        "id": name,
        "audio_path": str(root / f"{name}.wav"),
        "text": "example",
        "duration_seconds": 4.0,
        "phoneme_token_ids": [1, 2, 3],
        "ctc_minimum_input_length": 3,
        "estimated_ctc_input_length": 5,
        "source_corpus": "old",
        "release_source": "original_ready",
        "split": split,
        "language": "es" if split != "unsplit" else "es-419",
        "speaker_id": None,
        "training_ready": True,
        "issues": [],
    }


def fixture(root: Path) -> dict[str, Any]:
    pool, staging, manifest = (root / name for name in ("pool", "staging", "manifest"))
    for p in (pool, staging, manifest):
        p.mkdir()
    old = {s: raw(root, s, s) for s in ("train", "validation")}
    for split, row in old.items():
        rows(pool / f"full_ctc_{split}.jsonl", [row])
    summary = {
        "status": "pass",
        "test_set_used": False,
        "manifest_sha256": {s: _sha(pool / f"full_ctc_{s}.jsonl") for s in old}
        | {"test": "sealed"},
        "split_records": {s: 1 for s in old},
        "split_audio_hours": {s: 4 / 3600 for s in old},
    }
    dump(pool / "split_summary.json", summary)
    checks(pool)
    ready = [raw(root, f"new{i}") for i in range(5)]
    recover = raw(root, "recovery")
    recover.update(
        phoneme_token_ids=[1, 2, 3, 1, 2, 3],
        ctc_minimum_input_length=6,
        training_ready=False,
        issues=[{"reason": "ctc_length_infeasible"}],
    )
    blocked = raw(root, "blocked")
    blocked.update(training_ready=False, issues=[{"reason": "dictionary_missing"}])
    full = ready + [recover, blocked]
    source = [
        dict(
            r,
            id=f"source-{r['id']}",
            status="audio_text_candidate",
            split="unassigned",
            audio_file_sha256=f"hash-{r['id']}",
            source_batch="batch",
            directory_group_hint="G1",
        )
        for r in full
    ]
    rows(staging / "candidates.jsonl", source)
    (staging / "source.tsv").write_text("fixture only")
    dump(
        staging / "report.json",
        {
            "status": "completed",
            "candidate_records": len(full),
            "source_json_sha256": "source",
        },
    )
    checks(staging)
    for key, value in (("train_ready", ready), ("needs_review", [recover, blocked])):
        rows(manifest / f"{key}.jsonl", value)
    dump(
        manifest / "summary.json",
        {
            "status": "pass",
            "split": "unsplit",
            "source_records": len(full),
            "ready_records": 5,
            "review_records": 2,
            "ready_audio_hours": 20 / 3600,
            "total_audio_hours": 28 / 3600,
        },
    )
    dictionary, vocab = root / "dict", root / "vocab"
    dictionary.write_text("fixture dictionary")
    vocab.write_text("fixture vocab")
    dump(
        manifest / "build_config.json",
        {
            "tsv": {"path": str(staging / "source.tsv")},
            "dictionary": {"path": str(dictionary)},
            "vocab": {"path": str(vocab)},
        },
    )
    return {
        "pool": str(pool),
        "staging": str(staging),
        "manifest": str(manifest),
        "target_hours": 17 / 3600,
        "seed": 1,
        "source_json_sha256": "source",
        "expected_sha256": {"dictionary": _sha(dictionary), "vocab": _sha(vocab)},
    }


def test_plan_preserves_old_excludes_dirty_and_never_reads_audio_or_test(tmp_path: Path) -> None:
    cfg = fixture(tmp_path)
    out = tmp_path / "out"
    report = prepare_plan(cfg, out)
    assert report["training_ready"] is False
    assert report["test_manifest_content_read"] is False
    assert report["audio_read"] is False
    assert report["noah_available"]["records"] == 6
    assert report["old_train"]["records"] == 1
    chosen = [json.loads(line) for line in (out / "proposed_ids.jsonl").read_text().splitlines()]
    assert len({r["id"] for r in chosen}) == len(chosen)
    assert "train" in {r["id"] for r in chosen}
    assert "blocked" not in {r["id"] for r in chosen}
    assert 17 <= report["combined"]["hours"] * 3600 < 21
    assert not (out / "full_ctc_train.jsonl").exists()
    assert all(r["speaker_id"] is None for r in chosen if r["origin"] == "noah")
    assert list(out.glob("sha256.txt"))
    prepare_plan(cfg, tmp_path / "again")
    assert (out / "proposed_ids.jsonl").read_bytes() == (
        tmp_path / "again/proposed_ids.jsonl"
    ).read_bytes()


@pytest.mark.parametrize("mutation", ["checksum", "join", "duplicate_hash", "overlap", "count"])
def test_rejects_invalid_inputs_before_output(tmp_path: Path, mutation: str) -> None:
    cfg = fixture(tmp_path)
    staging = Path(cfg["staging"])
    path = staging / "candidates.jsonl"
    source = [json.loads(x) for x in path.read_text().splitlines()]
    if mutation == "checksum":
        path.write_text("tampered")
    elif mutation == "count":
        p = Path(cfg["manifest"]) / "summary.json"
        summary = json.loads(p.read_text())
        summary["ready_records"] += 1
        dump(p, summary)
    else:
        if mutation == "join":
            source[0]["text"] = "changed"
        elif mutation == "duplicate_hash":
            source[1]["audio_file_sha256"] = source[0]["audio_file_sha256"]
        elif mutation == "overlap":
            p = Path(cfg["manifest"]) / "train_ready.jsonl"
            rs = [json.loads(x) for x in p.read_text().splitlines()]
            rs[0]["audio_path"] = str(tmp_path / "validation.wav")
            source[0]["audio_path"] = rs[0]["audio_path"]
            rows(p, rs)
        rows(path, source)
        checks(staging)
    with pytest.raises(ValueError):
        prepare_plan(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_existing_output_and_insufficient_capacity(tmp_path: Path) -> None:
    cfg = fixture(tmp_path)
    with pytest.raises(FileExistsError):
        prepare_plan(cfg, Path(cfg["manifest"]))
    cfg["target_hours"] = 100
    with pytest.raises(ValueError, match="insufficient"):
        prepare_plan(cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_lengths_repeats_recovery_boundary_and_stratum_balance(tmp_path: Path) -> None:
    row = raw(tmp_path, "bad")
    row["phoneme_token_ids"] = [1, 1, 2]
    with pytest.raises(ValueError, match="minimum"):
        candidate(row, "source", "original_ready")
    row.update(phoneme_token_ids=[1, 2, 3] * 3, ctc_minimum_input_length=9)
    assert candidate(row, "source", "temporal_2x_recovery")["effective_ctc_ratio"] == 0.9
    row.update(phoneme_token_ids=[1, 2, 3] * 3 + [1], ctc_minimum_input_length=10)
    with pytest.raises(ValueError, match="recovery"):
        candidate(row, "source", "temporal_2x_recovery")
    data = [dict(candidate(raw(tmp_path, str(i)), str(i % 2), "original_ready")) for i in range(20)]
    chosen, quotas = stratified_select(data, 40, 7)
    assert len(chosen) == 10
    assert all(q["selected_hours"] == q["quota_hours"] for q in quotas.values())
    reverse, _ = stratified_select(list(reversed(data)), 40, 7)
    assert chosen == reverse
