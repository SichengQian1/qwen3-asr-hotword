from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.audio_conflict_audit import audit_conflicts
from qwen_hotword.training.ctc_overfit import load_experiment_records
from qwen_hotword.training.multilingual_480_repair import choose_replacements, repair_training
from qwen_hotword.training.spanish_480_freeze import audit_audio
from qwen_hotword.training.spanish_480_plan import candidate
from qwen_hotword.training.spanish_capacity import _sha


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value))


def rows(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def checks(root: Path) -> None:
    (root / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(root.iterdir()) if p.name != "sha256.txt")
    )


def make_row(root: Path, name: str, lang: str, seconds: float, data: bytes) -> dict[str, Any]:
    audio = root / f"{name}.wav"
    audio.write_bytes(data)
    return {
        "id": name,
        "audio_path": str(audio),
        "split": "train",
        "text": "fixture",
        "language": "en-US" if lang == "en" else "pt-BR" if lang == "pt" else "es",
        "source_corpus": f"{lang}_source",
        "duration_seconds": seconds,
        "release_source": "original_ready",
        "balanced_language_bucket": lang,
        "experiment": "full-ctc-v1",
        "dataset_version": "old",
        "phoneme_token_ids": [1, 2],
        "label_length": 2,
        "ctc_minimum_input_length": 2,
        "estimated_ctc_input_length": 10,
        "ctc_time_upsampling_factor": 2,
        "effective_ctc_input_length": 20,
        "effective_ctc_target_ratio": 0.1,
    }


def fixture(root: Path) -> dict[str, Any]:
    freeze, plan = root / "blocked", root / "plan"
    freeze.mkdir()
    plan.mkdir()
    pending = {
        "en": [make_row(root, f"e{i}", "en", 2, b"en-dup") for i in range(2)],
        "es": [make_row(root, f"s{i}", "es", 2, f"es-{i}".encode()) for i in range(2)],
        "pt": [
            make_row(
                root,
                f"p{i}",
                "pt",
                1,
                (b"pt-inconsistent" if i < 2 else b"heldout" if i == 2 else b"pt-unique"),
            )
            for i in range(4)
        ],
    }
    pending["pt"][1]["text"] = "different"
    val = root / "heldout.wav"
    val.write_bytes(b"heldout")
    for lang, values in pending.items():
        rows(freeze / f"{lang}.pending.jsonl", values)
    selected = {r["audio_path"]: {} for rs in pending.values() for r in rs}
    audit = audit_audio(selected, {str(val): {"validation"}}, freeze, workers=1)
    cfg: dict[str, Any] = {"target_hours": 4 / 3600, "pools": {}}
    for lang in ("en", "pt"):
        pool = root / lang
        pool.mkdir()
        additions = [
            make_row(
                root,
                f"{lang}_extra{i}",
                lang,
                2 if lang == "en" else 1,
                f"{lang}-extra-{i}".encode(),
            )
            for i in range(8)
        ]
        values = pending[lang] + additions
        path = pool / "full_ctc_train.jsonl"
        rows(path, values)
        hours = sum(r["duration_seconds"] for r in values) / 3600
        hashes = {"train": _sha(path), "validation": "v" * 64, "test": "t" * 64}
        dump(
            pool / "split_summary.json",
            {
                "status": "pass",
                "test_set_sealed": True,
                "test_set_used": False,
                "manifest_paths": {"train": str(path)},
                "manifest_sha256": hashes,
                "split_records": {"train": len(values)},
                "split_audio_hours": {"train": hours},
                "corpus_metrics": {
                    f"{lang}_source": {"split_train": {"records": len(values), "hours": hours}}
                },
            },
        )
        cfg["pools"][lang] = {
            "pool": str(pool),
            "sources": [f"{lang}_source"],
            "manifest_sha256": hashes,
            "records": len(values),
            "hours": hours,
        }
    dump(plan / "config.json", cfg)
    checks(plan)
    dump(
        freeze / "report.json",
        {
            "status": "blocked_audio_conflicts",
            "training_manifest_ready": False,
            "selection": {
                lang: {"records": len(rs), "hours": 4 / 3600} for lang, rs in pending.items()
            },
            "audio_identity_audit": audit,
            "plan_dir": str(plan),
            "heldout_references": {"unchanged": "sealed test manifest never opened"},
            "frozen_es_reference": {"sha256": "existing Spanish selection"},
        },
    )
    checks(freeze)
    diagnosis = audit_conflicts(freeze, target_hours=4 / 3600)
    return {
        "freeze_dir": str(freeze),
        "target_hours": 4 / 3600,
        "seed": 23,
        "removed_ids_sha256": diagnosis["proposed_removed_ids_sha256"],
        "train_pool_sha256": {
            lang: spec["manifest_sha256"]["train"] for lang, spec in cfg["pools"].items()
        },
        "expected_sha256": {Path(p).name: v["sha256"] for p, v in diagnosis["inputs"].items()},
    }


def test_repair_excludes_refills_reaudits_and_preserves_old_outputs(tmp_path: Path) -> None:
    cfg = fixture(tmp_path)
    old = {p.name: _sha(p) for p in (tmp_path / "blocked").iterdir()}
    result = repair_training(cfg, tmp_path / "fixed", workers=1)
    assert result["training_manifest_ready"]
    assert result["retained_audio_rehashed"] and not result["training_started"]
    assert result["audio_identity_audit"]["conflict_groups_by_reason"] == {}
    assert result["replacements"]["en"]["records"] == 1
    assert result["replacements"]["pt"]["records"] == 3
    assert all(stats["hours"] == pytest.approx(4 / 3600) for stats in result["selection"].values())
    assert result["artifacts"]["combined"]["records"] == 8
    manifest = tmp_path / "fixed/full_ctc_train.jsonl"
    final = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert not {"p0", "p1", "p2", "e1"} & {r["id"] for r in final}
    assert {r["id"] for r in final if r["language"] == "es"} == {"s0", "s1"}
    assert (
        len(load_experiment_records(manifest, num_classes=90, expected_experiment="full-ctc-v1"))
        == 8
    )
    assert {p.name: _sha(p) for p in (tmp_path / "blocked").iterdir()} == old
    for line in (tmp_path / "fixed/sha256.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert digest == _sha(tmp_path / "fixed" / name)


def test_replacements_reject_known_bytes_and_duplicate_new_bytes(tmp_path: Path) -> None:
    raw = [make_row(tmp_path, str(i), "pt", 1, b"same" if i < 2 else b"new") for i in range(3)]
    candidates = [candidate(r, r["source_corpus"], "original_ready") for r in raw]
    known = {_sha(Path(raw[0]["audio_path"]))}
    chosen, _ = choose_replacements(
        candidates, {candidates[0]["stratum"]: 1}, 1, known, set(), set(), seed=1
    )
    assert [r["id"] for r in chosen] == ["2"]
    with pytest.raises(ValueError, match="insufficient"):
        choose_replacements(
            candidates,
            {candidates[0]["stratum"]: 2},
            2,
            {_sha(Path(raw[0]["audio_path"]))},
            set(),
            set(),
            seed=1,
        )


@pytest.mark.parametrize(
    "failure", ["changed_retained_audio", "changed_pending", "exclusions", "existing"]
)
def test_repair_fails_closed(tmp_path: Path, failure: str) -> None:
    cfg = fixture(tmp_path)
    out = tmp_path / "out"
    if failure == "changed_retained_audio":
        (tmp_path / "s0.wav").write_bytes(b"changed after previous audit")
    elif failure == "changed_pending":
        (tmp_path / "blocked/pt.pending.jsonl").write_text("tampered")
    elif failure == "exclusions":
        cfg["removed_ids_sha256"] = "changed"
    else:
        out.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        repair_training(cfg, out, workers=1)
    assert not (out / "full_ctc_train.jsonl").exists()
