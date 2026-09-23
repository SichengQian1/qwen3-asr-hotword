from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.balanced_multilingual import (
    LanguagePool,
    _scan_train_manifest,
    _validate_pool_summary,
    _validate_scanned_train,
)
from qwen_hotword.training.spanish_480_freeze import freeze_plan, protected_audio
from qwen_hotword.training.spanish_480_plan import candidate
from qwen_hotword.training.spanish_capacity import _sha


def dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data))


def jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def checks(path: Path) -> None:
    (path / "sha256.txt").write_text(
        "".join(
            f"{_sha(p)}  {p.name}\n"
            for p in sorted(path.iterdir())
            if p.is_file() and p.name != "sha256.txt"
        )
    )


def tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def setup(root: Path, *, conflict: str | None = None) -> Path:
    pool, manifest, plan = (root / p for p in ("pool", "manifest", "plan"))
    for path in (pool, manifest, plan):
        path.mkdir()
    audio = {name: root / f"{name}.wav" for name in ("old", "new", "validation", "test")}
    for name, path in audio.items():
        path.write_bytes(name.encode())
    if conflict:
        audio["new"].write_bytes(audio[conflict].read_bytes())
    source = root / "source.tsv"
    tsv(
        source,
        [
            {
                "audio": str(audio[name]),
                "speaker_id": name,
                "source_split": "train" if name == "old" else name,
                "text": "unused metadata transcript",
            }
            for name in ("old", "validation", "test")
        ],
    )
    tsv(
        pool / "speaker_split_assignments.tsv",
        [
            {"speaker_id": name, "split": "train" if name == "old" else name}
            for name in ("old", "validation", "test")
        ],
    )
    dump(
        pool / "split_config.json",
        {"inputs": {"old": {"source_tsv": {"path": str(source), "sha256": _sha(source)}}}},
    )
    raw: dict[str, dict[str, Any]] = {}
    for name in ("old", "new"):
        raw[name] = {
            "id": name,
            "audio_path": str(audio[name]),
            "duration_seconds": 240 * 3600,
            "text": "fixture",
            "phoneme_token_ids": [1, 2, 3],
            "label_length": 3,
            "estimated_ctc_input_length": 5,
            "ctc_minimum_input_length": 3,
            "ctc_time_upsampling_factor": 2,
            "split": "train" if name == "old" else "unsplit",
            "language": "es" if name == "old" else "es-419",
            "speaker_id": "old" if name == "old" else None,
            "release_source": "original_ready",
            "source_corpus": name,
            "source_tsv": str(source),
            "split_hash": 0.1,
            "row_number": 1,
            "training_ready": True,
            "issues": [],
        }
    jsonl(pool / "full_ctc_train.jsonl", [raw["old"]])
    jsonl(pool / "full_ctc_validation.jsonl", [{"audio_path": str(audio["validation"])}])
    jsonl(manifest / "train_ready.jsonl", [raw["new"]])
    jsonl(manifest / "needs_review.jsonl", [])
    old: dict[str, Any] = {
        "status": "pass",
        "test_set_sealed": True,
        "test_set_used": False,
        "manifest_paths": {
            s: str(pool / f"full_ctc_{s}.jsonl") for s in ("train", "validation", "test")
        },
        "manifest_sha256": {
            "train": _sha(pool / "full_ctc_train.jsonl"),
            "validation": _sha(pool / "full_ctc_validation.jsonl"),
            "test": "f" * 64,
        },
        "split_records": {"train": 1, "validation": 1, "test": 1},
        "split_audio_hours": {"train": 240, "validation": 1, "test": 1},
    }
    dump(pool / "split_summary.json", old)
    checks(pool)
    choices = []
    for name in raw:
        item = candidate(raw[name], name, "original_ready")
        item["origin"] = "old_train" if name == "old" else "noah"
        if name == "new":
            item.update(
                source_id="original-new",
                audio_file_sha256=_sha(audio[name]),
                directory_group_hint="G1",
            )
        choices.append(item)
    jsonl(plan / "proposed_ids.jsonl", choices)
    dump(plan / "config.json", {"pool": str(pool), "manifest": str(manifest), "target_hours": 480})
    dump(
        plan / "report.json",
        {
            "status": "plan_completed",
            "training_ready": False,
            "target_hours": 480,
            "combined": {"records": 2, "hours": 480},
            "inputs": {
                str(p): {"sha256": _sha(p)}
                for p in [
                    pool / "full_ctc_train.jsonl",
                    manifest / "train_ready.jsonl",
                    manifest / "needs_review.jsonl",
                ]
            },
            "validation_reference": {"sha256": old["manifest_sha256"]["validation"]},
            "test_reference": {"sha256": "f" * 64},
        },
    )
    checks(plan)
    return plan


def test_freeze_only_after_identity_checks_without_test_manifest(tmp_path: Path) -> None:
    plan = setup(tmp_path)
    result = freeze_plan(plan, tmp_path / "frozen", workers=1)
    assert result["status"] == "completed"
    assert result["selection"]["hours"] == 480
    assert result["audio_identity_audit"]["checked_files"] == 4
    assert result["test_manifest_content_read"] is False
    assert result["test_audio_bytes_read_for_identity_only"] is True
    assert not (tmp_path / "pool/full_ctc_test.jsonl").exists()
    final = [json.loads(x) for x in Path(result["train_manifest"]).read_text().splitlines()]
    new = next(r for r in final if r["id"] == "new")
    assert new["language"] == "es" and new["source_language"] == "es-419"
    assert new["speaker_id"] is None
    assert new["effective_ctc_input_length"] == 10
    assert not (tmp_path / "frozen/train.pending.jsonl").exists()
    summary = json.loads((tmp_path / "frozen/split_summary.json").read_text())
    assert summary["manifest_sha256"]["train"] == _sha(Path(result["train_manifest"]))
    _validate_pool_summary(
        summary, LanguagePool("es", tmp_path / "frozen"), Path(result["train_manifest"])
    )
    scan = _scan_train_manifest(
        Path(result["train_manifest"]), language="es", seed=1, expected_tags=frozenset({"es"})
    )
    _validate_scanned_train(scan, summary, "es")
    assert summary["manifest_paths"]["validation"] == str(
        tmp_path / "pool/full_ctc_validation.jsonl"
    )


@pytest.mark.parametrize(
    "conflict,reason",
    [
        ("old", "duplicate_train_file_bytes"),
        ("validation", "train_holdout_file_overlap"),
        ("test", "train_holdout_file_overlap"),
    ],
)
def test_conflicts_never_publish_train(tmp_path: Path, conflict: str, reason: str) -> None:
    plan = setup(tmp_path, conflict=conflict)
    out = tmp_path / "out"
    result = freeze_plan(plan, out, workers=1)
    assert result["status"] == "blocked_audio_conflicts"
    assert result["audio_identity_audit"]["conflict_groups_by_reason"][reason] == 1
    assert not (out / "full_ctc_train.jsonl").exists()
    assert not (out / "split_summary.json").exists()
    assert (out / "train.pending.jsonl").exists()


def test_changed_noah_bytes_fail_closed(tmp_path: Path) -> None:
    plan = setup(tmp_path)
    (tmp_path / "new.wav").write_bytes(b"changed after plan")
    with pytest.raises(ValueError, match="staging SHA"):
        freeze_plan(plan, tmp_path / "out", workers=1)
    assert (tmp_path / "out/FAILED.txt").exists()
    assert not (tmp_path / "out/full_ctc_train.jsonl").exists()


def test_rejects_modified_plan_and_existing_output(tmp_path: Path) -> None:
    plan = setup(tmp_path)
    with pytest.raises(FileExistsError):
        freeze_plan(plan, plan)
    (plan / "proposed_ids.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        freeze_plan(plan, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_holdout_superset_uses_speaker_metadata_and_rejects_changed_source(tmp_path: Path) -> None:
    setup(tmp_path)
    source = tmp_path / "source.tsv"
    tsv(
        source,
        [
            {
                "audio": str(tmp_path / "validation.wav"),
                "speaker_id": "validation",
                "source_split": "unsplit",
            },
            {
                "audio": str(tmp_path / "extra.wav"),
                "speaker_id": "validation",
                "source_split": "unsplit",
            },
            {"audio": str(tmp_path / "test.wav"), "speaker_id": "test", "source_split": "test"},
        ],
    )
    pool = tmp_path / "pool"
    with pytest.raises(ValueError, match="source metadata identity"):
        protected_audio(pool, {})
    config = json.loads((pool / "split_config.json").read_text())
    config["inputs"]["old"]["source_tsv"]["sha256"] = _sha(source)
    dump(pool / "split_config.json", config)
    checks(pool)
    result = protected_audio(pool, {})
    assert result[str(tmp_path / "extra.wav")] == {"validation"}
