from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from scripts import repair_arabic_native_labels as run
from scripts.label_arabic_native import CONFIG, VOCAB


def original_run(source: Path) -> None:
    vocab_sha = sha(VOCAB)
    part = source / "parts/ar_msa/00000"
    part.mkdir(parents=True)
    rows = [
        {
            "group": "ar_msa",
            "source": "/original.json",
            "row": index,
            "text": text,
            "audio_path": "/audio-not-read.wav",
            "duration_seconds": 2.0,
            "issues": [],
        }
        for index, text in enumerate(("شكرا", "بيت", "طويل", "مجهول", "آخر"), 1)
    ]
    ready = {
        **rows[0],
        "words": ["شكرا"],
        "phones": ["ʃ"],
        "ids": [154],
        "vocab_sha256": vocab_sha,
    }
    ready_line = json.dumps(ready, ensure_ascii=False) + "\n"
    (source / "labels.jsonl").write_text(ready_line, encoding="utf-8")
    (part / "labels.jsonl").write_text(ready_line, encoding="utf-8")
    issues = [
        "phone_OOV:'ar_ipa:y'",
        "phone_OOV:'ar_ipa:yː'",
        "no_analysis:مجهول",
        "phone_OOV:'ar_ipa:unreviewed'",
    ]
    (part / "review.jsonl").write_text(
        "".join(
            json.dumps({**row, "label_issue": issue}, ensure_ascii=False) + "\n"
            for row, issue in zip(rows[1:], issues, strict=True)
        ),
        encoding="utf-8",
    )
    piece = {
        "group": "ar_msa",
        "rows": 5,
        "ready_rows": 1,
        "ready_hours": 2 / 3600,
        "reasons": {"phone_OOV": 3, "no_analysis": 1},
        "vocab_sha256": vocab_sha,
        "labels": "parts/ar_msa/00000/labels.jsonl",
        "review": "parts/ar_msa/00000/review.jsonl",
        "labels_sha256": sha(part / "labels.jsonl"),
        "review_sha256": sha(part / "review.jsonl"),
    }
    write_json(part / "completed.json", piece)
    identity = {
        "backend_sha256": run.OLD_BACKEND_SHA,
        "config_sha256": sha(CONFIG),
        "script_sha256": sha(run.ROOT / "scripts/label_arabic_native.py"),
        "inventories": {"ar_msa": "original inventory hash"},
    }
    write_json(source / "contract.json", identity)
    write_json(
        source / "report.json",
        {
            "status": "automatic_labels_completed",
            "vocab_sha256": vocab_sha,
            "num_classes": 167,
            "identity": identity,
            "labels_sha256": sha(source / "labels.jsonl"),
            "by_group": {
                "ar_msa": {
                    "source_rows": 5,
                    "ready_rows": 1,
                    "ready_hours": 2 / 3600,
                    "review_rows": 4,
                    "reasons": piece["reasons"],
                }
            },
        },
    )


def test_repair_reuses_ready_rows_retries_only_y_and_resumes(tmp_path: Path) -> None:
    source, output = tmp_path / "original", tmp_path / "repair"
    original_run(source)
    before = {p.relative_to(source): sha(p) for p in source.rglob("*") if p.is_file()}
    frontend = Mock()
    frontend.label.side_effect = [
        {"words": ["بيت"], "phones": ["b", "a", "j", "t"], "ids": [74, 72, 91, 116]},
        {"words": ["طويل"], "phones": ["b"] * 80, "ids": [74] * 80},
    ]
    with patch.object(run, "CalimaFrontend", return_value=frontend):
        report = run.repair({"assets": "unused"}, source, output, resume=False)
        assert frontend.label.call_count == 2
        assert list(frontend.label.call_args_list[0].args) == ["بيت"]
        assert report["repair"]["recovered_rows"] == 1
        assert report["repair"]["retry_failures"] == {"estimated_CTC_length_infeasible": 1}
        assert report["by_group"]["ar_msa"]["ready_rows"] == 2
        assert report["by_group"]["ar_msa"]["review_rows"] == 3
        records = list(iter_jsonl(output / "labels.jsonl"))
        assert records[0] == next(iter_jsonl(source / "labels.jsonl"))
        assert records[1]["ids"] == [74, 72, 91, 116]
        assert [r["row"] for r in iter_jsonl(output / "parts/ar_msa/00000/review.jsonl")] == [
            3,
            4,
            5,
        ]
        digest = sha(output / "labels.jsonl")
        resumed = run.repair({"assets": "unused"}, source, output, resume=True)
        assert resumed["labels_sha256"] == digest
        assert frontend.label.call_count == 2
    assert before == {p.relative_to(source): sha(p) for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize("workers", [1, 2])
def test_repair_rejects_changed_source_part_before_retry(tmp_path: Path, workers: int) -> None:
    source, output = tmp_path / "original", tmp_path / "repair"
    original_run(source)
    (source / "parts/ar_msa/00000/review.jsonl").write_text("changed", encoding="utf-8")
    with (
        patch.object(run, "ProcessPoolExecutor", side_effect=thread_pool),
        patch.object(run, "CalimaFrontend") as frontend,
        pytest.raises(ValueError, match="missing/changed pinned input"),
    ):
        run.repair({"assets": "unused"}, source, output, resume=False, workers=workers)
    frontend.assert_not_called()
    assert not (output / "report.json").exists()


def add_second_part(source: Path) -> None:
    first = source / "parts/ar_msa/00000"
    second = source / "parts/ar_msa/00001"
    second.mkdir()
    piece = json.loads((first / "completed.json").read_text())
    for key in ("labels", "review"):
        records = list(iter_jsonl(first / f"{key}.jsonl"))
        for row in records:
            row["row"] += 5
        target = second / f"{key}.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
        piece[key] = str(target.relative_to(source))
        piece[key + "_sha256"] = sha(target)
    write_json(second / "completed.json", piece)
    (source / "labels.jsonl").write_bytes(
        (first / "labels.jsonl").read_bytes() + (second / "labels.jsonl").read_bytes()
    )
    report = json.loads((source / "report.json").read_text())
    group = report["by_group"]["ar_msa"]
    for key in ("source_rows", "ready_rows", "ready_hours", "review_rows"):
        group[key] *= 2
    group["reasons"] = {key: count * 2 for key, count in group["reasons"].items()}
    report["labels_sha256"] = sha(source / "labels.jsonl")
    write_json(source / "report.json", report)


def fake_label(text: str) -> dict[str, Any]:
    if text == "بيت":
        return {"words": [text], "phones": ["b", "a", "j", "t"], "ids": [74, 72, 91, 116]}
    return {"words": [text], "phones": ["b"] * 80, "ids": [74] * 80}


def thread_pool(**kwargs: Any) -> ThreadPoolExecutor:
    # Run the same per-part worker scheduling with the injected phonetic test backend.
    kwargs.pop("mp_context")
    return ThreadPoolExecutor(**kwargs)


def test_parallel_completion_preserves_serial_label_order_and_counts(tmp_path: Path) -> None:
    source, serial, parallel = (tmp_path / name for name in ("original", "serial", "parallel"))
    original_run(source)
    add_second_part(source)
    frontend = Mock()
    frontend.label.side_effect = fake_label
    with (
        patch.object(run, "CalimaFrontend", return_value=frontend),
        patch.object(run, "ProcessPoolExecutor", side_effect=thread_pool) as pool,
    ):
        one = run.repair({"assets": "unused"}, source, serial, resume=False)
        many = run.repair({"assets": "unused"}, source, parallel, resume=False, workers=2)
    pool.assert_called_once()
    assert one["labels_sha256"] == many["labels_sha256"]
    assert one["by_group"] == many["by_group"]
    assert one["repair"] == many["repair"]
    assert many["repair"]["recovered_rows"] == 2
    assert many["execution"]["workers"] == 2


def test_reviewed_serial_upgrade_keeps_contract_and_completed_parts(tmp_path: Path) -> None:
    source, output = tmp_path / "original", tmp_path / "repair"
    original_run(source)
    frontend = Mock()
    frontend.label.side_effect = fake_label
    with patch.object(run, "CalimaFrontend", return_value=frontend):
        previous = run.repair({"assets": "unused"}, source, output, resume=False)
    contract = json.loads((output / "contract.json").read_text())
    contract["script_sha256"] = run.REVIEWED_SERIAL_SHA
    write_json(output / "contract.json", contract)
    contract_sha = sha(output / "contract.json")
    with (
        patch.object(run, "ProcessPoolExecutor", side_effect=thread_pool),
        patch.object(run, "CalimaFrontend") as model,
    ):
        resumed = run.repair({"assets": "unused"}, source, output, resume=True, workers=2)
    model.assert_not_called()
    assert resumed["labels_sha256"] == previous["labels_sha256"]
    assert resumed["identity"]["script_sha256"] == run.REVIEWED_SERIAL_SHA
    assert sha(output / "contract.json") == contract_sha
    contract["backend_sha256"] = "unreviewed change"
    write_json(output / "contract.json", contract)
    with pytest.raises(ValueError, match="inputs or code changed"):
        run.repair({"assets": "unused"}, source, output, resume=True, workers=2)
