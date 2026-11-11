from __future__ import annotations

import json
from pathlib import Path
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


def test_repair_rejects_changed_source_part_before_retry(tmp_path: Path) -> None:
    source, output = tmp_path / "original", tmp_path / "repair"
    original_run(source)
    (source / "parts/ar_msa/00000/review.jsonl").write_text("changed", encoding="utf-8")
    with (
        patch.object(run, "CalimaFrontend") as frontend,
        pytest.raises(ValueError, match="missing/changed pinned input"),
    ):
        run.repair({"assets": "unused"}, source, output, resume=False)
    frontend.assert_not_called()
    assert not (output / "report.json").exists()
