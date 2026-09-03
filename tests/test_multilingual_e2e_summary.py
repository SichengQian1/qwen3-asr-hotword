from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_hotword.inference.multilingual_e2e_summary import (
    D5_GATE,
    EXPECTED_CTC_CHECKPOINT_SHA256,
    LANGUAGES,
    summarize_multilingual_streaming_e2e,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quality(group: str, offset: int) -> dict[str, object]:
    matched = {"C": 1, "D": 2, "E": 2}[group]
    wrong_injected = 1 if group == "D" else 0
    return {
        "expected_hotwords": 2,
        "correct_prompt_injected_hotwords": 2 if group == "D" else 0,
        "prompt_hotword_recall": 1.0 if group == "D" else 0.0,
        "correct_prompt_adopted_hotwords": 2 if group == "D" else 0,
        "correct_prompt_adoption_rate": 1.0 if group == "D" else 0.0,
        "wrong_injected_hotwords": wrong_injected,
        "wrong_prompt_written_hotwords": 0,
        "wrong_prompt_landing_rate": 0.0,
        "final_hotword_recall": matched / 2,
        "final_hotword_precision": None if group == "C" else 1.0,
        "sample_hotword_hit_rate": matched / 2,
        "negative_hotword_hallucination_rate": 0.0,
        "wer": 0.1 + offset * 0.01,
        "cer": 0.05 + offset * 0.01,
        "mean_inference_seconds": 0.5,
        "prompt_causal_metrics": {
            "wrong_prompt_injected_hotwords": wrong_injected,
        },
    }


def _write_run(root: Path, code: str, checkpoint_sha: str, offset: int) -> None:
    root.mkdir()
    config = {
        "language": LANGUAGES[code],
        "groups": ["C", "D", "E"],
        "inputs": {"checkpoint": {"sha256": checkpoint_sha}},
        **D5_GATE,
    }
    summary = {
        "status": "pass",
        "test_set_used": False,
        "groups": {group: _quality(group, offset) for group in ("C", "D", "E")},
    }
    (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    for group in ("C", "D", "E"):
        for index in range(100):
            rows.append(
                {
                    "case_id": f"{code}-case-{index}",
                    "sample_id": f"{code}-sample-{index}",
                    "reference_text": "target one target two",
                    "experiment_group": group,
                }
            )
    (root / "sample_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    names = ("run_config.json", "summary.json", "sample_results.jsonl")
    (root / "sha256.txt").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def test_multilingual_summary_verifies_and_micro_aggregates(tmp_path: Path) -> None:
    checkpoint_sha = EXPECTED_CTC_CHECKPOINT_SHA256
    runs = {}
    for offset, code in enumerate(LANGUAGES):
        run = tmp_path / code
        _write_run(run, code, checkpoint_sha, offset)
        runs[code] = run

    output = tmp_path / "summary"
    report = summarize_multilingual_streaming_e2e(runs, output)

    assert report["status"] == "pass"
    assert report["total_sample_count"] == 300
    assert report["ctc_checkpoint_sha256"] == checkpoint_sha
    assert report["identity_checks"]["cross_language_sample_id_overlaps"] == {
        "en-es": 0,
        "en-pt": 0,
        "es-pt": 0,
    }
    aggregate = report["aggregate"]["groups"]
    assert aggregate["D"]["expected_hotwords"] == 6
    assert aggregate["D"]["correct_prompt_injected_hotwords"] == 6
    assert aggregate["D"]["wrong_prompt_injected_hotwords"] == 3
    assert aggregate["D"]["final_hotword_recall"] == 1.0
    assert aggregate["C"]["final_hotword_recall"] == 0.5
    assert aggregate["D"]["wer_macro"] == pytest.approx(0.11)
    assert (output / "sha256.txt").is_file()


def test_multilingual_summary_rejects_checkpoint_mismatch(tmp_path: Path) -> None:
    runs = {}
    for index, code in enumerate(LANGUAGES):
        run = tmp_path / code
        checkpoint = EXPECTED_CTC_CHECKPOINT_SHA256 if index == 0 else code[0] * 64
        _write_run(run, code, checkpoint, 0)
        runs[code] = run

    with pytest.raises(ValueError, match="sealed multilingual checkpoint"):
        summarize_multilingual_streaming_e2e(runs, tmp_path / "summary")
