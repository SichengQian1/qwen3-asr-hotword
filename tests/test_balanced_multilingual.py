from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.training.balanced_multilingual import (
    LanguagePool,
    build_balanced_multilingual_training,
)


def test_balanced_multilingual_reaches_target_and_keeps_spanish_core(
    tmp_path: Path,
) -> None:
    en = _write_pool(
        tmp_path,
        "en",
        "en-US",
        [(f"en_{index}", "swift_us_english", 10.0) for index in range(5)],
    )
    es = _write_pool(
        tmp_path,
        "es",
        "es",
        [
            ("es_slr", "slr61", 10.0),
            ("es_rio", "common_voice_rioplatense_v26", 10.0),
            ("es_aux_1", "common_voice_latam_auxiliary", 10.0),
            ("es_aux_2", "common_voice_latam_auxiliary", 10.0),
            ("es_aux_3", "common_voice_latam_auxiliary", 10.0),
        ],
    )
    pt = _write_pool(
        tmp_path,
        "pt",
        "pt-BR",
        [(f"pt_{index}", "noah_500h", 10.0) for index in range(5)],
    )

    output = tmp_path / "balanced"
    summary = build_balanced_multilingual_training(
        [en, es, pt],
        output,
        target_hours=0.01,
        seed=17,
        include_all_sources={
            "es": ("slr61", "common_voice_rioplatense_v26")
        },
    )

    for language in ("en", "es", "pt"):
        metrics = summary["language_metrics"][language]
        assert metrics["selected"]["hours"] >= 0.01
        assert metrics["overshoot_seconds"] <= 10.0
        assert len(_read_jsonl(output / f"full_ctc_train_{language}.jsonl")) == 4
    spanish_ids = {
        row["id"] for row in _read_jsonl(output / "full_ctc_train_es.jsonl")
    }
    assert {"es_slr", "es_rio"}.issubset(spanish_ids)
    combined = _read_jsonl(output / "full_ctc_train.jsonl")
    assert len(combined) == 12
    assert summary["combined_records"] == 12
    assert summary["duplicate_selected_ids"] == 0
    assert summary["duplicate_selected_audio_paths"] == 0
    assert summary["test_set_used"] is False
    assert summary["test_set_content_read"] is False
    assert (output / "sha256.txt").is_file()


def test_balanced_multilingual_rejects_used_test_pool(tmp_path: Path) -> None:
    pools = [
        _write_pool(
            tmp_path,
            language,
            {"en": "en-US", "es": "es", "pt": "pt-BR"}[language],
            [(f"{language}_0", f"{language}_source", 40.0)],
        )
        for language in ("en", "es", "pt")
    ]
    summary_path = pools[0].root / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["test_set_used"] = True
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="test boundary"):
        build_balanced_multilingual_training(
            pools,
            tmp_path / "output",
            target_hours=0.01,
        )


def _write_pool(
    root: Path,
    language: str,
    language_tag: str,
    rows: list[tuple[str, str, float]],
) -> LanguagePool:
    pool_root = root / language
    pool_root.mkdir()
    train_path = pool_root / "full_ctc_train.jsonl"
    payloads = [
        _record(sample_id, language_tag, source, duration)
        for sample_id, source, duration in rows
    ]
    train_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads),
        encoding="utf-8",
    )
    train_hours = sum(duration for _, _, duration in rows) / 3600.0
    dummy_hash = "0" * 64
    summary = {
        "status": "pass",
        "test_set_sealed": True,
        "test_set_used": False,
        "manifest_paths": {
            "train": str(train_path),
            "validation": str(pool_root / "sealed_validation.jsonl"),
            "test": str(pool_root / "sealed_test.jsonl"),
        },
        "manifest_sha256": {
            "train": _sha256(train_path),
            "validation": dummy_hash,
            "test": dummy_hash,
        },
        "split_records": {"train": len(rows), "validation": 1, "test": 1},
        "split_audio_hours": {
            "train": train_hours,
            "validation": 0.01,
            "test": 0.01,
        },
    }
    (pool_root / "split_summary.json").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    return LanguagePool(language, pool_root)


def _record(
    sample_id: str,
    language: str,
    source: str,
    duration: float,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "audio_path": f"/audio/{sample_id}.wav",
        "audio_relative": f"/audio/{sample_id}.wav",
        "text": sample_id,
        "language": language,
        "split": "train",
        "source_corpus": source,
        "release_source": "original_ready",
        "speaker_id": f"speaker_{sample_id}",
        "phoneme_token_ids": [1],
        "label_length": 1,
        "duration_seconds": duration,
        "ctc_time_upsampling_factor": 2,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
