from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    FeatureShardDescriptor,
)
from qwen_hotword.training.train_sampling import (
    PT_ORIGINAL_READY_EQUAL_RECORD_EXPOSURE,
    build_portuguese_original_ready_sampling_plan,
)


def _cache(tmp_path: Path, sample_ids: tuple[str, ...]) -> DiskFeatureCache:
    descriptor = FeatureShardDescriptor(
        shard_index=0,
        feature_path=tmp_path / "unused.pt",
        metadata_path=tmp_path / "unused.json",
        record_count=len(sample_ids),
        total_frames=len(sample_ids),
        total_target_tokens=len(sample_ids),
        feature_bytes=1,
        feature_sha256="0" * 64,
        sample_ids=sample_ids,
        ctc_time_upsampling_factor=2,
    )
    return DiskFeatureCache(
        split="train",
        root=tmp_path,
        sample_count=len(sample_ids),
        shard_count=1,
        feature_bytes=1,
        encoder_frozen_parameters=317_477_504,
        fingerprint="fixture-cache",
        sha256_verified=True,
        ctc_time_upsampling_factor=2,
        shards=(descriptor,),
    )


def _row(
    sample_id: str,
    language: str,
    release: str,
    *,
    source: str | None = None,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "split": "train",
        "balanced_language_bucket": language,
        "release_source": release,
        "source_corpus": source or f"{language}_source",
        "duration_seconds": 3.0,
    }


def test_portuguese_original_sampling_preserves_epoch_record_exposure(
    tmp_path: Path,
) -> None:
    rows = [
        _row("en-1", "en", "original_ready"),
        _row("en-2", "en", "original_ready"),
        _row("es-1", "es", "original_ready"),
        _row("es-2", "es", "original_ready"),
        _row("pt-o1", "pt", "original_ready", source="noah"),
        _row("pt-o2", "pt", "original_ready", source="noah"),
        _row("pt-o3", "pt", "original_ready", source="finance"),
        _row("pt-r1", "pt", "temporal_2x_recovery", source="noah"),
        _row("pt-r2", "pt", "temporal_2x_recovery", source="finance"),
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    cache = _cache(tmp_path, tuple(str(row["id"]) for row in rows))

    plan = build_portuguese_original_ready_sampling_plan(manifest, cache)
    repeats = plan.additional_repeat_counts(epoch=1, seed=7)

    assert plan.policy == PT_ORIGINAL_READY_EQUAL_RECORD_EXPOSURE
    assert plan.epoch_sample_count == len(rows)
    assert plan.additional_samples_per_epoch == 2
    assert set(plan.included_sample_ids) == {
        "en-1",
        "en-2",
        "es-1",
        "es-2",
        "pt-o1",
        "pt-o2",
        "pt-o3",
    }
    assert set(repeats) <= {"pt-o1", "pt-o2", "pt-o3"}
    assert sum(repeats.values()) == 2
    assert sum(repeats.get(sample_id, 0) for sample_id in ("pt-o1", "pt-o2")) == 1
    assert repeats["pt-o3"] == 1
    assert plan.summary["record_exposure_preserved"] is True
    assert plan.summary["test_set_used"] is False


def test_portuguese_original_sampling_rejects_manifest_cache_mismatch(
    tmp_path: Path,
) -> None:
    rows = [
        _row("en-1", "en", "original_ready"),
        _row("es-1", "es", "original_ready"),
        _row("pt-o1", "pt", "original_ready"),
        _row("pt-r1", "pt", "temporal_2x_recovery"),
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    cache = _cache(tmp_path, ("en-1", "es-1", "pt-o1", "missing"))

    with pytest.raises(ValueError, match="manifest IDs differ from cache IDs"):
        build_portuguese_original_ready_sampling_plan(manifest, cache)
