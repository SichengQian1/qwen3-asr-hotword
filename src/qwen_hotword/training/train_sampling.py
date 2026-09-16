from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qwen_hotword.training.sharded_ctc import DiskFeatureCache

PT_ORIGINAL_READY_EQUAL_RECORD_EXPOSURE = "pt_original_ready_equal_record_exposure"
SOURCE_STRATIFIED_EPOCH_SHUFFLE_VERSION = "source_stratified_epoch_shuffle_v1"


@dataclass(frozen=True)
class TrainingSamplingPlan:
    policy: str
    included_sample_ids: frozenset[str]
    oversample_pool_sample_ids: tuple[str, ...]
    oversample_strata: dict[str, tuple[str, ...]]
    additional_samples_by_stratum: dict[str, int]
    additional_samples_per_epoch: int
    epoch_sample_count: int
    fingerprint: str
    summary: dict[str, object]

    def additional_repeat_counts(self, *, epoch: int, seed: int) -> dict[str, int]:
        if epoch <= 0:
            raise ValueError("training sampling epoch must be positive")
        if not self.oversample_pool_sample_ids or self.additional_samples_per_epoch <= 0:
            return {}
        result: dict[str, int] = {}
        for stratum, stratum_ids in sorted(self.oversample_strata.items()):
            draws = self.additional_samples_by_stratum[stratum]
            if draws <= 0:
                continue
            ordered = list(stratum_ids)
            stratum_seed = int.from_bytes(
                hashlib.sha256(stratum.encode("utf-8")).digest()[:8],
                "big",
            )
            random.Random(seed + epoch + stratum_seed).shuffle(ordered)
            full_repeats, remainder = divmod(draws, len(ordered))
            for sample_id in ordered:
                if full_repeats > 0:
                    result[sample_id] = full_repeats
            for sample_id in ordered[:remainder]:
                result[sample_id] = result.get(sample_id, 0) + 1
        if sum(result.values()) != self.additional_samples_per_epoch:
            raise RuntimeError("training sampling repeat count is inconsistent")
        return result

    def identity_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "fingerprint": self.fingerprint,
            "included_sample_count": len(self.included_sample_ids),
            "oversample_pool_sample_count": len(self.oversample_pool_sample_ids),
            "oversample_strata": {
                stratum: {
                    "pool_sample_count": len(sample_ids),
                    "additional_samples_per_epoch": self.additional_samples_by_stratum[stratum],
                }
                for stratum, sample_ids in sorted(self.oversample_strata.items())
            },
            "additional_samples_per_epoch": self.additional_samples_per_epoch,
            "epoch_sample_count": self.epoch_sample_count,
        }


def build_portuguese_original_ready_sampling_plan(
    manifest_path: str | Path,
    cache: DiskFeatureCache,
) -> TrainingSamplingPlan:
    if cache.split != "train":
        raise ValueError("Portuguese original-ready sampling requires a train cache")
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {path}")

    cache_ids = {sample_id for descriptor in cache.shards for sample_id in descriptor.sample_ids}
    seen: set[str] = set()
    included: set[str] = set()
    pt_original: list[str] = []
    pt_original_by_source: defaultdict[str, list[str]] = defaultdict(list)
    language_counts: defaultdict[str, int] = defaultdict(int)
    language_seconds: defaultdict[str, float] = defaultdict(float)
    pt_release_counts: defaultdict[str, int] = defaultdict(int)
    pt_release_seconds: defaultdict[str, float] = defaultdict(float)
    pt_source_release_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    pt_source_release_seconds: defaultdict[tuple[str, str], float] = defaultdict(float)

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = _load_row(line, path, line_number)
            sample_id = _required_string(raw, "id", path, line_number)
            if sample_id in seen:
                raise ValueError(f"duplicate training sample ID: {sample_id}")
            seen.add(sample_id)
            if raw.get("split") != "train":
                raise ValueError(f"non-train row at {path}:{line_number}")
            language = _required_string(
                raw,
                "balanced_language_bucket",
                path,
                line_number,
            )
            if language not in {"en", "es", "pt"}:
                raise ValueError(f"unexpected language bucket {language!r}")
            duration = _required_duration(raw, path, line_number)
            language_counts[language] += 1
            language_seconds[language] += duration
            if language != "pt":
                included.add(sample_id)
                continue

            release = _required_string(raw, "release_source", path, line_number)
            if release not in {"original_ready", "temporal_2x_recovery"}:
                raise ValueError(f"unexpected Portuguese release source {release!r}")
            source = _required_string(raw, "source_corpus", path, line_number)
            pt_release_counts[release] += 1
            pt_release_seconds[release] += duration
            pt_source_release_counts[(source, release)] += 1
            pt_source_release_seconds[(source, release)] += duration
            if release == "original_ready":
                included.add(sample_id)
                pt_original.append(sample_id)
                pt_original_by_source[source].append(sample_id)

    if seen != cache_ids:
        raise ValueError(
            "training manifest IDs differ from cache IDs: "
            f"missing={len(cache_ids - seen)}, extra={len(seen - cache_ids)}"
        )
    if set(language_counts) != {"en", "es", "pt"}:
        raise ValueError("training manifest must contain en, es, and pt")
    if not pt_original or pt_release_counts["temporal_2x_recovery"] <= 0:
        raise ValueError("Portuguese sampling requires both original and recovery samples")

    additional = pt_release_counts["temporal_2x_recovery"]
    epoch_sample_count = len(included) + additional
    if epoch_sample_count != cache.sample_count:
        raise RuntimeError("Portuguese resampling must preserve the epoch sample count")
    recovery_by_source = {
        source: count
        for (source, release), count in sorted(pt_source_release_counts.items())
        if release == "temporal_2x_recovery" and count > 0
    }
    recovery_only_sources = set(recovery_by_source) - set(pt_original_by_source)
    if recovery_only_sources:
        raise ValueError(
            "Portuguese recovery sources have no original-ready oversample pool: "
            + ", ".join(sorted(recovery_only_sources))
        )
    oversample_strata = {
        source: tuple(sorted(pt_original_by_source[source])) for source in recovery_by_source
    }
    pool = tuple(
        sorted(sample_id for sample_ids in oversample_strata.values() for sample_id in sample_ids)
    )
    identity = {
        "schema_version": 1,
        "policy": PT_ORIGINAL_READY_EQUAL_RECORD_EXPOSURE,
        "sampling_algorithm": SOURCE_STRATIFIED_EPOCH_SHUFFLE_VERSION,
        "manifest_sha256": _sha256(path),
        "cache_fingerprint": cache.fingerprint,
        "included_sample_ids_sha256": _strings_sha256(sorted(included)),
        "oversample_pool_sample_ids_sha256": _strings_sha256(pool),
        "oversample_strata": {
            source: {
                "sample_ids_sha256": _strings_sha256(sample_ids),
                "pool_sample_count": len(sample_ids),
                "additional_samples_per_epoch": recovery_by_source[source],
            }
            for source, sample_ids in sorted(oversample_strata.items())
        },
        "excluded_sample_count": additional,
        "additional_samples_per_epoch": additional,
        "epoch_sample_count": epoch_sample_count,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "policy": PT_ORIGINAL_READY_EQUAL_RECORD_EXPOSURE,
        "sampling_algorithm": SOURCE_STRATIFIED_EPOCH_SHUFFLE_VERSION,
        "purpose": "Portuguese original-ready CTC training ablation",
        "manifest_path": str(path),
        "manifest_sha256": identity["manifest_sha256"],
        "cache_fingerprint": cache.fingerprint,
        "cache_sample_count": cache.sample_count,
        "included_unique_sample_count": len(included),
        "excluded_portuguese_recovery_sample_count": additional,
        "portuguese_original_unique_sample_count": len(pt_original),
        "portuguese_original_oversample_pool_count": len(pool),
        "additional_portuguese_original_draws_per_epoch": additional,
        "epoch_sample_count": epoch_sample_count,
        "language_input": {
            language: {
                "records": language_counts[language],
                "hours": language_seconds[language] / 3600.0,
            }
            for language in ("en", "es", "pt")
        },
        "portuguese_release_input": {
            release: {
                "records": pt_release_counts[release],
                "hours": pt_release_seconds[release] / 3600.0,
            }
            for release in ("original_ready", "temporal_2x_recovery")
        },
        "portuguese_source_release_input": {
            f"{source}::{release}": {
                "records": count,
                "hours": pt_source_release_seconds[(source, release)] / 3600.0,
            }
            for (source, release), count in sorted(pt_source_release_counts.items())
        },
        "sampling_semantics": (
            "Exclude every Portuguese temporal_2x_recovery sample. Each epoch, "
            "deterministically redraw the same number of records from Portuguese "
            "original_ready samples within each source corpus; English and Spanish "
            "samples remain exactly once."
        ),
        "unique_portuguese_hours": pt_release_seconds["original_ready"] / 3600.0,
        "record_exposure_preserved": True,
        "feature_cache_rebuilt": False,
        "test_set_used": False,
        "fingerprint": fingerprint,
    }
    return TrainingSamplingPlan(
        policy=PT_ORIGINAL_READY_EQUAL_RECORD_EXPOSURE,
        included_sample_ids=frozenset(included),
        oversample_pool_sample_ids=pool,
        oversample_strata=oversample_strata,
        additional_samples_by_stratum=recovery_by_source,
        additional_samples_per_epoch=additional,
        epoch_sample_count=epoch_sample_count,
        fingerprint=fingerprint,
        summary=summary,
    )


def _load_row(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"manifest row is not an object at {path}:{line_number}")
    return raw


def _required_string(
    raw: dict[str, Any],
    field: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {field} at {path}:{line_number}")
    return value.strip()


def _required_duration(raw: dict[str, Any], path: Path, line_number: int) -> float:
    value = raw.get("duration_seconds")
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"invalid duration_seconds at {path}:{line_number}")
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings_sha256(values: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
