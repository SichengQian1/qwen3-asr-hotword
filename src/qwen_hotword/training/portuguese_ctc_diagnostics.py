from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.ctc_diagnostics import diagnose_ctc_checkpoint
from qwen_hotword.training.sharded_ctc import (
    DiskFeatureCache,
    load_disk_feature_cache,
    load_feature_shard,
)

OUTPUT_FILES = (
    "portuguese_ctc_diagnostics.json",
    "top_error_samples.jsonl",
    "README.md",
    "sha256.txt",
)


@dataclass(frozen=True)
class PortugueseValidationMetadata:
    sample_id: str
    source_corpus: str
    release_source: str
    text: str
    audio_path: str
    duration_seconds: float


@dataclass(frozen=True)
class CtcSampleLengths:
    reference_tokens: int
    input_frames: int
    reference_tokens_per_input_frame: float
    minimum_ctc_frames: int
    minimum_ctc_ratio: float


def diagnose_portuguese_ctc_checkpoints(
    *,
    validation_cache_path: str | Path,
    validation_manifest_path: str | Path,
    vocab_path: str | Path,
    multilingual_checkpoint_path: str | Path,
    portuguese_checkpoint_path: str | Path,
    output_dir: str | Path,
    device: Any,
    batch_size: int = 256,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("diagnostic batch size must be positive")
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(f"diagnostic output already exists: {destination}")

    manifest = Path(validation_manifest_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    multilingual_checkpoint = Path(multilingual_checkpoint_path).expanduser()
    portuguese_checkpoint = Path(portuguese_checkpoint_path).expanduser()
    cache = load_disk_feature_cache(
        validation_cache_path,
        expected_split="validation",
        source_manifest_path=manifest,
        vocab_path=vocab_file,
        verify_sha256=True,
    )
    vocab = load_phoneme_vocab(vocab_file)
    metadata = load_portuguese_validation_metadata(manifest, cache)
    selected_ids = set(metadata)
    lengths = load_ctc_sample_lengths(cache, selected_ids, num_classes=len(vocab.tokens))
    pressure_groups, pressure_summary = build_pressure_tertiles(lengths)
    groupings = {
        "source_corpus": {sample_id: item.source_corpus for sample_id, item in metadata.items()},
        "release_source": {sample_id: item.release_source for sample_id, item in metadata.items()},
        "source_release": {
            sample_id: f"{item.source_corpus}::{item.release_source}"
            for sample_id, item in metadata.items()
        },
        "reference_tokens_per_input_frame_tertile": pressure_groups,
    }

    checkpoint_paths = {
        "multilingual": multilingual_checkpoint,
        "portuguese": portuguese_checkpoint,
    }
    full_reports: dict[str, dict[str, object]] = {}
    for label, checkpoint in checkpoint_paths.items():
        report = diagnose_ctc_checkpoint(
            checkpoint,
            cache,
            vocab,
            device=device,
            batch_size=batch_size,
            selected_sample_ids=selected_ids,
            sample_groupings=groupings,
            include_sample_metrics=True,
        )
        head_config = report.get("head_config")
        if (
            not isinstance(head_config, dict)
            or head_config.get("time_upsampling_factor") != cache.ctc_time_upsampling_factor
        ):
            raise ValueError(
                f"{label} checkpoint time factor differs from the feature cache contract"
            )
        full_reports[label] = report

    comparison = compare_checkpoint_reports(
        full_reports["multilingual"],
        full_reports["portuguese"],
    )
    top_rows = build_top_error_rows(full_reports, metadata, lengths)
    compact_reports = {
        label: {key: value for key, value in report.items() if key != "sample_metrics"}
        for label, report in full_reports.items()
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "same_sample_portuguese_ctc_head_stratified_validation_diagnostics",
        "test_set_used": False,
        "training_performed": False,
        "encoder_inference_performed": False,
        "selection": {
            "balanced_language_bucket": "pt",
            "sample_count": len(selected_ids),
            "source_corpora": sorted({item.source_corpus for item in metadata.values()}),
            "release_sources": sorted({item.release_source for item in metadata.values()}),
        },
        "pressure_stratification": pressure_summary,
        "inputs": {
            "validation_manifest": _file_identity(manifest),
            "vocab": _file_identity(vocab_file),
            "validation_cache": {
                "path": str(cache.root),
                "fingerprint": cache.fingerprint,
                "sample_count": cache.sample_count,
                "shard_count": cache.shard_count,
                "sha256_verified": cache.sha256_verified,
                "ctc_time_upsampling_factor": cache.ctc_time_upsampling_factor,
            },
            "checkpoints": {
                label: _file_identity(path) for label, path in checkpoint_paths.items()
            },
        },
        "checkpoints": compact_reports,
        "comparison": comparison,
        "top_error_sample_count": len(top_rows),
    }

    destination.mkdir(parents=True)
    _write_json(destination / "portuguese_ctc_diagnostics.json", summary)
    _write_jsonl(destination / "top_error_samples.jsonl", top_rows)
    (destination / "README.md").write_text(
        "# Portuguese CTC stratified diagnostics\n\n"
        "This output compares the multilingual and Portuguese-specific CTC Heads on "
        "the exact same Portuguese subset of the balanced validation feature cache. "
        "It performs no training, encoder inference, or sealed-test access.\n",
        encoding="utf-8",
    )
    _write_hashes(destination)
    return summary


def load_portuguese_validation_metadata(
    manifest_path: str | Path,
    cache: DiskFeatureCache,
) -> dict[str, PortugueseValidationMetadata]:
    path = Path(manifest_path).expanduser()
    cache_ids = {sample_id for descriptor in cache.shards for sample_id in descriptor.sample_ids}
    seen: set[str] = set()
    selected: dict[str, PortugueseValidationMetadata] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"validation row is not an object at {path}:{line_number}")
            sample_id = _required_string(raw, "id", path, line_number)
            if sample_id in seen:
                raise ValueError(f"duplicate validation sample ID: {sample_id}")
            seen.add(sample_id)
            if raw.get("split") != "validation":
                raise ValueError(f"non-validation row at {path}:{line_number}")
            if raw.get("balanced_language_bucket") != "pt":
                continue
            language = _required_string(raw, "language", path, line_number)
            if language not in {"pt", "pt-BR"}:
                raise ValueError(f"Portuguese bucket has unexpected language {language!r}")
            duration = raw.get("duration_seconds")
            if not isinstance(duration, int | float) or isinstance(duration, bool) or duration <= 0:
                raise ValueError(f"invalid duration_seconds at {path}:{line_number}")
            selected[sample_id] = PortugueseValidationMetadata(
                sample_id=sample_id,
                source_corpus=_required_string(raw, "source_corpus", path, line_number),
                release_source=_required_string(raw, "release_source", path, line_number),
                text=_required_string(raw, "text", path, line_number),
                audio_path=_required_string(raw, "audio_path", path, line_number),
                duration_seconds=float(duration),
            )
    if seen != cache_ids:
        raise ValueError(
            "validation manifest IDs differ from cache IDs: "
            f"missing={len(cache_ids - seen)}, extra={len(seen - cache_ids)}"
        )
    if not selected:
        raise ValueError("validation manifest contains no Portuguese samples")
    return selected


def load_ctc_sample_lengths(
    cache: DiskFeatureCache,
    selected_sample_ids: set[str],
    *,
    num_classes: int,
) -> dict[str, CtcSampleLengths]:
    lengths: dict[str, CtcSampleLengths] = {}
    for descriptor in cache.shards:
        for sample in load_feature_shard(descriptor, num_classes=num_classes):
            if sample.sample_id not in selected_sample_ids:
                continue
            reference_tokens = len(sample.token_ids)
            input_frames = int(sample.hidden_states.shape[0]) * cache.ctc_time_upsampling_factor
            minimum_frames = reference_tokens + sum(
                left == right
                for left, right in zip(sample.token_ids, sample.token_ids[1:], strict=False)
            )
            lengths[sample.sample_id] = CtcSampleLengths(
                reference_tokens=reference_tokens,
                input_frames=input_frames,
                reference_tokens_per_input_frame=reference_tokens / input_frames,
                minimum_ctc_frames=minimum_frames,
                minimum_ctc_ratio=minimum_frames / input_frames,
            )
    if set(lengths) != selected_sample_ids:
        raise RuntimeError("failed to collect lengths for every selected Portuguese sample")
    return lengths


def build_pressure_tertiles(
    lengths: dict[str, CtcSampleLengths],
) -> tuple[dict[str, str], dict[str, object]]:
    if len(lengths) < 3:
        raise ValueError("pressure tertiles require at least three samples")
    ordered = sorted(
        lengths,
        key=lambda sample_id: (
            lengths[sample_id].reference_tokens_per_input_frame,
            sample_id,
        ),
    )
    group_names = ("low", "medium", "high")
    groups = {
        sample_id: group_names[min(2, index * 3 // len(ordered))]
        for index, sample_id in enumerate(ordered)
    }
    summary_groups: dict[str, object] = {}
    for group in group_names:
        values = [
            lengths[sample_id].reference_tokens_per_input_frame
            for sample_id in ordered
            if groups[sample_id] == group
        ]
        summary_groups[group] = {
            "sample_count": len(values),
            "minimum_ratio": min(values),
            "maximum_ratio": max(values),
            "mean_ratio": sum(values) / len(values),
        }
    return groups, {
        "metric": "reference_tokens / effective_ctc_input_frames",
        "policy": "stable_sample_count_tertiles_sorted_by_ratio_then_sample_id",
        "groups": summary_groups,
    }


def compare_checkpoint_reports(
    multilingual: dict[str, object],
    portuguese: dict[str, object],
) -> dict[str, object]:
    multilingual_samples = _sample_metrics_by_id(multilingual)
    portuguese_samples = _sample_metrics_by_id(portuguese)
    if set(multilingual_samples) != set(portuguese_samples):
        raise ValueError("checkpoint reports do not cover identical samples")
    paired = {"multilingual_better": 0, "multilingual_worse": 0, "tied": 0}
    total_error_delta = 0
    for sample_id in multilingual_samples:
        new_errors = _required_metric_int(multilingual_samples[sample_id], "errors")
        old_errors = _required_metric_int(portuguese_samples[sample_id], "errors")
        total_error_delta += new_errors - old_errors
        if new_errors < old_errors:
            paired["multilingual_better"] += 1
        elif new_errors > old_errors:
            paired["multilingual_worse"] += 1
        else:
            paired["tied"] += 1

    multilingual_validation = _required_mapping(multilingual, "validation")
    portuguese_validation = _required_mapping(portuguese, "validation")
    multilingual_dimensions = _required_mapping(multilingual, "validation_by_dimension")
    portuguese_dimensions = _required_mapping(portuguese, "validation_by_dimension")
    by_dimension: dict[str, object] = {}
    for dimension, raw_groups in multilingual_dimensions.items():
        groups = _as_mapping(raw_groups, f"multilingual dimension {dimension}")
        reference_groups = _as_mapping(
            portuguese_dimensions.get(dimension),
            f"Portuguese dimension {dimension}",
        )
        if set(groups) != set(reference_groups):
            raise ValueError(f"checkpoint grouping differs for dimension {dimension}")
        by_dimension[dimension] = {
            group: _metric_delta(
                _as_mapping(groups[group], f"multilingual group {group}"),
                _as_mapping(reference_groups[group], f"Portuguese group {group}"),
            )
            for group in groups
        }
    return {
        "delta_definition": "multilingual_minus_portuguese_specific",
        "overall": _metric_delta(multilingual_validation, portuguese_validation),
        "paired_sample_counts": paired,
        "paired_total_error_delta": total_error_delta,
        "by_dimension": by_dimension,
    }


def build_top_error_rows(
    reports: dict[str, dict[str, object]],
    metadata: dict[str, PortugueseValidationMetadata],
    lengths: dict[str, CtcSampleLengths],
    *,
    limit: int = 25,
) -> list[dict[str, object]]:
    multilingual = _sample_metrics_by_id(reports["multilingual"])
    portuguese = _sample_metrics_by_id(reports["portuguese"])
    ordered = sorted(
        multilingual,
        key=lambda sample_id: (
            _required_metric_float(multilingual[sample_id], "phoneme_error_rate"),
            _required_metric_int(multilingual[sample_id], "errors"),
            sample_id,
        ),
        reverse=True,
    )
    regressions = sorted(
        (
            sample_id
            for sample_id in multilingual
            if _required_metric_int(multilingual[sample_id], "errors")
            > _required_metric_int(portuguese[sample_id], "errors")
        ),
        key=lambda sample_id: (
            _required_metric_int(multilingual[sample_id], "errors")
            - _required_metric_int(portuguese[sample_id], "errors"),
            sample_id,
        ),
        reverse=True,
    )
    selected = list(dict.fromkeys((*ordered[:limit], *regressions[:limit])))
    rows = []
    for sample_id in selected:
        item = metadata[sample_id]
        new_metrics = multilingual[sample_id]
        old_metrics = portuguese[sample_id]
        rows.append(
            {
                **asdict(item),
                "lengths": asdict(lengths[sample_id]),
                "multilingual": new_metrics,
                "portuguese": old_metrics,
                "multilingual_minus_portuguese_errors": (
                    _required_metric_int(new_metrics, "errors")
                    - _required_metric_int(old_metrics, "errors")
                ),
            }
        )
    return rows


def _metric_delta(
    multilingual: dict[str, object],
    portuguese: dict[str, object],
) -> dict[str, object]:
    reference_tokens = _required_metric_int(multilingual, "reference_tokens")
    if reference_tokens != _required_metric_int(portuguese, "reference_tokens"):
        raise ValueError("checkpoint reports use different reference token totals")
    result: dict[str, object] = {"reference_tokens": reference_tokens}
    for name in ("phoneme_error_rate", "substitution_rate", "deletion_rate", "insertion_rate"):
        new_value = _derived_rate(multilingual, name)
        old_value = _derived_rate(portuguese, name)
        result[f"multilingual_{name}"] = new_value
        result[f"portuguese_{name}"] = old_value
        result[f"delta_{name}"] = new_value - old_value
    return result


def _derived_rate(metrics: dict[str, object], name: str) -> float:
    if name == "phoneme_error_rate":
        value = metrics.get(name)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        raise ValueError("phoneme_error_rate is not numeric")
    count_name = {
        "substitution_rate": "substitutions",
        "deletion_rate": "deletions",
        "insertion_rate": "insertions",
    }[name]
    return _required_metric_int(metrics, count_name) / _required_metric_int(
        metrics, "reference_tokens"
    )


def _sample_metrics_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = report.get("sample_metrics")
    if not isinstance(rows, list):
        raise ValueError("checkpoint report has no sample metrics")
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("checkpoint sample metric is not an object")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise ValueError("checkpoint sample metric has an invalid sample ID")
        result[sample_id] = row
    return result


def _required_mapping(value: dict[str, object], name: str) -> dict[str, Any]:
    return _as_mapping(value.get(name), name)


def _as_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not an object")
    return value


def _required_metric_int(value: dict[str, object], name: str) -> int:
    raw = value.get(name)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"metric {name} is not an integer")
    return raw


def _required_metric_float(value: dict[str, object], name: str) -> float:
    raw = value.get(name)
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        raise ValueError(f"metric {name} is not numeric")
    return float(raw)


def _required_string(
    value: dict[str, object],
    name: str,
    path: Path,
    line_number: int,
) -> str:
    raw = value.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"missing {name} at {path}:{line_number}")
    return raw.strip()


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"input file does not exist: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_hashes(destination: Path) -> None:
    lines = []
    for name in OUTPUT_FILES:
        if name == "sha256.txt":
            continue
        path = destination / name
        lines.append(f"{_sha256(path)}  {name}\n")
    (destination / "sha256.txt").write_text("".join(lines), encoding="utf-8")
