from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_hotword.inference.streaming_core import boundary_bucket

REQUIRED_COVERAGE = {
    "chunk_middle",
    "boundary_before",
    "cross_boundary",
    "boundary_after",
    "tail_flush",
    "multiword_phrase",
    "long_multi_chunk",
    "multiple_hotwords",
    "negative",
}


def build_streaming_boundary_manifest(
    source_spec_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_size_sec: float = 2.0,
    minimum_hotword_start_sec: float = 4.0,
    require_complete_coverage: bool = True,
) -> dict[str, object]:
    if chunk_size_sec <= 0 or minimum_hotword_start_sec < 0:
        raise ValueError("chunk size must be positive and minimum start must not be negative")
    source = Path(source_spec_path).expanduser()
    destination = Path(output_dir).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"boundary source spec does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("boundary output directory is not empty")
    rows = _read_jsonl(source)
    variants: list[dict[str, object]] = []
    for line_number, row in enumerate(rows, start=1):
        variants.extend(
            _build_variants(
                row,
                line_number=line_number,
                chunk_size_sec=chunk_size_sec,
                minimum_hotword_start_sec=minimum_hotword_start_sec,
            )
        )
    coverage: Counter[str] = Counter()
    for row in variants:
        tags = row.get("coverage_tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise RuntimeError("generated boundary variant has invalid coverage tags")
        coverage.update(tags)
    missing = sorted(REQUIRED_COVERAGE - coverage.keys())
    if missing and require_complete_coverage:
        raise ValueError(
            "boundary source spec cannot satisfy required coverage: " + ", ".join(missing)
        )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "boundary_cases.jsonl"
    _write_jsonl(manifest_path, variants)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "pass" if not missing else "warn",
        "source_spec_path": str(source),
        "source_spec_sha256": _sha256(source),
        "manifest_path": str(manifest_path),
        "source_cases": len(rows),
        "generated_variants": len(variants),
        "chunk_size_sec": chunk_size_sec,
        "minimum_hotword_start_sec": minimum_hotword_start_sec,
        "coverage_counts": dict(sorted(coverage.items())),
        "missing_required_coverage": missing,
        "audio_generation": "dynamic_leading_silence_only",
        "original_audio_overwritten": False,
    }
    _write_json(destination / "summary.json", summary)
    return summary


def _build_variants(
    raw: Mapping[str, Any],
    *,
    line_number: int,
    chunk_size_sec: float,
    minimum_hotword_start_sec: float,
) -> list[dict[str, object]]:
    case_id = _required_string(raw, "case_id", line_number)
    sample_id = _required_string(raw, "sample_id", line_number)
    audio_path = Path(_required_string(raw, "audio_path", line_number)).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError(f"boundary source audio does not exist: {audio_path}")
    duration = _audio_duration(audio_path)
    expected = _strings(raw, "expected_hotword_ids", line_number, allow_empty=True)
    active = _strings(raw, "active_hotword_ids", line_number)
    if not set(expected).issubset(active):
        raise ValueError(f"expected hotwords are not active at boundary row {line_number}")
    raw_timings = raw.get("hotword_timings", [])
    if not isinstance(raw_timings, list):
        raise ValueError(f"boundary row {line_number} has invalid hotword_timings")
    timings = [_parse_timing(item, line_number, duration) for item in raw_timings]
    if expected and {item["hotword_id"] for item in timings} != set(expected):
        raise ValueError(f"boundary row {line_number} timings do not match expected hotwords")
    if not expected and timings:
        raise ValueError(f"negative boundary row {line_number} must not contain timings")
    source_tags = set(_strings(raw, "coverage_tags", line_number, allow_empty=True))
    base = {
        "sample_id": sample_id,
        "audio_path": str(audio_path),
        "reference_text": _required_string(raw, "reference_text", line_number),
        "language": _required_string(raw, "language", line_number),
        "expected_hotword_ids": list(expected),
        "active_hotword_ids": list(active),
        "source_case_id": case_id,
        "source_audio_duration_sec": duration,
    }
    if not expected:
        padding = max(0.0, minimum_hotword_start_sec)
        return [
            {
                **base,
                "case_id": f"{case_id}__negative",
                "leading_silence_sec": padding,
                "audio_duration_sec": duration + padding,
                "hotword_timings": [],
                "boundary_bucket": "negative",
                "coverage_tags": ["negative"],
            }
        ]

    primary = timings[0]
    width = _float_value(primary, "end_sec") - _float_value(primary, "start_sec")
    if width > chunk_size_sec:
        source_tags.add("long_multi_chunk")
    elif "long_multi_chunk" in source_tags:
        raise ValueError(f"boundary row {line_number} labels a short timing as long_multi_chunk")
    boundary = math.ceil((minimum_hotword_start_sec + width) / chunk_size_sec) * chunk_size_sec
    target_starts = {
        "chunk_middle": boundary + 0.5,
        "boundary_before": boundary + chunk_size_sec - width,
        "cross_boundary": boundary + chunk_size_sec - width / 2.0,
        "boundary_after": boundary + chunk_size_sec + 0.01,
    }
    variants = []
    for label, target_start in target_starts.items():
        padding = max(0.0, target_start - _float_value(primary, "start_sec"))
        variants.append(
            _variant(
                base,
                case_id=case_id,
                label=label,
                padding=padding,
                duration=duration,
                timings=timings,
                source_tags=source_tags,
                chunk_size_sec=chunk_size_sec,
            )
        )
    # Tail coverage is valid only when the source hotword is already close enough to
    # the end; leading silence changes phase but cannot change its distance to the end.
    distance_from_start_to_end = duration - _float_value(primary, "start_sec")
    if distance_from_start_to_end < chunk_size_sec:
        desired_duration_phase = min(
            chunk_size_sec - 0.05,
            distance_from_start_to_end + 0.05,
        )
        padding = (desired_duration_phase - (duration % chunk_size_sec)) % chunk_size_sec
        while _float_value(primary, "start_sec") + padding < minimum_hotword_start_sec:
            padding += chunk_size_sec
        tail = _variant(
            base,
            case_id=case_id,
            label="tail_flush",
            padding=padding,
            duration=duration,
            timings=timings,
            source_tags=source_tags,
            chunk_size_sec=chunk_size_sec,
        )
        if tail["boundary_bucket"] != "tail_flush":
            raise RuntimeError("tail phase construction did not produce a real tail bucket")
        variants.append(tail)
    return variants


def _variant(
    base: Mapping[str, object],
    *,
    case_id: str,
    label: str,
    padding: float,
    duration: float,
    timings: Sequence[Mapping[str, object]],
    source_tags: set[str],
    chunk_size_sec: float,
) -> dict[str, object]:
    shifted = [
        {
            **item,
            "start_sec": _float_value(item, "start_sec") + padding,
            "end_sec": _float_value(item, "end_sec") + padding,
        }
        for item in timings
    ]
    actual_duration = duration + padding
    primary = shifted[0]
    computed = boundary_bucket(
        hotword_start_sec=_float_value(primary, "start_sec"),
        hotword_end_sec=_float_value(primary, "end_sec"),
        audio_duration_sec=actual_duration,
        chunk_size_sec=chunk_size_sec,
    )
    bucket = computed
    tags = {bucket} | source_tags
    if len(timings) > 1:
        tags.add("multiple_hotwords")
    return {
        **base,
        "case_id": f"{case_id}__{label}",
        "leading_silence_sec": round(padding, 6),
        "audio_duration_sec": round(actual_duration, 6),
        "hotword_timings": shifted,
        "boundary_bucket": bucket,
        "coverage_tags": sorted(tags),
    }


def _parse_timing(raw: Any, line_number: int, duration: float) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"boundary row {line_number} contains an invalid timing")
    source = _required_string(raw, "timing_source", line_number)
    if source not in {"forced_alignment", "manual_confirmed"}:
        raise ValueError("timing_source must be forced_alignment or manual_confirmed")
    start = float(raw["start_sec"])
    end = float(raw["end_sec"])
    if not 0 <= start < end <= duration + 0.05:
        raise ValueError(f"boundary row {line_number} timing is outside the audio")
    return {
        "hotword_id": _required_string(raw, "hotword_id", line_number),
        "start_sec": start,
        "end_sec": end,
        "timing_source": source,
    }


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("soundfile is required to inspect boundary audio") from error
    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"boundary audio is empty or invalid: {path}")
    return float(info.frames / info.samplerate)


def _float_value(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"generated timing has invalid {key}")
    return float(value)


def _required_string(raw: Mapping[str, Any], key: str, line_number: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"boundary row {line_number} has invalid {key}")
    return value.strip()


def _strings(
    raw: Mapping[str, Any],
    key: str,
    line_number: int,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"boundary row {line_number} has invalid {key}")
    if not value and not allow_empty:
        raise ValueError(f"boundary row {line_number} has empty {key}")
    return tuple(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"boundary source row {line_number} is not an object")
            rows.append(raw)
    if not rows:
        raise ValueError("boundary source spec is empty")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
