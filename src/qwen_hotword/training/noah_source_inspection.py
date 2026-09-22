"""Inspect source identity/schema and bounded audio samples; never build training data."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qwen_hotword.training.experiment_a import read_audio_metadata


def iter_json_array(path: Path, *, chunk_size: int = 65536) -> Iterator[dict[str, Any]]:
    """Stream strict array-of-objects JSON with bounded record size, including one-line files."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as handle:
        buffer = ""
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            part = handle.read(chunk_size)
            eof = not part
            buffer += part

        def whitespace() -> None:
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()

        whitespace()
        if not buffer.startswith("["):
            raise ValueError("expected a top-level JSON array")
        buffer = buffer[1:]
        first = True
        while True:
            whitespace()
            if buffer.startswith("]"):
                buffer = buffer[1:]
                whitespace()
                if buffer or not eof:
                    raise ValueError("trailing content after JSON array")
                return
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("expected comma between JSON records")
                buffer = buffer[1:]
                whitespace()
                if buffer.startswith("]"):
                    raise ValueError("trailing comma in JSON array")
            while True:
                try:
                    record, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as error:
                    if eof or len(buffer) > 8 * 1024 * 1024:
                        raise ValueError(
                            "invalid/truncated JSON or record exceeds 8 MiB"
                        ) from error
                    fill()
            if not isinstance(record, dict):
                raise ValueError("each array element must be an object")
            buffer = buffer[end:]
            first = False
            yield record


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_audio_path(audio: str, prefixes: dict[str, str]) -> tuple[Path | None, str]:
    resolved = Path(audio)
    matches = []
    for old, new in prefixes.items():
        if not old.endswith("/") or not new.endswith("/"):
            raise ValueError("audio prefix rewrites must end with slash")
        if audio.startswith(old):
            matches.append(Path(new + audio[len(old) :]))
    if len(matches) > 1:
        raise ValueError("overlapping explicit audio prefix rewrites")
    if matches:
        resolved = matches[0]
    if not resolved.is_file():
        return None, "missing"
    return resolved, "explicit_prefix_rewrite" if matches else "original"


def inspect_noah_source(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["input"])
    actual = _hash(path)
    if actual != config["expected_sha256"]:
        raise ValueError("source JSON SHA256 differs from supplied identity")
    limit = int(config["samples_per_corpus"])
    if not 1 <= limit <= 100:
        raise ValueError("samples_per_corpus must be between 1 and 100")
    prefixes = config["audio_prefix_rewrites"]
    fields: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    corpora: Counter[str] = Counter()
    metadata: dict[str, Counter[str]] = {
        name: Counter()
        for name in ("speaker_id", "speaker", "country", "accent", "dialect", "split")
    }
    samples: defaultdict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    seen: dict[str, str] = {}
    count = 0
    for index, row in enumerate(iter_json_array(path), 1):
        count = index
        fields.update(row.keys())
        languages[str(row.get("language", "<missing>"))] += 1
        for name, counts in metadata.items():
            value = row.get(name)
            counts[str(value) if value is not None else "<missing>"] += 1
        audios = row.get("audios")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            issues["not_exactly_one_audio_path"] += 1
            continue
        audio = audios[0]
        if not audio or not Path(audio).is_absolute():
            issues["empty_or_relative_audio_path"] += 1
            continue
        parts = Path(audio).parts
        corpus = parts[parts.index("noah_esZ00") + 1] if "noah_esZ00" in parts[:-1] else "unknown"
        corpora[corpus] += 1
        if len(corpora) > 30:
            raise ValueError("more than 30 source folders; inspect grouping before audio probing")
        response = row.get("response")
        messages = row.get("messages")
        assistant = (
            [
                m.get("content")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if isinstance(messages, list)
            else []
        )
        if len(assistant) != 1 or assistant[0] != response:
            issues["assistant_response_not_identical"] += 1
        if row.get("language") != "Spanish":
            issues["language_not_Spanish"] += 1
        match = (
            re.fullmatch(r"language Spanish<asr_text>([\s\S]*)", response)
            if isinstance(response, str)
            else None
        )
        text = match.group(1).strip() if match else ""
        if not text:
            issues["missing_or_unexpected_transcript_format"] += 1
        fingerprint = hashlib.sha256(str(response).encode()).hexdigest()
        if audio in seen:
            issues["duplicate_audio_path"] += 1
            if seen[audio] != fingerprint:
                issues["duplicate_audio_conflicting_text"] += 1
            continue
        seen[audio] = fingerprint
        priority = int(hashlib.sha256(f"{config['seed']}\0{audio}".encode()).hexdigest(), 16)
        item = (-priority, index, audio, text[:240])
        heap = samples[corpus]
        heapq.heappush(heap, item)
        if len(heap) > limit:
            heapq.heappop(heap)
    if not count:
        raise ValueError("source array is empty")

    probe_reports: dict[str, Any] = {}
    for corpus, selected in sorted(samples.items()):
        statuses: Counter[str] = Counter()
        rates: Counter[str] = Counter()
        seconds: list[float] = []
        examples: list[dict[str, Any]] = []
        for _, index, audio, text in sorted(selected, reverse=True):
            resolved, status = resolve_audio_path(audio, prefixes)
            example: dict[str, Any] = {
                "source_row": index,
                "original_path": audio,
                "resolved_path": str(resolved) if resolved else None,
                "path_status": status,
                "text_preview": text,
            }
            if resolved is not None:
                try:
                    info = read_audio_metadata(resolved)
                    if not math.isfinite(info.duration_seconds) or info.duration_seconds <= 0:
                        raise ValueError("invalid audio duration")
                    seconds.append(info.duration_seconds)
                    rates[str(info.sample_rate)] += 1
                    example["duration_seconds"] = info.duration_seconds
                except Exception as error:
                    status = "unreadable_audio"
                    example["error"] = f"{type(error).__name__}: {error}"[:300]
            statuses[status] += 1
            if len(examples) < 3 or (
                status in {"missing", "unreadable_audio"} and len(examples) < 6
            ):
                examples.append(example)
        probe_reports[corpus] = {
            "sampled_unique_paths": len(selected),
            "status_counts": dict(statuses),
            "sample_rates": dict(rates),
            "readable_samples": len(seconds),
            "sample_audio_hours": sum(seconds) / 3600,
            "duration_seconds_min_max": [min(seconds), max(seconds)] if seconds else None,
            "examples": examples,
        }
    audio_ok = all(
        r["readable_samples"] == r["sampled_unique_paths"] for r in probe_reports.values()
    )
    return {
        "status": "inspection_completed" if audio_ok and not issues else "inspection_has_flags",
        "input": {
            "path": str(path),
            "sha256": actual,
            "sha256_verified": True,
            "size_bytes": path.stat().st_size,
        },
        "record_count": count,
        "unique_exact_audio_paths": len(seen),
        "field_counts": dict(fields),
        "language_counts": dict(languages),
        "source_folder_record_counts": dict(corpora),
        "issue_counts": dict(issues),
        "metadata_fields": {
            name: {"distinct_values_including_missing": len(c), "top5": c.most_common(5)}
            for name, c in metadata.items()
        },
        "audio_probe": probe_reports,
        "sampling": {
            "seed": config["seed"],
            "per_corpus": limit,
            "policy": "smallest_SHA256_seed_audio_path_without_replacement",
        },
        "provider_provenance": config.get("provenance", {}),
        "latin_american_scope_independently_verified": False,
        "full_corpus_measured_hours": None,
        "training_ready": False,
        "model_loaded": False,
        "files_written": False,
        "limitations": [
            "Audio checks cover only the fixed sample; no total-hours extrapolation.",
            "Exact path duplicates only; content, renamed files and holdout overlap not checked.",
            "Folder names, transcript topics and Spanish language tags do not prove dialect.",
            "Directory and filename components have not been certified as speaker identities.",
            "Human review is user-supplied provenance, not independent phoneme label validation.",
        ],
    }
