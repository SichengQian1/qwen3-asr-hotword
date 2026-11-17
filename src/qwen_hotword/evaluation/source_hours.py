"""Raw source duration audit from audio headers; no decoding, labeling or split selection."""

from __future__ import annotations

import math
import tarfile
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_hotword.training.en_ar_source_audit import read_records
from qwen_hotword.training.stage_progress import stage_progress

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aiff", ".aif"}


def header_seconds(handle: Any) -> float:
    """Read container frame counts only; compressed formats may require libsndfile support."""
    try:
        with wave.open(handle, "rb") as wav:
            seconds = wav.getnframes() / wav.getframerate()
    except (wave.Error, EOFError):
        handle.seek(0)
        import soundfile as sf

        info = sf.info(handle)
        seconds = info.frames / info.samplerate
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("nonpositive_or_nonfinite_duration")
    return float(seconds)


def stamp(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_dev, value.st_ino


def empty() -> dict[str, Any]:
    return {
        "audio_references": 0,
        "readable_references": 0,
        "known_seconds": 0.0,
        "failed_references": 0,
        "failure_types": Counter(),
        "failure_examples": [],
    }


def failure(result: dict[str, Any], reference: str, error: Exception) -> None:
    result["failed_references"] += 1
    result["failure_types"][type(error).__name__] += 1
    if len(result["failure_examples"]) < 12:
        result["failure_examples"].append({"reference": reference, "error": str(error)})


def finish(result: dict[str, Any]) -> dict[str, Any]:
    result["known_hours"] = result["known_seconds"] / 3600
    result["status"] = (
        "complete"
        if (
            result["audio_references"] > 0
            and not result["failed_references"]
            and not result.get("invalid_rows")
            and not result.get("scan_error")
            and not result.get("loose_audio_files_not_counted")
        )
        else "incomplete"
    )
    return result


def english_one(item: tuple[int, dict[str, Any]], fallbacks: dict[str, str]) -> dict[str, Any]:
    number, row = item
    value: dict[str, Any] = {"row": number}
    try:
        audios = row.get("audios")
        if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
            raise ValueError("expected exactly one audio path in audios; row not counted")
        original = Path(audios[0])
        if not original.is_absolute():
            raise ValueError("expected absolute audio path")
        value["original"] = str(original)
        path = original
        if not path.is_file():
            for old, new in fallbacks.items():
                if path.is_relative_to(Path(old)):
                    mapped = Path(new) / path.relative_to(Path(old))
                    if mapped.is_file():
                        path = mapped
                        break
        value["path"] = str(path.resolve())
        before = stamp(path)
        with path.open("rb") as stream:
            value["seconds"] = header_seconds(stream)
        if stamp(path) != before:
            raise ValueError("audio changed while reading its header")
    except Exception as error:
        value["error"] = error
    return value


def english_hours(path: Path, fallbacks: dict[str, str], workers: int) -> dict[str, Any]:
    result = empty()
    result.update(
        source=str(path),
        format="swift_json_audio_headers",
        rows=0,
        invalid_rows=0,
        duplicate_path_references=0,
        unique_readable_path_seconds=0.0,
    )
    seen: set[str] = set()
    before = stamp(path)
    _, records = read_records(path)
    numbered = enumerate(records, 1)
    try:
        with (
            ThreadPoolExecutor(max_workers=workers) as pool,
            stage_progress(f"Read audio headers: {path.name}", every=2000) as tick,
        ):
            while batch := list(islice(numbered, 512)):
                for value in pool.map(lambda item: english_one(item, fallbacks), batch):
                    result["rows"] += 1
                    if "original" in value:
                        result["audio_references"] += 1
                    else:
                        result["invalid_rows"] += 1
                    if "error" in value:
                        failure(result, value.get("path", f"row {value['row']}"), value["error"])
                    else:
                        result["readable_references"] += 1
                        result["known_seconds"] += value["seconds"]
                        if value["path"] in seen:
                            result["duplicate_path_references"] += 1
                        else:
                            seen.add(value["path"])
                            result["unique_readable_path_seconds"] += value["seconds"]
                    tick(result["rows"])
    except Exception as error:
        result["scan_error"] = f"{type(error).__name__}: {error}"
    if stamp(path) != before:
        result["scan_error"] = "JSON source changed during audit"
    result["unique_readable_path_hours"] = result["unique_readable_path_seconds"] / 3600
    return finish(result)


def archive_hours(path: Path) -> dict[str, Any]:
    result = empty()
    result["source"] = str(path)
    try:
        before = stamp(path)
        # Uncompressed tar seeks past payload; compressed tar may require decompression in memory.
        # No member is ever written to disk, and no waveform is decoded into samples.
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if Path(member.name).suffix.lower() not in AUDIO_SUFFIXES:
                    continue
                result["audio_references"] += 1
                try:
                    if not member.isfile():
                        raise ValueError("nonregular_audio_member")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError("missing_member_stream")
                    with stream:
                        seconds = header_seconds(stream)
                    result["readable_references"] += 1
                    result["known_seconds"] += seconds
                except Exception as error:
                    failure(result, member.name, error)
        if stamp(path) != before:
            raise ValueError("archive changed during header audit")
    except Exception as error:
        result["scan_error"] = f"{type(error).__name__}: {error}"
    return finish(result)


def chinese_hours(root: Path, workers: int) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"Chinese root missing or not mounted: {root}")
    archives = []
    loose_audio = 0
    with stage_progress(f"Locate shards: {root.name}"):
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
                archives.append(path)
            elif path.suffix.lower() in AUDIO_SUFFIXES:
                loose_audio += 1
    # Do not count a tar again merely because a symlink exposes the same path.
    unique = sorted({p.resolve() for p in archives})
    result = empty()
    result.update(
        source=str(root),
        format="tar_audio_headers",
        archives=len(unique),
        archive_path_aliases=len(archives) - len(unique),
        loose_audio_files_not_counted=loose_audio,
        failed_archives=0,
    )
    if not unique:
        result["scan_error"] = "no tar/tar.gz/tgz shards found; inspect source structure"
        return finish(result)
    parts = []
    with (
        ThreadPoolExecutor(max_workers=min(workers, 4)) as pool,
        stage_progress(
            f"Read shard audio headers: {root.name}", total=len(unique), every=1
        ) as tick,
    ):
        # Bounded submissions, at most eight archives; one handle per archive worker.
        for start in range(0, len(unique), 8):
            for part in pool.map(archive_hours, unique[start : start + 8]):
                parts.append(part)
                for key in (
                    "audio_references",
                    "readable_references",
                    "known_seconds",
                    "failed_references",
                ):
                    result[key] += part[key]
                result["failure_types"].update(part["failure_types"])
                for example in part["failure_examples"]:
                    if len(result["failure_examples"]) < 12:
                        result["failure_examples"].append({"archive": part["source"], **example})
                if part["status"] != "complete":
                    result["failed_archives"] += 1
                tick(len(parts))
    if result["failed_archives"]:
        result["scan_error"] = "one or more incomplete archives; inspect archive_details.jsonl"
    result["archive_details"] = parts
    return finish(result)


def totals(results: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    return {
        "sources": keys,
        "known_hours": sum(results[k].get("known_hours", 0.0) for k in keys),
        "complete": all(results[k]["status"] == "complete" for k in keys),
        "basis": "sum of source audio-reference durations; no cross-source content deduplication",
    }
