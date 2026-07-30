from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ASR_PREFIX = re.compile(r"^\s*language\s+[^<\s]+\s*<asr_text>\s*", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class SwiftJsonConversionSummary:
    input_path: str
    output_tsv_path: str
    source_records: int
    written_records: int
    skipped_records: int
    expected_language: str | None
    audio_prefix_rewrites: tuple[tuple[str, str], ...]
    rewritten_audio_paths: int
    language_counts: dict[str, int]
    audio_extension_counts: dict[str, int]
    issue_counts: dict[str, int]
    elapsed_seconds: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def convert_swift_json_to_tsv(
    input_path: str | Path,
    output_tsv_path: str | Path,
    *,
    expected_language: str | None = None,
    check_audio: bool = False,
    audio_prefix_rewrites: Sequence[tuple[str, str]] = (),
    progress_every_records: int = 10_000,
) -> SwiftJsonConversionSummary:
    source = Path(input_path).expanduser()
    destination = Path(output_tsv_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Swift JSON file does not exist: {source}")
    if progress_every_records < 0:
        raise ValueError("progress_every_records must be non-negative")
    rewrites = _validated_prefix_rewrites(audio_prefix_rewrites)

    destination.parent.mkdir(parents=True, exist_ok=True)
    issue_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    audio_extension_counts: Counter[str] = Counter()
    written_records = 0
    source_records = 0
    rewritten_audio_paths = 0
    started = time.monotonic()

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    print(f"loading Swift JSON: {source}", flush=True)
    records = _iter_swift_records(source)
    print(f"loaded Swift JSON records={len(records)}", flush=True)

    def print_progress() -> None:
        if not progress_every_records or source_records % progress_every_records:
            return
        elapsed = time.monotonic() - started
        rate = source_records / elapsed if elapsed > 0 else 0.0
        print(
            f"converted records={source_records}/{len(records)} "
            f"written={written_records} "
            f"skipped={source_records - written_records} "
            f"elapsed={elapsed:.1f}s records_per_second={rate:.1f}",
            flush=True,
        )

    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio", "text"], delimiter="\t")
        writer.writeheader()
        for record in records:
            source_records += 1
            language = _string_or_empty(record.get("language"))
            language_counts[language or "<missing>"] += 1
            if expected_language and language and language != expected_language:
                issue_counts["unexpected_language"] += 1

            audio = _extract_audio(record)
            audio, rewritten = _rewrite_audio_prefix(audio, rewrites)
            rewritten_audio_paths += int(rewritten)
            text = _extract_text(record)
            if not audio:
                issue_counts["missing_audio"] += 1
            if not text:
                issue_counts["missing_text"] += 1
            if audio and check_audio and not Path(audio).expanduser().is_file():
                issue_counts["missing_audio_file"] += 1
            if not audio or not text:
                print_progress()
                continue

            audio_extension_counts[Path(audio).suffix.lower() or "<none>"] += 1
            writer.writerow({"audio": audio, "text": text})
            written_records += 1
            print_progress()
    temporary.replace(destination)

    skipped_records = source_records - written_records
    conversion_clean = (
        written_records > 0
        and skipped_records == 0
        and issue_counts.get("unexpected_language", 0) == 0
        and (not check_audio or issue_counts.get("missing_audio_file", 0) == 0)
    )
    summary = SwiftJsonConversionSummary(
        input_path=str(source),
        output_tsv_path=str(destination),
        source_records=source_records,
        written_records=written_records,
        skipped_records=skipped_records,
        expected_language=expected_language,
        audio_prefix_rewrites=rewrites,
        rewritten_audio_paths=rewritten_audio_paths,
        language_counts=dict(sorted(language_counts.items())),
        audio_extension_counts=dict(sorted(audio_extension_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        elapsed_seconds=time.monotonic() - started,
        status="pass" if conversion_clean else "warn",
    )
    (destination.parent / "swift_json_conversion_summary.json").write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def clean_swift_asr_text(text: str) -> str:
    stripped = _ASR_PREFIX.sub("", text).strip()
    return _WHITESPACE.sub(" ", stripped)


def _iter_swift_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError("Swift JSON must be a top-level list of records")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Swift JSON record {index} is not an object")
        records.append(item)
    return records


def _extract_audio(record: dict[str, Any]) -> str:
    audios = record.get("audios")
    if not isinstance(audios, list) or not audios:
        return ""
    first_audio = audios[0]
    return _string_or_empty(first_audio)


def _extract_text(record: dict[str, Any]) -> str:
    response = _string_or_empty(record.get("response"))
    if response:
        return clean_swift_asr_text(response)

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = _string_or_empty(message.get("content"))
                if content:
                    return clean_swift_asr_text(content)
    return ""


def _validated_prefix_rewrites(
    rewrites: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    seen_sources: set[str] = set()
    for source, destination in rewrites:
        old_prefix = source.rstrip("/")
        new_prefix = destination.rstrip("/")
        if not old_prefix or old_prefix == "/" or not new_prefix:
            raise ValueError("audio prefix rewrites require non-root OLD and non-empty NEW")
        if old_prefix in seen_sources:
            raise ValueError(f"duplicate audio prefix rewrite source: {old_prefix}")
        seen_sources.add(old_prefix)
        normalized.append((old_prefix, new_prefix))
    return tuple(sorted(normalized, key=lambda item: len(item[0]), reverse=True))


def _rewrite_audio_prefix(
    audio: str,
    rewrites: Sequence[tuple[str, str]],
) -> tuple[str, bool]:
    for old_prefix, new_prefix in rewrites:
        if audio == old_prefix or audio.startswith(old_prefix + "/"):
            return new_prefix + audio[len(old_prefix) :], True
    return audio, False


def _string_or_empty(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
