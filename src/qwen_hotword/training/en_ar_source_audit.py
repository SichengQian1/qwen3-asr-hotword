"""Read-only corpus inventory before bilingual pronunciation labeling."""

from __future__ import annotations

import json
import re
import unicodedata
import wave
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path
from random import Random
from typing import Any

from qwen_hotword.training.experiment_a import read_audio_metadata
from qwen_hotword.training.noah_source_inspection import iter_json_array
from qwen_hotword.training.stage_progress import stage_progress

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}
ASR_TAG = re.compile(r"^\s*language\s+(English|Arabic)\s*<asr_text>\s*", re.IGNORECASE)


def read_records(path: Path) -> tuple[str, Iterator[dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as handle:
        first = ""
        while chunk := handle.read(4096):
            if stripped := chunk.lstrip():
                first = stripped[0]
                break
    if first == "[":
        return "json_array", iter_json_array(path)
    if first == "{":
        return "jsonl", iter_jsonl(path)
    raise ValueError(f"expected JSON array or JSONL: {path}")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def transcript_key(text: str) -> str:
    # Preserve Arabic marks, case and words while ignoring punctuation and spacing differences.
    return " ".join(
        "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in unicodedata.normalize("NFC", text)
        ).split()
    )


def inspect_record(
    item: tuple[int, dict[str, Any]], *, group: str, fallbacks: Mapping[Path, Path]
) -> dict[str, Any]:
    number, row = item
    language = group[:2]
    issues = []
    original_audio: Any = row.get("audio_path")
    text: Any = row.get("text")
    assistant_text = None
    if group == "en_us_donor":
        if row.get("split") != "train" or row.get("language") != "en-US":
            issues.append("unexpected_donor_split_or_language")
    else:
        audios = row.get("audios")
        original_audio = audios[0] if isinstance(audios, list) and len(audios) == 1 else None
        response = row.get("response")
        tag = ASR_TAG.match(response) if isinstance(response, str) else None
        if (
            not isinstance(response, str)
            or tag is None
            or tag.group(1).casefold() != LANGUAGE_NAMES[language].casefold()
        ):
            issues.append("missing_or_wrong_response_tag")
            text = None
        else:
            text = response[tag.end() :].strip()
        if str(row.get("language", "")).casefold() != LANGUAGE_NAMES[language].casefold():
            issues.append("unexpected_language_field")
        messages = row.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    content = message.get("content")
                    message_tag = ASR_TAG.match(content) if isinstance(content, str) else None
                    if (
                        isinstance(content, str)
                        and message_tag is not None
                        and message_tag.group(1).casefold() == LANGUAGE_NAMES[language].casefold()
                    ):
                        assistant_text = content[message_tag.end() :].strip()
                    break
        if assistant_text is None:
            issues.append("assistant_missing_or_wrong_tag")
        elif isinstance(text, str) and assistant_text != text:
            reason = (
                "assistant_punctuation_or_spacing_only"
                if transcript_key(text) == transcript_key(assistant_text)
                else "assistant_transcript_differs"
            )
            issues.append(reason)
    if not isinstance(text, str) or not transcript_key(text):
        issues.append("empty_transcript")
    audio = None
    seconds = None
    path_source = None
    if isinstance(original_audio, str) and Path(original_audio).is_absolute():
        source_path = Path(original_audio)
        audio = source_path
        path_source = "original"
        try:
            exists = audio.is_file()
            if not exists:
                for old, new in fallbacks.items():
                    if (
                        source_path.is_relative_to(old)
                        and (candidate := new / source_path.relative_to(old)).is_file()
                    ):
                        audio = candidate
                        exists = True
                        path_source = "verified_fallback"
                        break
            if exists:
                metadata = read_audio_metadata(audio)
                if metadata.frames <= 0 or metadata.sample_rate <= 0:
                    raise ValueError("empty or invalid audio header")
                seconds = metadata.duration_seconds
            else:
                issues.append("audio_path_unavailable")
        except (OSError, RuntimeError, ValueError, ZeroDivisionError, EOFError, wave.Error):
            issues.append("audio_metadata_unreadable")
    else:
        issues.append("expected_one_absolute_audio_path")
    return {
        "row": number,
        "audio_original": original_audio if isinstance(original_audio, str) else None,
        "audio_path": str(audio) if audio is not None else None,
        "path_source": path_source,
        "text": text if isinstance(text, str) else None,
        "assistant_text_if_different": assistant_text if assistant_text != text else None,
        "duration_seconds": seconds,
        "issues": issues,
        **{
            key: row[key]
            for key in ("id", "speaker_id", "session_id", "source_corpus")
            if key in row
        },
    }


def audit(
    sources: list[tuple[str, Path]],
    fallbacks: Mapping[Path, Path],
    output: Path,
    *,
    limit: int | None,
    workers: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {output}", flush=True)
    reports = []
    samples: list[dict[str, Any]] = []
    rng = Random(20261110)
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        (output / "source_inventory.jsonl").open("x", encoding="utf-8") as inventory,
    ):
        for index, (group, path) in enumerate(sources, 1):
            report: dict[str, Any] = {"group": group, "path": str(path)}
            issues: Counter[str] = Counter()
            examples_by_issue: dict[str, dict[str, Any]] = {}
            read_count = usable = processed = 0
            readable_seconds = usable_seconds = 0.0
            sample: list[dict[str, Any]] = []
            try:
                stat = path.stat()
                report["bytes"] = stat.st_size
                report["mtime_ns"] = stat.st_mtime_ns
                report["format"], records = read_records(path)
                selected = enumerate(islice(records, limit), 1)
                worker = partial(inspect_record, group=group, fallbacks=fallbacks)
                with stage_progress(
                    f"{index}/{len(sources)} {group} {path.name}", every=4096
                ) as step:
                    while chunk := list(islice(selected, 256)):
                        for row in executor.map(worker, chunk):
                            processed += 1
                            issues.update(row["issues"])
                            for issue in row["issues"]:
                                examples_by_issue.setdefault(
                                    issue, {"row": row["row"], "audio_path": row["audio_original"]}
                                )
                            seconds = row["duration_seconds"]
                            if seconds is not None:
                                read_count += 1
                                readable_seconds += seconds
                            if seconds is not None and not (
                                set(row["issues"]) - {"assistant_punctuation_or_spacing_only"}
                            ):
                                usable += 1
                                usable_seconds += seconds
                                slot = rng.randrange(usable) if len(sample) == 8 else len(sample)
                                if len(sample) < 8:
                                    sample.append(row)
                                elif slot < 8:
                                    sample[slot] = row
                            inventory.write(
                                json.dumps(
                                    {"source": str(path), "group": group, **row}, ensure_ascii=False
                                )
                                + "\n"
                            )
                            step(processed)
                        inventory.flush()
                current = path.stat()
                if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    raise ValueError("source manifest changed during audit")
                report["status"] = "checked" if usable else "no_usable_rows"
            except (OSError, ValueError) as error:
                report.update(status="failed", error=str(error))
            report.update(
                rows_processed=processed,
                readable_headers=read_count,
                structurally_usable_rows=usable,
                readable_hours=round(readable_seconds / 3600, 6),
                structurally_usable_hours=round(usable_seconds / 3600, 6),
                issue_counts=dict(issues),
                first_example_by_issue=examples_by_issue,
            )
            reports.append(report)
            samples.extend({"source": str(path), "group": group, **row} for row in sample)
    result = {
        "status": "checked"
        if all(report["status"] == "checked" for report in reports)
        else "review",
        "mode": "sample_probe" if limit is not None else "full_header_audit",
        "max_rows_per_source": limit,
        "output_dir": str(output),
        "audio_fallbacks": {str(old): str(new) for old, new in fallbacks.items()},
        "hours_basis": "audio headers only, before deduplication, holdout or G2P",
        "sources": reports,
    }
    for name, value in (("report.json", result), ("samples.json", samples)):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return result
