"""Stream the three explicit Swift sources, preserving train rows and test protection."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from qwen_hotword.training.noah_source_inspection import iter_json_array
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.swift_json import clean_swift_asr_text

PREFIX = re.compile(r"^\s*language\s+English\s*<asr_text>", re.IGNORECASE)


def audio_path(row: dict[str, Any], country: dict[str, str]) -> Path:
    audios = row.get("audios")
    if not isinstance(audios, list) or len(audios) != 1 or not isinstance(audios[0], str):
        raise ValueError("exactly one string audio path required")
    original = Path(audios[0])
    old, new = Path(country["old_prefix"]), Path(country["directory"])
    if not original.is_absolute() or ".." in original.parts:
        raise ValueError("absolute non-traversing source audio path required")
    if original.is_relative_to(old):
        mapped = new / original.relative_to(old)
    elif original.is_relative_to(new):
        mapped = original
    else:
        raise ValueError(f"audio outside this country's explicit prefixes: {original}")
    mapped = mapped.resolve()
    if not mapped.is_relative_to((new / "audios").resolve()):
        raise ValueError("mapped file outside country's audios directory")
    return mapped


def transcript(row: dict[str, Any]) -> str:
    response = row.get("response")
    if (
        not isinstance(response, str)
        or not PREFIX.match(response)
        or row.get("language", "English") != "English"
    ):
        raise ValueError("English response with language English<asr_text> prefix required")
    text = clean_swift_asr_text(response)
    if not text or "<asr_text>" in text or "<audio>" in text:
        raise ValueError("empty or embedded-tag response")
    messages = row.get("messages", [])
    assistants = [
        m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    if assistants and (
        not isinstance(assistants[-1], str) or clean_swift_asr_text(assistants[-1]) != text
    ):
        raise ValueError("assistant message/response transcripts disagree")
    return text


def stage_sources(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Every train row goes to TSV; malformed contracts stop, never silently skip."""
    if output.exists():
        raise FileExistsError("source staging requires a new directory")
    names = [c["name"] for c in config["countries"]]
    if len(names) != len(set(names)) or len(names) != 3:
        raise ValueError("three distinct explicit country sources required")
    output.mkdir(parents=True)
    identities = {
        str(Path(c["directory"]) / f"{split}.json"): _sha(Path(c["directory"]) / f"{split}.json")
        for c in config["countries"]
        for split in ("train", "test")
    }
    counts, total = {}, 0
    with (
        (output / "source.tsv").open("x", encoding="utf-8", newline="") as handle,
        (output / "source_rows.jsonl").open("x", encoding="utf-8") as provenance,
        (output / "test_audio_paths.jsonl").open("x", encoding="utf-8") as protected,
    ):
        writer = csv.DictWriter(handle, fieldnames=["audio", "text"], delimiter="\t")
        writer.writeheader()
        for country in config["countries"]:
            # The reused manifest builder counts the header as TSV line 1.
            first, count = total + 2, 0
            for number, row in enumerate(
                iter_json_array(Path(country["directory"]) / "train.json"), 1
            ):
                try:
                    path, text = audio_path(row, country), transcript(row)
                except ValueError as error:
                    raise ValueError(
                        f"{country['name']}/train.json row {number}: {error}"
                    ) from error
                total += 1
                count += 1
                writer.writerow({"audio": str(path), "text": text})
                provenance.write(
                    json.dumps(
                        {
                            "tsv_row": total + 1,
                            "country": country["name"],
                            "source_row": number,
                            "original_audio": row["audios"][0],
                            "audio_path": str(path),
                            "original_response": row["response"],
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if total % 10000 == 0:
                    print(f"Converted train rows: {total}", flush=True)
            if not count:
                raise ValueError(f"empty country train source: {country['name']}")
            tests = 0
            # No test response or assistant text is consumed for labels, vocabulary or selection.
            for row in iter_json_array(Path(country["directory"]) / "test.json"):
                path = audio_path(row, country)
                protected.write(
                    json.dumps({"audio_path": str(path), "country": country["name"]}) + "\n"
                )
                tests += 1
            counts[country["name"]] = {
                "train_rows": count,
                "test_paths": tests,
                "first_tsv_row": first,
                "last_tsv_row": total + 1,
            }
            print(f"{country['name']}: train={count}, test paths={tests}", flush=True)
    for name, sha in identities.items():
        if _sha(Path(name)) != sha:
            raise ValueError("source JSON changed during conversion")
    report = {
        "status": "completed",
        "source_records": total,
        "countries": counts,
        "source_sha256": identities,
        "test_labels_used": False,
        "missing_audio_policy": "retained in full manifest needs_review",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report
