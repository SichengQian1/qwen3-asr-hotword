from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from qwen_hotword.training.english_speaker_inventory import (
    audit_swift_english_speakers,
)


def test_audit_swift_english_speakers_builds_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    audio_rows = [
        "/audio/shards_1/US_101_F_8297_1051478.wav",
        "/audio/shards_2/US_101_F_8297_1051514.wav",
        "/audio/shards_2/US_202_M_1000_77.wav",
    ]
    _write_source(source, audio_rows)
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    _write_jsonl(
        manifest / "train_ready.jsonl",
        [_manifest_row(audio_rows[0], 2.0), _manifest_row(audio_rows[1], 3.0)],
    )
    _write_jsonl(
        manifest / "needs_review.jsonl",
        [_manifest_row(audio_rows[2], 4.0)],
    )
    (manifest / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "source_records": 3,
                "ready_records": 2,
                "review_records": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "output"
    summary = audit_swift_english_speakers(source, manifest, output)

    assert summary["status"] == "pass"
    assert summary["source_records"] == 3
    assert summary["speaker_count"] == 2
    assert summary["prefix_component_counts"] == {"4": 3}
    assert summary["first_component_counts"] == {"US": 3}
    assert summary["third_component_counts"] == {"F": 2, "M": 1}
    assert summary["speakers_across_multiple_shards"] == 1
    assert summary["parse_failure_records"] == 0
    assert summary["duplicate_speaker_utterance_keys"] == 0
    inventory = _read_tsv(output / "speaker_inventory.tsv")
    assert [row["speaker_id"] for row in inventory] == [
        "US_101_F_8297",
        "US_101_F_8297",
        "US_202_M_1000",
    ]
    assert [row["manifest_status"] for row in inventory] == [
        "ready",
        "ready",
        "needs_review",
    ]
    assert (output / "sha256.txt").is_file()


def test_audit_swift_english_speakers_warns_on_unparsed_basename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tsv"
    audio = "/audio/noseparator.wav"
    _write_source(source, [audio])
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    _write_jsonl(manifest / "train_ready.jsonl", [_manifest_row(audio, 1.0)])
    _write_jsonl(manifest / "needs_review.jsonl", [])
    (manifest / "summary.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "source_records": 1,
                "ready_records": 1,
                "review_records": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = audit_swift_english_speakers(source, manifest, tmp_path / "output")

    assert summary["status"] == "warn"
    assert summary["parse_failure_records"] == 1
    assert summary["speaker_count"] == 0


def _write_source(path: Path, audios: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio", "text"), delimiter="\t")
        writer.writeheader()
        for index, audio in enumerate(audios):
            writer.writerow({"audio": audio, "text": f"text {index}"})


def _manifest_row(audio: str, duration: float) -> dict[str, Any]:
    return {"audio_path": audio, "duration_seconds": duration}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
