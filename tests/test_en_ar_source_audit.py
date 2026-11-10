from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

from qwen_hotword.training.en_ar_source_audit import audit


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(16000 * 2))


def swift_row(
    audio: Path, response: str, assistant: str, language: str = "Arabic"
) -> dict[str, Any]:
    prefix = f"language {language}<asr_text>"
    return {
        "audios": [str(audio)],
        "response": prefix + response,
        "messages": [{"role": "assistant", "content": prefix + assistant}],
        "language": language,
    }


def test_inventory_formats_fallback_and_transcript_disagreement(tmp_path: Path) -> None:
    old_root, mounted = tmp_path / "unmounted", tmp_path / "mounted"
    actual, missing = mounted / "عينة.wav", old_root / "missing.wav"
    write_wav(actual)
    array, jsonl, donor = (tmp_path / filename for filename in ("egy.json", "msa.json", "us.jsonl"))
    array.write_text(
        json.dumps([swift_row(old_root / actual.name, "من فضلك", "من فضلك،")]), encoding="utf-8"
    )
    jsonl.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                swift_row(actual, "صباح الخير", "صباح الخير"),
                swift_row(missing, "قَلْب", "قُلْب"),
            ]
        ),
        encoding="utf-8",
    )
    donor.write_text(
        json.dumps(
            {
                "audio_path": str(actual),
                "text": "A sentence.",
                "split": "train",
                "language": "en-US",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = audit(
        [("ar_egy", array), ("ar_msa", jsonl), ("en_us_donor", donor)],
        {old_root: mounted},
        output,
        limit=None,
        workers=2,
    )
    assert result["status"] == "checked"
    assert result["sources"][0]["format"] == "json_array"
    assert result["sources"][1]["format"] == "jsonl"
    assert result["sources"][1]["readable_headers"] == 1
    assert [row["structurally_usable_rows"] for row in result["sources"]] == [1, 1, 1]
    inventory = [
        json.loads(line) for line in (output / "source_inventory.jsonl").read_text().splitlines()
    ]
    assert inventory[0]["path_source"] == "verified_fallback"
    assert inventory[0]["audio_path"] == str(actual)
    assert inventory[0]["duration_seconds"] == 1
    assert inventory[0]["issues"] == ["assistant_punctuation_or_spacing_only"]
    assert inventory[2]["issues"] == ["assistant_transcript_differs", "audio_path_unavailable"]
    assert len(json.loads((output / "samples.json").read_text())) == 3


def test_probe_limits_reads_and_full_audit_reports_malformed_source(tmp_path: Path) -> None:
    audio = tmp_path / "test.wav"
    write_wav(audio)
    records = tmp_path / "swift.json"
    valid = json.dumps(swift_row(audio, "Hi.", "Hi.", language="English"))
    records.write_text(valid + "\n" + valid + "\n{bad json}\n", encoding="utf-8")
    sources = [("en_egy", records)]
    probe = audit(sources, {}, tmp_path / "probe", limit=2, workers=1)
    assert probe["status"] == "checked"
    assert probe["sources"][0]["rows_processed"] == 2
    full = audit(sources, {}, tmp_path / "full", limit=None, workers=1)
    assert full["status"] == "review"
    assert "invalid JSONL" in full["sources"][0]["error"]
