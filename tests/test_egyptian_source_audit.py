from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import write_json
from qwen_hotword.training import egyptian_source_audit as m

ROOT = Path(__file__).resolve().parents[1]


def audio(path: Path, seconds: int = 1) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * 8000 * seconds)
    return path


def record(path: Path, response: str = "شكرا لكم", assistant: str | None = None) -> dict:
    return {
        "audios": [str(path)],
        "language": "Arabic",
        "response": response,
        "messages": [
            {
                "role": "assistant",
                "content": "language Arabic<asr_text>"
                + (assistant if assistant is not None else response),
            }
        ],
        "lengths": 999999,
    }


@pytest.mark.parametrize(
    ("response", "assistant", "expected"),
    [
        (
            "المهمه في السياسه التركيه",
            "المهمة في السياسة التركية.",
            "arabic_spelling_normalization_equivalent",
        ),
        ("شكرا لكم", "شكرا لكم!", "punctuation_or_spacing_only"),
        ("شكرا لكم", "شكرا لكم", "exact"),
        ("شكرا لكم", "مرحبا بك", "content_difference"),
    ],
)
def test_two_texts_are_preserved_not_silently_chosen(tmp_path, response, assistant, expected):
    row = m.inspect(record(audio(tmp_path / "a.wav"), response, assistant), {})
    assert row["text_relation"] == expected
    assert row["response_text"] == response and row["assistant_text"] == assistant
    assert row["text_for_labeling"] is None
    assert row["duration_seconds"] == 1


def test_tagged_response_wrong_language_and_missing_audio(tmp_path):
    row = record(tmp_path / "missing.wav")
    row["response"] = "language Arabic<asr_text>شكرا لكم"
    assert m.inspect(row, {})["response_format"] == "tagged"
    row["response"] = "language English<asr_text>hello"
    result = m.inspect(row, {})
    assert "response_wrong_language_tag" in result["issues"]
    assert "audio_missing" in result["issues"]
    assert result["duration_seconds"] is None


def test_only_explicit_audio_fallback_is_used(tmp_path):
    actual = audio(tmp_path / "a.wav")
    result = m.inspect(record(Path("/unmounted/a.wav")), {"/unmounted": str(tmp_path)})
    assert result["audio_path"] == str(actual.resolve())
    assert result["path_source"] == "verified_fallback"


def fixture(tmp_path):
    base = tmp_path / "data"
    groups = {}
    for name, priority, role in (
        ("open_source", 1, "train_pool"),
        ("gaoshan_train", 2, "train_pool"),
        ("coarse_train", 2, "train_pool"),
        ("gaoshan_test", 2, "original_test"),
    ):
        (base / name).mkdir(parents=True)
        groups[name] = {
            "directory": name,
            "priority": priority,
            "role": role,
            "expected": ["data.json" if name != "open_source" else "data.jsonl"],
        }
    shared = audio(tmp_path / "shared.wav", 2)
    heldout = audio(tmp_path / "heldout.wav", 3)
    extra = audio(tmp_path / "extra.wav", 4)
    (base / "open_source/data.jsonl").write_text(
        json.dumps(record(shared)) + "\n" + json.dumps(record(heldout)) + "\n"
    )
    write_json(base / "gaoshan_train/data.json", [record(shared)])
    write_json(base / "coarse_train/data.json", [record(extra)])
    write_json(base / "gaoshan_test/data.json", [record(heldout)])
    config = tmp_path / "config.json"
    write_json(config, {"root": str(base), "groups": groups, "audio_fallbacks": {}})
    return config


def test_full_headers_priority_dedup_test_protection_resume_and_no_model(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    output = tmp_path / "audit"
    report = m.audit(ROOT, config, output, 2, False)
    assert report["status"] == "checked"
    assert report["totals"]["rows"] == 5
    assert report["totals"]["readable_hours_including_duplicates_and_original_test"] == 14 / 3600
    assert report["totals"]["path_unique_hours_including_original_test"] == 9 / 3600
    assert report["totals"]["path_unique_train_pool_hours_excluding_original_test"] == 6 / 3600
    assert report["totals"]["train_original_test_overlapping_paths"] == 1
    assert report["path_unique_train_pool_attribution"]["open_source"]["hours"] == 2 / 3600
    assert not report["encoder_loaded"] and not report["msa_supplement_used"]
    assert not report["dialect_verified"] and not report["audio_content_hash_computed"]
    assert (output / "return_summary.zip").is_file()
    monkeypatch.setattr(m, "read_audio_metadata", lambda _: pytest.fail("resume reread audio"))
    assert m.audit(ROOT, config, output, 4, True) == report
    with pytest.raises(FileExistsError):
        m.audit(ROOT, config, output, 2, False)
    data = m.read(config)
    data["groups"]["open_source"]["priority"] = 9
    write_json(config, data)
    with pytest.raises(ValueError, match="identity changed"):
        m.audit(ROOT, config, output, 2, True)


def test_part_resume_avoids_audio_reads(tmp_path, monkeypatch):
    config = fixture(tmp_path)
    output = tmp_path / "audit"
    original = m.inspect
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "inspect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        m.audit(ROOT, config, output, 1, False)
    assert len(list((output / "parts").glob("*/*.json"))) == 1
    remaining = []

    def saved(*args, **kwargs):
        remaining.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "inspect", saved)
    assert m.audit(ROOT, config, output, 1, True)["status"] == "checked"
    assert len(remaining) == 3


def test_missing_expected_file_and_corrupt_json_are_not_complete(tmp_path):
    config = fixture(tmp_path)
    data = m.read(config)
    data["groups"]["open_source"]["expected"].append("absent.jsonl")
    write_json(config, data)
    (Path(data["root"]) / "coarse_train/data.json").write_text("[{broken]")
    report = m.audit(ROOT, config, tmp_path / "audit", 2, False)
    assert report["status"] == "review"
    assert any(s["status"] == "failed" for s in report["sources"])
    assert any(d["missing_expected_files"] for d in report["directories"])


def test_configuration_contains_all_fourteen_supplied_files_only_egyptian():
    config = m.read(ROOT / "configs/egyptian_sources_audit.workzone.json")
    assert sum(len(s["expected"]) for s in config["groups"].values()) == 14
    assert config["groups"]["open_source"]["priority"] == 1
    assert all("msa" not in s["directory"] for s in config["groups"].values())
