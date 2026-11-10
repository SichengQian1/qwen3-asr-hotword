from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from scripts import run_en_ar_training as run


def wav(path: Path, *, heldout: bool, seed: int) -> str:
    for n in range(seed, seed + 1000):
        with wave.open(str(path), "wb") as output:
            output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            output.writeframes(n.to_bytes(2, "little") * 16000)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (int(digest[:12], 16) % 100 < 3) == heldout:
            return digest
    raise AssertionError("test waveform could not be assigned")


def test_freeze_gates_audio_and_reuses_english_g2p(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run,
        "VALIDATION_HOURS",
        {"en_egy": 0.0002, "en_us_donor": 0, "ar_egy": 0.0002, "ar_msa": 0.0002},
    )
    source_rows, arabic = [], []
    for number, (group, heldout) in enumerate(
        [
            ("en_egy", True),
            ("en_egy", False),
            ("en_us_donor", False),
            ("ar_egy", True),
            ("ar_egy", False),
            ("ar_msa", True),
        ],
        1,
    ):
        path = tmp_path / f"audio-{number}.wav"
        wav(path, heldout=heldout, seed=number * 1000)
        native_text = "Hi." if group.startswith("en") else "مش"
        item = {
            "group": group,
            "row": number,
            "source": f"{group}.json",
            "text": native_text,
            "audio_path": str(path),
            "duration_seconds": 1.0,
            "issues": [],
        }
        source_rows.append(item)
        if group.startswith("ar"):
            arabic.append({**item, "romanized": "mish", "ids": [97, 91, 72]})
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
    ar = tmp_path / "arabic.jsonl"
    ar.write_text("".join(json.dumps(row) + "\n" for row in arabic), encoding="utf-8")
    en = tmp_path / "english.json"
    en.write_text(
        json.dumps(
            {
                "status": "english_g2p_completed",
                "vocab_sha256": run.VOCAB_SHA,
                "donor_sha256": run.DONOR_SHA,
                "inventory_sha256": sha(inventory),
                "words": {
                    "hi": {
                        "status": "ok",
                        "candidates": [{"usable": True, "oov_units": [], "token_ids": [20, 60]}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    inputs = dict.fromkeys(run.AUDITS, inventory)
    output = tmp_path / "release"
    output.mkdir()
    report = run.freeze(output, inputs, en, ar, 0.0002)
    assert report["status"] == "labels_complete"
    training = list(iter_jsonl(output / "train.jsonl"))
    validation = list(iter_jsonl(output / "validation.jsonl"))
    assert {row["language"] for row in training} == {"en", "ar"}
    assert {row["source_group"] for row in validation} == {"en_egy", "ar_egy", "ar_msa"}
    assert {row["audio_sha256"] for row in training}.isdisjoint(
        row["audio_sha256"] for row in validation
    )
    assert {tuple(row["phoneme_token_ids"]) for row in training} == {(20, 60), (97, 91, 72)}

    ar.write_text("", encoding="utf-8")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    assert run.freeze(blocked, inputs, en, ar, 0.0002)["status"] == "insufficient_labeled_hours"


def test_arabic_input_requires_ascii_and_audit_identity(tmp_path: Path) -> None:
    path = tmp_path / "arabic.jsonl"
    row = {"group": "ar_egy", "row": 12941, "source": "egy.json", "text": "مش"}
    path.write_text(json.dumps({**row, "romanized": "mish"}) + "\n", encoding="utf-8")
    assert list(run.romanizations(path))[0]["text"] == "مش"
    path.write_text(json.dumps({**row, "romanized": "مش"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid Arabic"):
        list(run.romanizations(path))
