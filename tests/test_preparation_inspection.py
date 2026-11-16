from __future__ import annotations

from pathlib import Path

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training import preparation_inspection as inspection
from qwen_hotword.training.zh_en_common import finish, read, save_rows


def make_run(root: Path) -> Path:
    source = root / "run"
    source.mkdir()
    config = {
        "hours": {
            "train": {"zh": 480, "us": 336, "uk": 144},
            "validation": {"zh": 10, "us": 5, "uk": 5},
        },
        "minimum_hours": {
            "train": {"zh": 432, "us": 302.4, "uk": 129.6},
            "validation": {"zh": 5, "us": 2.5, "uk": 2.5},
        },
    }
    write_json(source / "config.json", config)
    write_json(source / "contract.json", {"config": config})
    write_json(
        source / "identity.json",
        {"git_commit": "fixture", "contract_sha256": sha(source / "contract.json")},
    )
    write_json(source / "g2p_report.json", {"fixture": True})
    totals = {}
    for i, group in enumerate(("zh", "us", "uk")):
        values = []
        for split in ("test", "train", "validation"):
            values.append(
                {
                    "id": f"{group}-{split}",
                    "group": group,
                    "split": split,
                    "source": group + ".tar",
                    "text": "sample",
                    "seconds": 3600,
                    "issues": [],
                    "phoneme_token_ids": [] if split == "test" else [2, 3],
                    "audio": {"path": "not-read.wav"},
                }
            )
        values.append(
            {
                "id": group + "-rejected",
                "group": group,
                "split": "train",
                "source": group + ".tar",
                "text": "bad",
                "seconds": 1800,
                "issues": ["g2p:word:empty", "g2p:other:empty"],
            }
        )
        folder = source / "labels" / f"part-{i:06d}"
        save_rows(folder / "rows.jsonl", values)
        finish(
            folder,
            {"g2p_report": sha(source / "g2p_report.json")},
            [folder / "rows.jsonl"],
            {"rows": len(values)},
        )
        totals[group] = len(values)
    write_json(source / "label_report.json", {"totals": totals})
    return source


def test_saved_duration_usable_and_excluded_are_not_confused(tmp_path):
    source = make_run(tmp_path)
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "inspection"
    result = inspection.save_inspection(source, output, headers_per_group=0)
    assert result["by_source_group"]["zh"]["known_original_hours"] == 3.5
    assert result["availability"]["zh/train"]["available_hours"] == 1
    assert result["availability"]["zh/train"]["minimum_met"] is False
    report = read(output / "report.json")
    assert report["by_split"]["zh/train"]["records"] == 2
    assert report["by_split"]["zh/train"]["excluded_records"] == 1
    assert report["by_split"]["zh/train"]["excluded_known_seconds"] == 1800
    assert report["by_split"]["zh/test"]["usable_records"] == 0
    assert (output / "return_bundle.zip").is_file()
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}


def test_all_parts_required_and_checksums_verified(tmp_path):
    source = make_run(tmp_path)
    path = source / "labels/part-000000/rows.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError):
        inspection.inspect_preparation(source, headers_per_group=0)


def test_missing_part_not_silently_audited_as_full_run(tmp_path):
    source = make_run(tmp_path)
    (source / "labels/part-000000/rows.jsonl").unlink()
    with pytest.raises(ValueError, match="not all original label rows"):
        inspection.inspect_preparation(source, headers_per_group=0)


def test_header_samples_never_read_test_audio(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    source = make_run(tmp_path)
    seen = []

    def load(row):
        assert row["split"] != "test"
        seen.append(row["id"])
        return b"fixture"

    monkeypatch.setattr(inspection, "audio_bytes", load)
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(info=lambda _: SimpleNamespace(frames=16000, samplerate=16000, channels=1)),
    )
    result = inspection.inspect_preparation(source, headers_per_group=2)
    assert len(seen) == 6
    assert all(r["status"] == "mismatch" for r in result["header_checks"])


def test_separate_output_required(tmp_path):
    source = make_run(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        inspection.save_inspection(source, source / "inspection", headers_per_group=0)
