from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_hotword.training.pt_validation_metadata import audit_pt_validation_metadata


def _fixture(tmp_path: Path) -> dict[str, str]:
    root = tmp_path / "cv"
    root.mkdir()
    texts = {
        "exact": "Olá!", "normalized": "  BOM   DIA ", "different": "Bom dia",
        "missing": "Olá", "duplicate": "Olá", "empty": "Olá",
        "quote": '"Uma frase sem aspas finais', "long": '"' + "a" * 140_000,
    }
    manifest = tmp_path / "validation.jsonl"
    records = [
        {"id": name, "audio_path": str(root / "clips" / f"{name}.mp3"),
         "text": text, "source_corpus": "common_voice", "split": "validation"}
        for name, text in texts.items()
    ]
    # Same basename outside the CV root must not match CV metadata.
    records.append({
        "id": "other_root", "audio_path": str(tmp_path / "other" / "exact.mp3"),
        "text": "Olá!", "source_corpus": "common_voice", "split": "validation",
    })
    manifest.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    header = "client_id\tpath\tsentence\tup_votes\tdown_votes\taccents\tlocale\n"
    lines = []
    for name, text in texts.items():
        if name == "missing":
            continue
        original = {"normalized": "bom dia", "different": "Boa tarde", "empty": ""}.get(
            name, text
        )
        line = f"speaker\t{name}.mp3\t{original}\t2\t0\tBrasil\tpt\n"
        lines.append(line)
        if name == "duplicate":
            lines.append(line)
    (root / "validated.tsv").write_text("\ufeff" + header + "".join(lines), encoding="utf-8")
    fleurs = tmp_path / "fleurs_train.tsv"
    fleurs.write_text("1\tclip.wav\tOlá\n2\tclip2.wav\tBom dia\n", encoding="utf-8")
    return {
        "validation_manifest": str(manifest),
        "validation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "common_voice_root": str(root),
        "fleurs_train_tsv": str(fleurs),
    }


def _run(config: dict[str, str]) -> dict[str, object]:
    return audit_pt_validation_metadata(
        validation_manifest=Path(config["validation_manifest"]),
        expected_sha256=config["validation_sha256"],
        common_voice_root=Path(config["common_voice_root"]),
        fleurs_train_tsv=Path(config["fleurs_train_tsv"]),
    )


def test_literal_quotes_long_fields_and_conservative_join(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = _run(config)
    assert report["cv_candidates"] == 9
    assert report["cv_checks"] == {
        "exact_text_match": 3, "case_space_unicode_only": 1, "text_difference": 1,
        "not_in_validated": 2, "duplicate_metadata": 1, "empty_original_text": 1,
    }
    assert report["cv_known_speakers"] == 1
    assert report["cv_votes_top20"] == [("up=2,down=0", 6)]
    assert report["status"] == "completed"
    assert report["files_written"] is False
    assert len(json.dumps(report)) < 10_000
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("row", ["a\tb\n", "a\tb\tc\td\te\tf\tg\th\n", "\n"])
def test_bad_tsv_width_stops_instead_of_silently_skipping(tmp_path: Path, row: str) -> None:
    config = _fixture(tmp_path)
    path = Path(config["common_voice_root"]) / "validated.tsv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)
    with pytest.raises(ValueError, match="No rows skipped"):
        _run(config)


def test_identity_failure_before_opening_metadata(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["validation_sha256"] = "0" * 64
    config["common_voice_root"] = str(tmp_path / "not_present")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _run(config)


def test_non_validation_record_is_rejected(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = Path(config["validation_manifest"])
    path.write_text(path.read_text().replace('"validation"', '"train"'))
    config["validation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="expected validation record"):
        _run(config)


def test_duplicate_candidate_and_metadata_header_rejected(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = Path(config["validation_manifest"])
    text = path.read_text()
    path.write_text(text + text.splitlines()[0] + "\n")
    config["validation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate Common Voice candidate"):
        _run(config)
    path.write_text(text)
    config["validation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (Path(config["common_voice_root"]) / "validated.tsv").write_text("path\tpath\n")
    with pytest.raises(ValueError, match="missing required or duplicate columns"):
        _run(config)


def test_cli_runs_from_another_directory_and_returns_error_on_wrong_sha(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    script = Path(__file__).resolve().parents[1] / "scripts/audit_pt_validation_metadata.py"
    command = [sys.executable, "-B", str(script), "--config", str(config_path)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["cv_checks"]["exact_text_match"] == 3
    config["validation_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr
    assert result.stdout == ""
