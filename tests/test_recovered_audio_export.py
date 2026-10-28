from __future__ import annotations

import json
import struct
import wave
import zipfile
from pathlib import Path

import pytest

from qwen_hotword.evaluation.recovered_audio_export import (
    candidate,
    export_saved_cases,
    sha,
    table_digests,
)
from qwen_hotword.inference.hotword_prompt import normalize_match_words


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def write_rows(path: Path, objects: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objects), encoding="utf-8"
    )


def fixture(tmp_path: Path, language: str = "pt") -> tuple[Path, Path, dict, dict]:
    table = tmp_path / "table.json"
    write_json(table, {"entries": [{"surface": "São Paulo"}, {"surface": "DHL"}]})
    source = tmp_path / "outputs" / language
    source.mkdir(parents=True)
    audio = tmp_path / f"original-{language}.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<h", 7 if language == "pt" else 9) * 1600)
    common = {
        "case_id": language + "-case",
        "sample_id": language + "-sample",
        "audio_path": str(audio),
        "audio_sha256": sha(audio),
        "reference_text": "Eu vou para São Paulo com DHL",
        "language": language,
    }
    before = {
        **common,
        "prediction": "eu vou para são pablo com de agá ele",
        "injected_hotwords": [],
    }
    after = {
        **common,
        "prediction": "Eu vou para São Paulo com DHL",
        "injected_hotwords": ["São Paulo", "DHL"],
    }
    write_json(
        source / "run_config.json",
        {"inputs": {"hotwords": {"path": str(table), "sha256": sha(table)}}, "language": language},
    )
    write_rows(source / "baseline_predictions.jsonl", [before])
    write_rows(source / "retrieved_predictions.jsonl", [after])
    return source, table, before, after


def export(tmp_path: Path, table: Path, *, count: int = 50) -> dict:
    return export_saved_cases(
        tmp_path,
        [tmp_path / "outputs"],
        table,
        tmp_path / "export",
        count=count,
        expected_table_sha256=sha(table),
    )


def test_exports_original_wav_and_complete_evidence_without_changing_sources(
    tmp_path: Path,
) -> None:
    source, table, before, _ = fixture(tmp_path)
    source_files = [p for p in source.rglob("*") if p.is_file()] + [Path(before["audio_path"])]
    original = {str(p): sha(p) for p in source_files}
    result = export(tmp_path, table, count=1)
    assert result["status"] == "completed" and result["exported_cases"] == 1
    assert result["export_pairs_checked"] == 1
    assert result["runs_with_verified_latest_table"] == 1
    output = tmp_path / "export"
    cases = [json.loads(s) for s in (output / "cases.jsonl").read_text().splitlines()]
    assert cases[0]["recovered_hotwords"] == ["DHL", "São Paulo"]
    assert (output / cases[0]["exported_audio"]).read_bytes() == Path(
        before["audio_path"]
    ).read_bytes()
    doc = (output / "案例说明.md").read_text()
    assert "不加热词" in doc and "加热词后" in doc and "São Paulo" in doc
    assert {str(p): sha(p) for p in source_files} == original
    with zipfile.ZipFile(result["return_bundle"]) as archive:
        for line in archive.read("sha256.txt").decode().splitlines():
            digest, name = line.split(maxsplit=1)
            import hashlib

            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export(tmp_path, table, count=1)


def test_exact_phrase_reference_and_actual_injection_are_all_required(tmp_path: Path) -> None:
    _, table, base, hot = fixture(tmp_path)
    words = {normalize_match_words("São Paulo"): "São Paulo", normalize_match_words("DHL"): "DHL"}
    config = {"language": "pt"}
    same = {**base, "prediction": "SÃO PAULO, com DHL!"}
    assert candidate(same, hot, config, words, {}, tmp_path) is None
    substring = {**hot, "prediction": "são pablo DHLexpress"}
    assert candidate(base, substring, config, words, {}, tmp_path) is None
    not_injected = {**hot, "injected_hotwords": []}
    assert candidate(base, not_injected, config, words, {}, tmp_path) is None
    reference_absent = "viajar para Recife"
    assert (
        candidate(
            {**base, "reference_text": reference_absent},
            {**hot, "reference_text": reference_absent},
            config,
            words,
            {},
            tmp_path,
        )
        is None
    )
    with pytest.raises(ValueError, match="baseline already"):
        candidate({**base, "injected_hotwords": ["DHL"]}, hot, config, words, {}, tmp_path)
    with pytest.raises(ValueError, match="oracle"):
        candidate(base, {**hot, "mode": "oracle"}, config, words, {}, tmp_path)
    assert sha(table)


def test_old_table_results_are_not_relabelled_as_latest(tmp_path: Path) -> None:
    source, table, _, _ = fixture(tmp_path)
    write_json(source / "run_config.json", {"inputs": {"hotwords": {"sha256": "0" * 64}}})
    result = export(tmp_path, table)
    assert result["status"] == "insufficient_verified_cases"
    assert result["exported_cases"] == 0 and result["shortfall"] == 50
    assert result["export_pairs_checked"] == 0 and result["saved_asr_pairs_seen"] == 1
    report = json.loads((tmp_path / "export/report.json").read_text())
    item = report["run_inventory"][0]
    assert item["reason"] == "different_recorded_table_identity"
    assert item["recorded_table_digests"][0]["sha256"] == "0" * 64
    assert item["metadata_files"][0]["path"].endswith("run_config.json")
    assert item["transcript_inspection"]["paired_rows_seen"] == 1
    assert item["transcript_inspection"]["text_gain_pairs_before_identity_and_audio_checks"] == 1
    assert item["transcript_inspection"]["diagnostic_only"] is True
    assert item["transcript_inspection"]["examples"][0]["text_recovered_words"] == [
        "DHL",
        "São Paulo",
    ]


@pytest.mark.parametrize("change", ["audio_path", "audio_hash", "language", "header"])
def test_different_or_changed_audio_and_language_are_rejected(tmp_path: Path, change: str) -> None:
    source, table, base, hot = fixture(tmp_path)
    if change == "audio_path":
        hot["audio_path"] = str(tmp_path / "other.wav")
    elif change == "audio_hash":
        hot["audio_sha256"] = "0" * 64
    elif change == "language":
        hot["language"] = "en"
    else:
        Path(base["audio_path"]).write_bytes(b"this is not a wav")
    write_rows(source / "retrieved_predictions.jsonl", [hot])
    result = export(tmp_path, table)
    assert result["exported_cases"] == 0
    assert json.loads((tmp_path / "export/report.json").read_text())["rejected_pair_reasons"]


def test_deduplicates_audio_bytes_and_reports_shortfall(tmp_path: Path) -> None:
    source, table, base, hot = fixture(tmp_path)
    write_rows(source / "baseline_predictions.jsonl", [base, {**base, "case_id": "second"}])
    write_rows(source / "retrieved_predictions.jsonl", [hot, {**hot, "case_id": "second"}])
    result = export(tmp_path, table)
    assert result["exported_cases"] == 1 and result["shortfall"] == 49


def test_streaming_cd_uses_authenticated_audio_manifest_and_excludes_oracle(tmp_path: Path) -> None:
    source, table, base, hot = fixture(tmp_path)
    manifest = tmp_path / "validation.jsonl"
    write_rows(
        manifest,
        [
            {
                "id": base["sample_id"],
                "audio_path": base["audio_path"],
                "audio_sha256": base["audio_sha256"],
            }
        ],
    )
    write_json(
        source / "run_config.json",
        {
            "language": "pt",
            "inputs": {
                "hotwords": {"sha256": sha(table)},
                "validation": {"path": str(manifest), "sha256": sha(manifest)},
            },
        },
    )
    for row in (base, hot):
        row.pop("audio_path")
        row.pop("audio_sha256")
    (source / "baseline_predictions.jsonl").unlink()
    (source / "retrieved_predictions.jsonl").unlink()
    write_rows(
        source / "sample_results.jsonl",
        [
            {**base, "experiment_group": "C"},
            {**hot, "experiment_group": "D"},
            {**hot, "experiment_group": "E"},
        ],
    )
    assert export(tmp_path, table, count=1)["exported_cases"] == 1
    config = json.loads((source / "run_config.json").read_text())
    config["inputs"]["validation"].pop("sha256")
    write_json(source / "run_config.json", config)
    result = export_saved_cases(
        tmp_path,
        [tmp_path / "outputs"],
        table,
        tmp_path / "unauthenticated_mapping",
        count=1,
        expected_table_sha256=sha(table),
    )
    assert result["exported_cases"] == 0


def test_corrupt_saved_result_checksum_is_not_ignored(tmp_path: Path) -> None:
    source, table, _, _ = fixture(tmp_path)
    (source / "sha256.txt").write_text("0" * 64 + "  retrieved_predictions.jsonl\n")
    result = export(tmp_path, table)
    assert result["exported_cases"] == 0
    assert (
        "checksum mismatch"
        in json.loads((tmp_path / "export/report.json").read_text())["run_inventory"][0]["reason"]
    )


def test_no_saved_asr_results_produces_an_honest_empty_inventory(tmp_path: Path) -> None:
    table = tmp_path / "table.json"
    write_json(table, {"entries": [{"surface": "DHL"}]})
    result = export(tmp_path, table)
    assert result["candidate_run_directories"] == 0 and result["exported_cases"] == 0
    assert "0 条不同原始音频" in (tmp_path / "export/案例说明.md").read_text()
    assert table_digests({"unrelated_sha256": "a" * 64}) == []
    with pytest.raises(ValueError, match="pinned latest"):
        export_saved_cases(tmp_path, [], table, tmp_path / "wrong", expected_table_sha256="f" * 64)


def test_packages_exactly_fifty_distinct_original_wavs_across_three_languages(
    tmp_path: Path,
) -> None:
    source, table, base, hot = fixture(tmp_path)
    left, right = [], []
    for i in range(54):
        audio = tmp_path / f"case-{i}.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack("<h", i) * 1600)
        common = {
            "case_id": f"case-{i:03d}",
            "sample_id": f"sample-{i:03d}",
            "language": ("en", "es", "pt")[i % 3],
            "audio_path": str(audio),
            "audio_sha256": sha(audio),
        }
        left.append({**base, **common})
        right.append({**hot, **common})
    write_rows(source / "baseline_predictions.jsonl", left)
    write_rows(source / "retrieved_predictions.jsonl", right)
    result = export(tmp_path, table)
    assert result["status"] == "completed" and result["exported_cases"] == 50
    assert result["by_language"] == {"en": 17, "es": 17, "pt": 16}
    with zipfile.ZipFile(result["return_bundle"]) as archive:
        assert len([name for name in archive.namelist() if name.endswith(".wav")]) == 50


@pytest.mark.parametrize("state", ["missing", "conflicting"])
def test_table_gate_failure_preserves_exact_metadata_and_transcript_evidence(
    tmp_path: Path, state: str
) -> None:
    source, table, _, _ = fixture(tmp_path)
    config = {"language": "pt", "inputs": {}}
    if state == "conflicting":
        config["inputs"] = {
            "hotwords": {"path": str(table), "sha256": sha(table)},
            "hotword_families": {"sha256": "f" * 64},
        }
    write_json(source / "run_config.json", config)
    result = export(tmp_path, table)
    assert result["exported_cases"] == 0 and result["saved_asr_pairs_seen"] == 1
    report = json.loads((tmp_path / "export/report.json").read_text())
    item = report["run_inventory"][0]
    expected = (
        "missing_recorded_table_identity"
        if state == "missing"
        else "conflicting_recorded_table_identities"
    )
    assert item["reason"] == expected
    assert report["table_identity_status_counts"] == {expected: 1}
    assert item["metadata_evidence"]["inputs"] == config["inputs"]
    assert item["transcript_inspection"]["examples"]
    assert not list((tmp_path / "export/audio").iterdir())


def test_old_table_and_unsupported_prediction_schema_are_both_reported(tmp_path: Path) -> None:
    source, table, _, _ = fixture(tmp_path)
    write_json(source / "run_config.json", {"inputs": {"hotwords": {"sha256": "f" * 64}}})
    (source / "baseline_predictions.jsonl").unlink()
    (source / "retrieved_predictions.jsonl").unlink()
    write_rows(source / "predictions.jsonl", [{"unknown_schema": "text"}])
    result = export(tmp_path, table)
    assert result["exported_cases"] == 0
    item = json.loads((tmp_path / "export/report.json").read_text())["run_inventory"][0]
    assert item["reason"] == "different_recorded_table_identity"
    assert "schema needs explicit adapter" in item["transcript_inspection_error"]
