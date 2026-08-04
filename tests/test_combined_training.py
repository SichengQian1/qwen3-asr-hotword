from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.training.combined_training import (
    CombinedCorpusInput,
    build_temporal2x_combined_training,
    parse_combined_corpus_spec,
)


def _row(
    index: int,
    split_hash: float,
    *,
    corpus: str,
    ready: bool,
    estimated: int = 10,
    minimum: int = 8,
    issues: list[str] | None = None,
    language: str = "pt-BR",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": corpus,
        "training_ready": ready,
        "label_status": "ready" if ready else "needs_review",
        "issues": [{"reason": reason, "detail": "fixture"} for reason in (issues or [])],
        "id": f"{corpus}-{index:04d}",
        "audio_path": f"/audio/{corpus}/{index:04d}.wav",
        "audio_relative": f"{corpus}/{index:04d}.wav",
        "text": "bom dia",
        "language": language,
        "phoneme_token_ids": [2, 3, 4],
        "label_length": 3,
        "ctc_minimum_input_length": minimum,
        "estimated_ctc_input_length": estimated,
        "duration_seconds": 1.0,
        "source_tsv": f"/data/{corpus}.tsv",
        "row_number": index + 2,
        "split_hash": split_hash,
    }


def _write_corpus(
    root: Path,
    name: str,
    ready: list[dict[str, object]],
    review: list[dict[str, object]],
) -> CombinedCorpusInput:
    manifest_dir = root / name
    manifest_dir.mkdir()
    (manifest_dir / "train_ready.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ready),
        encoding="utf-8",
    )
    (manifest_dir / "needs_review.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in review),
        encoding="utf-8",
    )
    return CombinedCorpusInput(name, manifest_dir)


def _read_jsonl(path: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_combined_training_builds_new_stable_splits_and_releases_only_safe_rows(
    tmp_path: Path,
) -> None:
    ready_rows = [
        _row(index, (index + 0.5) / 100, corpus="noah", ready=True) for index in range(95)
    ]
    recovery_rows = [
        _row(
            index,
            (index + 0.5) / 100,
            corpus="external",
            ready=False,
            estimated=5,
            minimum=8,
            issues=["ctc_length_infeasible"],
            language="pt",
        )
        for index in range(95, 100)
    ]
    excluded_rows = [
        _row(
            101,
            0.1,
            corpus="external",
            ready=False,
            estimated=5,
            minimum=10,
            issues=["ctc_length_infeasible"],
        ),
        _row(
            102,
            0.1,
            corpus="external",
            ready=False,
            estimated=5,
            minimum=8,
            issues=["ctc_length_infeasible", "dictionary_missing"],
        ),
    ]
    corpora = [
        _write_corpus(tmp_path, "noah", ready_rows, []),
        _write_corpus(tmp_path, "external", [], recovery_rows + excluded_rows),
    ]

    summary = build_temporal2x_combined_training(
        corpora,
        tmp_path / "combined",
        print_progress=False,
    )

    assert summary.status == "pass"
    assert summary.source_records == 100
    assert summary.original_ready_records == 95
    assert summary.recovered_records == 5
    assert summary.split_records == {"train": 96, "validation": 2, "test": 2}
    assert summary.source_manifests_modified is False
    assert summary.test_set_used is False
    assert summary.test_set_sealed is True
    rows = [row for path in summary.manifest_paths.values() for row in _read_jsonl(path)]
    assert len({str(row["id"]) for row in rows}) == 100
    assert len({str(row["audio_path"]) for row in rows}) == 100
    assert {int(row["ctc_time_upsampling_factor"]) for row in rows} == {2}
    assert all(
        int(row["effective_ctc_input_length"]) == int(row["estimated_ctc_input_length"]) * 2
        for row in rows
    )
    recovered = [row for row in rows if row["release_source"] == "temporal_2x_recovery"]
    assert len(recovered) == 5
    assert {row["language"] for row in recovered} == {"pt"}


def test_combined_training_refuses_nonempty_output_and_cross_corpus_duplicate_audio(
    tmp_path: Path,
) -> None:
    first_row = _row(1, 0.1, corpus="first", ready=True)
    second_row = _row(2, 0.2, corpus="second", ready=True)
    second_row["audio_path"] = first_row["audio_path"]
    corpora = [
        _write_corpus(tmp_path, "first", [first_row], []),
        _write_corpus(tmp_path, "second", [second_row], []),
    ]
    output = tmp_path / "combined"

    with pytest.raises(ValueError, match="duplicate audio path"):
        build_temporal2x_combined_training(corpora, output, print_progress=False)

    (output / "existing.txt").write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_temporal2x_combined_training(corpora, output, print_progress=False)


def test_parse_combined_corpus_spec() -> None:
    parsed = parse_combined_corpus_spec("noah_500h=outputs/noah_pt_full_500h")
    assert parsed.name == "noah_500h"
    assert parsed.manifest_dir == Path("outputs/noah_pt_full_500h")

    with pytest.raises(ValueError, match="NAME=MANIFEST_DIR"):
        parse_combined_corpus_spec("missing-separator")
