import csv
from pathlib import Path

from qwen_hotword.training.spanish_sources import (
    CANONICAL_FIELDS,
    convert_common_voice_rioplatense_to_tsv,
    convert_slr61_argentinian_to_tsv,
)


def _write_cv_split(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "client_id",
        "path",
        "sentence_id",
        "sentence",
        "sentence_domain",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accents",
        "variant",
        "locale",
        "segment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _cv_row(audio: str, speaker: str) -> dict[str, str]:
    return {
        "client_id": speaker,
        "path": audio,
        "sentence_id": f"sentence-{audio}",
        "sentence": f"Texto para {audio}",
        "sentence_domain": "",
        "up_votes": "2",
        "down_votes": "0",
        "age": "twenties",
        "gender": "male_masculine",
        "accents": "Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay",
        "variant": "",
        "locale": "es",
        "segment": "",
    }


def test_convert_slr61_includes_first_headerless_rows_and_excludes_es_es(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "slr61"
    downloads = source_root / "downloads"
    extracted = source_root / "extracted"
    downloads.mkdir(parents=True)
    (extracted / "es-ar").mkdir(parents=True)
    (extracted / "es-es").mkdir(parents=True)
    (downloads / "line_index_female.tsv").write_text(
        "arf_05679_0001\t¿Me podés ayudar?\n"
        "arf_02485_0001\tHace doce grados con sol\n",
        encoding="utf-8",
    )
    (downloads / "line_index_male.tsv").write_text(
        "arm_09697_0001\tTengo un nuevo jabón\n", encoding="utf-8"
    )
    (downloads / "es_ar_line_index_weather.tsv").write_text(
        "arf_02485_0001\tHace doce grados con sol\n", encoding="utf-8"
    )
    (extracted / "arf_05679_0001.wav").write_bytes(b"audio")
    (extracted / "arf_02485_0001.wav").write_bytes(b"weather")
    (extracted / "arm_09697_0001.wav").write_bytes(b"audio")
    (extracted / "es-ar" / "arf_02485_0001.wav").write_bytes(b"weather")
    (extracted / "es-es" / "esw_03397_0001.wav").write_bytes(b"excluded")
    output_tsv = tmp_path / "output" / "source.tsv"

    summary = convert_slr61_argentinian_to_tsv(
        source_root,
        output_tsv,
        check_audio=True,
        scan_audio_inventory=True,
    )

    assert summary.status == "pass"
    assert summary.source_records == 4
    assert summary.written_records == 3
    assert summary.input_record_counts == {
        "female": 2,
        "male": 1,
        "weather_es_ar": 1,
    }
    assert summary.duplicate_source_ids == 1
    assert summary.duplicate_weather_alias_records == 1
    assert summary.verified_duplicate_weather_audio_files == 1
    assert summary.unexpected_duplicate_source_ids == 0
    assert summary.audio_files_under_root == 5
    assert summary.indexed_audio_files == 3
    assert summary.excluded_duplicate_argentinian_weather_audio_files == 1
    assert summary.excluded_peninsular_weather_audio_files == 1
    assert summary.unexpected_unindexed_audio_files == 0
    with output_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert tuple(rows[0]) == CANONICAL_FIELDS
    assert [row["source_id"] for row in rows] == [
        "arf_05679_0001",
        "arf_02485_0001",
        "arm_09697_0001",
    ]
    assert {row["language"] for row in rows} == {"es"}
    assert {row["dialect"] for row in rows} == {"argentinian"}


def test_convert_slr61_warns_when_weather_alias_differs(tmp_path: Path) -> None:
    source_root = tmp_path / "slr61"
    downloads = source_root / "downloads"
    extracted = source_root / "extracted"
    downloads.mkdir(parents=True)
    (extracted / "es-ar").mkdir(parents=True)
    (downloads / "line_index_female.tsv").write_text(
        "arf_02485_0001\tHace doce grados con sol\n", encoding="utf-8"
    )
    (downloads / "line_index_male.tsv").write_text(
        "arm_09697_0001\tTexto masculino\n", encoding="utf-8"
    )
    (downloads / "es_ar_line_index_weather.tsv").write_text(
        "arf_02485_0001\tTexto diferente\n", encoding="utf-8"
    )
    (extracted / "arf_02485_0001.wav").write_bytes(b"original")
    (extracted / "arm_09697_0001.wav").write_bytes(b"audio")
    (extracted / "es-ar" / "arf_02485_0001.wav").write_bytes(b"different")

    summary = convert_slr61_argentinian_to_tsv(
        source_root,
        tmp_path / "source.tsv",
        check_audio=True,
        scan_audio_inventory=True,
    )

    assert summary.status == "warn"
    assert summary.duplicate_weather_text_mismatches == 1
    assert summary.duplicate_weather_audio_mismatches == 1
    assert summary.verified_duplicate_weather_audio_files == 0


def test_convert_common_voice_preserves_official_splits_and_speakers(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "es-Rioplatense"
    clips = corpus_root / "clips"
    clips.mkdir(parents=True)
    split_rows = {
        "train": [_cv_row("train.mp3", "speaker-train")],
        "dev": [_cv_row("dev.mp3", "speaker-dev")],
        "test": [_cv_row("test.mp3", "speaker-test")],
    }
    for split, rows in split_rows.items():
        _write_cv_split(corpus_root / f"{split}.tsv", rows)
        (clips / rows[0]["path"]).write_bytes(b"audio")
    output_tsv = tmp_path / "output" / "source.tsv"

    summary = convert_common_voice_rioplatense_to_tsv(
        corpus_root,
        output_tsv,
        check_audio=True,
        scan_audio_inventory=True,
    )

    assert summary.status == "pass"
    assert summary.split_record_counts == {"test": 1, "train": 1, "validation": 1}
    assert summary.speaker_count == 3
    assert set(summary.cross_split_speaker_overlaps.values()) == {0}
    assert summary.audio_files_under_root == 3
    assert summary.unreferenced_audio_files == 0
    with output_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["source_split"] for row in rows] == ["train", "validation", "test"]
    assert [row["speaker_id"] for row in rows] == [
        "speaker-train",
        "speaker-dev",
        "speaker-test",
    ]


def test_convert_common_voice_warns_on_cross_split_speaker_overlap(
    tmp_path: Path,
) -> None:
    corpus_root = tmp_path / "es-Rioplatense"
    clips = corpus_root / "clips"
    clips.mkdir(parents=True)
    for split in ("train", "dev", "test"):
        row = _cv_row(f"{split}.mp3", "shared-speaker" if split != "test" else "test")
        _write_cv_split(corpus_root / f"{split}.tsv", [row])
        (clips / row["path"]).write_bytes(b"audio")

    summary = convert_common_voice_rioplatense_to_tsv(
        corpus_root,
        tmp_path / "source.tsv",
        check_audio=True,
    )

    assert summary.status == "warn"
    assert summary.cross_split_speaker_overlaps["train__validation"] == 1
    assert summary.issue_counts == {"cross_split_speaker_overlap": 1}
