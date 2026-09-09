from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.external_keyword_retrieval import (
    ExternalAudioRecord,
    SourceSpec,
    build_external_dataset,
    load_keyword_bias,
    parse_source_spec,
    run_external_keyword_retrieval,
    summarize_retrieval,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab


def _write_keyword_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "prompt_type": "keyword_bias",
                "language": "pt",
                "phoneme_source": "mfa",
                "keyword_sets": {"hard_k266": ["São Paulo", "Lisboa"]},
                "keyword_phonemes": {
                    "São Paulo": "/s a w p a w l u/",
                    "Lisboa": "/l i s b o ɐ/",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_source_parser_preserves_absolute_paths() -> None:
    source = parse_source_spec("mls=/data/audio,/data/transcripts.txt")
    assert source.name == "mls"
    assert source.audio_dir == Path("/data/audio")
    assert source.transcript_path == Path("/data/transcripts.txt")
    with pytest.raises(ValueError, match="NAME=AUDIO_DIR,TRANSCRIPTS"):
        parse_source_spec("invalid")


def test_keyword_bias_is_tokenized_against_current_vocab(tmp_path: Path) -> None:
    keyword_path = tmp_path / "keywords.json"
    _write_keyword_file(keyword_path)
    vocab = load_phoneme_vocab(Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"))
    bundle = load_keyword_bias(keyword_path, vocab=vocab)
    assert [entry.surface for entry in bundle.entries] == ["São Paulo", "Lisboa"]
    assert bundle.audit["keyword_count"] == 2
    assert bundle.phonemes["São Paulo"] == "/s a w p a w l u/"


def test_keyword_bias_rejects_ctc_oov(tmp_path: Path) -> None:
    keyword_path = tmp_path / "keywords.json"
    _write_keyword_file(keyword_path)
    value = json.loads(keyword_path.read_text(encoding="utf-8"))
    value["keyword_phonemes"]["Lisboa"] = "/☃/"
    keyword_path.write_text(json.dumps(value), encoding="utf-8")
    vocab = load_phoneme_vocab(Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"))
    with pytest.raises(ValueError, match="phoneme audit failed"):
        load_keyword_bias(keyword_path, vocab=vocab)


def test_dataset_matches_flac_stems_and_rejects_cross_source_collisions(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.flac").write_bytes(b"fake")
    (second / "b.WAV").write_bytes(b"fake")
    first_txt = tmp_path / "first.txt"
    second_txt = tmp_path / "second.txt"
    first_txt.write_text("a\ttexto um\n", encoding="utf-8")
    second_txt.write_text("b texto dois\n", encoding="utf-8")
    records, audit = build_external_dataset(
        (
            SourceSpec("first", first, first_txt),
            SourceSpec("second", second, second_txt),
        )
    )
    assert [(record.sample_id, record.source) for record in records] == [
        ("a", "first"),
        ("b", "second"),
    ]
    assert audit["sample_count"] == 2
    sources = audit["sources"]
    assert isinstance(sources, list)
    assert sources[0]["audio_format_counts"] == {"flac": 1, "wav": 0}
    assert sources[1]["audio_format_counts"] == {"flac": 0, "wav": 1}

    (second / "b.WAV").rename(second / "a.wav")
    second_txt.write_text("a|texto dois\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cross-source duplicate"):
        build_external_dataset(
            (
                SourceSpec("first", first, first_txt),
                SourceSpec("second", second, second_txt),
            )
        )


def test_dataset_rejects_unmatched_transcript(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "a.flac").write_bytes(b"fake")
    transcript = tmp_path / "transcripts.txt"
    transcript.write_text("other texto\n", encoding="utf-8")
    with pytest.raises(ValueError, match="audio/transcript mismatch"):
        build_external_dataset((SourceSpec("source", audio, transcript),))


def test_summary_builds_exact_downstream_shape_and_metrics(tmp_path: Path) -> None:
    keyword_path = tmp_path / "keywords.json"
    _write_keyword_file(keyword_path)
    vocab = load_phoneme_vocab(Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"))
    bundle = load_keyword_bias(keyword_path, vocab=vocab)
    sao, lisboa = bundle.entries

    def match(entry: HotwordEntry) -> dict[str, object]:
        surface = entry.surface
        hotword_id = entry.hotword_id
        return {
            "hotword_id": hotword_id,
            "surface": surface,
            "score": 0.9,
            "edit_ratio": 0.0,
            "posterior_confidence": 0.9,
        }

    base_timing = {
        "audio_load_seconds": 0.01,
        "ctc_processor_seconds": 0.01,
        "ctc_encoder_seconds": 0.02,
        "ctc_head_seconds": 0.01,
        "ctc_decode_seconds": 0.001,
        "anchor_query_seconds": 0.002,
        "hotword_matching_seconds": 0.003,
        "hotword_sorting_seconds": 0.001,
        "hotword_selection_seconds": 0.001,
        "pure_retrieval_seconds": 0.007,
        "model_retrieval_seconds": 0.047,
        "audio_to_result_seconds": 0.057,
    }
    rows = [
        {
            "sample_id": "a",
            "source": "first",
            "expected_hotword_ids": [sao.hotword_id],
            "raw_hit_expected_hotword_ids": [sao.hotword_id],
            "selected_hotword_ids": [sao.hotword_id, lisboa.hotword_id],
            "final_hit_expected_hotword_ids": [sao.hotword_id],
            "raw_missed_expected_hotword_ids": [],
            "final_missed_expected_hotword_ids": [],
            "wrong_selected_hotword_ids": [lisboa.hotword_id],
            "selected_matches": [match(sao), match(lisboa)],
            "audio": {"duration_seconds": 2.0},
            "timing": base_timing,
        },
        {
            "sample_id": "b",
            "source": "second",
            "expected_hotword_ids": [],
            "raw_hit_expected_hotword_ids": [],
            "selected_hotword_ids": [],
            "final_hit_expected_hotword_ids": [],
            "raw_missed_expected_hotword_ids": [],
            "final_missed_expected_hotword_ids": [],
            "wrong_selected_hotword_ids": [],
            "selected_matches": [],
            "audio": {"duration_seconds": 2.0},
            "timing": base_timing,
        },
    ]
    output, _, summary, failures = summarize_retrieval(rows, bundle=bundle)
    assert output["a"] == [
        {"word": "São Paulo", "phoneme": "/s a w p a w l u/"},
        {"word": "Lisboa", "phoneme": "/l i s b o ɐ/"},
    ]
    assert output["b"] == []
    overall = summary["overall"]
    assert isinstance(overall, dict)
    assert overall["raw_recall_at_5"] == 1.0
    assert overall["final_retrieval_precision"] == 0.5
    assert len(failures) == 1


def test_record_manifest_shape() -> None:
    row = ExternalAudioRecord(
        sample_id="sample",
        source="delivery",
        audio_path=Path("/data/sample.flac"),
        reference_text="texto",
    ).to_dict()
    assert row["language"] == "pt-BR"
    assert row["audio_path"] == "/data/sample.flac"


def test_full_mock_run_writes_resumable_downstream_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "Qwen3-ASR-1.7B"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    checkpoint = tmp_path / "ctc_head_best.pt"
    checkpoint.write_bytes(b"checkpoint")
    keyword_path = tmp_path / "keywords.json"
    _write_keyword_file(keyword_path)
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "sample.flac").write_bytes(b"fake")
    transcripts = tmp_path / "transcripts.txt"
    transcripts.write_text("sample\tSão Paulo aparece\n", encoding="utf-8")
    output = tmp_path / "output"

    monkeypatch.setattr(
        "qwen_hotword.inference.external_keyword_retrieval._load_audio",
        lambda _path: (
            object(),
            {
                "original_sample_rate": 16_000,
                "format": "FLAC",
                "target_sample_rate": 16_000,
                "channels": 1,
                "resampled": False,
                "samples": 32_000,
                "duration_seconds": 2.0,
                "audio_load_seconds": 0.01,
            },
        ),
    )

    def retrieve(_waveform: object, _detector: object, _vocab: object) -> dict[str, object]:
        return {
            "raw_ranked_matches": [
                {
                    "hotword_id": "external_pt_0000",
                    "surface": "São Paulo",
                    "score": 0.9,
                    "edit_ratio": 0.0,
                    "posterior_confidence": 0.9,
                }
            ],
            "selected_matches": [
                {
                    "hotword_id": "external_pt_0000",
                    "surface": "São Paulo",
                    "score": 0.9,
                    "edit_ratio": 0.0,
                    "posterior_confidence": 0.9,
                }
            ],
            "decoded_token_ids": [1],
            "decoded_tokens": ["a"],
            "decoded_confidences": [0.9],
            "suppressed_reason": None,
            "shortlist_candidates": 1,
            "postings_visited": 1,
            "timing": {
                "ctc_processor_seconds": 0.01,
                "ctc_encoder_seconds": 0.02,
                "ctc_head_seconds": 0.01,
                "ctc_decode_seconds": 0.001,
                "anchor_query_seconds": 0.002,
                "hotword_matching_seconds": 0.003,
                "hotword_sorting_seconds": 0.001,
                "hotword_selection_seconds": 0.001,
                "pure_retrieval_seconds": 0.007,
                "model_retrieval_seconds": 0.047,
            },
        }

    arguments = {
        "model_path": model,
        "checkpoint_path": checkpoint,
        "vocab_path": Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"),
        "keyword_bias_path": keyword_path,
        "sources": (SourceSpec("source", audio, transcripts),),
        "output_dir": output,
        "retrieve_function": retrieve,
        "print_progress": False,
    }
    summary = run_external_keyword_retrieval(**arguments)
    assert summary["status"] == "pass"
    assert json.loads((output / "retrieval_output.json").read_text())["sample"] == [
        {"word": "São Paulo", "phoneme": "/s a w p a w l u/"}
    ]
    assert (output / "sha256.txt").is_file()
    with pytest.raises(FileExistsError):
        run_external_keyword_retrieval(**arguments)
    run_external_keyword_retrieval(**arguments, resume=True)
