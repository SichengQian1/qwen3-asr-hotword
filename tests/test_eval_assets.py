from __future__ import annotations

import json
from pathlib import Path

from qwen_hotword.evaluation.cases import build_eval_cases
from qwen_hotword.evaluation.config import load_eval_asset_config
from qwen_hotword.evaluation.hotwords import build_hotwords
from qwen_hotword.evaluation.scanner import scan_sources


def test_build_eval_assets_from_small_corpora(tmp_path: Path) -> None:
    libri_dir = tmp_path / "LibriSpeech" / "test-clean" / "1" / "2"
    libri_dir.mkdir(parents=True)
    (libri_dir / "1-2-0001.flac").write_bytes(b"fake")
    (libri_dir / "1-2.trans.txt").write_text(
        "1-2-0001 THE SAINT MICHAEL PROJECT ARRIVED TODAY\n",
        encoding="utf-8",
    )
    flat_libri_dir = tmp_path / "FlatLibriSpeech"
    (flat_libri_dir / "clean").mkdir(parents=True)
    (flat_libri_dir / "clean" / "clean001.wav").write_bytes(b"fake")
    (flat_libri_dir / "trans_clean.tsv").write_text(
        "id\tpath\ttext\nclean001\tclean/clean001.wav\tTHE OXFORD ROBOT ARRIVED\n",
        encoding="utf-8",
    )

    cv_dir = tmp_path / "CommonVoice" / "en"
    (cv_dir / "clips").mkdir(parents=True)
    (cv_dir / "clips" / "cv001.mp3").write_bytes(b"fake")
    (cv_dir / "test.tsv").write_text(
        "path\tsentence\ncv001.mp3\tThe Lisbon Harbor team arrived\n",
        encoding="utf-8",
    )

    fleurs_dir = tmp_path / "Fleurs" / "English"
    (fleurs_dir / "validation").mkdir(parents=True)
    (fleurs_dir / "validation" / "fleurs001.wav").write_bytes(b"fake")
    (fleurs_dir / "validation.tsv").write_text(
        "path\ttranscription\nfleurs001.wav\tThe Cambridge Atlas project starts tomorrow\n",
        encoding="utf-8",
    )

    mls_dir = tmp_path / "MLS"
    (mls_dir / "test" / "audio").mkdir(parents=True)
    (mls_dir / "test" / "audio" / "mls001.flac").write_bytes(b"fake")
    (mls_dir / "test" / "transcripts.txt").write_text(
        "mls001\tA estação central fica perto de São Paulo\n",
        encoding="utf-8",
    )
    (mls_dir / "dev" / "audio").mkdir(parents=True)
    (mls_dir / "dev" / "audio" / "mls-dev001.flac").write_bytes(b"fake")
    (mls_dir / "dev" / "transcripts.txt").write_text(
        "mls-dev001\tO aeroporto internacional recebe navios antigos\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"""
output_dir: {tmp_path / "out"}
sampling:
  seed: 7
  max_utterances_per_source: 10
hotwords:
  max_per_language: 20
  min_chars: 4
  min_words: 1
  max_words: 3
  max_occurrences: 10
  source_utt_limit: 5
cases:
  cases_per_language: 8
  active_hotwords_per_case: 4
  positive_ratio: 0.5
  negative_ratio: 0.25
  confusable_ratio: 0.25
  no_hotword_ratio: 0.0
sources:
  - name: librispeech
    dataset: librispeech
    language: en
    root: {tmp_path / "LibriSpeech"}
    include_splits: [test-clean]
  - name: flat_librispeech
    dataset: librispeech
    language: en
    root: {tmp_path / "FlatLibriSpeech"}
    include_splits: [clean]
  - name: commonvoice_en
    dataset: commonvoice
    language: en
    root: {tmp_path / "CommonVoice"}
    include_locales: [en]
    include_splits: [test]
  - name: fleurs_en
    dataset: fleurs
    language: en
    root: {tmp_path / "Fleurs" / "English"}
    include_splits: [validation]
  - name: mls_pt
    dataset: mls
    language: pt-BR
    root: {tmp_path / "MLS"}
    include_splits: [test]
  - name: mls_pt_aux_hotwords
    dataset: mls
    language: pt-BR
    root: {tmp_path / "MLS"}
    include_splits: [dev]
    use_for_hotwords: true
    use_for_cases: false
""",
        encoding="utf-8",
    )

    config = load_eval_asset_config(config_path)
    utterances = scan_sources(config)
    assert len(utterances) == 6
    assert {utterance.language for utterance in utterances} == {"en", "pt-BR"}
    assert all(Path(utterance.audio_path).is_file() for utterance in utterances)
    assert any(utterance.dataset == "fleurs_en" for utterance in utterances)
    assert any(
        utterance.split == "dev" and utterance.metadata["use_for_cases"] == "false"
        for utterance in utterances
    )

    hotword_utterances = [
        utterance
        for utterance in utterances
        if utterance.metadata["use_for_hotwords"] == "true"
    ]
    case_utterances = [
        utterance
        for utterance in utterances
        if utterance.metadata["use_for_cases"] == "true"
    ]
    assert all(utterance.split != "dev" for utterance in case_utterances)

    hotwords = build_hotwords(
        hotword_utterances,
        config.hotwords,
        phonemizer_backend="none",
        require_ipa=False,
    )
    assert any(hotword.normalized == "saint michael" for hotword in hotwords)
    assert any(hotword.normalized == "aeroporto internacional" for hotword in hotwords)
    assert any(hotword.language == "pt-BR" for hotword in hotwords)

    ctc_cases = build_eval_cases(
        case_utterances,
        hotwords,
        config.cases,
        seed=config.sampling.seed,
        eval_stage="ctc",
    )
    asr_cases = build_eval_cases(
        case_utterances,
        hotwords,
        config.cases,
        seed=config.sampling.seed,
        eval_stage="asr_injection",
    )
    assert ctc_cases
    assert asr_cases
    assert any(case.expected_hotword_ids for case in ctc_cases)
    assert all(case.eval_stage == "asr_injection" for case in asr_cases)


def test_eval_config_template_is_valid() -> None:
    config = load_eval_asset_config("configs/eval_sources.example.yaml")
    assert config.sources
    assert config.hotwords.max_per_language == 800
    assert json.dumps(config.sources[0].name)
