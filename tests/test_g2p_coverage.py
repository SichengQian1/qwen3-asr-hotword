from __future__ import annotations

import importlib.util
from pathlib import Path

from qwen_hotword.phonemes.coverage import (
    espeak_language_code,
    load_phoneme_vocab,
    normalize_ipa_for_vocab,
    tokenize_ipa_to_vocab,
)

VOCAB_PATH = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")
SCRIPT_PATH = Path("scripts/scan_g2p_coverage.py")


def _load_scan_script():
    spec = importlib.util.spec_from_file_location("scan_g2p_coverage", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenize_ipa_matches_multichar_and_nasal_phones() -> None:
    vocab = load_phoneme_vocab(VOCAB_PATH)

    result = tokenize_ipa_to_vocab("ˈt͡ʃ ẽ d͡ʒ ɐ̃", vocab)

    assert result.tokens == ["tʃ", "ẽ", "dʒ", "ɐ̃"]
    assert result.oov_units == []
    assert len(result.token_ids) == 4


def test_tokenize_ipa_reports_unknown_units() -> None:
    vocab = load_phoneme_vocab(VOCAB_PATH)

    result = tokenize_ipa_to_vocab("k q", vocab)

    assert result.tokens == ["k"]
    assert result.oov_units == ["q"]


def test_normalize_ipa_maps_common_espeak_diphthongs() -> None:
    assert normalize_ipa_for_vocab("aɪ aʊ eɪ oʊ ɔɪ") == "aj aw ej ow ɔj"


def test_espeak_language_code_maps_project_languages() -> None:
    assert espeak_language_code("en") == "en-us"
    assert espeak_language_code("es-419") == "es"
    assert espeak_language_code("pt-BR") == "pt-br"


def test_parse_input_spec_accepts_per_file_language() -> None:
    script = _load_scan_script()

    path, language = script.parse_input_spec("/data/train_es.jsonl::es-419")

    assert str(path) == "/data/train_es.jsonl"
    assert language == "es-419"


def test_iter_config_records_reuses_eval_scanner_paths(tmp_path: Path) -> None:
    script = _load_scan_script()
    root_dir = tmp_path / "CommonVoice"
    data_dir = root_dir / "en"
    (data_dir / "clips").mkdir(parents=True)
    (data_dir / "clips" / "cv001.mp3").write_bytes(b"fake")
    (data_dir / "test.tsv").write_text(
        "path\tsentence\ncv001.mp3\tThe Qwen hotword appears\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        f"""
output_dir: {tmp_path / "out"}
sampling:
  seed: 1
  max_utterances_per_source: 0
hotwords:
  max_per_language: 10
  min_chars: 4
  min_words: 1
  max_words: 3
  max_occurrences: 10
  source_utt_limit: 5
cases:
  cases_per_language: 2
  active_hotwords_per_case: 2
  positive_ratio: 0.5
  negative_ratio: 0.25
  confusable_ratio: 0.25
  no_hotword_ratio: 0.0
sources:
  - name: cv_en
    dataset: commonvoice
    language: en
    root: {root_dir}
    include_locales: [en]
    include_splits: [test]
""",
        encoding="utf-8",
    )

    records = list(script.iter_config_records(config_path))

    assert len(records) == 1
    assert records[0]["language"] == "en"
    assert records[0]["text"] == "The Qwen hotword appears"
