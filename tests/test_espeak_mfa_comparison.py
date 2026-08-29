from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.phonemes.espeak_mfa_comparison import (
    ESPEAK_LANGUAGE_CODES,
    align_phone_sequences,
    compare_espeak_mfa,
    parse_named_path,
)

VOCAB_PATH = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")
WORDS = (
    "alpha",
    "bravo",
    "switch",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
)


def test_compare_espeak_mfa_writes_reproducible_independent_reports(
    tmp_path: Path,
) -> None:
    manifests: dict[str, Path] = {}
    dictionaries: dict[str, Path] = {}
    for language in ("en", "es", "pt"):
        manifest = tmp_path / f"{language}.jsonl"
        manifest.write_text(
            json.dumps(
                    {
                        "id": language,
                        "split": "train",
                        "language": {"en": "en-US", "es": "es", "pt": "pt-BR"}[
                            language
                        ],
                        "text": " ".join(WORDS) + " alpha alpha bravo",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        dictionary = tmp_path / f"{language}.dict"
        dictionary.write_text(
            "".join(f"{word}\ta b\n" for word in WORDS),
            encoding="utf-8",
        )
        manifests[language] = manifest
        dictionaries[language] = dictionary

    calls: list[tuple[list[str], str]] = []

    def fake_phonemize(words: list[str], language: str) -> list[str]:
        calls.append((words, language))
        result: list[str] = []
        for word in words:
            if word == "alpha":
                result.append("a q")
            elif word == "bravo":
                result.append("a p")
            elif word == "switch":
                result.append("(en)a b")
            else:
                result.append("a b")
        return result

    output = tmp_path / "selection"
    summary = compare_espeak_mfa(
        manifests,
        dictionaries,
        VOCAB_PATH,
        output,
        phonemize_batch=fake_phonemize,
        sample_size=10,
        seed=17,
        tool_metadata={"espeak_ng_version": "fixture"},
    )

    assert [language for _, language in calls] == ["en-us", "es-419", "pt-br"]
    assert all(len(words) == 10 for words, _ in calls)
    assert summary["total_sampled_words"] == 30
    for language in ("en", "es", "pt"):
        metrics = summary["by_language"][language]
        assert metrics["words"] == 10
        assert metrics["exact_token_matches"] == 8
        assert metrics["words_with_espeak_oov"] == 1
        assert metrics["words_with_language_switch"] == 1
        assert summary["source_stats"][language]["eligible_words"] == 10

    comparison_rows = _read_jsonl(output / "word_comparisons.jsonl")
    assert len(comparison_rows) == 30
    spanish_switch = next(
        row
        for row in comparison_rows
        if row["language"] == "es" and row["word"] == "switch"
    )
    assert spanish_switch["espeak_language_switch_flags"] == ["(en)"]
    assert spanish_switch["exact_token_match"] is True
    assert (output / "manual_review.tsv").is_file()
    assert "es\tq\tU+0071\t1" in (output / "oov_units.tsv").read_text(encoding="utf-8")
    _verify_sha256(output)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        compare_espeak_mfa(
            manifests,
            dictionaries,
            VOCAB_PATH,
            output,
            phonemize_batch=fake_phonemize,
            sample_size=10,
        )


def test_compare_espeak_mfa_rejects_missing_language_and_insufficient_words(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"text": "alpha", "split": "train", "language": "en"}) + "\n",
        encoding="utf-8",
    )
    dictionary = tmp_path / "mfa.dict"
    dictionary.write_text("alpha\ta\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly en, es, and pt"):
        compare_espeak_mfa(
            {"en": manifest},
            {"en": dictionary},
            VOCAB_PATH,
            tmp_path / "missing",
            phonemize_batch=lambda words, language: ["a"] * len(words),
        )

    paths: dict[str, Path] = {}
    for language in ("en", "es", "pt"):
        language_manifest = tmp_path / f"{language}_train.jsonl"
        language_manifest.write_text(
            json.dumps({"text": "alpha", "split": "train", "language": language}) + "\n",
            encoding="utf-8",
        )
        paths[language] = language_manifest
    dictionaries = {language: dictionary for language in ("en", "es", "pt")}
    with pytest.raises(ValueError, match="only 1 eligible en words"):
        compare_espeak_mfa(
            paths,
            dictionaries,
            VOCAB_PATH,
            tmp_path / "insufficient",
            phonemize_batch=lambda words, language: ["a"] * len(words),
            sample_size=2,
        )
    assert not (tmp_path / "insufficient").exists()


def test_phone_alignment_and_named_paths_are_deterministic() -> None:
    assert align_phone_sequences(["a", "b", "c"], ["a", "p", "c", "d"]) == [
        ("a", "a"),
        ("b", "p"),
        ("c", "c"),
        (None, "d"),
    ]
    assert parse_named_path("es=/data/train.jsonl").language == "es"
    assert parse_named_path("pt=/data/train.jsonl").path == Path("/data/train.jsonl")
    with pytest.raises(ValueError, match="en=PATH"):
        parse_named_path("fr=/data/train.jsonl")
    assert ESPEAK_LANGUAGE_CODES == {"en": "en-us", "es": "es-419", "pt": "pt-br"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_sha256(directory: Path) -> None:
    for line in (directory / "sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split("  ", maxsplit=1)
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
