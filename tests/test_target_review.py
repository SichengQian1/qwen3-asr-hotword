from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import build_target_review, compile_policy
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()
POLICY = Path("configs/multilingual_basic_words.v1.json").resolve()
TARGETS = {
    "en": ["A", "THE", "Me", "I.", "Fire", "Will", "May", "The Who", "Me Too", "AI", "Shared"],
    "es": ["a", "el", "él", "me", "El Paso", "La Paz", "Di", "AI", "Shared", "The"],
    "pt": ["a", "o", "me", "não", "No", "São", "São Paulo", "Banco do Brasil", "Di", "Shared"],
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def reseal(directory):
    (directory / "sha256.txt").unlink(missing_ok=True)
    seal(directory)


def fixture(root):
    write(root / "policy.json", json.loads(POLICY.read_text()))
    sources = {}
    for language, surfaces in TARGETS.items():
        directory = root / language
        names = (
            ["en_chile", "en_colombia", "en_mexico"]
            if language == "en"
            else ["wave1", "wave2", "wave3", "wave4"]
        )
        table = {
            "language": {"en": "English", "es": "Spanish", "pt": "pt-BR"}[language],
            "keyword_sets": {"all_keywords": surfaces + ["not_a_target", "extra_filler"]},
            "keyword_phonemes": {
                w: "/b a/" if language == "es" else "/a b/"
                for w in surfaces + ["not_a_target", "extra_filler"]
            },
        }
        write(directory / "table.json", table)
        write(
            directory / "groups.json",
            {name: sorted(" ".join(normalize_match_words(s)) for s in surfaces) for name in names},
        )
        write(
            directory / "report.json",
            {
                "status": "completed",
                "tables_created": True,
                "mandatory_union": len(surfaces),
                "languages": {language: {"mandatory": len(surfaces)}},
            },
        )
        seal(directory)
        sources[language] = {
            "directory": language,
            "table": "table.json",
            "target_groups": "groups.json",
            "expected_groups": names,
            "expected_targets": len(surfaces),
            "expected_table_size": len(surfaces) + 2,
        }
    config = {
        "output": "review",
        "basic_word_policy": "policy.json",
        "vocab": str(VOCAB),
        "vocab_sha256": _sha(VOCAB),
        "sources": sources,
    }
    write(root / "config.json", config)
    return root / "config.json", config


def load(path):
    return json.loads(path.read_text())


def test_declared_targets_only_and_reviewable_language_safe_union(tmp_path):
    config, _ = fixture(tmp_path)
    input_hashes = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    report = build_target_review(tmp_path, config)
    decisions = [
        json.loads(x)
        for x in (tmp_path / "review/all_target_decisions.jsonl").read_text().splitlines()
    ]
    assert report["input_language_entries"] == sum(map(len, TARGETS.values()))
    assert report["input_language_entries"] == (
        report["retained_language_entries"] + report["removed_language_entries"]
    )
    assert report["filler_entries_added"] == 0
    assert not report["model_loaded"] and not report["inference_run"]
    assert not report["human_approved"] and not report["transcripts_or_predictions_read"]
    kept = {(r["language"], r["surface"]) for r in decisions if r["decision"] == "retained"}
    removed = {(r["language"], r["surface"]) for r in decisions if r["decision"] == "removed_basic"}
    assert {("en", w) for w in ["A", "THE", "Me", "I.", "Will", "May"]} <= removed
    assert {("en", w) for w in ["Fire", "The Who", "Me Too", "AI", "Shared"]} <= kept
    assert {
        ("es", "El Paso"),
        ("es", "La Paz"),
        ("pt", "São Paulo"),
        ("pt", "Banco do Brasil"),
    } <= kept
    assert ("pt", "São") in removed and ("es", "él") in removed
    assert ("es", "The") in kept  # No cross-language stopword deletion.
    assert all(r["surface"] not in {"not_a_target", "extra_filler"} for r in decisions)
    entries = load(tmp_path / "review/keyword_table.json")["entries"]
    shared = next(e for e in entries if e["normalized"] == "shared")
    assert shared["languages"] == ["en", "es", "pt"]
    assert [v["pronunciation"] for v in shared["variants"]] == ["/a b/", "/b a/", "/a b/"]
    assert len(entries) < report["retained_language_entries"]
    assert "同形专名" in (tmp_path / "review/removed_basic_words.md").read_text()
    assert all(_sha(p) == h for p, h in input_hashes.items())
    for line in (tmp_path / "review/sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(tmp_path / "review" / name) == digest


def test_determinism_refuse_overwrite_and_keep_override(tmp_path):
    config, _ = fixture(tmp_path)
    build_target_review(tmp_path, config)
    build_target_review(tmp_path, config, Path("second"))
    for name in ("keyword_table.json", "all_target_decisions.jsonl", "review.md", "keywords.txt"):
        assert (tmp_path / "review" / name).read_bytes() == (
            tmp_path / "second" / name
        ).read_bytes()
    with pytest.raises(FileExistsError):
        build_target_review(tmp_path, config)
    policy = load(tmp_path / "policy.json")
    policy["languages"]["en"]["keep_overrides"] = ["Will"]
    write(tmp_path / "policy.json", policy)
    build_target_review(tmp_path, config, Path("third"))
    assert "Will" in (tmp_path / "third/keywords.txt").read_text().splitlines()
    assert "Will" not in (tmp_path / "review/keywords.txt").read_text().splitlines()


@pytest.mark.parametrize(
    "failure",
    [
        "table_sha",
        "group_sha",
        "report_sha",
        "missing_target",
        "duplicate_target",
        "noncanonical",
        "wrong_groups",
        "wrong_count",
        "wrong_report",
        "language",
        "oov",
        "pin",
    ],
)
def test_invalid_source_fails_before_output(tmp_path, failure):
    config, cfg = fixture(tmp_path)
    directory = tmp_path / "en"
    table = load(directory / "table.json")
    groups = load(directory / "groups.json")
    report = load(directory / "report.json")
    if failure.endswith("_sha"):
        name = {"table_sha": "table.json", "group_sha": "groups.json", "report_sha": "report.json"}[
            failure
        ]
        with (directory / name).open("a") as handle:
            handle.write(" ")
    else:
        if failure == "missing_target":
            table["keyword_sets"]["all_keywords"].remove("Fire")
            table["keyword_sets"]["all_keywords"].append("replacement")
            table["keyword_phonemes"]["replacement"] = "a b"
        if failure == "duplicate_target":
            groups["en_chile"].append("fire")
        if failure == "noncanonical":
            groups["en_chile"][0] = " Ａ "
        if failure == "wrong_groups":
            groups["wave1"] = groups.pop("en_chile")
        if failure == "wrong_count":
            cfg["sources"]["en"]["expected_targets"] += 1
            write(config, cfg)
        if failure == "wrong_report":
            report["mandatory_union"] += 1
        if failure == "language":
            table["language"] = "Portuguese"
        if failure == "oov":
            table["keyword_phonemes"]["Fire"] = "☃"
        if failure == "pin":
            cfg["sources"]["en"]["table_sha256"] = "0" * 64
            write(config, cfg)
        write(directory / "table.json", table)
        write(directory / "groups.json", groups)
        write(directory / "report.json", report)
        reseal(directory)
    with pytest.raises(ValueError):
        build_target_review(tmp_path, config)
    assert not (tmp_path / "review").exists()


@pytest.mark.parametrize("failure", ["phrase", "duplicate", "override", "language", "category"])
def test_policy_is_explicit_single_token_and_language_scoped(failure):
    policy = load(POLICY)
    if failure == "phrase":
        policy["languages"]["en"]["remove"]["articles"].append("The Who")
    if failure == "duplicate":
        policy["languages"]["en"]["remove"]["articles"].append("Ａ.")
    if failure == "override":
        policy["languages"]["en"]["keep_overrides"] = ["Fire"]
    if failure == "language":
        del policy["languages"]["pt"]
    if failure == "category":
        policy["languages"]["en"]["remove"]["unknown"] = ["example"]
    with pytest.raises(ValueError):
        compile_policy(policy)


def test_shipped_config_uses_expected_sealed_target_scopes():
    config = load(Path("configs/multilingual_target_review.workzone.json"))
    assert {lang: c["expected_targets"] for lang, c in config["sources"].items()} == {
        "en": 773,
        "es": 316,
        "pt": 295,
    }
    assert _sha(VOCAB) == config["vocab_sha256"]
    assert config["sources"]["en"]["table_sha256"] == (
        "6e2a7016d16bf86cb1f31f3254213e58f5becd4810cc02a166fd604d1cd9b655"
    )
    assert set(compile_policy(load(POLICY))) == {"en", "es", "pt"}


def test_output_cannot_modify_a_sealed_source_directory(tmp_path):
    config, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="inside an existing source"):
        build_target_review(tmp_path, config, Path("en/new_review"))
    assert not (tmp_path / "en/new_review").exists()
