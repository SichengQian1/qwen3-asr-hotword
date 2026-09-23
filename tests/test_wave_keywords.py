from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.evaluation.wave_keywords import WAVES, build_language, freeze_wave_keywords
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.spanish_capacity import _sha

VOCAB_PATH = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()
VOCAB = load_phoneme_vocab(VOCAB_PATH)


def inputs(lang="es"):
    phone = "/ĩ a/"
    primary = {
        wave: {
            "language": "Spanish" if lang == "es" else "Portuguese",
            "keyword_sets": {"baseline": [], "all_keywords": ["Casa"]},
            "keyword_phonemes": {"Casa": phone},
        }
        for wave in WAVES
    }
    neighbors = {
        wave: {
            "language": lang,
            "keyword_phonemes": {"Caza": "☃"},
            "neighbors": {
                "casa": [{"word": "Caza", "sim": 0.9, "count": 2, "phoneme": "k a s a"}],
                "unrelated": [{"word": "unwanted", "phoneme": "k a"}],
            },
        }
        for wave in WAVES
    }
    fillers = []
    for word, phone in (("Caza", "k a s a"), ("Mesa", "m e s a"), ("Otro", "o t r o")):
        tokens = tokenize_ipa_to_vocab(phone, VOCAB)
        fillers.append(
            {
                "surface": word,
                "normalized": word.lower(),
                "language": lang,
                "pronunciation": phone,
                "token_ids": tokens.token_ids,
                "phoneme_tokens": [VOCAB.tokens[i] for i in tokens.token_ids],
            }
        )
    return primary, neighbors, fillers


def build(lang="es", **kwargs):
    return build_language(
        lang, *inputs(lang), VOCAB, target_size=4, seed="fixed", expected_primary=1, **kwargs
    )


@pytest.mark.parametrize(
    "lang,declared",
    [
        ("es", "Spanish"),
        ("es", "es"),
        ("es", "es-419"),
        ("es", " SPANISH "),
        ("pt", "Portuguese"),
        ("pt", "pt"),
        ("pt", "pt-BR"),
        ("pt", " portuguese "),
    ],
)
def test_primary_language_aliases_preserve_table(lang, declared):
    primary, neighbors, fillers = inputs(lang)
    for raw in primary.values():
        raw["language"] = declared
    result = build_language(
        lang, primary, neighbors, fillers, VOCAB, target_size=4, seed="fixed", expected_primary=1
    )
    assert result == build(lang)


@pytest.mark.parametrize(
    "lang,declared",
    [
        ("es", "Portuguese"),
        ("pt", "Spanish"),
        ("es", "pt-BR"),
        ("pt", "es-419"),
        ("es", "unknown"),
        ("pt", ""),
        ("es", None),
        ("pt", ["Portuguese"]),
    ],
)
def test_primary_language_mismatch_reports_actual_value(lang, declared):
    primary, neighbors, fillers = inputs(lang)
    primary["wave3"]["language"] = declared
    with pytest.raises(ValueError) as exc:
        build_language(
            lang,
            primary,
            neighbors,
            fillers,
            VOCAB,
            target_size=4,
            seed="fixed",
            expected_primary=1,
        )
    assert f"wave3/{lang}: primary language mismatch" in str(exc.value)
    assert repr(declared) in str(exc.value)


def test_primary_missing_language_keeps_previous_behavior():
    primary, neighbors, fillers = inputs()
    for raw in primary.values():
        del raw["language"]
    assert (
        build_language(
            "es",
            primary,
            neighbors,
            fillers,
            VOCAB,
            target_size=4,
            seed="fixed",
            expected_primary=1,
        )
        == build()
    )


def test_exact_unique_union_provenance_and_language_specific_cleanup():
    es, pt = build(), build("pt")
    for result in (es, pt):
        entries = {e["normalized"]: e for e in result["entries"]}
        assert set(entries) == {"casa", "caza", "mesa", "otro"}
        assert entries["caza"]["source"] == "neighbor"
        assert entries["caza"]["pronunciation"] == "k a s a"  # not root keyword_phonemes
        assert len(entries["casa"]["origins"]) == 4
        assert result["summary"]["all_targets_retained"]
        assert result["summary"]["ignored_non_target_parent_occurrences"] == 4
        assert result["targets_by_wave"] == {w: ["casa"] for w in WAVES}
    assert es["entries"][0]["pronunciation"] == "i a"
    assert pt["entries"][0]["token_ids"] != es["entries"][0]["token_ids"]
    assert es["summary"]["primary_cleanup_occurrences"] == {"combining_tilde_removed": 4}
    assert pt["summary"]["primary_cleanup_occurrences"] == {}


@pytest.mark.parametrize("problem", ["oov", "conflict", "missing", "union_count"])
def test_mandatory_errors_block_without_dropping_targets(problem):
    primary, neighbors, fillers = inputs()
    if problem == "missing":
        primary["wave4"]["keyword_phonemes"].clear()
    elif problem == "union_count":
        primary["wave4"]["keyword_sets"]["other"] = ["Other"]
        primary["wave4"]["keyword_phonemes"]["Other"] = "a"
    else:
        primary["wave4"]["keyword_phonemes"]["Casa"] = "☃" if problem == "oov" else "a"
    with pytest.raises(ValueError):
        build_language(
            "es",
            primary,
            neighbors,
            fillers,
            VOCAB,
            target_size=4,
            seed="fixed",
            expected_primary=1,
        )


def test_optional_conflicts_quarantined_and_invalid_skipped_then_old_fill():
    primary, neighbors, fillers = inputs()
    neighbors["wave1"]["neighbors"]["casa"] += [
        {"word": "Caza", "phoneme": "s a"},
        {"word": "OOV", "phoneme": "☃"},
        {"word": "Casa", "phoneme": "s a"},
    ]
    result = build_language(
        "es", primary, neighbors, fillers, VOCAB, target_size=3, seed="fixed", expected_primary=1
    )
    assert {e["normalized"] for e in result["entries"]} == {"casa", "mesa", "otro"}
    assert result["summary"]["optional_conflicting_surfaces"] == 1
    assert result["summary"]["audit_decisions"]["excluded_invalid"] == 1
    assert result["summary"]["audit_decisions"]["mandatory_precedence"] == 1
    assert any(r["decision"] == "excluded_conflict" for r in result["audit"])


def test_deterministic_selection_never_uses_supplied_similarity_or_input_order():
    primary, neighbors, fillers = inputs()
    first = build_language(
        "es", primary, neighbors, fillers, VOCAB, target_size=3, seed="fixed", expected_primary=1
    )
    for raw in neighbors.values():
        raw["neighbors"]["casa"][0]["sim"] = 0.001
    second = build_language(
        "es",
        dict(reversed(list(primary.items()))),
        neighbors,
        list(reversed(fillers)),
        VOCAB,
        target_size=3,
        seed="fixed",
        expected_primary=1,
    )
    assert [e["normalized"] for e in first["entries"]] == [
        e["normalized"] for e in second["entries"]
    ]


@pytest.mark.parametrize("problem", ["capacity", "old_tokens", "neighbor_schema", "language"])
def test_invalid_inputs_fail(problem):
    primary, neighbors, fillers = inputs()
    if problem == "old_tokens":
        fillers[0]["token_ids"] = [0]
    elif problem == "neighbor_schema":
        neighbors["wave4"] = {"language": "es", "items": []}
    elif problem == "language":
        neighbors["wave4"]["language"] = "pt"
    with pytest.raises(ValueError):
        build_language(
            "es",
            primary,
            neighbors,
            fillers,
            VOCAB,
            target_size=10 if problem == "capacity" else 4,
            seed="fixed",
            expected_primary=1,
        )


def fixture_files(root):
    inventory = {"vocab_sha256": _sha(VOCAB_PATH), "json_files": {}}
    config = {
        "inventory_dir": "inventory",
        "vocab": str(VOCAB_PATH),
        "vocab_sha256": _sha(VOCAB_PATH),
        "filler_sha256": {},
        "expected_primary_counts": {"es": 1, "pt": 1},
        "target_size": 4,
        "seed": "fixed",
    }
    for lang in ("es", "pt"):
        primary, neighbors, fillers = inputs(lang)
        for wave in WAVES:
            base = root / wave / lang
            base.mkdir(parents=True)
            for kind, data in (
                ("keyword_bias_phoneme", primary),
                ("phonetic_neighbors_phoneme", neighbors),
            ):
                path = base / f"{lang}_{kind}.json"
                path.write_text(json.dumps(data[wave]))
                inventory["json_files"][f"{wave}/{lang}/{kind}"] = {"sha256": _sha(path)}
        old = root / (
            "outputs/en_es_pt_streaming_e2e_4k_formal100_v1/"
            f"capacity_{lang}/representative/size_4000/hotwords.jsonl"
        )
        old.parent.mkdir(parents=True)
        old.write_text("\n".join(json.dumps(r) for r in fillers))
        config["filler_sha256"][lang] = _sha(old)
    (root / "inventory").mkdir()
    report = root / "inventory/report.json"
    report.write_text(json.dumps(inventory))
    (root / "inventory/sha256.txt").write_text(f"{_sha(report)}  report.json\n")
    path = root / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_freeze_both_tables_read_only_inputs_checksums_and_no_overwrite(tmp_path):
    config = fixture_files(tmp_path)
    before = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    output = tmp_path / "release"
    result = freeze_wave_keywords(tmp_path, config, output)
    assert result["tables_created"] and not result["model_loaded"]
    assert before == {p: _sha(p) for p in before}
    for lang in ("es", "pt"):
        table = json.loads((output / lang / "keyword_bias_phoneme.json").read_text())
        assert len(table["keyword_sets"]["all_keywords"]) == 4
        assert len(table["keyword_phonemes"]) == 4
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(output / name) == digest
    with pytest.raises(FileExistsError):
        freeze_wave_keywords(tmp_path, config, output)
    other = tmp_path / "release2"
    freeze_wave_keywords(tmp_path, config, other)
    assert _sha(output / "es/hotwords.jsonl") == _sha(other / "es/hotwords.jsonl")


def test_changed_source_fails_before_any_table_written(tmp_path):
    config = fixture_files(tmp_path)
    (tmp_path / "wave4/pt/pt_keyword_bias_phoneme.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        freeze_wave_keywords(tmp_path, config, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_4000_exact_and_distinct_homophones_preserved():
    primary, neighbors, fillers = inputs()
    template = fillers[1]
    fillers = [{**template, "surface": f"item {i}", "normalized": f"item {i}"} for i in range(4401)]
    result = build_language(
        "es", primary, neighbors, fillers, VOCAB, target_size=4000, seed="fixed", expected_primary=1
    )
    assert result["summary"]["total"] == 4000
    assert result["summary"]["selected_by_source"] == {
        "primary": 1,
        "neighbor": 1,
        "old_table": 3998,
    }
    assert result["summary"]["homophone_groups_retained"] == 1
    assert len({e["normalized"] for e in result["entries"]}) == 4000


def test_neighbor_fallback_when_old_pool_too_small():
    primary, neighbors, _ = inputs()
    neighbors["wave1"]["neighbors"]["casa"] += [
        {"word": "Other", "phoneme": "o"},
        {"word": "Last", "phoneme": "a"},
    ]
    result = build_language(
        "es", primary, neighbors, [], VOCAB, target_size=4, seed="fixed", expected_primary=1
    )
    assert result["summary"]["selected_by_source"] == {"primary": 1, "neighbor": 3}
