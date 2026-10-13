from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_reviewed_retrieval import VOCAB, fixture, mocks, read, retrieval, write

from qwen_hotword.evaluation.joint_fillers import (
    balanced_select,
    build_joint_fillers,
    lexical_rejection,
    load_joint_fillers,
)
from qwen_hotword.evaluation.reviewed_targets import import_review
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.inference.reviewed_joint import joint_bundle
from qwen_hotword.inference.reviewed_retrieval import run_reviewed_retrieval
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha


def setup(tmp_path, monkeypatch):
    config, cfg = fixture(tmp_path)
    _, kw = mocks(monkeypatch)
    run_reviewed_retrieval(tmp_path, config, tmp_path / "new_run", **kw)
    build = {
        "reviewed_table": cfg["reviewed_table"],
        "reviewed_table_sha256": _sha(tmp_path / cfg["reviewed_table"]),
        "reference_run": "new_run",
        "reference_config": str(config),
        "target_size": 11,
        "seed": "fixed",
        "min_letters": 4,
        "min_phonemes": 4,
        "min_train_occurrences": 5,
        "vocab": str(VOCAB),
        "vocab_sha256": _sha(VOCAB),
        "basic_word_policy": "policy.json",
        "training_sources": {},
    }
    write(
        tmp_path / "policy.json",
        {
            "version": "v1",
            "categories": {"basic": "basic"},
            "languages": {
                lang: {"remove": {"basic": ["because"]}, "keep_overrides": []}
                for lang in ("en", "es", "pt")
            },
        },
    )
    for lang, words in {
        "en": ["planet", "garden", "orange", "marble"],
        "es": ["playas", "plátano", "camino", "persona"],
        "pt": ["praias", "coração", "estrada", "pessoa"],
    }.items():
        words += [
            "nothing",
            "deleteme",
            "because",
            "inn",
            "xyz",
            "brief",
            "ambiguous",
            "shared",
            "garbage",
            "rareword",
            "dash-word",
            "東京旅行",
        ]
        train = tmp_path / f"train_{lang}.jsonl"
        train.write_text(
            "".join(
                json.dumps(
                    {
                        "split": "train",
                        "language": "pt-BR" if lang == "pt" else lang,
                        "text": " ".join(w for w in words if n == 0 or w != "rareword"),
                    }
                )
                + "\n"
                for n in range(5)
            )
        )
        dictionary = tmp_path / f"{lang}.dict"
        dictionary.write_text(
            "".join(
                f"{w}\t" + ("a b" if w == "brief" else "☃" if w == "garbage" else "a b a b") + "\n"
                for w in words
            )
            + "ambiguous\tb a b a\n"
        )
        build["training_sources"][lang] = {
            "manifest": str(train),
            "manifest_sha256": _sha(train),
            "dictionary": str(dictionary),
            "dictionary_sha256": _sha(dictionary),
        }
    write(tmp_path / "build.json", build)
    return config, cfg, tmp_path / "build.json", build


def test_build_is_reproducible_balanced_and_preserves_targets(tmp_path, monkeypatch):
    _, cfg, path, _ = setup(tmp_path, monkeypatch)
    before = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "expanded"
    report = build_joint_fillers(tmp_path, path, out)
    assert report["unique_surfaces"] == 11
    assert report["fillers_by_language"] == {"en": 2, "es": 2, "pt": 2}
    assert report["selected_filler_transcript_overlap"] == 0
    assert sum(report["transcript_counts"].values()) == 22
    for stats in report["training_pools"].values():
        assert stats["eligible"] == 4
        assert stats["present_in_any_test_transcript"] == 1
        assert stats["original_target_or_deleted_target"] == 2
        assert stats["basic_word"] == 1
        assert stats["short_phoneme_sequence"] == 1
        assert stats["low_train_frequency"] == 1
        assert stats["missing_or_ambiguous_dictionary"] == 1
        assert stats["phoneme_oov_or_empty"] == 1
    table = read(out / "keyword_table.json")
    original = read(tmp_path / cfg["reviewed_table"])
    assert table["entries"][:5] == original["entries"]
    assert all(_sha(p) == digest for p, digest in before.items())
    build_joint_fillers(tmp_path, path, tmp_path / "expanded_again")
    assert (out / "keyword_table.json").read_bytes() == (
        tmp_path / "expanded_again/keyword_table.json"
    ).read_bytes()
    for line in (out / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(out / name) == digest
    with pytest.raises(FileExistsError):
        build_joint_fillers(tmp_path, path, out)


def test_4000_size_and_cross_language_collision_selection():
    pools = {
        lang: {str(n): {"surface": str(n), "language": lang} for n in range(5000)}
        for lang in ("en", "es", "pt")
    }
    selected = balanced_select(pools, 2642, "seed")
    assert len({r["surface"] for r in selected}) + 1358 == 4000
    assert [sum(r["language"] == lang for r in selected) for lang in pools] == [881, 881, 880]
    assert balanced_select(pools, 2642, "seed") == selected


@pytest.mark.parametrize(
    "word,reason",
    [
        ("coração", "present_in_any_test_transcript"),
        ("planet", "present_in_any_test_transcript"),
        ("because", "basic_word"),
        ("inn", "short_word"),
        ("bad-word", "not_plain_latin_single_word"),
        ("東京旅行", "not_plain_latin_single_word"),
        ("actual", ""),
    ],
)
def test_global_transcript_exclusion_and_short_words(word, reason):
    from qwen_hotword.inference.hotword_prompt import normalize_match_words

    # Unicode composition, case and punctuation match the inference reference policy.
    forbidden = set(normalize_match_words("“PLANET”, CORAC\u0327A\u0303O!"))
    assert lexical_rejection(word, forbidden, {"because"}, 4) == reason


@pytest.mark.parametrize("change", ["capacity", "train_split", "train_sha", "reference", "review"])
def test_invalid_input_has_no_partial_table(tmp_path, monkeypatch, change):
    _, cfg, path, build = setup(tmp_path, monkeypatch)
    if change == "capacity":
        build["target_size"] = 4000
    elif change in {"train_split", "train_sha"}:
        train = tmp_path / "train_en.jsonl"
        train.write_text(train.read_text().replace('"train"', '"test"'))
        if change == "train_split":
            build["training_sources"]["en"]["manifest_sha256"] = _sha(train)
    elif change == "reference":
        text = tmp_path / "audio/en_chile/transcripts.txt"
        text.write_text(text.read_text() + "unexpected\tadditional\n")
    else:
        table = tmp_path / cfg["reviewed_table"]
        data = read(table)
        data["entries"].pop()
        write(table, data)
    write(path, build)
    with pytest.raises(ValueError):
        build_joint_fillers(tmp_path, path, tmp_path / "expanded")
    assert not (tmp_path / "expanded").exists()


def test_expansion_runs_same_targets_22_exports_and_safe_resume(tmp_path, monkeypatch):
    config, cfg, path, _ = setup(tmp_path, monkeypatch)
    out = tmp_path / "expanded"
    build_joint_fillers(tmp_path, path, out)
    cfg.update(
        expanded_table="expanded/keyword_table.json",
        reviewed_baseline="new_run",
        output="expanded_run",
    )
    config = tmp_path / "expanded_run.json"
    write(config, cfg)
    calls = []

    def factory(**kwargs):
        assert len({e.normalized for e in kwargs["hotwords"]}) == 11
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        calls.append(1)
        return retrieval(detector.hotwords, waveform == b"negative")

    kw = {"detector_factory": factory, "retrieve_function": retrieve}
    audit = run_reviewed_retrieval(
        tmp_path, config, tmp_path / "expanded_run", audit_only=True, **kw
    )
    assert audit["active_unique_surfaces_per_audio"] == 11
    assert audit["filler_entries_added"] == 6 and not calls
    result = run_reviewed_retrieval(tmp_path, config, tmp_path / "expanded_run", **kw)
    assert result["sample_count"] == 22 and result["delivery_file_count"] == 22
    for metrics in result["groups"].values():
        for k in ("top5", "top7"):
            assert metrics["current"][k]["expected_target_sample_pairs"] == 3
            assert metrics["reviewed_baseline_same_targets"][k]["expected_target_sample_pairs"] == 3
    assert all("reviewed_baseline_same_targets" in m for m in result["by_language"].values())
    run_reviewed_retrieval(tmp_path, config, tmp_path / "expanded_run", resume=True, **kw)
    assert len(calls) == 22
    # Even a harmless edit of the sealed expansion cannot reuse sample shards.
    with (out / "keyword_table.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="SHA256"):
        run_reviewed_retrieval(tmp_path, config, tmp_path / "expanded_run", resume=True, **kw)
    assert len(calls) == 22


def test_loader_rechecks_filler_labels_and_no_target_changes(tmp_path, monkeypatch):
    _, cfg, path, _ = setup(tmp_path, monkeypatch)
    out = tmp_path / "expanded"
    build_joint_fillers(tmp_path, path, out)
    reviewed = tmp_path / cfg["reviewed_table"]
    vocab = load_phoneme_vocab(VOCAB)
    bundles, _, _ = import_review(reviewed, vocab, {})
    expanded, _ = load_joint_fillers(out / "keyword_table.json", reviewed, bundles, vocab, {})
    assert len(joint_bundle(expanded)[0].entries) == 11
    # Resealing an unsupported manual target edit does not bypass semantic validation.
    table = read(out / "keyword_table.json")
    table["entries"][0]["variants"][0]["pronunciation"] = "b b b b"
    write(out / "keyword_table.json", table)
    (out / "sha256.txt").unlink()  # Only this disposable tampering-test fixture.
    seal(out)
    with pytest.raises(ValueError, match="identity or schema"):
        load_joint_fillers(out / "keyword_table.json", reviewed, bundles, vocab, {})
