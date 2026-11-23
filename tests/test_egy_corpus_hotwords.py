from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.evaluation.egy_corpus_hotwords import (
    assert_disjoint,
    build_table,
    entity_surfaces,
    expected_ids,
    phrase_present,
    select_audio,
    summarize,
)

spec = importlib.util.spec_from_file_location(
    "egy_corpus_runner",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_egy_corpus_hotwords.py",
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row(identifier: str, group: str = "open_source", text: str = "زار القاهرة") -> dict[str, Any]:
    return dict(
        id=identifier,
        source_group=group,
        text=text,
        source="source.json",
        source_row=1,
        split="test",
        language="ar",
        audio_path="/audio/" + identifier,
        audio_sha256=identifier,
        duration_seconds=10.0,
        recording_key=identifier,
        grouping_basis="recognized_recording",
    )


def test_selection_uses_duration_not_words_and_is_deterministic() -> None:
    rows = [row(str(i)) for i in range(20)]
    rows += [row("g" + str(i), "gaoshan_train", "") for i in range(2)]
    rows += [row("c" + str(i), "coarse_train", "") for i in range(12)]
    selected, report = select_audio(rows)
    assert [r["id"] for r in selected] == [r["id"] for r in select_audio(rows[::-1])[0]]
    assert len(selected) == 20
    assert [
        report["by_source"][g]["samples"] for g in ("open_source", "gaoshan_train", "coarse_train")
    ] == [16, 2, 2]
    assert not report["selection_uses_keywords"]
    changed = [dict(r, text="完全不同") for r in rows]
    assert [r["id"] for r in selected] == [r["id"] for r in select_audio(changed)[0]]


@pytest.mark.parametrize(
    "change",
    [dict(audio_sha256="a"), dict(audio_path="/audio/a"), dict(recording_key="a"), dict(id="a")],
)
def test_overlap_refused(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint([row("a")], [dict(row("b"), **change)])


def test_unknown_recording_keys_do_not_falsely_merge_files() -> None:
    assert_disjoint(
        [dict(row("a"), grouping_basis="file_only", recording_key="")],
        [dict(row("b"), grouping_basis="file_only", recording_key="")],
    )


def test_ner_uses_real_span_and_rejects_partial_word() -> None:
    text = "زار القاهرة ومحمد"
    predictions = [
        dict(start=4, end=11, entity_group="LOC", score=0.98),
        dict(start=13, end=17, entity_group="PERS", score=0.99),
    ]
    assert [r["surface"] for r in entity_surfaces(text, predictions)] == ["القاهرة"]
    assert not entity_surfaces(text, [dict(start=4, end=11, entity_group="LOC", score=0.89)])


def test_actual_new_entity_and_only_observed_terms_and_distractors() -> None:
    evaluation = [row("a", text="زار أسوان لعلاج مرض السكري")]
    training = [row("b", text="أسوان والإسكندرية")]
    extracted = {
        "a": [dict(surface="أسوان", category="LOC", origin="NER")],
        "b": [
            dict(surface="أسوان", category="LOC", origin="NER"),
            dict(surface="والإسكندرية", category="LOC", origin="NER"),
        ],
    }
    terms = [dict(surface=s, category="医疗", gloss_zh="术语") for s in ["مرض السكري", "ضغط الدم"]]
    table = build_table(evaluation, training, extracted, terms)
    assert {t["surface"]: t["role"] for t in table} == {
        "أسوان": "target",
        "مرض السكري": "target",
        "والإسكندرية": "distractor",
    }
    assert all(t["examples"] for t in table)
    assert not phrase_present("والقاهرة", "القاهرة")


def test_nested_phrases_and_repeated_mentions_count_once() -> None:
    table = [dict(id="a", surface="بنك مصر"), dict(id="b", surface="مصر")]
    assert expected_ids("بنك مصر بنك مصر", table) == {"a"}
    assert expected_ids("بنك مصر في مصر", table) == {"a", "b"}
    assert not expected_ids("مصري", table)


def test_metrics_keep_inactive_gold_negatives_and_capacity() -> None:
    matches = [
        dict(hotword_id=str(i), score=0.9, edit_ratio=0.1, posterior_confidence=0.9)
        for i in range(7)
    ]
    matches[0]["posterior_confidence"] = 0.1
    rows = [
        dict(
            expected_ids=[str(i) for i in range(9)],
            raw_ranked_matches=matches,
            reference_present_ids=[str(i) for i in range(9)],
        ),
        dict(expected_ids=[], raw_ranked_matches=matches[:1], reference_present_ids=[]),
    ]
    report = summarize(rows)
    assert report["expected_target_sample_pairs"] == 9
    assert report["negative_sample_count"] == 1
    assert report["raw_ranked"]["top7"]["target_recall"] == 7 / 9
    assert report["operating"]["top7"]["target_recall"] == 6 / 9
    assert report["raw_ranked"]["top5"]["capacity_upper_bound"] == 5 / 9
    assert report["raw_ranked"]["top7"]["negative_audio_with_candidates"] == 1


def test_seal_and_resume_reject_changed_identity(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    runner.sealed(path, {"a": 1})
    runner.sealed(path, {"a": 1})
    with pytest.raises(ValueError, match="identity changed"):
        runner.sealed(path, {"a": 2})
    runner.write_json(path, {"identity": "old"})
    with pytest.raises(ValueError, match="identity changed"):
        runner.result_file(path, "new")


def test_whole_audio_retrieval_never_receives_gold_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qwen_hotword.inference.external_keyword_retrieval as backend

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audiofixture")
    sample = dict(row("a"), audio_path=str(audio), audio_sha256=runner.sha(audio))
    table = [dict(id="cairo", surface="القاهرة")]
    for name, value in [
        ("selected_audio.json", [sample]),
        ("table.json", table),
        ("contract.json", {"sealed": True}),
    ]:
        runner.write_json(tmp_path / name, value)
    calls = []
    monkeypatch.setattr(runner, "load_detector", lambda *args: "detector")
    monkeypatch.setattr(
        backend, "_load_audio", lambda path: ("entire_audio", {"duration_seconds": 10})
    )

    def infer(detector: Any, waveform: Any, vocab: Any, **kwargs: Any) -> dict[str, Any]:
        assert detector == "detector" and waveform == "entire_audio"
        assert kwargs == {"saved_raw_rank_depth": 64}
        calls.append(1)
        return {"raw_ranked_matches": []}

    monkeypatch.setattr(backend, "_retrieve_waveform", infer)
    vocab = Path("configs/phonemes/en_ar_native_ipa_vocab.v1.1.json")
    first = runner.retrieve(tmp_path, tmp_path, tmp_path, vocab)
    second = runner.retrieve(tmp_path, tmp_path, tmp_path, vocab)
    assert first == second and calls == [1]
    assert first[0]["expected_ids"] == ["cairo"]


def test_pinned_ner_official_person_label() -> None:
    assert entity_surfaces("محمد", [dict(start=0, end=4, entity_group="PERS", score=0.99)])
    config = json.loads(Path("configs/egy_corpus_retrieval.workzone.json").read_text())
    assert len(config["ner_revision"]) == 40
    assert all(len(s) == 64 for s in config["ner_files"].values())


def test_reject_duplicate_test_audio() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        select_audio([row("a"), row("a")])


def test_download_only_upgrade_preserves_old_contract(tmp_path: Path) -> None:
    script = str(Path(runner.__file__))
    old_sha = "15158ba7292ba8ff4799402ae8f630249df7177b721aa9c7c9f63e71dead4ee2"
    old = {"inputs": {script: old_sha, "model": "unchanged"}, "policy": "fixed"}
    current = {"inputs": {script: "new", "model": "unchanged"}, "policy": "fixed"}
    path = tmp_path / "contract.json"
    runner.write_json(path, old)
    runner.seal_run_contract(path, current)
    assert runner.read(path) == old
    assert (tmp_path / "download_upgrade.json").exists()
    current["policy"] = "changed"
    with pytest.raises(ValueError, match="identity changed"):
        runner.seal_run_contract(path, current)
