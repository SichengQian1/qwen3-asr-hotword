from __future__ import annotations

from copy import deepcopy
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_reviewed_retrieval as fixtures
from test_joint_fillers import setup

from qwen_hotword.evaluation.english_failure_cases import (
    collect_cases,
    keep_example,
    stratum,
    write_cases,
)
from qwen_hotword.evaluation.joint_fillers import build_joint_fillers
from qwen_hotword.evaluation.retrieval_case_trace import (
    decoded_sequence,
    make_index,
    scoring_config,
    trace_case,
)
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.hotwords.scoring import profile_anchor_guided_decoded_hotwords
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.reviewed_retrieval import run_reviewed_retrieval
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha

GATE = {
    "threshold": 0.75,
    "minimum_phonemes": 1,
    "posterior_weight": 0.25,
    "maximum_edit_ratio": 0.35,
    "minimum_posterior_confidence": 0.5,
    "minimum_top1_margin": 0.0,
}
ORIGINAL_RETRIEVAL = fixtures.retrieval
CONFIG = {
    "retrieval": {
        "anchor_ngram_sizes": [2, 3, 4],
        "anchors_per_entry": 24,
        "anchor_offset_tolerance": 1,
        "shortlist_size": 64,
        "rerank_start_radius": 2,
    }
}


def entry(identifier, ids):
    return HotwordEntry(
        hotword_id=identifier,
        surface=identifier,
        normalized=identifier,
        language="en",
        words=(identifier,),
        pronunciation=" ".join(map(str, ids)),
        token_ids=tuple(ids),
        phoneme_tokens=tuple(map(str, ids)),
        source="unit_test",
        validation_occurrences=0,
    )


def saved_row(entries, ids, *, config=CONFIG, phones=None):
    run = {
        "variants": tuple(entries),
        "index": {e.hotword_id: e for e in entries},
        "mapping": {e.hotword_id: e for e in entries},
    }
    index = make_index(run, config)
    row = {
        "decoded_token_ids": ids,
        "decoded_tokens": phones or list(map(str, ids)),
        "decoded_confidences": [0.9] * len(ids),
    }
    query = index.query(
        ids,
        confidences=row["decoded_confidences"],
        maximum_candidates=config["retrieval"]["shortlist_size"],
    )
    matches = profile_anchor_guided_decoded_hotwords(
        decoded_sequence(row),
        effective_time_steps=max(1, len(ids)),
        hotwords=tuple(run["index"][c.hotword_id] for c in query.candidates),
        start_hints={c.hotword_id: c.best_offset for c in query.candidates},
        maximum_start_delta=config["retrieval"]["rerank_start_radius"],
        config=scoring_config(GATE),
    ).result
    row.update(
        raw_ranked_matches=[m.to_dict() for m in matches.ranked_matches],
        selected_matches=[m.to_dict() for m in matches.selected_matches],
        suppressed_reason=matches.suppressed_reason,
        shortlist_candidates=len(query.candidates),
        postings_visited=query.postings_visited,
        timing={"pure_retrieval_seconds": 0.0, "model_retrieval_seconds": 0.0},
    )
    return run, index, collapse_variants(row, tuple(entries), run["mapping"], GATE)


def test_sampling_is_order_independent_and_distinct_audio_with_shortfall():
    items = [
        {"sample_id": s, "selection_key": k}
        for s, k in (("same", "9"), ("same", "1"), ("second", "2"), ("third", "3"))
    ]
    for order in permutations(items):
        slots = []
        for item in order:
            keep_example(slots, item, 2)
        assert [(r["sample_id"], r["selection_key"]) for r in slots] == [
            ("same", "1"),
            ("second", "2"),
        ]
    assert stratum("gate:score") == "gate_score_only"
    assert stratum("gate:score+posterior") == "gate_other"
    assert stratum("selected") == "selected_control"
    assert stratum("qualified_below_topk") is None


def test_trace_distinguishes_no_anchor_pruning_exact_single_phone_and_gate():
    entries = [
        entry("long", [1, 2, 3]),
        entry("short", [1, 2]),
        entry("single", [3]),
        entry("absent", [8, 9]),
    ]
    config = deepcopy(CONFIG)
    config["retrieval"]["shortlist_size"] = 1
    run, index, row = saved_row(entries, [1, 2, 3], config=config)
    traces = {
        e.hotword_id: trace_case(run, row, e.hotword_id, config, GATE, index) for e in entries
    }
    assert traces["absent"]["variants"][0]["anchor_state"] == "no_anchor_and_no_exact_hit"
    exact = [traces[k]["variants"][0] for k in ("long", "short", "single")]
    assert sorted(v["anchor_rank_before_cap"] for v in exact) == [1, 2, 3]
    assert sum(v["anchor_state"] == "outside_shortlist_cap" for v in exact) == 2
    assert traces["single"]["variants"][0]["anchor_evidence"]["exact_match"]
    # A high-confidence 3/4-phoneme match fails only the score gate.
    run, index, row = saved_row([entry("target", [1, 2, 3, 4])], [1, 2, 3])
    trace = trace_case(run, row, "target", CONFIG, GATE, index)
    m = trace["saved_surface_match"]
    assert m["failed_gates"] == ["score"]
    assert m["score"] == pytest.approx(0.73125)
    assert m["edits_table_to_ctc"] == [
        {
            "operation": "delete",
            "query_index": 3,
            "span_index": 3,
            "query_phone": "4",
            "ctc_phone": None,
        }
    ]
    assert "start_step" not in m  # Synthetic ordinal steps cannot masquerade as audio timing.
    assert trace["variants"][0]["diagnostic_full_start_search"]["score"] == m["score"]


def test_trace_preserves_all_pronunciations_and_rejects_changed_saved_score():
    entries = [entry("v1", [1, 2]), entry("v2", [3, 4])]
    run, index, row = saved_row(entries, [1, 2])
    run["mapping"]["v2"] = run["mapping"]["v1"]
    trace = trace_case(run, row, "v1", CONFIG, GATE, index)
    assert len(trace["variants"]) == 2
    assert trace["variants"][1]["saved_guided_match"] is None
    row["raw_variant_ranked_matches"][0]["score"] -= 0.01
    with pytest.raises(ValueError, match="does not reproduce"):
        trace_case(run, row, "v1", CONFIG, GATE, index)


def real_retrieval(entries, negative=False):
    vocab = load_phoneme_vocab(fixtures.VOCAB)
    ids = [] if negative else [vocab.token_to_id[p] for p in ("a", "b")]
    _, _, collapsed = saved_row(entries, ids, phones=[vocab.tokens[i] for i in ids])
    # The actual runner does its own joint surface collapse.
    collapsed["raw_ranked_matches"] = collapsed.pop("raw_variant_ranked_matches")
    collapsed["shortlist_candidates"] = collapsed.pop("variant_shortlist_candidates")
    collapsed["selected_matches"] = collapsed["raw_ranked_matches"][:5]
    collapsed["timing"] = ORIGINAL_RETRIEVAL(entries, negative)["timing"]
    return collapsed


def real_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "retrieval", real_retrieval)
    _, cfg, build_path, _ = setup(tmp_path, monkeypatch)
    build_joint_fillers(tmp_path, build_path, tmp_path / "expanded")
    cfg.update(expanded_table="expanded/keyword_table.json", reviewed_baseline="new_run")
    config = tmp_path / "expanded_run.json"
    fixtures.write(config, cfg)
    run_reviewed_retrieval(
        tmp_path,
        config,
        tmp_path / "expanded_run",
        detector_factory=lambda **kw: SimpleNamespace(**kw),
        retrieve_function=lambda detector, waveform, vocab: real_retrieval(
            detector.hotwords, waveform == b"negative"
        ),
    )
    return tmp_path / "new_run", tmp_path / "expanded_run"


def test_sealed_end_to_end_and_compact_bundle_without_audio_or_models(tmp_path, monkeypatch):
    old, new = real_runs(tmp_path, monkeypatch)
    hashes = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("no audio or models in case extraction")

    monkeypatch.setattr(fixtures.external, "_load_audio", forbidden)
    report, cases = collect_cases(tmp_path, old, new)
    assert report["source_metrics_reproduced"] and report["example_count"] == 3
    assert all(c["new_state"] == "selected" for c in cases)
    assert all(c["old"]["saved_ranking_and_scores_reproduced"] for c in cases)
    assert report["by_dataset"]["en_chile"]["strata"]["not_in_saved_shortlist"]["shortfall"] == 2
    output = tmp_path / "cases"
    write_cases(report, cases, output)
    import zipfile

    with zipfile.ZipFile(output / "return_bundle.zip") as archive:
        assert set(archive.namelist()) == {
            "summary.json",
            "examples.json",
            "examples.md",
            "target_terms.txt",
            "sha256.txt",
        }
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(output / name) == digest
    assert all(_sha(p) == digest for p, digest in hashes.items())
    with pytest.raises(FileExistsError):
        write_cases(report, cases, output)
    with pytest.raises(ValueError, match="outside source"):
        write_cases(report, cases, new / "cases")
    report["inputs"][str(new / "report.json")]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="source changed"):
        write_cases(report, cases, tmp_path / "not_written")
    assert not (tmp_path / "not_written").exists()


def test_live_replay_code_must_match_sealed_run(tmp_path):
    from qwen_hotword.evaluation.english_failure_cases import REPLAY_CODE, check_local_code

    package = Path(fixtures.__file__).parents[1] / "src/qwen_hotword"
    run = {"plan": {"code_sha256": {p: _sha(package / p) for p in REPLAY_CODE}}}
    check_local_code(run, {})
    run["plan"]["code_sha256"]["hotwords/scoring.py"] = "wrong"
    with pytest.raises(ValueError, match="SHA256"):
        check_local_code(run, {})
