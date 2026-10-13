from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import test_reviewed_retrieval as fixtures
from test_joint_fillers import setup

from qwen_hotword.evaluation.joint_diagnostic_inputs import check_pair, load_run, validate_row
from qwen_hotword.evaluation.joint_diagnostics import (
    _profile,
    _update,
    diagnose_joint_runs,
    gate_failures,
    target_state,
    write_diagnostics,
)
from qwen_hotword.evaluation.joint_fillers import build_joint_fillers
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.reviewed_retrieval import run_reviewed_retrieval
from qwen_hotword.inference.wave_retrieval import _row_digest
from qwen_hotword.training.spanish_capacity import _sha

GATE = {
    "threshold": 0.75,
    "maximum_edit_ratio": 0.35,
    "minimum_posterior_confidence": 0.5,
    "minimum_top1_margin": 0.0,
}


def match(identifier, score=0.9, edit=0.0, posterior=0.9):
    return {
        "hotword_id": identifier,
        "matched_variant_id": identifier,
        "score": score,
        "edit_ratio": edit,
        "posterior_confidence": posterior,
    }


def replay(raw, expected, k=7):
    return {
        "expected_hotword_ids": expected,
        f"top{k}_selected_hotword_ids": [m["hotword_id"] for m in _select(raw, gate=GATE, top_k=k)],
        f"raw_top{k}_hotword_ids": [m["hotword_id"] for m in raw[:k]],
    }


def test_disjoint_loss_stages_and_gate_backfill():
    raw = [match("gate", 0.99, 0.6, 0.1)] + [match(str(n)) for n in range(8)]
    selected = {m["hotword_id"] for m in _select(raw, gate=GATE, top_k=7)}
    assert "6" in selected  # Raw rank 8 is allowed to backfill after a failed gate.
    row = {"raw_ranked_matches": raw}
    assert target_state("6", row, selected, GATE) == "selected"
    assert target_state("7", row, selected, GATE) == "qualified_below_topk"
    assert target_state("missing", row, selected, GATE) == "not_in_saved_shortlist"
    assert target_state("gate", row, selected, GATE) == "gate:edit_ratio+posterior"
    assert gate_failures(match("all", 0.1, 0.9, 0.1), GATE) == ["score", "edit_ratio", "posterior"]
    assert not gate_failures(match("boundary", 0.75, 0.35, 0.5), GATE)


def test_paired_losses_gains_origins_and_saved_list_counterfactual():
    # One formerly selected target is displaced; another target is newly recalled.
    # Net recall stays flat, so counts must not report zero changed targets.
    old_raw = [match("lost"), match("stays"), match("old_bad")] + [
        match(f"shared_bad{n}") for n in range(4)
    ]
    new_raw = (
        [match("filler", 0.99), match("gained", 0.98), match("stays", 0.97)]
        + [match(f"shared_bad{n}") for n in range(4)]
        + [match("lost", 0.8)]
    )
    all_ids = {m["hotword_id"] for m in old_raw + new_raw}
    run = {
        "index": {
            i: SimpleNamespace(language="pt-BR" if i == "filler" else "en", token_ids=(1, 2, 3, 4))
            for i in all_ids
        }
    }
    targets = {"lost", "gained", "stays"}
    profile = _profile()
    lost, gained, added = _update(
        profile,
        {"raw_ranked_matches": old_raw},
        {"raw_ranked_matches": new_raw},
        replay(old_raw, sorted(targets)),
        replay(new_raw, sorted(targets)),
        targets,
        run,
        {"filler"},
        GATE,
        7,
    )
    assert lost == {"lost"} and gained == {"gained"} and added == {"filler"}
    assert profile["old"]["gated_target_hits"] == profile["new"]["gated_target_hits"] == 2
    assert profile["lost_target_destinations"] == {"qualified_below_topk": 1}
    assert profile["paired"]["lost_targets_recovered_removing_fillers_from_saved_list"] == 1
    assert profile["paired"]["removed_absent_candidates"] == 1
    assert profile["added_absent_by_origin_language"] == {"filler::pt": 1}
    assert profile["current_absent_by_origin_language"] == {
        "reviewed_target::en": 4,
        "filler::pt": 1,
    }
    assert profile["added_absent_phone_lengths"] == {"4-5": 1}
    assert profile["added_absent_edit_ratio_zero"] == 1


def paired_runs(tmp_path, monkeypatch):
    original_retrieval = fixtures.retrieval

    def with_decoded(entries, negative=False):
        result = original_retrieval(entries, negative)
        result.update(
            decoded_token_ids=[1, 2], decoded_tokens=["a", "b"], decoded_confidences=[0.9, 0.8]
        )
        for m in result["raw_ranked_matches"]:
            m.update(decoded_start=0, decoded_end=2, start_step=0, end_step=4)
        return result

    monkeypatch.setattr(fixtures, "retrieval", with_decoded)
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
        retrieve_function=lambda detector, waveform, vocab: with_decoded(
            detector.hotwords, waveform == b"negative"
        ),
    )
    return tmp_path / "new_run", tmp_path / "expanded_run"


def test_sealed_runs_metrics_reproduce_without_audio_or_model(tmp_path, monkeypatch):
    old, new = paired_runs(tmp_path, monkeypatch)
    hashes = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics cannot open audio or load a model")

    monkeypatch.setattr(fixtures.external, "_load_audio", forbidden)
    report, examples = diagnose_joint_runs(old, new)
    assert report["source_metrics_reproduced"]
    assert (report["old_surfaces"], report["new_surfaces"], report["new_fillers"]) == (5, 11, 6)
    assert sum(v["top7"]["new"]["sample_count"] for v in report["by_language"].values()) == 22
    assert report["ctc_sequence_comparison"]["en"]["identical_token_sequences"] == 6
    assert any(e["category"] == "new_absent_filler" for e in examples)
    assert all(e["word_after"]["ctc_context"] == ["a", "b"] for e in examples)
    output = tmp_path / "diagnostic"
    write_diagnostics(report, examples, output)
    assert all(_sha(p) == digest for p, digest in hashes.items())
    for line in (output / "sha256.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert _sha(output / name) == digest
    with pytest.raises(FileExistsError):
        write_diagnostics(report, examples, output)
    with pytest.raises(ValueError, match="outside source"):
        write_diagnostics(report, examples, new / "diagnostic")


@pytest.mark.parametrize("change", ["targets", "audio", "gate", "code", "labels"])
def test_mismatched_pair_is_rejected(tmp_path, monkeypatch, change):
    old_dir, new_dir = paired_runs(tmp_path, monkeypatch)
    old, new = load_run(old_dir, {}), load_run(new_dir, {})
    group = new["plan"]["groups"]["en_chile"]
    if change == "targets":
        group["target_ids"].pop()
    elif change == "audio":
        path = group["records"][0]["audio_path"]
        new["plan"]["inputs"][path]["sha256"] = "changed"
    elif change == "gate":
        group["config"]["threshold"] = 0.2
    elif change == "code":
        new["plan"]["code_sha256"]["hotwords/scoring.py"] = "changed"
    else:
        new["table"]["entries"][0]["variants"][0]["pronunciation"] = "b b b"
    with pytest.raises(ValueError):
        check_pair(old, new)


def test_shortlist_hash_and_source_summary_fail_closed(tmp_path, monkeypatch):
    old, new = paired_runs(tmp_path, monkeypatch)
    run = load_run(new, {})
    group = run["plan"]["groups"]["en_chile"]
    import json

    path = new / "languages/en/en_chile/retrieval_details.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    bad = deepcopy(row)
    bad["shortlist_candidates"] += 1
    bad["row_sha256"] = _row_digest(bad)
    with pytest.raises(ValueError, match="incomplete saved shortlist"):
        validate_row(bad, group["records"][0], run, "en_chile")
    bad = deepcopy(row)
    bad["reference_text"] += "changed"
    with pytest.raises(ValueError, match="identity/checksum"):
        validate_row(bad, group["records"][0], run, "en_chile")
    saved = fixtures.read(new / "report.json")
    saved["groups"]["en_chile"]["current"]["top7"]["gated_target_hits"] += 1
    fixtures.write(new / "report.json", saved)
    with pytest.raises(ValueError, match="SHA256"):
        diagnose_joint_runs(old, new)
    (new / "sha256.txt").unlink()  # Disposable test fixture only: a forged summary still fails.
    seal(new)
    with pytest.raises(ValueError, match="counts do not reproduce"):
        diagnose_joint_runs(old, new)
