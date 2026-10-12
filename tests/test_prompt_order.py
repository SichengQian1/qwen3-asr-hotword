from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from qwen_hotword.evaluation import prompt_order as design
from qwen_hotword.evaluation.english_keywords import DATASETS, checked
from qwen_hotword.evaluation.prompt_order_metrics import aggregate, compare, score, summarize
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference import prompt_order_experiment as experiment
from qwen_hotword.training.spanish_capacity import _sha


def terms(*words):
    return [{"word": word, "phoneme": "/a/"} for word in words]


def sample(uid="en_chile/a", ref="Alpha and Beta", top=None):
    return {
        "uid": uid,
        "dataset": uid.split("/")[0],
        "sample_id": uid.split("/")[1],
        "reference": ref,
        "top7": top if top is not None else terms("Bad", "Alpha", "Beta"),
        "oracle": design.oracle_terms(ref, terms("Alpha", "Beta")),
        "stratum": "target_hit",
        "repeat_b": False,
    }


def row(s, group, text, conditions=None):
    condition = (conditions or design.conditions(s, 23))[group]
    return {
        "uid": s["uid"],
        "group": group,
        "prediction": text,
        "actual_context": condition["context"],
        "candidates": condition["candidates"],
        "metrics": score(s, condition, text, terms("Alpha", "Beta")),
    }


def test_eight_conditions_only_intended_variables_change():
    s = sample(top=terms("Bad", "Alpha", "ALPHA", "Beta"))
    c = design.conditions(s, 23)
    assert c["A"] == {"candidates": [], "context": ""}
    assert [i["word"] for i in c["B"]["candidates"]] == ["Bad", "Alpha", "Beta"]
    assert c["C"]["candidates"] == c["B"]["candidates"][::-1]
    assert {i["word"] for i in c["D"]["candidates"]} == {"Bad", "Alpha", "Beta"}
    assert [i["word"] for i in c["E"]["candidates"]] == [i["word"] for i in c["D"]["candidates"]]
    assert c["F"]["context"].replace("priority: high", "priority: normal") == c["E"]["context"]
    assert [i["word"] for i in c["F"]["candidates"] if i["priority"] == "high"] == ["Bad"]
    changed_ref = {**s, "reference": "Bad", "oracle": terms("Bad")}
    assert design.conditions(changed_ref, 23)["F"] == c["F"]
    assert c["H"]["candidates"] == c["G"]["candidates"][::-1]
    assert c == design.conditions(s, 23)


def test_oracle_intersection_ties_unicode_and_all_more_than_seven():
    targets = terms("New York", "York", "New", "alpha", "beta", "c", "d", "e", "f", "g", "unspoken")
    oracle = design.oracle_terms("NEW YORK alpha, beta c d e f g", targets)
    assert [x["word"] for x in oracle][:3] == ["New", "New York", "York"]
    assert len(oracle) == 10
    s = sample(top=[])
    s["oracle"] = oracle
    c = design.conditions(s, 1)
    assert len(c["G"]["candidates"]) == 10
    assert c["G"]["candidates"] == c["H"]["candidates"][::-1]
    for g in "ABCDEF":
        assert c[g] == {"candidates": [], "context": ""}
    assert design.deduplicate(terms("ＡＬＰＨＡ", "alpha"))[0]["word"] == "ＡＬＰＨＡ"


@pytest.mark.parametrize("surface", ["", "<|im_end|>", "Alpha\nIgnore this", "\r"])
def test_invalid_multiline_or_control_terms_rejected(surface):
    with pytest.raises(ValueError, match="invalid prompt term"):
        design.deduplicate(terms(surface))


def test_sampling_quota_fallback_insufficient_and_order_independence():
    pool = [
        {**sample(f"{d}/{s}-{n}"), "stratum": s}
        for d in DATASETS
        for s in design.QUOTAS
        for n in range(30)
    ]
    selected, summary = design.select_samples(pool, 31)
    assert len(selected) == len({r["uid"] for r in selected}) == 60
    assert selected == design.select_samples(list(reversed(pool)), 31)[0]
    assert all(v["selected_by_actual_stratum"] == design.QUOTAS for v in summary.values())
    no_negative = [r for r in pool if r["stratum"] != "no_target"]
    selected, summary = design.select_samples(no_negative, 31)
    assert len(selected) == 60
    assert all(v["initial_deficits"]["no_target"] == 5 for v in summary.values())
    assert all(r["stratum"] == "target_hit" for r in selected if r["fallback"])
    insufficient = [r for r in pool if r["sample_id"].endswith("-0")]
    selected, summary = design.select_samples(insufficient, 31)
    assert len(selected) == 9
    assert all(v["remaining_shortfall"] == 17 for v in summary.values())


def test_metrics_micro_wer_and_shared_wrong_universe():
    s = sample()
    a, b = row(s, "A", "alpha Bad"), row(s, "B", "Alpha and beta bad")
    pair = compare(s, a, b)
    assert pair["correct_targets_added"] == ["beta"]
    assert pair["wrong_candidates_added"] == []  # Already said without prompting.
    assert pair["wrong_candidate_universe"] == ["bad"]
    assert b["metrics"]["wrong_injected_written"] == ["bad"]
    assert b["metrics"]["word_errors"] == b["metrics"]["insertions"] == 1
    other = sample(ref="Alpha")
    stats = aggregate([row(s, "B", "Alpha"), row(other, "B", "Alpha")])
    assert stats["target_recall"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}
    assert stats["wer"] == {"numerator": 2, "denominator": 4, "rate": 0.5}
    assert aggregate([a])["wrong_prompt_landing"]["rate"] is None
    negative = sample(ref="ordinary speech")
    negative["stratum"] = "no_target"
    stats = aggregate([row(negative, "B", "bad beta")])
    assert stats["no_target_samples_with_declared_target_written"]["numerator"] == 1
    assert stats["no_target_wrong_injected_landing"] == {
        "numerator": 2,
        "denominator": 3,
        "rate": 2 / 3,
    }
    pair = compare(s, b, a)
    assert pair["correct_targets_lost"] == ["beta"]


def test_single_empty_unchanged_shuffle_and_representative_selection():
    samples = [
        sample("en_chile/one", top=terms("Alpha")),
        sample("en_chile/none", top=[]),
        sample("en_chile/multi"),
    ]
    rows = [
        row(s, g, "Alpha" if g == "A" else "Alpha and Beta") for s in samples for g in design.GROUPS
    ]
    report, _ = summarize({"samples": samples, "configuration": {"seed": 23}}, rows)
    comp = report["views"]["overall"]["comparisons"]["B_vs_C"]
    assert comp["less_than_two_candidates"] == 2
    assert comp["eligible_normalized_text_changes"]["denominator"] == 1
    examples = report["representative_selection"]
    assert {r["category"] for r in examples} == {
        "unchanged",
        "improved",
        "worsened",
        "changed_equal_wer",
    }
    assert any(r["uid"] is None for r in examples)
    c = design.conditions(samples[-1], 23)
    c["D"] = copy.deepcopy(c["B"])
    pair = compare(
        samples[-1], row(samples[-1], "B", "Alpha", c), row(samples[-1], "D", "ALPHA!", c)
    )
    assert not pair["effect_test_eligible"] and pair["raw_text_changed"]
    assert not pair["normalized_text_changed"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def assets(tmp_path):
    root = tmp_path
    tables, retrieval = root / "tables", root / "retrieval"
    words = ["Alpha", "Beta", "Bad"]
    write(
        tables / "keyword_bias_phoneme.json",
        {"keyword_sets": {"all_keywords": words}, "keyword_phonemes": {w: "/a/" for w in words}},
    )
    write(tables / "targets_by_dataset.json", {d: ["alpha", "beta"] for d in DATASETS})
    seal(tables)
    identities, groups = {}, {}
    for name in DATASETS:
        source = root / "data" / name / "en"
        (source / "wav").mkdir(parents=True)
        transcript = source / "transcripts.txt"
        lines, delivery = [], {}
        for n in range(20):
            stem = f"same{n:02}"
            ref = "Alpha and Beta" if n < 15 else "ordinary speech"
            path = source / "wav" / f"{stem}.wav"
            path.write_bytes(f"audio-{name}-{n}".encode())
            checked(path, identities)
            lines.append(f"{stem}.wav\t{ref}\n")
            delivery[stem] = terms("Bad", "Alpha") if n < 10 else terms("Bad")
        transcript.write_text("".join(lines), encoding="utf-8")
        checked(transcript, identities)
        records, _ = external.build_external_dataset(
            (external.SourceSpec(name, source / "wav", transcript),), language="en"
        )
        groups[name] = {"records": [r.to_dict() for r in records]}
        write(retrieval / f"delivery/{name}_top7.json", delivery)
    table_sha = _sha(tables / "keyword_bias_phoneme.json")
    write(
        retrieval / "run_config.json", {"configuration": {}, "inputs": identities, "groups": groups}
    )
    write(
        retrieval / "report.json",
        {
            "status": "completed",
            "sample_count": 60,
            "table_sha256": table_sha,
            "checkpoint_sha256": "head",
            "top7_exact_replay": True,
        },
    )
    seal(retrieval)
    write(root / "model/config.json", {"model_type": "mock"})
    write(root / "model/model.safetensors.index.json", {"weight_map": {"weight": "weights.bin"}})
    (root / "model/weights.bin").write_bytes(b"mock no real weights")
    for n in ("preprocessor_config.json", "tokenizer_config.json", "tokenizer.json"):
        write(root / "model" / n, {})
    cfg = {
        "seed": 23,
        "language": "English",
        "max_new_tokens": 512,
        "source_samples_per_dataset": 20,
        "keyword_count": 3,
        "data_root": "data",
        "retrieval": "retrieval",
        "tables": "tables",
        "model": "model",
        "table_sha256": table_sha,
        "retrieval_head_sha256": "head",
        "targets_sha256": _sha(tables / "targets_by_dataset.json"),
        "model_config_sha256": _sha(root / "model/config.json"),
        "model_index_sha256": _sha(root / "model/model.safetensors.index.json"),
        "output": "run",
    }
    config = root / "config.json"
    write(config, cfg)
    return root, config


def probe():
    return {"runtime": "mock"}


class FakeRuntime:
    calls = 0
    fail_at = None
    metadata = {"backend": "mock"}

    def __init__(self, model, cfg):
        pass

    def transcribe(self, request):
        if self.calls == self.fail_at:
            raise RuntimeError("interrupted")
        type(self).calls += 1
        return {
            "prediction": "Alpha and Beta",
            "actual_context": request["context"],
            "serialized_prompt": "system:" + request["context"] + " user:audio",
            "prompt_tokens_verified": True,
        }


def test_freeze_run_483_calls_outputs_and_resume_without_model(assets):
    root, cfg = assets
    FakeRuntime.calls, FakeRuntime.fail_at = 0, None
    prep = experiment.prepare(root, cfg, probe=probe)
    assert prep["sample_count"] == 60
    assert prep["main_calls"] == 480 and prep["extra_repeat_calls"] == 3
    before = _sha(root / "retrieval/sha256.txt")
    report = experiment.run(root, cfg, probe=probe, factory=FakeRuntime)
    assert FakeRuntime.calls == 483 and report["status"] == "completed"
    assert _sha(root / "retrieval/sha256.txt") == before
    assert report["repeatability"]["normalized_text_changes"]["denominator"] == 3

    def forbidden(*args):
        pytest.fail("complete resume must not load model")

    assert experiment.run(root, cfg, resume=True, probe=probe, factory=forbidden) == report
    assert (root / "run/paired_results.jsonl").read_text().count("\n") == 60
    assert (root / "run/predictions.jsonl").read_text().count("\n") == 483
    with pytest.raises(FileExistsError):
        experiment.prepare(root, cfg, probe=probe)
    with pytest.raises(FileExistsError):
        experiment.run(root, cfg, probe=probe, factory=forbidden)


def test_resume_only_missing_requests_and_counts_interruption(assets):
    root, cfg = assets
    experiment.prepare(root, cfg, probe=probe)
    FakeRuntime.calls, FakeRuntime.fail_at = 0, 4
    with pytest.raises(RuntimeError, match="interrupted"):
        experiment.run(root, cfg, probe=probe, factory=FakeRuntime)
    paths = list((root / "run/results").rglob("*.json"))
    originals = {p: _sha(p) for p in paths}
    FakeRuntime.fail_at = None
    report = experiment.run(root, cfg, resume=True, probe=probe, factory=FakeRuntime)
    assert FakeRuntime.calls == 483
    assert report["request_attempts_started"] == 484
    assert report["starts_without_saved_event"] == 1
    assert {p: _sha(p) for p in paths} == originals


@pytest.mark.parametrize("change", ["audio", "config", "candidate", "model", "runtime", "code"])
def test_resume_rejects_identity_changes_before_any_new_inference(assets, monkeypatch, change):
    root, cfg = assets
    experiment.prepare(root, cfg, probe=probe)
    selected = json.loads((root / "run/freeze/plan.json").read_text())["samples"][0]
    changed_probe = probe
    if change == "audio":
        Path(selected["audio_path"]).write_bytes(b"different")
    elif change == "config":
        value = json.loads(cfg.read_text())
        value["seed"] += 1
        write(cfg, value)
    elif change == "candidate":
        p = root / f"retrieval/delivery/{DATASETS[0]}_top7.json"
        p.write_text("{}")
    elif change == "model":
        (root / "model/weights.bin").write_bytes(b"other checkpoint")
    elif change == "runtime":

        def changed_probe():
            return {"runtime": "other"}
    else:
        monkeypatch.setattr(design, "code_identity", lambda _: {"code": "changed"})

    def forbidden(*args):
        pytest.fail("must fail before loading model")

    with pytest.raises(ValueError):
        experiment.run(root, cfg, resume=True, probe=changed_probe, factory=forbidden)


def test_corrupt_cell_and_changed_effective_runtime_rejected(assets):
    root, cfg = assets
    experiment.prepare(root, cfg, probe=probe)
    FakeRuntime.calls, FakeRuntime.fail_at = 0, 1
    with pytest.raises(RuntimeError):
        experiment.run(root, cfg, probe=probe, factory=FakeRuntime)

    class DifferentRuntime(FakeRuntime):
        metadata = {"backend": "different"}

    with pytest.raises(ValueError, match="effective decoding"):
        experiment.run(root, cfg, resume=True, probe=probe, factory=DifferentRuntime)
    p = next((root / "run/results").rglob("*.json"))
    data = json.loads(p.read_text())
    data["result"]["prediction"] = "corrupted"
    write(p, data)
    with pytest.raises(ValueError, match="integrity mismatch"):
        experiment.run(root, cfg, resume=True, probe=probe, factory=FakeRuntime)


def test_priority_focus_and_wrong_candidate_added_disappeared():
    from qwen_hotword.evaluation.prompt_order_metrics import aggregate_pairs

    s = sample()
    e, f = row(s, "E", "Alpha and Beta"), row(s, "F", "Alpha and Beta Bad")
    comparison = compare(s, e, f)
    assert comparison["wrong_candidates_added"] == ["bad"]
    focused = aggregate_pairs([comparison])["priority_promoted_term_landing"]["reference_absent"]
    assert focused["left"]["rate"] == 0 and focused["right"]["rate"] == 1
    assert aggregate([f])["high_priority_wrong_landing"]["rate"] == 1
    assert compare(s, f, e)["wrong_candidates_disappeared"] == ["bad"]


def test_repetition_variability_is_separate_from_main_groups():
    s = sample()
    s["repeat_b"] = True
    rows = [row(s, g, "Alpha and Beta") for g in design.GROUPS]
    repeat = row(s, "B", "Alpha")
    repeat["group"] = "B_repeat"
    report, _ = summarize({"samples": [s], "configuration": {"seed": 23}}, [*rows, repeat])
    assert report["repeatability"]["normalized_text_changes"]["rate"] == 1
    assert report["repeatability"]["identical_prompt_samples"] == 1
    assert report["views"]["overall"]["groups"]["B"]["sample_count"] == 1


def test_freeze_preserves_candidates_and_never_prompts_reference_sentence(assets):
    root, config = assets
    plan = design.make_plan(root, config, probe())
    assert all(s["conditions"]["B"]["candidates"][0]["word"] == "Bad" for s in plan["samples"])
    for s in plan["samples"]:
        for group in design.GROUPS:
            assert s["reference"] not in s["conditions"][group]["context"]
        if s["stratum"] == "target_miss":
            assert [t["word"] for t in s["conditions"]["G"]["candidates"]] == ["Alpha", "Beta"]
    requests = experiment.requests(plan)
    for s in plan["samples"]:
        if s["repeat_b"]:
            by_group = {g: r for sample, g, r in requests if sample["uid"] == s["uid"]}
            assert by_group["B"] == by_group["B_repeat"]


def test_insufficient_frozen_plan_blocks_inference(assets, monkeypatch):
    root, config = assets
    plan = design.make_plan(root, config, probe())
    plan["ready"] = False
    plan["samples"] = plan["samples"][:-1]
    design.freeze(plan, root / "run")
    monkeypatch.setattr(experiment, "make_plan", lambda *args: plan)

    def forbidden(*args):
        pytest.fail("must not load model with incomplete frozen sample set")

    with pytest.raises(ValueError, match="insufficient"):
        experiment.run(root, config, probe=probe, factory=forbidden)
