from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy

import pytest

from qwen_hotword.evaluation.joint_latency import (
    STAGE_FIELDS,
    analyze_run,
    distribution,
    observation,
    summarize,
    write_report,
)
from qwen_hotword.inference.stream_context_contract import digest


def seal(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "sha256.txt")
    (folder / "sha256.txt").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder).as_posix()}\n"
            for p in files
        )
    )


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def payload(group="en_chile", total=0.02, anchor=0.008):
    return {
        "group": group,
        "sample_id": group,
        "step": 0,
        "audio_end_seconds": 2.0,
        "source_end_frame": 32000,
        "source_sample_rate": 16000,
        "audio_sha256": "audio-not-present",
        "key": f"{group}@@{group[-2:]}",
        "retrieval": {
            "raw_variant_ranked_matches": [{"hotword_id": "v"}],
            "variant_shortlist_candidates": 1,
            "shortlist_candidates": 1,
            "decoded_token_ids": [2, 3, 4],
            "postings_visited": 100,
            "timing": {
                **{field: 0.0001 for field in STAGE_FIELDS.values()},
                "anchor_query_seconds": anchor,
                "hotword_matching_seconds": 0.005,
                "pure_retrieval_seconds": total,
            },
        },
    }


@pytest.fixture
def evidence(tmp_path):
    assets, run = tmp_path / "assets", tmp_path / "run"
    table = {"entries": [{"normalized": "test", "variants": [{"token_ids": [2, 3, 4]}]}]}
    put(assets / "reviewed_keyword_table.json", table)
    table_sha = hashlib.sha256((assets / "reviewed_keyword_table.json").read_bytes()).hexdigest()
    index = {"hotword_id": "v", "normalized": "test", "language": "en", "token_ids": [2, 3, 4]}
    (assets / "indexed_pronunciations.jsonl").write_text(json.dumps(index) + "\n")
    index_sha = hashlib.sha256((assets / "indexed_pronunciations.jsonl").read_bytes()).hexdigest()
    seal(assets)
    samples = []
    for lang in ("en", "es", "pt"):
        group = f"group_{lang}"
        row = payload(group)
        row["sha256"] = digest(row)
        put(run / "prefix_shards" / group / "hash" / "000000.json", row)
        samples.append(
            {
                "group": group,
                "sample_id": group,
                "language": lang,
                "frames": 32000,
                "sample_rate": 16000,
                "step_count": 1,
                "audio_sha256": "audio-not-present",
                "key": row["key"],
            }
        )
    put(
        run / "run_config.json",
        {
            "source_configuration": {"table_sha256": table_sha},
            "samples": samples,
            "selected_samples": 3,
            "prefix_count": 3,
            "checkpoint": {"sha256": "not-opened"},
            "inputs": {"/unavailable/indexed_pronunciations.jsonl": {"sha256": index_sha}},
        },
    )
    put(run / "report.json", {"status": "completed"})
    seal(run)
    return run, assets, table_sha, {"v": index}


def test_units_workload_and_weighted_time_shares(evidence):
    _, _, _, index = evidence
    first = observation(payload(), index)
    second = observation(payload(total=0.08, anchor=0.06), index)
    assert first["anchor_ms"] == 8
    assert first["candidate_phone_square_sum"] == 9
    result = summarize([first, second])
    assert result["timing_ms"]["pure_ms"]["mean"] == 50
    assert result["time_share_of_sum"]["anchor_ms"] == pytest.approx(0.68)
    assert result["over_50ms"] == 1
    assert distribution([1.0, 2.0, 3.0])["p95"] == 3


def test_complete_sealed_run_and_output_without_opening_referenced_assets(evidence, tmp_path):
    run, assets, table_sha, _ = evidence
    originals = {
        p: p.read_bytes() for root in (run, assets) for p in root.rglob("*") if p.is_file()
    }
    report = analyze_run(run, assets, table_sha256=table_sha)
    assert report["model_loaded"] is False and report["audio_opened"] is False
    assert report["by_language"]["en"]["queries"] == 1
    assert report["last_prefix_by_language"]["en"]["queries"] == 1
    assert report["by_prefix_duration"]["es"]["(0,2]s"]["queries"] == 1
    assert report["evidence"]["run"]["files_read"] == 5
    assert report["source_backend_and_gc_policy"].startswith("not inferred")
    output = tmp_path / "new_output"
    write_report(report, output, (run, assets))
    with zipfile.ZipFile(output / "return_bundle.zip") as archive:
        assert set(archive.namelist()) == {"report.json", "summary.json", "sha256.txt"}
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    with pytest.raises(FileExistsError):
        write_report(report, output, (run, assets))
    with pytest.raises(ValueError, match="separate"):
        write_report(report, run / "nested", (run, assets))


def test_prefix_queries_and_last_full_audio_queries_are_separate(evidence):
    run, assets, sha, _ = evidence
    plan = json.loads((run / "run_config.json").read_text())
    sample = plan["samples"][0]
    sample.update(frames=64000, step_count=2)
    plan["prefix_count"] = 4
    put(run / "run_config.json", plan)
    row = payload(sample["group"], total=0.04)
    row.update(step=1, audio_end_seconds=4.0, source_end_frame=64000)
    row["sha256"] = digest(row)
    put(run / "prefix_shards" / sample["group"] / "hash" / "000001.json", row)
    seal(run)
    report = analyze_run(run, assets, table_sha256=sha)
    assert report["by_language"]["en"]["queries"] == 2
    assert report["by_language"]["en"]["timing_ms"]["pure_ms"]["mean"] == 30
    last = report["last_prefix_by_language"]["en"]
    assert last["queries"] == 1 and last["timing_ms"]["pure_ms"]["mean"] == 40


def test_missing_prefix_is_not_a_three_language_comparison(evidence):
    run, assets, sha, _ = evidence
    next((run / "prefix_shards").glob("*/*/*.json")).unlink()
    with pytest.raises(ValueError, match="incomplete/extra"):
        analyze_run(run, assets, table_sha256=sha)


def test_corrupted_record_is_not_used(evidence):
    run, assets, sha, _ = evidence
    path = next((run / "prefix_shards").glob("*/*/*.json"))
    row = json.loads(path.read_text())
    row["retrieval"]["timing"]["pure_retrieval_seconds"] = 0.5
    put(path, row)
    with pytest.raises(ValueError, match="checksum"):
        analyze_run(run, assets, table_sha256=sha)
    seal(run)
    with pytest.raises(ValueError, match="corrupt"):
        analyze_run(run, assets, table_sha256=sha)


@pytest.mark.parametrize("field,value", [("table_sha256", "wrong"), ("index", "wrong")])
def test_different_table_or_index_is_rejected(evidence, field, value):
    run, assets, sha, _ = evidence
    path = run / "run_config.json"
    plan = json.loads(path.read_text())
    if field == "index":
        plan["inputs"]["/unavailable/indexed_pronunciations.jsonl"]["sha256"] = value
    else:
        plan["source_configuration"][field] = value
    put(path, plan)
    seal(run)
    with pytest.raises(ValueError, match="different"):
        analyze_run(run, assets, table_sha256=sha)


@pytest.mark.parametrize("change", ["missing_stage", "nan", "truncated", "unknown", "overlap"])
def test_invalid_timing_or_shortlist_is_not_silently_imputed(evidence, change):
    _, _, _, index = evidence
    row = deepcopy(payload())
    data = row["retrieval"]
    if change == "missing_stage":
        del data["timing"]["ctc_decode_seconds"]
    elif change == "nan":
        data["timing"]["ctc_decode_seconds"] = float("nan")
    elif change == "truncated":
        data["variant_shortlist_candidates"] = 64
    elif change == "unknown":
        data["raw_variant_ranked_matches"][0]["hotword_id"] = "unknown"
    else:
        data["timing"]["pure_retrieval_seconds"] = 0.001
    with pytest.raises((ValueError, KeyError)):
        observation(row, index)
