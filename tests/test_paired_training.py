from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training import paired_gpu as gpu
from qwen_hotword.training import paired_labels as labels
from qwen_hotword.training import paired_run as pipeline
from qwen_hotword.training import paired_sources as sources
from qwen_hotword.training.paired_gpu_worker import claim
from qwen_hotword.training.paired_selection import manifests
from qwen_hotword.training.zh_en_common import digest, finish, read, rows, save_rows

ROOT = Path(__file__).resolve().parents[1]


def row(identity, pcm=None, family=None, split="train", group="zh", seconds=360):
    return dict(
        id=identity,
        pcm_sha256=pcm or identity,
        family=family or identity,
        split=split,
        group=group,
        seconds=seconds,
        text="word",
        issues=[],
        phoneme_token_ids=[2, 3],
        audio={"path": identity + ".wav"},
        samples_16k=int(seconds * 16000),
        ctc_minimum_input_length=2,
        ctc_time_upsampling_factor=2,
    )


def test_legacy_parts_require_exact_identity_and_leave_source_unchanged(tmp_path):
    source = tmp_path / "old"
    config = read(ROOT / "configs/zh_en_noise70_dual.workzone.json")
    config["source_run"] = "old"
    foundation = {"fixture": True}
    write_json(
        source / "contract.json", {"foundation": foundation, "config": config, "code_sha256": {}}
    )
    config["source_contract_sha256"] = sha(source / "contract.json")
    write_json(source / "g2p_report.json", {})
    path = source / "labels/part-000000/rows.jsonl"
    save_rows(path, [row("a")])
    finish(path.parent, {"g2p_report": sha(source / "g2p_report.json")}, [path], {"rows": 1})
    write_json(source / "label_report.json", {"totals": {"zh": 1}})
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    assert sources.legacy_parts(tmp_path, tmp_path / "new", config, foundation) == [path]
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="pinned"):
        sources.legacy_parts(tmp_path, tmp_path / "new", config, foundation)


def test_duplicate_linked_families_inherit_holdouts_and_conflicts_quarantined(tmp_path):
    old = [row("v", pcm="duplicate", split="validation"), row("t", split="test"), row("c")]
    new = [
        row("king_dup", pcm="duplicate", family="A"),
        row("king_a", family="A"),
        row("king_t", pcm="t", family="B"),
        row("king_b", family="B"),
        row("king_c", pcm="c", family="C"),
    ]
    new[-1]["text"] = "conflicting transcript"
    old_path, new_path = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    save_rows(old_path, old)
    save_rows(new_path, new)
    before = sha(old_path)
    old_outputs, new_outputs = sources.merge_splits(tmp_path, [old_path], [new_path], 4)
    prior = list(rows(old_outputs[0]))
    added = {r["id"]: r for r in rows(new_outputs[0])}
    assert sha(old_path) == before
    assert [r["split"] for r in prior] == ["validation", "test", "train"]
    assert [r["phoneme_token_ids"] for r in prior] == [[2, 3]] * 3
    assert added["king_a"]["split"] == "validation"
    assert added["king_b"]["split"] == "test"
    assert "cross_source_or_new_duplicate_pcm" in added["king_dup"]["issues"]
    assert "merged_pcm_conflicting_text" in prior[-1]["issues"]
    assert "merged_pcm_conflicting_text" in added["king_c"]["issues"]


def test_transitive_duplicate_links_cannot_join_legacy_train_and_test(tmp_path):
    old = [row("train"), row("test", split="test")]
    new = [
        row("king_a1", pcm="train", family="A"),
        row("king_a2", pcm="link", family="A"),
        row("king_b1", pcm="link", family="B"),
        row("king_b2", pcm="test", family="B"),
        row("king_free", family="B"),
    ]
    save_rows(tmp_path / "old.jsonl", old)
    save_rows(tmp_path / "new.jsonl", new)
    _, added = sources.merge_splits(tmp_path, [tmp_path / "old.jsonl"], [tmp_path / "new.jsonl"], 5)
    for r in rows(added[0]):
        assert "new_family_links_incompatible_legacy_splits" in r["issues"]


def selection_fixture(tmp_path):
    cfg = {"seed": 5, "part_records": 4, "hours": {}, "minimum_hours": {}}
    items = []
    for split in ("train", "validation"):
        cfg["hours"][split] = dict.fromkeys(("zh", "us", "uk"), 1)
        cfg["minimum_hours"][split] = dict.fromkeys(("zh", "us", "uk"), 1)
        for group in ("zh", "us", "uk"):
            items += [row(f"{group}_{split}_{i}", split=split, group=group) for i in range(10)]
    # Extra Chinese data should not displace eligible old labels.
    items += [row("king_extra", seconds=3600), row("sealed", split="test", seconds=100000)]
    path = tmp_path / "labels.jsonl"
    save_rows(path, items)
    return cfg, path


def test_selection_keeps_old_priority_exact_hours_noise_and_paired_validation(tmp_path):
    cfg, path = selection_fixture(tmp_path)
    result = manifests(tmp_path, [path], cfg)
    chosen = [r for part in result["parts"] for r in rows(Path(part["path"]))]
    train = [r for r in chosen if r["split"] == "train"]
    assert len(train) == 30
    assert all(r["id"] not in ("king_extra", "sealed") for r in chosen)
    for group in ("zh", "us", "uk"):
        assert sum(r["noisy"] for r in train if r["group"] == group) == 7
    assert len([r for r in chosen if r["split"] == "validation"]) == 60
    assert manifests(tmp_path, [path], cfg) == result


def test_all_six_shortfalls_reported_before_any_training(tmp_path):
    cfg, path = selection_fixture(tmp_path)
    for split in cfg["minimum_hours"]:
        cfg["minimum_hours"][split] = dict.fromkeys(("zh", "us", "uk"), 3)
    with pytest.raises(ValueError, match="all six"):
        manifests(tmp_path, [path], cfg)
    report = read(tmp_path / "label_report.json")
    assert len(report["availability"]) == 6
    assert all(not r["passed"] for r in report["availability"].values())
    assert not list(tmp_path.glob("prepared/train/*"))


def test_incremental_g2p_only_requests_unknown_words_and_is_resumable(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    new.mkdir()
    worker = dict(models={}, vocab_sha256="v", g2p_num_pronunciations=1, environment={})
    write_json(old / "g2p_worker_config.json", worker)
    path = old / "g2p/part-000000/predictions.jsonl"
    save_rows(
        path,
        [
            dict(language="zh", word="known", status="ok", candidates=[]),
            dict(language="zh", word="bad", status="unsupported", candidates=[]),
        ],
    )
    write_json(
        path.with_suffix(".completed.json"),
        {
            "identity": {"worker_config_sha256": sha(old / "g2p_worker_config.json")},
            "output_sha256": sha(path),
        },
    )
    write_json(old / "g2p_report.json", {"prediction_sha256": digest([sha(path)])})
    source = new / "text.jsonl"
    values = [
        dict(issues=[], split="train", frontend=[{"language": "zh", "word": w}])
        for w in ("known", "bad", "new", "new")
    ]
    values.append(dict(issues=[], split="test", frontend=[{"language": "zh", "word": "sealed"}]))
    save_rows(source, values)
    requests = []

    def fake(root, worker, assets, prefix, request, destination):
        items = list(rows(request))
        requests.extend(items)
        save_rows(destination, [{**r, "status": "ok", "candidates": []} for r in items])

    monkeypatch.setattr(labels, "run_worker", fake)
    before = sha(path)
    db_path = labels.incremental_g2p(tmp_path, new, [source], old, worker, tmp_path, [], 64)
    assert requests == [{"language": "zh", "word": "new"}]
    assert read(new / "g2p_report.json")["reused_unique_words"] == 2
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT count(*) FROM words").fetchone()[0] == 3
        assert (
            json.loads(db.execute("SELECT payload FROM words WHERE word='bad'").fetchone()[0])[
                "status"
            ]
            == "unsupported"
        )
    assert sha(path) == before
    worker["vocab_sha256"] = "changed"
    with pytest.raises(ValueError, match="policy differs"):
        labels.incremental_g2p(tmp_path, new, [source], old, worker, tmp_path, [], 1)


def test_cpu_ceiling_respects_affinity_and_rejects_over64(monkeypatch):
    monkeypatch.setattr(sources.os, "sched_getaffinity", lambda _: {0, 1, 2}, raising=False)
    assert 1 <= sources.cpu_limit(64) <= 3
    with pytest.raises(ValueError):
        sources.cpu_limit(65)


def test_selects_fastest_complete_warmed_trials_not_most_processes():
    trials = [
        dict(per_gpu=p, status="completed", measured_seconds=s, audio_seconds=1000)
        for p, s in ((1, 10), (1, 12), (2, 6), (2, 7), (4, 1))
    ]
    trials.append(dict(per_gpu=4, status="resource_unavailable"))
    assert gpu.choose(trials, 2) == 2
    with pytest.raises(gpu.ResourceFailure):
        gpu.choose([], 2)
    assert (
        gpu.choose(
            [
                dict(per_gpu=p, status="completed", measured_seconds=1, audio_seconds=1)
                for p in (1, 1, 2, 2)
            ],
            2,
        )
        == 1
    )


def test_shared_queue_claims_each_shard_once(tmp_path):
    queue = tmp_path / "queue.sqlite"
    with sqlite3.connect(queue) as db:
        db.execute(
            "CREATE TABLE jobs (id INTEGER PRIMARY KEY,payload TEXT,seconds REAL,state TEXT)"
        )
        db.executemany(
            "INSERT INTO jobs VALUES (?,?,?,?)", [(i, "{}", i, "pending") for i in range(64)]
        )

    def work(_):
        seen = []
        while result := claim(queue):
            seen.append(result["job_id"])
        return seen

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [v for values in pool.map(work, range(8)) for v in values]
    assert sorted(ids) == list(range(64))


def test_fixed_probe_covers_groups_noise_lengths_and_excludes_test(tmp_path):
    path = tmp_path / "prepared.jsonl"
    values = [
        {**row(f"{g}_{n}_{i}", group=g, seconds=duration), "noisy": n}
        for g in ("zh", "us", "uk")
        for n in (False, True)
        for i, duration in enumerate((3, 7, 15, 99))
    ]
    save_rows(path, values)
    prepared = {
        "parts": [
            {"path": str(path), "sha256": sha(path), "split": "train"},
            {"path": "/never/read/test", "split": "test"},
        ]
    }
    cfg = dict(seed=7, benchmark_per_stratum=16, benchmark_records_per_job=16)
    jobs = gpu.probe_jobs(tmp_path, prepared, cfg)
    identity = sha(tmp_path / "probe_cohort.json")
    assert len(read(tmp_path / "probe_cohort.json")["strata"]) == 18
    assert sum(len(list(rows(Path(j["manifest"])))) for j in jobs) == 18
    assert gpu.probe_jobs(tmp_path, prepared, cfg) == jobs
    assert sha(tmp_path / "probe_cohort.json") == identity


FAKE_WORKER = """import json, os, sqlite3, sys, time
from pathlib import Path
args=sys.argv
spec_path=Path(args[args.index('--spec')+1]); spec=json.loads(spec_path.read_text())
id=int(args[args.index('--worker-id')+1]); folder=spec_path.parent/f'worker-{id}'
folder.mkdir()
(folder/'ready.json').write_text('{}')
while not (spec_path.parent/'go.json').exists(): time.sleep(.01)
while True:
    with sqlite3.connect(spec_path.parent/'queue.sqlite',timeout=60) as db:
        db.execute('BEGIN IMMEDIATE')
        found=db.execute("SELECT id FROM jobs WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
        if found is None: break
        db.execute("UPDATE jobs SET state='done' WHERE id=?",found)
    time.sleep(.01)
(folder/'exit.json').write_text(json.dumps({'status':'completed','gpu':os.environ['CUDA_VISIBLE_DEVICES']}))
"""


def test_coordinator_real_subprocess_barrier_two_cards_and_cleanup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    script = root / "scripts/run_zh_en_dual.py"
    script.parent.mkdir(parents=True)
    script.write_text(FAKE_WORKER)
    monkeypatch.setattr(gpu, "gpu_inventory", lambda ids: {g: {"free_gib": 100} for g in ids})
    cfg = dict(gpu_ids=[4, 5], gpu_headroom_gib=8, estimated_worker_gpu_gib=12)
    jobs = [dict(seconds=3, manifest="fixture") for _ in range(20)]
    result = gpu.run_workers(root, tmp_path / "out", cfg, {}, jobs, 2)
    assert [w["gpu"] for w in result["workers"]] == ["4", "4", "5", "5"]
    assert result["jobs"] == 20 and result["audio_seconds"] == 60
    assert result["startup_seconds"] > 0 and result["measured_seconds"] > 0
    gpu.ensure_no_workers(tmp_path / "out")


def test_coordinator_propagates_failure_and_stops_only_owned_workers(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    script = root / "scripts/run_zh_en_dual.py"
    script.parent.mkdir(parents=True)
    script.write_text('raise ValueError("fixture failure")')
    monkeypatch.setattr(gpu, "gpu_inventory", lambda ids: {g: {"free_gib": 100} for g in ids})
    cfg = dict(gpu_ids=[4, 5], gpu_headroom_gib=8, estimated_worker_gpu_gib=12)
    with pytest.raises(RuntimeError, match="worker failed"):
        gpu.run_workers(root, tmp_path / "out", cfg, {}, [dict(seconds=3, manifest="fixture")], 1)
    assert len(list((tmp_path / "out").glob("gpu_runs/*/worker-*/exit.json"))) == 2
    gpu.ensure_no_workers(tmp_path / "out")


def test_source_identity_guard_and_safe_new_output(tmp_path):
    cfg = read(ROOT / "configs/zh_en_noise70_dual.workzone.json")
    foundation = read(ROOT / cfg["foundation_config"])
    with pytest.raises(ValueError, match="overlaps"):
        pipeline.establish(ROOT, (ROOT / cfg["source_run"]).resolve(), cfg, foundation)
    output = tmp_path / "new_run"
    pipeline.establish(ROOT, output, cfg, foundation)
    pipeline.establish(ROOT, output, cfg, foundation)
    write_json(output / "dual_code_identity.json", {})
    with pytest.raises(ValueError, match="source changed"):
        pipeline.establish(ROOT, output, cfg, foundation)


def test_new_inventory_resume_no_reinspection(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "part.tar").write_bytes(b"fixture")
    values = [row("x"), row("y")]
    calls = []
    monkeypatch.setattr(sources, "chinese_rows", lambda *args: iter([dict(v) for v in values]))

    def inspect(value):
        calls.append(value["id"])
        return value

    monkeypatch.setattr(sources, "inspect_audio", inspect)
    cfg = dict(added_chinese_root=str(raw), part_records=1, cpu_worker_limit=64)
    output = tmp_path / "out"
    paths = sources.new_inventory(output, cfg)
    assert calls == ["king_x", "king_y"]
    assert sources.new_inventory(output, cfg) == paths
    assert calls == ["king_x", "king_y"]
    assert next(rows(paths[0]))["family"] == "king:x"


def test_worker_reuses_only_matching_samples_and_prefetch_preserves_order(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from qwen_hotword.training import (
        ctc_overfit,
        feature_cache,
        paired_gpu_worker,
        sharded_ctc,
        zh_en_run,
    )

    items = [dict(row(f"s{i}", seconds=1), base_id=f"s{i}", noisy=False) for i in range(3)]
    manifest = tmp_path / "manifest.jsonl"
    save_rows(manifest, items)
    probe = tmp_path / "probe.json"
    write_json(probe, dict(path="probe", manifest="probe.jsonl", fingerprint="verified"))

    def sample(identity):
        return ctc_overfit.CachedSample(identity, SimpleNamespace(shape=(13, 1024)), (2, 3))

    loaded = SimpleNamespace(fingerprint="verified", shards=["shard"], feature_bytes=10)
    monkeypatch.setattr(sharded_ctc, "load_disk_feature_cache", lambda *a, **k: loaded)
    monkeypatch.setattr(sharded_ctc, "load_feature_shard", lambda *a, **k: [sample("s1")])
    decoded = []

    def audio(self, record):
        decoded.append(record.sample_id)
        return record.sample_id

    monkeypatch.setattr(zh_en_run.MixedExtractor, "load", audio)
    forwarded = []

    def extract(records, wrapper, waveform_loader, **kwargs):
        result = []
        for record in records:
            forwarded.append(waveform_loader(record))
            result.append(sample(record.sample_id))
        return result, 317477504, 1

    monkeypatch.setattr(ctc_overfit, "extract_frozen_features", extract)
    captured = []

    def cache(records, wrapper, folder, extractor, **kwargs):
        values, frozen, _ = extractor(records, wrapper, encoder_batch_size=1)
        captured.extend(s.sample_id for s in values)
        assert frozen == 317477504

    monkeypatch.setattr(feature_cache, "cache_feature_split", cache)
    result = paired_gpu_worker.process_job(
        dict(
            manifest=str(manifest),
            manifest_sha256=sha(manifest),
            destination=str(tmp_path / "features"),
            split="train",
        ),
        None,
        dict(seed=1, model="model", part_records=512),
        {"vocab": "vocab"},
        tmp_path,
        {},
        {"s1": str(probe)},
    )
    assert captured == ["s0", "s1", "s2"]
    assert forwarded == ["s0", "s2"]
    assert decoded == ["s0", "s2"]
    assert result["probe_samples_reused"] == 1


def test_prepare_complete_skips_all_old_and_new_g2p(tmp_path, monkeypatch):
    cfg, path = selection_fixture(tmp_path)
    report = manifests(tmp_path, [path], cfg)
    write_json(tmp_path / "prepared.json", report)
    write_json(
        tmp_path / "prepare_completed.json",
        {
            "files": {
                "prepared.json": sha(tmp_path / "prepared.json"),
                "prepared/completed.json": sha(tmp_path / "prepared/completed.json"),
            }
        },
    )
    monkeypatch.setattr(pipeline, "verify_assets", lambda *args: None)

    def forbidden(*args):
        raise AssertionError("must not regenerate labels")

    monkeypatch.setattr(pipeline, "legacy_parts", forbidden)
    monkeypatch.setattr(pipeline, "new_inventory", forbidden)
    monkeypatch.setattr(pipeline, "incremental_g2p", forbidden)
    assert pipeline.prepare(tmp_path, tmp_path, cfg, {}) == report
    chosen = Path(report["parts"][0]["path"])
    chosen.write_text("corrupt")
    with pytest.raises(ValueError, match="pinned"):
        pipeline.prepare(tmp_path, tmp_path, cfg, {})
