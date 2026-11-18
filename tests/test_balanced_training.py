from collections import Counter
from pathlib import Path

import pytest

from qwen_hotword.training.balanced_selection import select
from qwen_hotword.training.paired_run import verify_prepared
from qwen_hotword.training.zh_en_common import rows, save_rows


def setup_pool(tmp_path):
    groups = {"us": 5, "uk": 5, "0025": 1, "0007": 2, "0009": 3, "0059": 4}
    roots = {k: str(tmp_path / k) for k in groups if k.isdigit()}
    source = tmp_path / "pool.jsonl"
    raw = []
    for split in ("train", "validation", "test"):
        for group, units in groups.items():
            # US has exactly half its train target; UK has half its validation target.
            limited = (split, group) in (("train", "us"), ("validation", "uk"))
            for i in range(units * (10 if limited else 20)):
                uid = f"{split}-{group}-{i}"
                raw.append(
                    dict(
                        id=uid,
                        split=split,
                        group="zh" if group in roots else group,
                        source=str(tmp_path / group / "shard.tar"),
                        issues=[],
                        seconds=180,
                        pcm_sha256=uid,
                    )
                )
    raw.append(dict(raw[0], id="invalid", issues=["incomplete_label"]))
    save_rows(source, raw)
    return source, dict(
        quotas=dict(train=groups, validation=groups),
        chinese_roots=roots,
        seed=1,
        part_records=7,
        scale_to_us=True,
    )


def test_balanced_hours_paired_views_and_noise(tmp_path):
    from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json

    path, cfg = setup_pool(tmp_path)
    report = select(tmp_path, [path], cfg)
    counts = Counter()
    views = {}
    for part in report["parts"]:
        for row in rows(Path(part["path"])):
            assert row["split"] != "test" and not row["issues"]
            assert row["group"] in {"us", "uk", "zh"}
            counts[row["source_group"], row["condition"], row["noisy"]] += 1
            if row["split"] == "validation":
                views.setdefault(row["base_id"], set()).add(row["condition"])
    assert all(v == {"original", "noise10db"} for v in views.values())
    for g, h in cfg["quotas"]["train"].items():
        assert report["groups"]["train/" + g]["hours"] == h / 2
        assert report["groups"]["validation/" + g]["hours"] == h / 2
        assert counts[g, "mixed", True] == h * 7
        assert counts[g, "original", False] == counts[g, "noise10db", True] == h * 10
    write_json(tmp_path / "prepared.json", report)
    write_json(
        tmp_path / "prepare_completed.json",
        {"files": {"prepared.json": sha(tmp_path / "prepared.json")}},
    )
    assert verify_prepared(tmp_path) == report


def test_strict_shortfall(tmp_path):
    path, cfg = setup_pool(tmp_path)
    cfg["scale_to_us"] = False
    with pytest.raises(ValueError, match="train/us"):
        select(tmp_path, [path], cfg)


def test_reject_duplicate_pcm(tmp_path):
    path, cfg = setup_pool(tmp_path)
    data = list(rows(path))
    data[1]["pcm_sha256"] = data[0]["pcm_sha256"]
    save_rows(path, data)
    with pytest.raises(ValueError, match="duplicate selected PCM"):
        select(tmp_path, [path], cfg)


def test_previous_run_reuses_full_label_pools_and_rejects_drift(tmp_path):
    import runpy

    from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json

    script = Path(__file__).resolve().parents[1] / "scripts/run_balanced_zh_en.py"
    load = runpy.run_path(str(script))["previous_labels"]
    cfg = dict(frontend_policy="v1", frontend_packages=[], seed=1, head={}, num_classes=183)
    foundation = {"vocab_sha256": "v1"}
    write_json(tmp_path / "contract.json", dict(config=cfg, foundation=foundation))
    parts = [
        tmp_path / "legacy_labels/part-000000.jsonl",
        tmp_path / "labels/part-000000/rows.jsonl",
    ]
    for i, path in enumerate(parts):
        save_rows(path, [dict(id=i)])
    marker = tmp_path / "prepared/completed.json"
    write_json(marker, dict(identity={"labels": [sha(p) for p in parts]}))
    write_json(
        tmp_path / "prepare_completed.json", dict(files={"prepared/completed.json": sha(marker)})
    )
    assert load(tmp_path, foundation, cfg) == parts
    parts[1].write_text("changed")
    with pytest.raises(ValueError):
        load(tmp_path, foundation, cfg)
