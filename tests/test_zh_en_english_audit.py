from collections import Counter

from qwen_hotword.evaluation.zh_en_english_audit import (
    choose_cases,
    count_probe,
    failed_gates,
    neighbors,
    text_contains,
)


def test_joint_pronunciation_neighbors_exclude_self_but_include_other_languages():
    def word(name, phones, src):
        return {"surface": name, "pronunciations": [{"token_ids": phones, "sources": [src]}]}

    table = [
        word("A", [1, 2, 3, 4, 5], "probe"),
        word("B", [1, 2, 3, 4, 8], "other"),
        word("C", [1, 2, 3], "other"),
        word("D", [1, 2, 3, 4, 5], "other"),
    ]
    assert neighbors(table, "probe") == {"a": {"b", "d"}}


def test_ranking_miss_splits_and_co_targets():
    counts = Counter()
    for ranks in (["p", "q"], ["q"], ["z"], ["q", "a"]):
        count_probe("a", {"a", "q"}, ranks, {"p", "q"}, counts)
    assert counts == dict(
        pairs=4,
        hits=1,
        misses=3,
        miss_with_non_gold_neighbor=1,
        miss_only_other_gold_neighbor=1,
        miss_without_ranked_neighbor=1,
    )


def test_all_gate_failures_and_inclusive_threshold():
    assert failed_gates(dict(score=0.75, edit_ratio=0.35, posterior_confidence=0.5)) == []
    assert failed_gates(dict(score=0.65, edit_ratio=0.4, posterior_confidence=0.2)) == [
        "score",
        "edit_ratio",
        "posterior_confidence",
    ]
    assert text_contains("Visit AL–MALAZ 503 today.", "Al-Malaz 503")
    assert not text_contains("Visit Al Malaz five hundred and three.", "Al-Malaz 503")
    assert not text_contains("Abroad", "road")


def test_bounded_varied_cases_and_distinct_utterances():
    cases = [
        dict(group=g, sample_id=g + str(i), target=w, rank=r)
        for g in ("v1", "v2")
        for i, w, r in ((1, "x", 1), (1, "y", 5), (2, "x", 1), (3, "y", 3))
    ]
    got = choose_cases(reversed(cases), 2)
    assert [(c["group"], c["target"]) for c in got] == [
        ("v1", "x"),
        ("v1", "y"),
        ("v2", "x"),
        ("v2", "y"),
    ]


def test_cpu_scan_joins_final_wav_ids_and_reads_verified_cache(tmp_path, monkeypatch):
    import hashlib
    import importlib
    import json
    import zipfile
    from pathlib import Path

    from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
    from qwen_hotword.training.zh_en_common import digest

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    cli = importlib.import_module("diagnose_zh_en_480_english")
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    manifest = {}
    common = []
    probe_id = hashlib.md5(b"original/path").hexdigest()[:16]
    source_rows = []
    for i, group in enumerate((cli.PROBE, *cli.HOME)):
        uid = probe_id if group == cli.PROBE else "val_" + group + "-uid"
        s = dict(
            sample_id=uid,
            audio_sha256=str(i) * 64,
            language="en",
            group=group,
            audio_path=uid + ".wav",
        )
        manifest[group + "@@en"] = {uid: s}
        if group != cli.PROBE:
            common.append({**s, "targets": ["name"]})
            source_rows.append((group, uid, Path(s["audio_path"]), ["name"]))
        result = dict(raw_last=["Rival"] if group == cli.PROBE else ["Name"], steps=[[]])
        write_json(
            cli.cached(src, s),
            dict(
                run="run",
                audio=s["audio_sha256"],
                language="en",
                digest=digest(result),
                result=result,
            ),
        )
    with zipfile.ZipFile(src / "delivery.zip", "w") as z:
        z.writestr("delivery/audio_manifest.json", json.dumps(manifest))
    idx = tmp_path / "idx"
    idx.write_text(
        json.dumps(
            dict(
                audio_path="original/path",
                text="name",
                entities=[dict(surface="Name", ipa="x", category="place")],
            )
        )
    )
    meta = tmp_path / "meta"
    meta.write_text(
        "\n".join(
            json.dumps(dict(id=r[1], text="name", keywords=["Name"], label="name"))
            for r in source_rows
        )
    )
    monkeypatch.setattr(cli.base, "inputs", lambda cfg: iter(source_rows))
    report = {
        "plan_sha256": "run",
        "by_dataset": {
            group: {
                "raw_top7_last": {"target_pairs": 1, "target_hits": int(group != cli.PROBE)},
                "gated_top7_last": {"target_hits": 0},
            }
            for group in (cli.PROBE, *cli.HOME)
        },
    }
    table = [
        {
            "surface": w,
            "pronunciations": [
                {
                    "token_ids": [1, 2, 3, 4, i],
                    "phoneme_tokens": ["x"],
                    "sources": [cli.PROBE] if w == "Name" else ["other"],
                }
            ],
        }
        for i, w in enumerate(["Name", "Rival"])
    ]
    cli.scan(
        src,
        out,
        {"audit": {cli.PROBE: {"index_sha256": sha(idx)}}, "common_samples_sha256": digest(common)},
        report,
        {"work": str(tmp_path), "home_meta": str(meta), "en_index": "idx"},
        table,
    )
    cpu = json.loads((out / "cpu.json").read_text())
    assert cpu["probe_pairs"]["miss_with_non_gold_neighbor"] == 1
    assert len(cpu["selected"]) == 2
    assert cpu["source_checks"]["index_ipa_vs_registry"] == {"same_tokens": 1}
