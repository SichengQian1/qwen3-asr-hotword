from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import pytest

from qwen_hotword.evaluation import ziyan_control as control
from qwen_hotword.evaluation import ziyan_control_replay as replay
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import combine_variants
from qwen_hotword.evaluation.ziyan_assets import NAME
from qwen_hotword.evaluation.ziyan_identity import digest, verify_seal
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.wave_retrieval import _row_digest
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def variant(word, lang="en", role="filler", pronunciation="t a"):
    p = tokenize_ipa_to_vocab(pronunciation, load_phoneme_vocab(VOCAB))
    return {
        "surface": word,
        "normalized": word,
        "language": lang,
        "role": role,
        "pronunciation": pronunciation,
        "phoneme_tokens": list(p.tokens),
        "token_ids": list(p.token_ids),
        "decision": "retained",
        "source_groups": [],
    }


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Five frozen targets prove that the control path has no old 300-500 constraint.
    targets = ["alpha", "bravo", "charlie", "delta", "echo"]
    target_path = tmp_path / "config/targets.txt"
    target_path.parent.mkdir()
    target_path.write_text("\n".join(targets) + "\n")
    dataset = tmp_path / "data/input.json"
    records, raw = [], []
    for i, text in enumerate(("alpha alpha bravo", "hello", " ".join(targets))):
        audio = tmp_path / f"audio/item_{i}.wav"
        audio.parent.mkdir(exist_ok=True)
        audio.write_bytes(f"mock audio {i}; never decoded".encode())
        raw.append({"audios": [str(audio)], "response": text})
        records.append(external.ExternalAudioRecord(audio.stem, NAME, audio, text, "en").to_dict())
    write(dataset, raw)
    head = tmp_path / "model/head.pt"
    head.parent.mkdir()
    head.write_bytes(b"mock Head; never loaded")
    write(head.parent / "config.json", {})
    assets, results = tmp_path / "assets", tmp_path / "original"
    cfg = {
        "dataset": str(dataset),
        "dataset_sha256": digest(dataset),
        "expected_samples": 3,
        "targets": str(target_path),
        "targets_sha256": digest(target_path),
        "expected_targets": 5,
        "vocab": str(VOCAB),
        "vocab_sha256": digest(VOCAB),
        "checkpoint": str(head),
        "checkpoint_sha256": digest(head),
        "assets": str(assets),
        "output": str(results),
    }
    config = tmp_path / "config/base.json"
    write(config, cfg)
    table = {
        "entries": combine_variants(
            [variant(w, role="target") for w in targets]
            + [variant("warnings"), variant("connec"), variant("connec", "es")]
        )
    }
    write(assets / "config.json", cfg)
    write(assets / "targets.json", targets)
    write(assets / "keyword_table.json", table)
    (assets / "english_raw.dict").write_text("mock original dictionary\n")
    external._write_jsonl(assets / "dataset_manifest.jsonl", records)
    selection = {"unique_surfaces": len(table["entries"])}
    write(
        assets / "report.json",
        {
            "status": "completed",
            "table_ready": True,
            "targets": targets,
            "selection": selection,
        },
    )
    seal(assets)
    monkeypatch.setattr(external, "_git_commit", lambda: "test-cpu-only")
    detector_cfg = external._run_config(
        model=head.parent,
        checkpoint=head,
        vocab_path=VOCAB,
        keyword_path=assets / "keyword_table.json",
        sources=(),
        keyword_set="all_keywords",
        device="cuda:0",
        dtype="bfloat16",
        language="English",
        saved_raw_rank_depth=64,
    )
    runtime = control._runtime(
        table, assets / "keyword_table.json", load_phoneme_vocab(VOCAB), targets
    )
    replay.prepare_runtime(runtime, detector_cfg)
    ids = list(tokenize_ipa_to_vocab("t a", load_phoneme_vocab(VOCAB)).token_ids)
    rows = []
    for record in records:
        decoded = {
            "decoded_token_ids": ids,
            "decoded_tokens": [load_phoneme_vocab(VOCAB).tokens[t] for t in ids],
            "decoded_confidences": [0.95] * len(ids),
        }
        retrieved = replay.retrieve(decoded, runtime, detector_cfg)
        row = external._build_result_row(
            external.ExternalAudioRecord(
                record["sample_id"],
                NAME,
                Path(record["audio_path"]),
                record["reference_text"],
                "en",
            ),
            bundle=runtime["bundle"],
            retrieval={**retrieved, **decoded, "timing": {"model_retrieval_seconds": 0.01}},
            audio_timing={"audio_load_seconds": 0.01},
        )
        row.update(
            raw_variant_ranked_matches=retrieved["raw_variant_ranked_matches"],
            variant_shortlist_candidates=retrieved["variant_shortlist_candidates"],
            audio_sha256=digest(Path(record["audio_path"])),
        )
        row["row_sha256"] = _row_digest(row)
        rows.append(row)
    table_sha = digest(assets / "keyword_table.json")
    inputs = {
        str(p): {"sha256": digest(p)}
        for p in [dataset, target_path, VOCAB, head, assets / "keyword_table.json"]
        + [Path(r["audio_path"]) for r in records]
    }
    write(
        results / "run_config.json",
        {
            "configuration": cfg,
            "inputs": inputs,
            "group": {"config": detector_cfg, "target_ids": runtime["target_ids"]},
            "records": records,
            "code_sha256": {},
            "duplicate_audio_file_hashes": 0,
            "audio_metadata": [{"duration_seconds": 1} for _ in records],
        },
    )
    write(results / "keyword_table.json", table)
    write(
        results / "report.json",
        {
            "status": "completed",
            "candidate_scope": "joint_multilingual",
            "qwen_decoder_used": False,
            "table_sha256": table_sha,
            "checkpoint_sha256": digest(head),
            "table_replacement": selection,
            "sample_count": 3,
            "keyword_count": len(table["entries"]),
            "metrics": {},
        },
    )
    (results / NAME).mkdir()
    external._write_jsonl(results / NAME / "retrieval_details.jsonl", rows)
    for k in (5, 7):
        write(results / "delivery" / f"{NAME}_top{k}.json", {})
    seal(results)
    controlled = {
        "baseline_config": str(config),
        "source_table_sha256": table_sha,
        "maximum_targets_per_audio": 4,
        "keep_zero_target_samples": True,
        "expected_selection": {
            "samples": 2,
            "positive_samples": 1,
            "negative_samples": 1,
            "target_sample_pairs": 2,
            "excluded_samples": 1,
        },
        "remove_english_non_targets": {"fragments": ["connec"], "neighbors": ["warnings"]},
        "output": "controlled",
    }
    control_config = tmp_path / "config/controlled.json"
    write(control_config, controlled)
    return tmp_path, control_config, cfg, rows


def test_prune_preserves_foreign_variants_and_rejects_target_removal():
    table = {
        "entries": combine_variants(
            [variant("connec"), variant("connec", "es"), variant("alpha", role="target")]
        )
    }
    before = deepcopy(table)
    new, report = control.prune_table(table, {"review": ["connec"]}, ["alpha"])
    assert table == before
    assert report["removed_english_surfaces_retained_in_other_languages"] == ["connec"]
    assert next(e for e in new["entries"] if e["normalized"] == "connec")["languages"] == ["es"]
    assert [v for e in new["entries"] for v in e["variants"] if v["language"] == "es"] == [
        variant("connec", "es")
    ]
    for words in (["alpha"], ["absent"], ["connec", "connec"]):
        with pytest.raises(ValueError):
            control.prune_table(table, {"review": words}, ["alpha"])


def test_selection_counts_distinct_whole_words_and_includes_zero_and_four():
    targets = ["alpha", "bravo", "charlie", "delta", "echo"]
    records = [
        {"reference_text": text}
        for text in ("alpha alpha", "alphas", "alpha bravo charlie delta", " ".join(targets))
    ]
    included, excluded, report = control.select_records(records, targets, 4)
    assert [r["target_count"] for r in included] == [1, 0, 4]
    assert len(excluded) == 1 and report["target_sample_pairs"] == 5


def test_production_controls_match_agreed_37_and_returned_histogram():
    cfg = json.loads(Path("configs/ziyan_control_le4_drop37.workzone.json").read_text())
    groups = cfg["remove_english_non_targets"]
    assert len(groups["suspected_fragments"]) == 13 and len(groups["non_target_neighbors"]) == 24
    targets = set(Path("configs/ziyan_english_targets.v1.txt").read_text().splitlines())
    words = [w for values in groups.values() for w in values]
    assert len(set(words)) == 37 and not targets.intersection(words)
    histogram = {
        0: 299,
        1: 175,
        2: 101,
        3: 44,
        4: 35,
        5: 17,
        6: 14,
        7: 5,
        8: 2,
        9: 6,
        10: 2,
        11: 5,
        12: 2,
        13: 2,
        15: 1,
        18: 1,
        19: 1,
    }
    assert sum(n for k, n in histogram.items() if k <= 4) == 654
    assert sum(k * n for k, n in histogram.items() if k <= 4) == 649


def test_complete_cpu_pair_and_sealed_safe_resume(source, monkeypatch):
    root, cfg, original, rows = source
    inputs_before = {p: digest(p) for p in root.rglob("*") if p.is_file()}
    # Access to actual audio is unnecessary, even after identity hashes have been recorded.
    for row in rows:
        Path(row["audio_path"]).unlink()
    output = root / "controlled"
    report = control.run(root, cfg, output)
    assert report["selection"]["samples"] == 2
    assert report["selection"]["target_sample_pairs"] == 2
    assert report["baseline_reproduced_for_all_selected_samples"]
    assert report["index_sizes"]["old_surfaces"] == 7
    assert report["index_sizes"]["new_surfaces"] == 6  # connec remains Spanish only
    assert report["index_sizes"]["old_indexed_pronunciations"] == 7
    assert report["index_sizes"]["new_indexed_pronunciations"] == 6
    assert report["by_target_count"]["0"]["new"]["top7"]["target_negative_sample_count"] == 1
    new_rows = [
        json.loads(s) for s in (output / NAME / "retrieval_details.jsonl").read_text().splitlines()
    ]
    assert all(
        m["start_step"] is None and m["end_step"] is None
        for r in new_rows
        for m in r["raw_ranked_matches"]
    )
    for k in (5, 7):
        delivery = json.loads((output / "delivery" / f"{NAME}_top{k}.json").read_text())
        assert set(delivery) == {"item_0", "item_1"}
        assert not any(m["word"] == "warnings" for values in delivery.values() for m in values)
    verify_seal(output)
    with ZipFile(output / "return_bundle.zip") as z:
        assert "report.json" in z.namelist()
        assert not any("model" in s or "sample_shards" in s for s in z.namelist())
    monkeypatch.setattr(
        replay, "retrieve", lambda *a: pytest.fail("completed resume must not replay")
    )
    assert control.run(root, cfg, output, resume=True) == report
    with pytest.raises(FileExistsError):
        control.run(root, cfg, output)
    for path, value in inputs_before.items():
        if path.exists():
            assert digest(path) == value


def test_baseline_mismatch_fails_before_new_result(source, monkeypatch):
    root, cfg, _, _ = source
    retrieve = replay.retrieve

    def wrong(*args):
        result = retrieve(*args)
        result["raw_variant_ranked_matches"][0]["score"] -= 0.05
        return result

    monkeypatch.setattr(replay, "retrieve", wrong)
    with pytest.raises(ValueError, match="baseline replay differs"):
        control.run(root, cfg, root / "bad")
    assert not (root / "bad/report.json").exists()


def test_changed_sources_and_corrupt_completed_outputs_refused(source):
    root, cfg, _, _ = source
    output = root / "controlled"
    control.run(root, cfg, output)
    (output / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed file hash differs"):
        control.run(root, cfg, output, resume=True)


def test_resume_code_or_config_drift_refused(source):
    root, cfg, _, _ = source
    output = root / "controlled"
    control.run(root, cfg, output)
    value = json.loads(cfg.read_text())
    value["maximum_targets_per_audio"] = 3
    write(cfg, value)
    with pytest.raises(ValueError, match="resume configuration/source/code differs"):
        control.run(root, cfg, output, resume=True)


def test_source_directory_and_missing_ctc_refused(source):
    root, cfg, base, rows = source
    with pytest.raises(ValueError, match="separate"):
        control.run(root, cfg, Path(base["output"]) / "new")
    row = deepcopy(rows[0])
    row["decoded_confidences"] = []
    with pytest.raises(ValueError, match="invalid saved CTC"):
        control._check_ctc(row, load_phoneme_vocab(VOCAB))


def test_interrupted_resume_reuses_only_verified_pairs(source, monkeypatch):
    root, cfg, _, _ = source
    original = replay.retrieve
    calls = []

    def interrupt(*args):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("interrupted after first complete pair")
        return original(*args)

    monkeypatch.setattr(replay, "retrieve", interrupt)
    output = root / "controlled"
    with pytest.raises(RuntimeError, match="interrupted"):
        control.run(root, cfg, output)
    assert len(list((output / "sample_shards").glob("*.json"))) == 1
    calls.clear()
    report = control.run(root, cfg, output, resume=True)
    assert report["status"] == "completed" and len(calls) == 2


def test_rebuilt_index_can_add_candidates_missing_from_original_top64(source):
    root, _, base, _ = source
    detector = json.loads((Path(base["output"]) / "run_config.json").read_text())["group"]["config"]
    vocab = load_phoneme_vocab(VOCAB)
    words = ["distractor" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(70)]
    table = {"entries": combine_variants([variant(w) for w in words])}
    old = control._runtime(table, root / "unused.json", vocab, [])
    replay.prepare_runtime(old, detector)
    ids = list(tokenize_ipa_to_vocab("t a", vocab).token_ids)
    row = {"decoded_token_ids": ids, "decoded_confidences": [0.95] * len(ids)}
    before = replay.retrieve(row, old, detector)
    saved = {m["surface"] for m in before["raw_ranked_matches"]}
    assert len(saved) == 64
    missing = sorted(set(words) - saved)[0]
    removed = sorted(saved)[:10]
    pruned, _ = control.prune_table(table, {"neighbors": removed}, [missing])
    new = control._runtime(pruned, root / "new.json", vocab, [missing])
    replay.prepare_runtime(new, detector)
    after = replay.retrieve(row, new, detector)
    assert missing in {m["surface"] for m in after["raw_ranked_matches"]}
    assert len(after["raw_ranked_matches"]) == 60


def test_partial_report_packaging_can_resume_without_overwrite(source, monkeypatch):
    root, cfg, _, _ = source
    original = control._package
    monkeypatch.setattr(
        control, "_package", lambda _: (_ for _ in ()).throw(RuntimeError("packaging"))
    )
    output = root / "controlled"
    with pytest.raises(RuntimeError, match="packaging"):
        control.run(root, cfg, output)
    original_report_sha = digest(output / "report.json")
    monkeypatch.setattr(control, "_package", original)
    monkeypatch.setattr(replay, "retrieve", lambda *a: pytest.fail("verified pairs must be reused"))
    assert control.run(root, cfg, output, resume=True)["status"] == "completed"
    assert digest(output / "report.json") == original_report_sha
