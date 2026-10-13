from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_hotword.evaluation.reviewed_targets import import_review
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import combine_variants
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.reviewed_joint import (
    collapse_variants,
    export_joint_group,
    joint_bundle,
)
from qwen_hotword.inference.reviewed_retrieval import run_reviewed_retrieval
from qwen_hotword.inference.wave_retrieval import export_group
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.spanish_capacity import _sha

VOCAB = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json").resolve()
SCOPE = "multilingual_declared_targets_review_only"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def read(path):
    return json.loads(path.read_text())


def retrieval(entries, negative=False):
    matches = (
        []
        if negative
        else [
            {
                "hotword_id": e.hotword_id,
                "surface": e.surface,
                "score": 0.99 - n * 0.01,
                "edit_ratio": 0.0,
                "posterior_confidence": 0.9,
            }
            for n, e in enumerate(entries)
        ]
    )
    return {
        "raw_ranked_matches": matches,
        "selected_matches": matches[:5],
        "shortlist_candidates": len(matches),
        "postings_visited": 2,
        "timing": {
            key: 0.01
            for key in (
                "ctc_processor_seconds",
                "ctc_encoder_seconds",
                "ctc_head_seconds",
                "ctc_decode_seconds",
                "anchor_query_seconds",
                "hotword_matching_seconds",
                "hotword_sorting_seconds",
                "hotword_selection_seconds",
                "pure_retrieval_seconds",
                "model_retrieval_seconds",
            )
        },
    }


def fixture(root):
    cfg = read(Path("configs/reviewed_target_retrieval.workzone.json"))
    cfg.update(
        reviewed_table="review/keyword_table.json",
        model="model",
        checkpoint="head.pt",
        vocab=str(VOCAB),
        output="new_run",
        baselines={lang: f"baseline_{lang}" for lang in ("en", "es", "pt")},
    )
    (root / "head.pt").write_bytes(b"head")
    cfg["checkpoint_sha256"] = _sha(root / "head.pt")
    write(root / "model/config.json", {})
    vocab = load_phoneme_vocab(VOCAB)
    decisions = []
    for lang in ("en", "es", "pt"):
        # Distinct pronunciation for cross-language Shared is intentionally retained.
        ipa = "b a" if lang == "es" else "a b"
        tokens = tokenize_ipa_to_vocab(ipa, vocab)
        groups = [v["source_group"] for v in cfg["datasets"].values() if v["language"] == lang]
        for surface, decision in (
            ("A", "removed_basic"),
            ("Shared", "retained"),
            ("Fire", "retained"),
            ("Only" + lang, "retained"),
            ("DeleteMe", "retained"),
        ):
            decisions.append(
                {
                    "language": lang,
                    "surface": surface,
                    "normalized": surface.casefold(),
                    "pronunciation": ipa,
                    "phoneme_tokens": tokens.tokens,
                    "token_ids": tokens.token_ids,
                    "source_groups": groups,
                    "decision": decision,
                }
            )
        legacy = root / f"table_{lang}.json"
        surfaces = [f"Filler{n}" for n in range(6)] + [
            "Fire",
            "Shared",
            "Only" + lang,
            "DeleteMe",
            "A",
        ]
        write(
            legacy,
            {
                "language": lang,
                "keyword_sets": {"all_keywords": surfaces},
                "keyword_phonemes": {s: ipa for s in surfaces},
            },
        )
        bundle = external.load_keyword_bias(
            legacy, vocab=vocab, keyword_set="all_keywords", language=lang
        )
        baseline = root / cfg["baselines"][lang]
        (baseline / "delivery").mkdir(parents=True)
        plan = {"configuration": {}, "inputs": {}, "groups": {}}
        for name, info in cfg["datasets"].items():
            if info["language"] != lang:
                continue
            info["expected_samples"] = 2
            base = root / "audio" / name
            (base / "wav").mkdir(parents=True)
            (base / "wav/same.wav").write_bytes(b"positive")
            (base / "wav/negative.wav").write_bytes(b"negative")
            (base / "transcripts.txt").write_text(
                f"same\tFire Shared Only{lang} A\nnegative\tnothing here\n"
            )
            source = external.SourceSpec(name, base / "wav", base / "transcripts.txt")
            records, audit = external.build_external_dataset(
                (source,), language="pt-BR" if lang == "pt" else lang
            )
            group = {
                "config": external._run_config(
                    model=root / "model",
                    checkpoint=root / "head.pt",
                    vocab_path=VOCAB,
                    keyword_path=legacy,
                    sources=(source,),
                    keyword_set="all_keywords",
                    device="cuda:0",
                    dtype="bfloat16",
                    language=lang,
                    saved_raw_rank_depth=64,
                ),
                "audit": audit,
                "records": [r.to_dict() for r in records],
                "target_ids": [
                    e.hotword_id for e in bundle.entries if not e.surface.startswith("Filler")
                ],
            }
            if lang == "en":
                group["config"]["table_filler_exclusion_used_transcripts"] = True
            plan["groups"][name] = group
            rows = []
            for record in records:
                plan["inputs"][str(record.audio_path)] = {"sha256": _sha(record.audio_path)}
                rows.append(
                    external._build_result_row(
                        record,
                        bundle=bundle,
                        retrieval=retrieval(bundle.entries, record.sample_id == "negative"),
                        audio_timing={"duration_seconds": 1.0, "audio_load_seconds": 0.01},
                    )
                )
            (baseline / name).mkdir()
            export_group(baseline, name, group, bundle, rows)
        write(baseline / "run_config.json", plan)
        write(
            baseline / "report.json",
            {
                "status": "completed",
                "checkpoint_sha256": cfg["checkpoint_sha256"],
                "top5_reproduced": True,
                "top7_exact_replay": True,
            },
        )
        seal(baseline)
    directory = root / "review"
    write(
        directory / "keyword_table.json",
        {
            "schema_version": 1,
            "scope": SCOPE,
            "entries": combine_variants(decisions),
        },
    )
    (directory / "all_target_decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decisions)
    )
    write(directory / "report.json", {"scope": SCOPE})
    seal(directory)
    # Simulate the user's authorized edit; deliberately do NOT reseal the source.
    edited = read(directory / "keyword_table.json")
    edited["entries"] = [e for e in edited["entries"] if e["surface"] != "DeleteMe"]
    write(directory / "keyword_table.json", edited)
    write(root / "config.json", cfg)
    return root / "config.json", cfg


def mocks(monkeypatch):
    calls = {"loads": [], "retrieved": 0}

    def factory(**kwargs):
        lang = {"English": "en", "Spanish": "es", "Portuguese": "pt-BR"}[kwargs["language"]]
        assert {e.language for e in kwargs["hotwords"]} == {"en", "es", "pt-BR"}
        assert lang in {e.language for e in kwargs["hotwords"]}
        assert {e.surface for e in kwargs["hotwords"]} == {
            "Fire",
            "Shared",
            "Onlyen",
            "Onlyes",
            "Onlypt",
        }
        calls["loads"].append(kwargs["language"])
        return SimpleNamespace(**kwargs)

    def retrieve(detector, waveform, vocab):
        calls["retrieved"] += 1
        return retrieval(detector.hotwords, waveform == b"negative")

    monkeypatch.setattr(
        external,
        "_load_audio",
        lambda path: (
            path.read_bytes(),
            {"duration_seconds": 1.0, "audio_load_seconds": 0.01},
        ),
    )
    return calls, {"detector_factory": factory, "retrieve_function": retrieve}


def test_review_edited_checksum_is_accepted_but_labels_provenance_are_unchanged(tmp_path):
    _, cfg = fixture(tmp_path)
    table = tmp_path / cfg["reviewed_table"]
    identities = {}
    bundles, targets, audit = import_review(table, load_phoneme_vocab(VOCAB), identities)
    assert audit["reviewed_table_sha256"] != audit["original_generated_table_sha256"]
    assert audit["manually_deleted_unique_surfaces"] == 1
    assert audit["keywords_by_language"] == {"en": 3, "es": 3, "pt": 3}
    assert all(len(v) == 3 for by_group in targets.values() for v in by_group.values())
    assert bundles["en"].phonemes["Shared"] != bundles["es"].phonemes["Shared"]


def test_eleven_groups_22_deliveries_same_target_baseline_and_exact_resume(tmp_path, monkeypatch):
    config, cfg = fixture(tmp_path)
    hashes = {p: _sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    calls, kw = mocks(monkeypatch)
    output = tmp_path / "new_run"
    audit = run_reviewed_retrieval(tmp_path, config, output, audit_only=True, **kw)
    assert not output.exists() and calls == {"loads": [], "retrieved": 0}
    assert audit["delivery_file_count"] == 22
    report = run_reviewed_retrieval(tmp_path, config, output, **kw)
    assert report["sample_count"] == 22 and report["delivery_file_count"] == 22
    assert calls == {"loads": ["English", "Spanish", "Portuguese"], "retrieved": 22}
    for name, info in cfg["datasets"].items():
        directory = output / "languages" / info["language"]
        for k in (5, 7):
            delivered = read(directory / "delivery" / f"{name}_top{k}.json")
            assert set(delivered) == {"same", "negative"}
            assert delivered["negative"] == [] and len(delivered["same"]) == 5
            assert all(set(item) == {"word", "phoneme"} for item in delivered["same"])
            metrics = report["groups"][name]
            assert metrics["current"][f"top{k}"]["expected_target_sample_pairs"] == 3
            assert (
                metrics["baseline_same_retained_targets"][f"top{k}"]["expected_target_sample_pairs"]
                == 3
            )
            assert metrics["current"][f"top{k}"]["gated_target_hits"] == 3
        assert (
            report["groups"][name]["baseline_same_retained_targets"]["top7"]["gated_target_hits"]
            == 1
        )
    assert report["by_language"]["en"]["current"]["top7"]["gated_target_hits"] == 9
    assert all(_sha(p) == h for p, h in hashes.items())
    for line in (output / "sha256.txt").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        assert _sha(output / name) == sha
    assert run_reviewed_retrieval(tmp_path, config, output, resume=True, **kw) == report
    assert calls["retrieved"] == 22 and len(calls["loads"]) == 3
    with pytest.raises(FileExistsError):
        run_reviewed_retrieval(tmp_path, config, output, **kw)
    shard = next((output / "languages/en/en_chile/sample_shards").glob("*.json"))
    data = read(shard)
    data["selected_hotword_ids"] = ["invented"]
    write(shard, data)
    with pytest.raises(ValueError, match="corrupted sample"):
        run_reviewed_retrieval(tmp_path, config, output, resume=True, **kw)


@pytest.mark.parametrize("change", ["addition", "pronunciation", "duplicate", "decision", "empty"])
def test_review_rejects_unsupported_edits_before_inference(tmp_path, monkeypatch, change):
    config, cfg = fixture(tmp_path)
    table = tmp_path / cfg["reviewed_table"]
    data = read(table)
    if change == "addition":
        data["entries"][0]["id"] = "invented"
    elif change == "pronunciation":
        data["entries"][0]["variants"][0]["pronunciation"] = "b b"
    elif change == "duplicate":
        data["entries"].append(data["entries"][0])
    elif change == "empty":
        data["entries"] = []
    else:
        with (table.parent / "all_target_decisions.jsonl").open("a") as f:
            f.write(" ")
    write(table, data)
    calls, kw = mocks(monkeypatch)
    with pytest.raises(ValueError):
        run_reviewed_retrieval(tmp_path, config, tmp_path / "new_run", **kw)
    assert not (tmp_path / "new_run").exists() and not calls["loads"]


@pytest.mark.parametrize("change", ["review", "audio", "head", "transcript", "baseline"])
def test_resume_cannot_reuse_previous_vocabulary_or_modified_inputs(tmp_path, monkeypatch, change):
    config, cfg = fixture(tmp_path)
    calls, kw = mocks(monkeypatch)
    output = tmp_path / "new_run"
    run_reviewed_retrieval(tmp_path, config, output, **kw)
    if change == "review":
        table = tmp_path / cfg["reviewed_table"]
        data = read(table)
        data["entries"].pop()
        write(table, data)
    else:
        path = (
            tmp_path
            / {
                "audio": "audio/en_chile/wav/same.wav",
                "head": "head.pt",
                "transcript": "audio/en_chile/transcripts.txt",
                "baseline": "baseline_en/report.json",
            }[change]
        )
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="identity differs|SHA256"):
        run_reviewed_retrieval(tmp_path, config, output, resume=True, **kw)
    assert calls["retrieved"] == 22


def test_interruption_resumes_missing_samples_only(tmp_path, monkeypatch):
    config, _ = fixture(tmp_path)
    calls, kw = mocks(monkeypatch)
    retrieve = kw["retrieve_function"]

    def interrupted(*args):
        if calls["retrieved"] == 3:
            raise RuntimeError("interrupted")
        return retrieve(*args)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_reviewed_retrieval(
            tmp_path,
            config,
            tmp_path / "new_run",
            detector_factory=kw["detector_factory"],
            retrieve_function=interrupted,
        )
    assert not (tmp_path / "new_run/report.json").exists()
    report = run_reviewed_retrieval(tmp_path, config, tmp_path / "new_run", resume=True, **kw)
    assert calls["retrieved"] == 22 and report["delivery_file_count"] == 22


def test_shipped_config_and_no_writing_inside_existing_assets(tmp_path):
    cfg = read(Path("configs/reviewed_target_retrieval.workzone.json"))
    assert sum(v["expected_samples"] for v in cfg["datasets"].values()) == 6800
    assert cfg["candidate_scope"] == "joint_multilingual"
    config, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        run_reviewed_retrieval(tmp_path, config, tmp_path / "review/run", audit_only=True)


def test_joint_variants_take_one_slot_and_deliver_winning_ipa(tmp_path):
    _, cfg = fixture(tmp_path)
    table = tmp_path / cfg["reviewed_table"]
    bundles, _, _ = import_review(table, load_phoneme_vocab(VOCAB), {})
    bundle, variants, mapping = joint_bundle(bundles)
    # Equal EN/PT pronunciations do not occupy duplicate Anchor shortlist slots.
    assert len(variants) == 7 and len(bundle.entries) == 5
    spanish_shared = next(e for e in variants if e.surface == "Shared" and e.language == "es")
    ordered = (spanish_shared,) + tuple(e for e in variants if e != spanish_shared)
    config = read(tmp_path / "baseline_en/run_config.json")["groups"]["en_chile"]["config"]
    from qwen_hotword.inference.external_keyword_topk_replay import _validated_source_gate

    collapsed = collapse_variants(
        retrieval(ordered), variants, mapping, _validated_source_gate(config)
    )
    assert len(collapsed["selected_matches"]) == 5
    assert len({m["surface"] for m in collapsed["selected_matches"]}) == 5
    assert collapsed["selected_matches"][0]["matched_language"] == "es"
    record = external.ExternalAudioRecord("one", "en_chile", tmp_path / "wav", "Shared", "en")
    row = external._build_result_row(
        record,
        bundle=bundle,
        retrieval=collapsed,
        audio_timing={"duration_seconds": 1.0, "audio_load_seconds": 0.01},
    )
    out = tmp_path / "joint_export"
    (out / "delivery").mkdir(parents=True)
    (out / "en_chile").mkdir()
    export_joint_group(
        out,
        "en_chile",
        {
            "config": config,
            "target_ids": [mapping[spanish_shared.hotword_id].hotword_id],
        },
        bundle,
        [row],
    )
    delivered = read(out / "delivery/en_chile_top7.json")["one"]
    assert delivered[0] == {"word": "Shared", "phoneme": "b a"}
    assert len(delivered) == 5
    incomplete = retrieval(ordered)
    incomplete["raw_ranked_matches"].pop()
    with pytest.raises(ValueError, match="complete scored"):
        collapse_variants(incomplete, variants, mapping, _validated_source_gate(config))
