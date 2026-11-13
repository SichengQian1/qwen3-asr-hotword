from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.arabic_per_diagnostics import (
    agreement_scores,
    decoded_summary,
    exposure,
    fold_phones,
)
from qwen_hotword.training.ctc_overfit import CachedSample, ExperimentRecord
from qwen_hotword.training.feature_cache import cache_feature_split
from qwen_hotword.training.sharded_ctc import load_disk_feature_cache, train_sharded_ctc_head
from scripts import arabic_diagnostic_common as common
from scripts import check_arabic_text_agreement as agreement
from scripts import diagnose_arabic_labels as labels
from scripts import diagnose_frozen_ctc as diagnostic
from scripts import run_en_ar_merged as merged
from scripts.run_en_ar_training import NATIVE_VOCAB, VOCAB, combine_caches


def write_rows(path: Path, values: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))


@pytest.fixture
def source_run(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    source = tmp_path / "source"
    model = tmp_path / "model"
    model.mkdir()
    write_json(model / "config.json", {})
    write_json(model / "model.safetensors.index.json", {})
    manifests, cache_views = {}, {}

    def extract(records: list[ExperimentRecord], _: Any, **kwargs: Any) -> Any:
        return (
            [CachedSample(r.sample_id, torch.ones((6, 1024)), r.token_ids) for r in records],
            317477504,
            0.01,
        )

    for split in ("train", "validation"):
        parts = []
        for language, vocab in (("en", VOCAB), ("ar", NATIVE_VOCAB)):
            directory = source / "english" if language == "en" else source
            manifest = directory / f"{split}.jsonl"
            records, data = [], []
            for i in range(2):
                sample_id = f"{language}-{split}-{i}"
                token_ids = (2, 3) if language == "en" else (72, 73)
                records.append(
                    ExperimentRecord(
                        sample_id, tmp_path / f"{sample_id}.wav", "في", language, token_ids, 2, 2
                    )
                )
                data.append(
                    {
                        "id": sample_id,
                        "split": split,
                        "language": language,
                        "source_group": ("ar_egy" if i == 0 else "ar_msa")
                        if language == "ar"
                        else "en_egy",
                        "source": f"source-{language}-{i}.json",
                        "source_row": 1 if split == "train" else 2,
                        "audio_path": str(records[-1].audio_path),
                        "audio_sha256": sample_id,
                        "text": "في",
                        "duration_seconds": 1.0,
                        "phoneme_token_ids": list(token_ids),
                        "ctc_time_upsampling_factor": 2,
                    }
                )
            write_rows(manifest, data)
            manifests[f"{language}/{split}"] = {
                "path": str(manifest),
                "sha256": sha(manifest),
                "samples": 2,
            }
            cache_feature_split(
                records,
                None,
                directory / "cache" / split,
                split=split,
                source_manifest_path=manifest,
                model_path=model,
                model_dtype="bfloat16",
                vocab_path=vocab,
                samples_per_shard=2,
                extractor=extract,
            )
            parts.append(
                load_disk_feature_cache(
                    directory / "cache" / split,
                    expected_split=split,
                    source_manifest_path=manifest,
                    vocab_path=vocab,
                )
            )
        cache_views[split] = combine_caches(parts, source)
    train_sharded_ctc_head(
        cache_views["train"],
        cache_views["validation"],
        load_phoneme_vocab(NATIVE_VOCAB),
        source / "head",
        vocab_path=NATIVE_VOCAB,
        device="cpu",
        epochs=1,
        minimum_epochs=1,
        validation_sample_groups={
            r["id"]: r["language"]
            for key, m in manifests.items()
            if key.endswith("validation")
            for r in common.rows(Path(m["path"]))
        },
        head_type="temporal_upsample",
        head_hidden_dimension=4,
        head_dropout=0.1,
        early_stopping_metric="validation_macro_per",
        checkpoint_selection_metric="validation_macro_per",
    )
    plan = {
        "model": {
            "view": str(model),
            "files": {
                "config.json": {"sha256": sha(model / "config.json")},
                "model.safetensors": {"sha256": "fixture"},
            },
            "index_sha256": sha(model / "model.safetensors.index.json"),
        },
        "vocabularies": {
            lang: {"path": str(v), "sha256": sha(v)}
            for lang, v in (("en", VOCAB), ("ar", NATIVE_VOCAB))
        },
        "manifests": manifests,
        "code": [{"path": "old.py", "sha256": "historical-only"}],
    }
    config = common.read(source / "cache/train/cache_config.json")
    write_json(source / "plan.json", plan)
    write_json(
        source / "cache_sources.json",
        {
            "merged_weight_sha256": "fixture",
            "target_vocab_sha256": sha(NATIVE_VOCAB),
            "encoder_identity": {
                k: config[k] for k in ("model", "tap_module", "hidden_size", "feature_dtype")
            },
            **{f"{s}_fingerprint": c.fingerprint for s, c in cache_views.items()},
        },
    )
    write_json(
        source / "report.json",
        {
            "status": "completed",
            "head": common.read(source / "head/report.json"),
            "test_set_used": False,
            "merged_weight_sha256": "fixture",
            "paired_manifest_sha256": {k: v["sha256"] for k, v in manifests.items()},
        },
    )
    return source


def test_diagnostics_and_up3_reuse_real_tiny_caches(
    source_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {str(p): sha(p) for p in source_run.rglob("*") if p.is_file()}
    output = tmp_path / "diag"
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose",
            "--source-run",
            str(source_run),
            "--device",
            "cpu",
            "--grouping",
            "source_group=source_group",
            "--grouping",
            "source=source",
            "--per-sample-output",
            str(output / "rows.jsonl"),
            "--output",
            str(output / "report.json"),
        ],
    )
    assert diagnostic.main() == 0
    report = common.read(output / "report.json")
    decoded = list(common.rows(output / "rows.jsonl"))
    assert len(decoded) == 4
    assert "source_group/ar_msa" in report["checkpoints"][0]["sample_breakdown"]
    assert all(r["input_frames"] == 12 and r["reference"] for r in decoded)
    assert diagnostic.main() == 1  # never overwrites diagnostic evidence
    monkeypatch.setattr(merged, "wait_for_gpu", lambda *a: None)
    import qwen_hotword.training.sharded_ctc as trainer

    original = trainer.train_sharded_ctc_head
    calls = []

    def cpu_train(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["head_time_upsampling_factor"] == 3
        calls.append(kwargs.copy())
        kwargs["device"] = "cpu"
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "train_sharded_ctc_head", cpu_train)
    target = tmp_path / "up3"
    with pytest.raises(ValueError, match=">= 2"):
        merged.retrain_head(source_run, target, 1, 4, False)
    with pytest.raises(ValueError, match="protected"):
        merged.retrain_head(source_run, source_run / "new-head", 3, 4, False)
    result = merged.retrain_head(source_run, target, 3, 4, False)
    assert result["head"]["head_config"]["time_upsampling_factor"] == 3
    assert not (target / "cache").exists()
    base = common.read(source_run / "head/report.json")
    expected = common.training_options(base, 3)
    assert all(calls[0][key] == value for key, value in expected.items())
    with pytest.raises(FileExistsError):
        merged.retrain_head(source_run, target, 3, 4, False)
    merged.retrain_head(source_run, target, 3, 4, True)
    assert calls[-1]["resume"] is True
    with pytest.raises(ValueError, match="changed"):
        merged.retrain_head(source_run, target, 2, 4, True)
    assert before == {str(p): sha(p) for p in source_run.rglob("*") if p.is_file()}
    shard = source_run / "cache/validation/shards/shard-000000.pt"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        common.load_cache(common.completed_run(source_run), "validation", output)


def test_label_stats_only_selected_rows_and_verified_review(
    source_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = common.completed_run(source_run)
    old, new = tmp_path / "old", tmp_path / "new"
    selected = common.manifest_rows(run, "train", ("ar",)) + common.manifest_rows(
        run, "validation", ("ar",)
    )
    generated = [
        {
            "group": r["source_group"],
            "source": r["source"],
            "row": r["source_row"],
            "text": r["text"],
            "ids": r["phoneme_token_ids"],
            "words": ["في"],
            "diacritized_words": ["فِي"],
        }
        for r in selected
    ]
    generated.append({**generated[-1], "row": 99})  # must not count unselected labels
    sources, identities, totals = {}, {}, {}
    for group in ("ar_egy", "ar_msa"):
        inventory = tmp_path / f"{group}.jsonl"
        subset = [r for r in generated if r["group"] == group]
        write_rows(inventory, subset)
        sources[group] = inventory
        identities[group] = sha(inventory)
        totals[group] = {
            "source_rows": len(subset) + 1,
            "ready_rows": len(subset),
            "review_rows": 1,
        }
    monkeypatch.setattr(labels, "SOURCES", sources)
    for directory in (old, new):
        write_rows(directory / "labels.jsonl", generated)
        identity = {"inventories": identities}
        if directory == new:
            identity["source_report_sha256"] = sha(old / "report.json")
        report = {
            "status": "automatic_labels_completed",
            "identity": identity,
            "labels_sha256": sha(directory / "labels.jsonl"),
            "vocab_sha256": sha(NATIVE_VOCAB),
            "by_group": totals,
        }
        write_json(directory / "report.json", report)
        write_json(directory / "contract.json", identity)
        for group in totals:
            relative = Path("parts") / group / "00000"
            review = directory / relative / "review.jsonl"
            write_rows(
                review,
                [
                    {
                        "label_issue": "unsupported_Arabic_letter:'_'"
                        if group == "ar_egy"
                        else "no_analysis:abc"
                    }
                ],
            )
            write_json(
                directory / relative / "completed.json",
                {
                    "group": group,
                    "rows": totals[group]["source_rows"],
                    "ready_rows": totals[group]["ready_rows"],
                    "review": str(relative / "review.jsonl"),
                    "review_sha256": sha(review),
                },
            )
    before = {str(p): sha(p) for d in (old, new) for p in d.rglob("*") if p.is_file()}
    report = labels.stats(run, old, new)
    assert report["matched_rows"] == 4
    assert report["selected_label_exposure"]["train/ar_msa"]["final_i_y (fake j)/words"] == 1
    assert report["review"]["ar_egy"]["details"]["unsupported_Arabic_letter"] == [("_", 1)]
    assert before == {str(p): sha(p) for d in (old, new) for p in d.rglob("*") if p.is_file()}
    (new / "parts/ar_msa/00000/review.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="pinned"):
        labels.stats(run, old, new)


def test_fold_and_agreement_semantics() -> None:
    phones = ["ar_ipa:aː", "ar_ipa:bː", "ar_ipa:ʔ", "en_ipa:aː", "en_ipa:ʔ"]
    assert fold_phones(phones, "F1") == ["ar_ipa:a", *phones[1:]]
    assert fold_phones(phones, "F2") == [phones[0], "ar_ipa:b", *phones[2:]]
    assert fold_phones(phones, "F3") == phones[:2] + phones[3:]
    summary = decoded_summary(
        [
            {
                "reference": phones,
                "hypothesis": phones[:2] + phones[3:],
                "errors": 1,
                "phoneme_error_rate": 0.2,
                "groups": {"source_group": "ar_egy"},
                "duration_seconds": 1.0,
            }
        ]
    )["all"]
    assert summary["exact_per"] == 0.2
    assert summary["folds"]["F3"]["per"] == 0
    assert summary["folds"]["F3"]["reference_tokens"] == 4
    assert agreement_scores("إِلَى مَدْرَسَةٍ!", "الي مدرسه")["cer"] == 0
    hit = exposure(
        {"group": "ar_egy", "words": ["عشان"], "diacritized_words": ["عَشَأْن"], "ids": [1, 2]}
    )
    assert hit["ashaan/estimated_units"] == 2


def test_agreement_resume_and_review_dedup(tmp_path: Path) -> None:
    samples = [
        {
            "id": str(i),
            "sample_id": str(i),
            "split": "train",
            "source_group": "ar_egy",
            "language": "ar",
            "source": "s",
            "source_row": i,
            "audio_path": f"{i}.wav",
            "audio_sha256": str(i).zfill(3),
            "text": "في",
            "phoneme_error_rate": i / 200,
        }
        for i in range(120)
    ]
    chosen = labels.select_review(samples)
    assert len(chosen) == len({r["id"] for r in chosen}) == 100
    assert chosen[0]["id"] == "119" and chosen[50]["id"] == "0"
    subset = agreement.select_samples(samples, [], 3)
    calls = []

    def transcribe(**kwargs: Any) -> list[Any]:
        calls.append(kwargs)
        return [SimpleNamespace(text="في") for _ in kwargs["audio"]]

    model = SimpleNamespace(transcribe=transcribe)
    result = agreement.run_predictions(model, subset, tmp_path, 2)
    (tmp_path / "parts/000002.json.pending").write_text("partial ignored")
    resumed = agreement.run_predictions(model, subset, tmp_path, 2)
    assert result == resumed and len(calls) == 2
    assert calls[0]["language"] == "Arabic" and result[0]["wer"] == 0
    changed = [{**subset[0], "text": "غير"}, *subset[1:]]
    with pytest.raises(ValueError, match="identity"):
        agreement.run_predictions(model, changed, tmp_path, 2)
    with pytest.raises(FileExistsError):
        agreement.execute(argparse.Namespace(output_dir=tmp_path, resume=False))


def test_text_agreement_full_entry_resume(
    source_run: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "base"
    model_dir.mkdir()
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        write_json(model_dir / name, {"model_type": "qwen3_asr"})
    (model_dir / "model.safetensors").write_bytes(b"fake; never loaded")
    calls = []

    def decode(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return [SimpleNamespace(text="في") for _ in kwargs["audio"]]

    monkeypatch.setattr(agreement, "load_qwen", lambda *a: SimpleNamespace(transcribe=decode))
    args = argparse.Namespace(
        output_dir=tmp_path / "agreement",
        source_run=source_run,
        model=model_dir,
        per_source=1,
        batch_size=2,
        gpu=4,
        max_new_tokens=1024,
        resume=False,
        per_sample=None,
    )
    result = agreement.execute(args)
    assert result["samples"] == 4 and len(calls) == 2
    assert result["by_split_group_source"]["validation/all"]["corpus_WER"] == 0
    args.resume = True
    assert agreement.execute(args) == result
    assert len(calls) == 2
    args.max_new_tokens = 512
    with pytest.raises(ValueError, match="resume"):
        agreement.execute(args)
