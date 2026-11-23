from __future__ import annotations

import json
from pathlib import Path

import pytest

from qwen_hotword.evaluation.egy_hotword_review import (
    load_candidates,
    nested_pairs,
    scan_training,
    tokens,
    verify_run,
    write_review,
)
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json


def candidate(surface="القاهرة", identifier="a", aliases=None):
    return {
        "id": identifier,
        "surface": surface,
        "category": "地名",
        "gloss_zh": "开罗",
        "aliases": aliases or [],
        "review_status": "pending",
    }


def row(text, bucket="open_source", **kwargs):
    return {
        "id": "sample",
        "text": text,
        "language": "ar",
        "split": "train",
        "source_group": bucket,
        "source": "original.json",
        "source_row": 1,
        "duration_seconds": 5,
        **kwargs,
    }


def test_exact_words_punctuation_and_prefix_review_are_separate():
    candidates = [candidate(), candidate("جامعة القاهرة", "b")]
    result, stats = scan_training(
        [
            row("القاهرة القاهرة"),
            row("بالقاهرة"),
            row("جامعة القاهرة"),
            row("جامعة، القاهرة"),
            row("القاهرة الجديدة"),
            row("القاهرة", language="en"),
        ],
        candidates,
        lambda _: None,
    )
    assert result[0]["training_evidence"]["exact_samples"] == 4
    assert result[0]["training_evidence"]["exact_occurrences"] == 5
    assert result[0]["training_evidence"]["attached_prefix_review_samples"] == 1
    assert result[1]["training_evidence"]["exact_samples"] == 1
    assert stats["english_rows_skipped"] == 1
    assert nested_pairs(candidates) == [
        {"short_id": "a", "long_id": "b", "short": "القاهرة", "long": "جامعة القاهرة"}
    ]


def test_no_substring_and_no_destructive_arabic_merging():
    candidates = [candidate("مصر"), candidate("الإسكندرية", "b")]
    result, _ = scan_training([row("مصري المصري الاسكندرية")], candidates, lambda _: None)
    assert all(r["training_evidence"]["exact_samples"] == 0 for r in result)
    assert tokens("الْقَاهِرَة") == tokens("القاهرة")
    assert tokens("أ إ ا آ ى ي ة ه") == ("أ", "إ", "ا", "آ", "ى", "ي", "ة", "ه")


def test_explicit_aliases_count_once_and_unseen_candidates_retained():
    candidates = [candidate("الإسكندرية", aliases=["الاسكندرية"]), candidate("أسوان", "b")]
    result, _ = scan_training(
        [row("الاسكندرية الإسكندرية", "gaoshan_train")], candidates, lambda _: None
    )
    ev = result[0]["training_evidence"]
    assert ev["exact_samples"] == 1 and ev["exact_occurrences"] == 2
    assert ev["by_source_group"] == {"gaoshan_train": 1}
    assert result[1]["corpus_status"] == "not_observed_under_conservative_match"
    assert all(r["approved_for_retrieval"] is False for r in result)


@pytest.mark.parametrize("split", ["test", "validation", None])
def test_heldout_records_rejected(split):
    with pytest.raises(ValueError, match="train records only"):
        scan_training([row("القاهرة", split=split)], [candidate()], lambda _: None)


def test_candidate_collisions_rejected(tmp_path):
    p = tmp_path / "c.json"
    write_json(
        p,
        {
            "language": "ar",
            "schema_version": 1,
            "candidates": [candidate(), candidate("الْقاهرة", "b")],
        },
    )
    with pytest.raises(ValueError, match="multiple candidates"):
        load_candidates(p)


def test_editorial_preview_does_not_claim_measured_coverage(tmp_path):
    output = tmp_path / "out"
    write_review(output, [candidate()], {"corpus_scan_completed": False})
    report = json.loads((output / "report.json").read_text())
    assert report["test_set_read"] is False and report["retrieval_started"] is False
    assert "待工区核实" in (output / "候选词表.md").read_text()
    with pytest.raises(FileExistsError):
        write_review(output, [candidate()], {})


def fixture_config(root):
    run = root / "run"
    (run / "head").mkdir(parents=True)
    (run / "head/ctc_head_best.pt").write_bytes(b"test head")
    (run / "train.jsonl").write_text(json.dumps(row("القاهرة")) + "\n")
    (root / "vocab.json").write_text("{}")
    expected = {
        "best_epoch": 29,
        "num_classes": 167,
        "time_upsampling_factor": 3,
        "checkpoint_sha256": sha(run / "head/ctc_head_best.pt"),
        "train_manifest_sha256": sha(run / "train.jsonl"),
        "vocab_sha256": sha(root / "vocab.json"),
        "model_weight_sha256": "r15",
    }
    write_json(
        run / "report.json",
        {
            "status": "completed",
            **expected,
            "head": {
                "best_epoch": 29,
                "head_config": {"time_upsampling_factor": 3, "num_classes": 167},
            },
            "selection": {
                "arabic_msa_used": False,
                "manifests": {"train": {"sha256": expected["train_manifest_sha256"], "samples": 1}},
            },
        },
    )
    return {"source_run": "run", "vocab": "vocab.json", "expected": expected}


def test_run_verification_never_needs_heldout_or_encoder_weights(tmp_path):
    config = fixture_config(tmp_path)
    verified = verify_run(tmp_path, config)
    assert verified["expected_rows"] == 1
    assert not (tmp_path / "run/test.jsonl").exists()
    (tmp_path / "run/train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="pinned input"):
        verify_run(tmp_path, config)


def test_wrong_epoch_or_encoder_is_rejected(tmp_path):
    config = fixture_config(tmp_path)
    config["expected"]["best_epoch"] = 24
    with pytest.raises(ValueError, match="epoch 29"):
        verify_run(tmp_path, config)
    config["expected"]["best_epoch"] = 29
    config["expected"]["model_weight_sha256"] = "base"
    with pytest.raises(ValueError, match="model_weight"):
        verify_run(tmp_path, config)


def test_committed_seed_is_valid():
    root = Path(__file__).resolve().parents[1]
    rows = load_candidates(root / "configs/hotwords/egy_entities_terms.review.v1.json")
    assert len(rows) == 164
    assert all(r["review_status"] == "pending" for r in rows)


def test_full_review_build_reads_only_train_and_packages_real_evidence(tmp_path):
    import base64
    import hashlib
    import zipfile

    from scripts.prepare_egy_hotword_candidates import build

    config = fixture_config(tmp_path)
    config.update(
        {
            "candidate_file": "candidates.json",
            "future_evaluation": {
                "arabic_duration_ratios": {
                    "open_source": 0.8,
                    "gaoshan_train": 0.1,
                    "coarse_train": 0.1,
                },
                "requires_user_candidate_approval": True,
            },
        }
    )
    write_json(tmp_path / "config.json", config)
    write_json(
        tmp_path / "candidates.json",
        {
            "schema_version": 1,
            "language": "ar",
            "candidates": [candidate()],
        },
    )
    backend = tmp_path / "src/qwen_hotword/evaluation/egy_hotword_review.py"
    backend.parent.mkdir(parents=True)
    backend.write_text("# test identity")
    # If build tries to open either held-out path it cannot parse these directories.
    (tmp_path / "run/test.jsonl").mkdir()
    (tmp_path / "run/validation.jsonl").mkdir()
    output = tmp_path / "review"
    result = build(tmp_path, tmp_path / "config.json", output, preview=False)
    assert result["training_counts"]["candidates_with_exact_training_evidence"] == 1
    assert result["retrieval_started"] is False
    packed = output / "return_review.zip"
    with zipfile.ZipFile(packed) as bundle:
        data = json.loads(bundle.read("candidates.json"))
        assert data["candidates"][0]["training_evidence"]["examples"][0]["text"] == "القاهرة"
        assert not any(name.endswith(".pt") for name in bundle.namelist())
    chunks = []
    for part in sorted(output.glob("return_part_*.txt")):
        assert part.stat().st_size <= 32768
        header, payload = part.read_text().splitlines()
        assert (
            json.loads(header)["archive_sha256"] == hashlib.sha256(packed.read_bytes()).hexdigest()
        )
        chunks.append(payload)
    assert base64.b64decode("".join(chunks)) == packed.read_bytes()
    with pytest.raises(ValueError, match="output exists"):
        build(tmp_path, tmp_path / "config.json", output, preview=False)
