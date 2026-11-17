from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.evaluation.aishell_reviewed import (
    assemble_entries,
    select_dataset,
    selections,
    summarize,
)
from qwen_hotword.evaluation.aishell_run import check_checkpoint_metadata, infer, overlap_audit
from qwen_hotword.hotwords.scoring import HotwordMatch
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.zh_en_common import digest, save_rows

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json"


def source(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "kw").mkdir()
    (tmp_path / "kw/kw_candidates.tsv").write_text(
        "word\tfreq\n北京\t6\n上海\t8\n中国\t9\n学习\t30\n", encoding="utf-8"
    )
    records = [
        {"utt_id": "a", "text": "北京北京上海", "keywords": [{"word": "北京"}, {"word": "上海"}]},
        {
            "utt_id": "b",
            "text": "其中国际代表在北京学习",
            "keywords": [{"word": "中国"}, {"word": "北京"}, {"word": "学习"}],
        },
        {"utt_id": "c", "text": "学习", "keywords": [{"word": "学习"}]},
    ]
    save_rows(tmp_path / "val/kw_subset/kw.jsonl", records)
    return {
        "source_sha256": {
            n: sha(tmp_path / n) for n in ("kw/kw_candidates.tsv", "val/kw_subset/kw.jsonl")
        },
        "candidate_surfaces": ["北京", "上海", "中国"],
        "excluded_target_pairs": [{"utt_id": "b", "word": "中国"}],
        "expected_counts": {"candidates": 3, "utterances": 2, "targets": 2, "target_pairs": 3},
    }


def test_fixed_selection_keeps_distractors_without_adding_targets(tmp_path: Path) -> None:
    policy = source(tmp_path)
    words, records = select_dataset(tmp_path, policy)
    assert words == ["北京", "上海", "中国"]
    assert [r["targets"] for r in records] == [["北京", "上海"], ["北京"]]
    assert len(records) == 2


@pytest.mark.parametrize("change", ["count", "hash", "exclusion", "duplicate"])
def test_selection_rejects_changed_identity(tmp_path: Path, change: str) -> None:
    policy = source(tmp_path)
    if change == "count":
        policy["expected_counts"]["target_pairs"] = 4
    elif change == "hash":
        policy["source_sha256"]["kw/kw_candidates.tsv"] = "0" * 64
    elif change == "exclusion":
        policy["excluded_target_pairs"].append({"utt_id": "missing", "word": "北京"})
    else:
        policy["candidate_surfaces"].append("北京")
    with pytest.raises(ValueError):
        select_dataset(tmp_path, policy)


def test_safe_ids(tmp_path: Path) -> None:
    policy = source(tmp_path)
    p = tmp_path / "val/kw_subset/kw.jsonl"
    data = p.read_text().replace('"a"', '"../a"')
    p.write_text(data)
    policy["source_sha256"]["val/kw_subset/kw.jsonl"] = sha(p)
    with pytest.raises(ValueError, match="utterance ID"):
        select_dataset(tmp_path, policy)


def test_top1_rejects_partial_labels_and_keeps_homophones() -> None:
    vocab = load_phoneme_vocab(VOCAB)
    prepared = [
        {"surface": w, "units": [{"language": "zh", "word": "same"}], "errors": []}
        for w in ["甲", "乙"]
    ]
    prediction = {
        "language": "zh",
        "word": "same",
        "status": "ok",
        "candidates": [{"usable": True, "token_ids": [2, 3]}],
    }
    entries, failed = assemble_entries(prepared, [prediction], vocab)
    assert len(entries) == 2 and entries[0].token_ids == entries[1].token_ids and not failed
    prepared[0]["units"].append({"language": "zh", "word": "missing"})
    entries, failed = assemble_entries(prepared, [prediction], vocab)
    assert len(entries) == 1 and failed[0]["surface"] == "甲"
    prediction["candidates"] = [
        {"usable": False, "token_ids": [2]},
        {"usable": True, "token_ids": [2, 3]},
    ]
    assert not assemble_entries(prepared, [prediction], vocab)[0]


def test_rank_and_gate_are_independent() -> None:
    matches = [
        HotwordMatch(str(i), str(i), "zh", 0.99 - i * 0.04, 1.0, 0, 0.0, 0.9, 0, 2, 0, 2).to_dict()
        for i in range(8)
    ]
    cfg = json.loads((ROOT / "configs/aishell_retrieval.workzone.json").read_text())["scoring"]
    selected = selections({"raw_ranked_matches": matches}, cfg)
    assert selected["raw_top5"] == list("01234")
    assert selected["raw_top7"] == list("0123456")
    assert selected["gated_top7"] == list("0123456")
    matches[0]["posterior_confidence"] = 0.1
    selected = selections({"raw_ranked_matches": matches}, cfg)
    assert selected["raw_top5"] == list("01234")
    assert selected["gated_top5"] == list("12345")
    assert selected["gated_top7"] == list("123456")


def test_target_pair_denominator_and_macro_not_substrings() -> None:
    records = [
        {
            "sample_id": "a",
            "targets": ["北京", "上海"],
            "reference_text": "北京北京上海中国",
            "selections": {},
        },
        {"sample_id": "b", "targets": ["北京"], "reference_text": "北京", "selections": {}},
    ]
    for key in ("raw_top5", "raw_top7", "gated_top5", "gated_top7"):
        records[0]["selections"][key] = ["北京", "中国", "天津"]
        records[1]["selections"][key] = ["北京市"]
    report = summarize(records)
    metric = report["metrics"]["raw_top7"]
    assert report["target_pairs"] == 3 and report["target_words"] == 2
    assert metric["target_hits"] == 1 and metric["recall"] == pytest.approx(1 / 3)
    assert metric["macro_word_recall"] == 0.25
    assert metric["non_target_words_present_but_unannotated_or_excluded"] == 1
    assert metric["non_target_words_absent_from_reference"] == 2


def checkpoint() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    vocab = load_phoneme_vocab(VOCAB)
    head = {
        "head_type": "temporal_upsample",
        "input_dimension": 1024,
        "num_classes": 183,
        "time_upsampling_factor": 2,
    }
    payload = {
        "vocab_tokens": vocab.tokens,
        "num_classes": 183,
        "blank_id": 0,
        "head_config": head,
        "epoch_metrics": {"epoch": 23},
    }
    marker = {
        "status": "fresh_head_training_completed",
        "num_classes": 183,
        "encoder_frozen": True,
        "head_initialization": "random",
        "test_used": False,
        "best_epoch": 23,
    }
    report = {
        "status": "completed",
        "test_set_used": False,
        "num_classes": 183,
        "best_epoch": 23,
        "head_config": copy.deepcopy(head),
    }
    return payload, marker, report


def test_checkpoint_accepts_exact_chinese_head() -> None:
    check_checkpoint_metadata(*checkpoint(), load_phoneme_vocab(VOCAB))


@pytest.mark.parametrize("change", ["vocab", "old100", "up3", "epoch", "unfinished"])
def test_wrong_head_rejected(change: str) -> None:
    payload, marker, report = checkpoint()
    if change == "vocab":
        tokens = list(payload["vocab_tokens"])
        tokens[4], tokens[5] = tokens[5], tokens[4]
        payload["vocab_tokens"] = tokens
    elif change == "old100":
        payload["num_classes"] = 100
    elif change == "up3":
        payload["head_config"]["time_upsampling_factor"] = 3
    elif change == "epoch":
        payload["epoch_metrics"]["epoch"] = 22
    else:
        marker["status"] = "running"
    with pytest.raises(ValueError):
        check_checkpoint_metadata(payload, marker, report, load_phoneme_vocab(VOCAB))


def test_resume_skips_inference_and_rejects_corruption(tmp_path: Path) -> None:
    plan = {"configuration": {}, "checkpoint": "fixed"}
    record = {"sample_id": "a", "targets": ["北京"]}
    detail = {"sample_id": "a", "targets": ["北京"], "result": "stored"}
    saved = {"plan_sha256": digest(plan), "record": record, "detail": detail}
    path = tmp_path / "sample_shards/a.json"
    write_json(path, {**saved, "row_sha256": digest(saved)})
    assert infer(tmp_path, plan, [record], (), load_phoneme_vocab(VOCAB), 4) == [detail]
    saved["detail"]["result"] = "corrupt"
    write_json(path, {**saved, "row_sha256": "0" * 64})
    with pytest.raises(ValueError, match="corrupt"):
        infer(tmp_path, plan, [record], (), load_phoneme_vocab(VOCAB), 4)


def test_pcm_overlap_checked_independent_of_path(tmp_path: Path) -> None:
    part = tmp_path / "train_part.jsonl"
    save_rows(
        part,
        [
            {
                "id": "different_path",
                "pcm_sha256": "same-audio",
                "audio_sha256": "different-container",
            }
        ],
    )
    write_json(
        tmp_path / "prepared.json",
        {"parts": [{"path": str(part), "sha256": sha(part), "split": "train"}]},
    )
    inventory = [{"pcm_sha256": "same-audio", "audio_sha256": "test-container"}]
    with pytest.raises(ValueError, match="overlap audit"):
        overlap_audit(tmp_path, inventory, tmp_path)
    inventory[0]["pcm_sha256"] = "other-audio"
    assert overlap_audit(tmp_path, inventory, tmp_path)["overlap_count"] == 0


def test_inference_uses_audio_only_then_reuses_saved_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwen_hotword.evaluation import aishell_run as pipeline
    from qwen_hotword.inference import streaming_backends

    wav = tmp_path / "source.wav"
    wav.write_bytes(b"fake-wave")
    vocab = load_phoneme_vocab(VOCAB)
    entries, _ = assemble_entries(
        [{"surface": "北京", "units": [{"language": "zh", "word": "beijing"}], "errors": []}],
        [
            {
                "language": "zh",
                "word": "beijing",
                "status": "ok",
                "candidates": [{"usable": True, "token_ids": [2, 3]}],
            }
        ],
        vocab,
    )
    cfg = json.loads((ROOT / "configs/aishell_retrieval.workzone.json").read_text())
    plan = {"configuration": cfg, "model_path": "model", "checkpoint_path": "new183"}
    record = {
        "sample_id": "a",
        "reference_text": "北京",
        "targets": ["北京"],
        "audio_path": str(wav),
        "audio_sha256": sha(wav),
    }
    loaded = []

    def load(**kwargs: Any) -> object:
        loaded.append(kwargs)
        assert kwargs["hotwords"] == tuple(entries)
        assert kwargs["language"] == "Chinese" and kwargs["checkpoint_path"] == "new183"
        return object()

    audio = object()

    def retrieve(
        detector: object, waveform: object, observed_vocab: object, *, saved_raw_rank_depth: int
    ) -> dict[str, Any]:
        assert waveform is audio and observed_vocab is vocab and saved_raw_rank_depth == 64
        return {
            "raw_ranked_matches": [
                HotwordMatch("a", "北京", "zh", 0.9, 1.0, 0, 0.0, 0.9, 0, 2, 0, 2).to_dict()
            ]
        }

    monkeypatch.setattr(pipeline, "check_gpu", lambda gpu, minimum: None)
    monkeypatch.setattr(streaming_backends, "load_cumulative_ctc_detector", load)
    monkeypatch.setattr(pipeline.external, "_load_audio", lambda path: (audio, {}))
    monkeypatch.setattr(pipeline.external, "_retrieve_waveform", retrieve)
    result = infer(tmp_path, plan, [record], tuple(entries), vocab, 4)
    assert result[0]["selections"]["gated_top7"] == ["北京"]
    assert infer(tmp_path, plan, [record], tuple(entries), vocab, 4) == result
    assert len(loaded) == 1
    plan["checkpoint_path"] = "different"
    with pytest.raises(ValueError, match="changed/corrupt"):
        infer(tmp_path, plan, [record], tuple(entries), vocab, 4)


def original_policy(tmp_path: Path) -> dict[str, Any]:
    policy = source(tmp_path)
    policy.pop("candidate_surfaces")
    policy.pop("excluded_target_pairs")
    policy["selection_mode"] = "original"
    policy["expected_counts"] = {"candidates": 4, "utterances": 3, "targets": 4, "target_pairs": 6}
    return policy


def test_original_preserves_common_words_substrings_and_all_rows(tmp_path: Path) -> None:
    policy = original_policy(tmp_path)
    surfaces, records = select_dataset(tmp_path, policy)
    assert set(surfaces) == {"北京", "上海", "中国", "学习"}
    assert [r["sample_id"] for r in records] == ["a", "b", "c"]
    assert records[1]["targets"] == ["中国", "北京", "学习"]
    assert records[2]["targets"] == ["学习"]
    assert sum(len(r["targets"]) for r in records) == 6


@pytest.mark.parametrize("field", ["candidate_surfaces", "excluded_target_pairs"])
def test_original_cannot_apply_editorial_filters(tmp_path: Path, field: str) -> None:
    policy = original_policy(tmp_path)
    policy[field] = ["北京"] if field == "candidate_surfaces" else [{"utt_id": "b", "word": "中国"}]
    with pytest.raises(ValueError, match="whitelist or target exclusions"):
        select_dataset(tmp_path, policy)


def test_original_does_not_silently_drop_out_of_table_targets(tmp_path: Path) -> None:
    policy = original_policy(tmp_path)
    path = tmp_path / "kw/kw_candidates.tsv"
    path.write_text(path.read_text().replace("中国\t9\n", ""))
    policy["source_sha256"]["kw/kw_candidates.tsv"] = sha(path)
    with pytest.raises(ValueError, match="outside full table"):
        select_dataset(tmp_path, policy)


def test_configured_original_denominator_and_model_identity() -> None:
    from qwen_hotword.evaluation.aishell_reviewed import check_result_counts

    policy = json.loads((ROOT / "configs/aishell_original_2012.v1.json").read_text())
    cfg = json.loads((ROOT / "configs/aishell_original_retrieval.workzone.json").read_text())
    reviewed = json.loads((ROOT / "configs/aishell_retrieval.workzone.json").read_text())
    expected = {"candidates": 2012, "utterances": 2135, "targets": 630, "target_pairs": 2782}
    assert policy["expected_counts"] == expected
    assert policy["selection_mode"] == "original"
    assert (
        cfg["required_checkpoint_sha256"]
        == "955e31483b54040c5fce9da6e9287f604545087cdddf15db434a2ce7557991d1"
    )
    for field in (
        "training_run",
        "training_config",
        "vocab_sha256",
        "num_classes",
        "dtype",
        "anchor",
        "scoring",
    ):
        assert cfg[field] == reviewed[field]
    assert cfg["output"] != reviewed["output"]
    check_result_counts({"sample_count": 2135, "target_words": 630, "target_pairs": 2782}, expected)
    with pytest.raises(ValueError, match="denominator changed"):
        check_result_counts(
            {"sample_count": 1245, "target_words": 271, "target_pairs": 1489}, expected
        )
