from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.inference.external_keyword_retrieval import load_keyword_bias
from qwen_hotword.inference.external_keyword_topk_replay import (
    replay_external_keyword_top7,
    replay_rows_top7,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

VOCAB_PATH = Path("configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json")
GATE = {
    "threshold": 0.75,
    "minimum_phonemes": 1,
    "maximum_edit_ratio": 0.35,
    "posterior_weight": 0.25,
    "minimum_posterior_confidence": 0.5,
    "minimum_top1_margin": 0.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _keyword_bundle(tmp_path: Path) -> tuple[Path, Any]:
    words = ["alfa", "beta", "gama", "delta", "epsilon", "zeta", "eta", "teta"]
    path = tmp_path / "keywords.json"
    _write_json(
        path,
        {
            "prompt_type": "keyword_bias",
            "language": "pt",
            "phoneme_source": "mfa",
            "keyword_sets": {"hard_k266": words},
            "keyword_phonemes": {word: "/a/" for word in words},
        },
    )
    vocab = load_phoneme_vocab(VOCAB_PATH)
    return path, load_keyword_bias(path, vocab=vocab)


def _match(hotword_id: str, surface: str, score: float) -> dict[str, object]:
    return {
        "hotword_id": hotword_id,
        "surface": surface,
        "score": score,
        "edit_ratio": 0.1,
        "posterior_confidence": 0.9,
    }


def _row(source: str, sample_id: str, bundle: Any) -> dict[str, object]:
    raw = [
        _match(entry.hotword_id, entry.surface, 0.95 - index * 0.01)
        for index, entry in enumerate(bundle.entries)
    ]
    return {
        "sample_id": sample_id,
        "source": source,
        "expected_hotword_ids": [bundle.entries[6].hotword_id],
        "selected_hotword_ids": [entry.hotword_id for entry in bundle.entries[:5]],
        "raw_ranked_matches": raw,
        "shortlist_candidates": len(raw),
    }


def _distribution() -> dict[str, object]:
    return {
        "count": 2,
        "mean": 0.01,
        "median": 0.01,
        "p50": 0.01,
        "p90": 0.01,
        "p95": 0.01,
        "p99": 0.01,
        "max": 0.01,
    }


def _source_metrics(sample_count: int) -> dict[str, object]:
    return {
        "sample_count": sample_count,
        "expected_hotwords": sample_count,
        "raw_retrieved_expected_hotwords_at_5": 0,
        "raw_recall_at_5": 0.0,
        "selected_hotwords": sample_count * 5,
        "selected_true_positive_hotwords": 0,
        "final_retrieval_recall": 0.0,
        "final_retrieval_precision": 0.0,
        "negative_sample_count": 0,
        "negative_sample_false_positives": 0,
        "negative_sample_false_positive_rate": None,
        "mean_selected_hotwords_per_sample": 5.0,
        "latency_seconds": {
            "pure_retrieval_seconds": _distribution(),
            "audio_to_result_seconds": _distribution(),
        },
        "audio_to_result_real_time_factor": 0.01,
    }


def test_replay_adds_sixth_and_seventh_matches_without_changing_top5(
    tmp_path: Path,
) -> None:
    _, bundle = _keyword_bundle(tmp_path)
    rows = replay_rows_top7([_row("mls_portuguese", "sample", bundle)], gate=GATE, bundle=bundle)
    assert rows[0]["top5_selected_hotword_ids"] == [
        entry.hotword_id for entry in bundle.entries[:5]
    ]
    assert rows[0]["top7_selected_hotword_ids"] == [
        entry.hotword_id for entry in bundle.entries[:7]
    ]
    assert rows[0]["top7_hit_expected_hotword_ids"] == [bundle.entries[6].hotword_id]


def test_replay_rejects_truncated_rank_that_cannot_prove_top7(tmp_path: Path) -> None:
    _, bundle = _keyword_bundle(tmp_path)
    row = _row("mls_portuguese", "sample", bundle)
    row["raw_ranked_matches"] = row["raw_ranked_matches"][:6]  # type: ignore[index]
    row["shortlist_candidates"] = 64
    with pytest.raises(ValueError, match="exact raw Top-7"):
        replay_rows_top7([row], gate=GATE, bundle=bundle)


def test_replay_rejects_source_top5_selection_mismatch(tmp_path: Path) -> None:
    _, bundle = _keyword_bundle(tmp_path)
    row = _row("mls_portuguese", "sample", bundle)
    row["selected_hotword_ids"] = []
    with pytest.raises(ValueError, match="cannot reproduce stored Top-5"):
        replay_rows_top7([row], gate=GATE, bundle=bundle)


def test_full_replay_writes_split_delivery_files_and_comparison(tmp_path: Path) -> None:
    keyword_path, bundle = _keyword_bundle(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    config = {
        "schema_version": 1,
        "git_commit": "source-commit",
        "vocab": _identity(VOCAB_PATH),
        "keyword_bias": _identity(keyword_path),
        "keyword_set": "hard_k266",
        "gate": {**GATE, "top_k": 5},
        "retrieval": {"backend": "anchor_guided", "saved_raw_rank_depth": 20},
    }
    rows = [
        _row("mls_portuguese", "mls_sample", bundle),
        _row("delivery_20260706_ptbr", "delivery_sample", bundle),
    ]
    summary = {
        "evaluation_scope": "complete_audio_to_ctc_anchor_retrieval_no_qwen_decoder",
        "overall": _source_metrics(2),
        "by_source": {
            source_name: _source_metrics(1)
            for source_name in ("mls_portuguese", "delivery_20260706_ptbr")
        },
    }
    _write_json(source / "run_config.json", config)
    (source / "retrieval_details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    _write_json(source / "evaluation_summary.json", summary)
    (source / "sha256.txt").write_text(
        "".join(
            f"{_sha256(source / name)}  {name}\n"
            for name in ("run_config.json", "retrieval_details.jsonl", "evaluation_summary.json")
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    comparison = replay_external_keyword_top7(
        source_run=source,
        vocab_path=VOCAB_PATH,
        keyword_bias_path=keyword_path,
        output_dir=output,
    )
    assert comparison["status"] == "pass"
    assert (
        json.loads((output / "mls_retrieval_output.json").read_text())["mls_sample"][-1]["word"]
        == bundle.entries[6].surface
    )
    assert list(json.loads((output / "delivery_retrieval_output.json").read_text())) == [
        "delivery_sample"
    ]
    assert "| MLS | Top-7 |" in (output / "topk_comparison.md").read_text()
    assert (output / "sha256.txt").is_file()
    with pytest.raises(FileExistsError):
        replay_external_keyword_top7(
            source_run=source,
            vocab_path=VOCAB_PATH,
            keyword_bias_path=keyword_path,
            output_dir=output,
        )
