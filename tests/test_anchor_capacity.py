from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import qwen_hotword.hotwords.anchor_capacity as anchor_capacity_module
from qwen_hotword.hotwords.anchor_capacity import benchmark_anchor_hotword_capacity
from qwen_hotword.hotwords.anchor_index import AnchorIndexConfig, PhonemeAnchorIndex
from qwen_hotword.hotwords.registry import HotwordEntry, write_hotword_table
from qwen_hotword.hotwords.scoring import profile_decoded_hotwords


def _entry(hotword_id: str, token_ids: tuple[int, ...]) -> HotwordEntry:
    tokens = tuple(f"p{token_id}" for token_id in token_ids)
    return HotwordEntry(
        hotword_id=hotword_id,
        language="pt-BR",
        surface=hotword_id,
        normalized=hotword_id,
        words=(hotword_id,),
        pronunciation=" ".join(tokens),
        phoneme_tokens=tokens,
        token_ids=token_ids,
        source="test",
        validation_occurrences=1,
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_anchor_index_prioritizes_positionally_aligned_near_match() -> None:
    target = _entry("target", (1, 2, 3, 4, 5, 6))
    scattered = _entry("scattered", (1, 2, 8, 9, 3, 4, 10, 11, 5, 6))
    index = PhonemeAnchorIndex(
        [target, scattered],
        config=AnchorIndexConfig(anchors_per_entry=24, offset_tolerance=1),
    )

    result = index.query((1, 2, 3, 12, 5, 6), maximum_candidates=2)

    assert [candidate.hotword_id for candidate in result.candidates] == [
        "target",
        "scattered",
    ]
    assert result.candidates[0].alignment_score > result.candidates[1].alignment_score
    assert result.exact_hotword_ids == ()


def test_anchor_index_is_active_only_nested_and_deterministic() -> None:
    entries = [
        _entry("exact", (1, 2, 3, 4)),
        _entry("near", (1, 2, 3, 5)),
        _entry("inactive", (1, 2, 3, 6)),
        _entry("tail", (7, 8, 3, 4)),
    ]
    index = PhonemeAnchorIndex(entries)
    active = ("exact", "near", "tail")

    first = index.query((1, 2, 3, 4), active_hotword_ids=active, maximum_candidates=3)
    second = index.query((1, 2, 3, 4), active_hotword_ids=active, maximum_candidates=2)

    first_ids = tuple(candidate.hotword_id for candidate in first.candidates)
    second_ids = tuple(candidate.hotword_id for candidate in second.candidates)
    assert first_ids[0] == "exact"
    assert "inactive" not in first_ids
    assert second_ids == first_ids[:2]
    assert first.exact_hotword_ids == ("exact",)


def test_anchor_capacity_benchmark_reports_reference_coverage_and_latency_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = ["<blank>"] + [f"p{index}" for index in range(1, 41)]
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps({"tokens": tokens}), encoding="utf-8")
    entries = [_entry("target", (1, 2, 3, 4, 5, 6))]
    for index in range(99):
        entries.append(
            _entry(
                f"filler-{index:03d}",
                (7, 8, 9, 10 + index // 10, 20 + index % 10),
            )
        )
    active_ids = [entry.hotword_id for entry in entries]
    assets = tmp_path / "assets"
    level = assets / "representative" / "size_100"
    level.mkdir(parents=True)
    (assets / "asset_summary.json").write_text(
        json.dumps({"status": "pass", "test_set_used": False}), encoding="utf-8"
    )
    write_hotword_table(level / "hotwords.jsonl", entries)
    cases: list[dict[str, object]] = [
        {
            "case_id": "positive",
            "sample_id": "sample-positive",
            "reference_text": "target",
            "normalized_reference_text": "target",
            "language": "pt-BR",
            "primary_group": "single_hotword",
            "expected_hotword_ids": ["target"],
            "active_hotword_ids": active_ids,
        },
        {
            "case_id": "negative",
            "sample_id": "sample-negative",
            "reference_text": "negative",
            "normalized_reference_text": "negative",
            "language": "pt-BR",
            "primary_group": "negative",
            "expected_hotword_ids": [],
            "active_hotword_ids": active_ids,
        },
    ]
    _write_jsonl(level / "cases.jsonl", cases)
    replay = tmp_path / "replay.jsonl"
    replay_rows = []
    for case_id, token_ids in (
        ("positive", (1, 2, 3, 12, 5, 6)),
        ("negative", (31, 32, 33, 34)),
    ):
        replay_rows.append(
            {
                "case_id": case_id,
                "sample_id": f"sample-{case_id}",
                "expected_hotword_ids": ["target"] if case_id == "positive" else [],
                "chunk_id": 0,
                "cumulative_audio_sec": 2.0,
                "is_final": True,
                "is_tail_flush": False,
                "effective_time_steps": len(token_ids),
                "decoded": [
                    {
                        "token_id": token_id,
                        "confidence": 0.99,
                        "start_step": position,
                        "end_step": position + 1,
                    }
                    for position, token_id in enumerate(token_ids)
                ],
            }
        )
    _write_jsonl(replay, replay_rows)

    events: list[str] = []
    original_query = PhonemeAnchorIndex.query
    original_reference = profile_decoded_hotwords

    def tracked_query(self: PhonemeAnchorIndex, *args: Any, **kwargs: Any) -> Any:
        events.append("anchor")
        return original_query(self, *args, **kwargs)

    def tracked_reference(*args: Any, **kwargs: Any) -> Any:
        events.append("reference")
        return original_reference(*args, **kwargs)

    monkeypatch.setattr(PhonemeAnchorIndex, "query", tracked_query)
    monkeypatch.setattr(anchor_capacity_module, "profile_decoded_hotwords", tracked_reference)

    output = tmp_path / "benchmark"
    report = benchmark_anchor_hotword_capacity(
        assets_root=assets,
        replay_path=replay,
        vocab_path=vocab,
        output_dir=output,
        sizes=(100,),
        shortlist_sizes=(8, 16, 32),
        warmup_queries=0,
        print_progress=False,
    )

    assert report["status"] == "pass"
    assert events == ["anchor", "anchor", "reference", "reference"]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    level_summary = summary["levels"]["representative"]["100"]
    assert level_summary["quality"]["expected_recall_at_8"] == 1.0
    assert level_summary["quality"]["anchor_raw_recall_at_5"] == 1.0
    assert level_summary["quality"]["anchor_raw_precision_at_5"] == 1.0
    assert level_summary["quality"]["reference_raw_recall_at_7"] == 1.0
    assert level_summary["quality"]["reference_raw_precision_at_10"] == pytest.approx(0.05)
    assert level_summary["quality"]["reference_operating_recall_at_5"] == 0.0
    assert level_summary["quality"]["reference_operating_precision_at_5"] is None
    assert level_summary["quality"]["reference_negative_case_false_positive_rate"] == 0.0
    assert level_summary["quality"]["reference_top5_coverage_at_32"] == 0.1
    assert level_summary["quality"]["reference_top5_positive_coverage_at_32"] == 0.2
    assert level_summary["performance"]["latency_scope"] == (
        "anchor_index_query_only_excludes_full_scan_reference"
    )
    assert summary["timing_protocol"] == "all_anchor_queries_before_full_scan_reference"
    rows = [
        json.loads(line)
        for line in (output / "query_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["candidate_ids_at_8"] == rows[0]["candidate_ids_at_16"][:8]
    assert rows[0]["candidate_ids_at_16"] == rows[0]["candidate_ids_at_32"][:16]
    assert rows[0]["anchor_top5_ids"] == rows[0]["candidate_ids_at_8"][:5]
    assert len(rows[0]["reference_raw_top7_ids"]) == 7
    assert len(rows[0]["reference_raw_top10_ids"]) == 10
    assert rows[0]["reference_operating_ids"] == []
    assert (output / "diagnostic_cases.jsonl").is_file()
    diagnostic = json.loads((output / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert diagnostic["maximum_shortlist"] == 32
    assert diagnostic["test_set_used"] is False
    assert (output / "sha256.txt").is_file()
