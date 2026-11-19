from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qwen_hotword.evaluation import zh_en_delivery as ev
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

ROOT = Path(__file__).resolve().parents[1]


def test_joint_index_and_same_step(monkeypatch: Any) -> None:
    vocab = load_phoneme_vocab(ROOT / "configs/phonemes/zh_en_mfa_g2p_vocab.v1.0.json")
    entries, owners = ev.registry(ROOT / "configs/zh_en_joint2351.v1.jsonl", vocab)
    pm = [e for e in entries if e.surface == "PM2.5"]
    assert len(pm) == 2 and owners[pm[0].hotword_id] == owners[pm[1].hotword_id]
    assert len({e.surface for e in entries}) == 2351
    assert len({e.token_ids for e in entries}) < len(entries)
    homophones: dict[tuple[int, ...], list[Any]] = {}
    for entry in entries:
        homophones.setdefault(entry.token_ids, []).append(entry)
    pair = next(es[:2] for es in homophones.values() if len({e.surface for e in es}) > 1)
    monkeypatch.setattr(ev, "audio_prefixes", lambda *_: [{"waveform": 1}, {"waveform": 2}])

    def infer(detector: Any, wav: int, vocab: Any, saved_raw_rank_depth: int) -> dict[str, Any]:
        assert detector.language == "Chinese" and saved_raw_rank_depth == 64
        results = (
            []
            if wav == 1
            else [
                dict(hotword_id=e.hotword_id, score=s, edit_ratio=0.0, posterior_confidence=1.0)
                for e, s in zip([pm[1], pm[0], *pair], [0.99, 0.98, 0.9, 0.7], strict=True)
            ]
        )
        return {
            "raw_ranked_matches": results,
            "shortlist_candidates": len(results),
            "timing": {"pure_retrieval_seconds": 0.0, "model_retrieval_seconds": 0.0},
        }

    sample = {
        "audio_path": "mock.wav",
        "language": "zh",
        "group": "zh-example",
        "step_count": 2,
        "targets": ["pm2.5"],
    }
    result = ev.retrieve(SimpleNamespace(), vocab, entries, owners, sample, infer=infer)
    assert result["steps"][0] == []
    assert [m["word"] for m in result["steps"][1]] == ["PM2.5", pair[0].surface]
    assert result["raw_last"] == ["PM2.5", pair[0].surface, pair[1].surface]
    assert result["steps"][1][0]["phoneme"] == pm[1].pronunciation
    ctx = {m: {"zh-example@@zh": {"id": result["steps"]}} for m in MODES}
    assert validate_contexts(ctx, {"zh-example@@zh": {"id": 2.5}})["unique_utterances"] == 1


def test_counts_for_negative_oov_and_early_hit() -> None:
    totals: dict[str, dict[str, Counter[str]]] = {}
    result = {"steps": [[{"word": "APPLE"}], []], "raw_last": []}
    sample = {"language": "en", "group": "fixture", "targets": ["apple", "missing"]}
    ev.score(result, sample, totals)
    ev.score(result, {**sample, "targets": []}, totals)
    assert ev.rates(totals["overall"]["gated_top5_last"])["target_recall"] == 0
    ever = ev.rates(totals["overall"]["gated_top7_ever"])
    assert ever["target_recall"] == 0.5 and ever["negative_prompt_rate"] == 1
    assert ev.rates(Counter())["target_recall"] is None
    assert sum(ev.COUNTS.values()) == 3465
