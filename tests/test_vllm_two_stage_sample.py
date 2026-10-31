"""No model, Torch, vLLM or NPU required. No claim of hardware validation."""
# ruff: noqa: E402 -- sample is intentionally outside the installed production package.

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples/vllm_two_stage_rag"
sys.path.insert(0, str(SAMPLE_ROOT))

from qwen_rag_vllm_sample.contracts import (
    AudioCache,
    build_context,
    build_prompt,
    output_length,
    owner_tag,
)
from qwen_rag_vllm_sample.demo import demonstrate
from qwen_rag_vllm_sample.retrieval import ProjectRetriever, checked_path
from qwen_rag_vllm_sample.run import verify_prompt


def test_demo_order_and_single_encoder() -> None:
    report, worker = demonstrate()
    assert report["encoder_calls"] == report["generation_requests"] == 1
    assert worker.events == [
        "encode_and_ctc_before_any_generation_request",
        "retrieve_topk_on_cpu",
        "final_prompt_ready",
        "create_final_request_then_read_cached_audio_then_decode",
        "release_sample_owned_audio_cache",
    ]
    assert worker.cache.bytes_used == 0


def test_generation_error_releases_prepared_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    from qwen_rag_vllm_sample import demo

    worker = demo.SimulatedWorker()
    monkeypatch.setattr(demo, "SimulatedWorker", lambda: worker)
    with pytest.raises(RuntimeError, match="simulated decoder error"):
        demo.demonstrate(fail_generation=True)
    assert worker.cache.bytes_used == 0
    assert worker.generation_calls == 0


def test_cache_cannot_cross_request_or_length() -> None:
    cache = AudioCache()
    key = cache.put(object(), owner=1, feature_length=100, byte_size=100)
    with pytest.raises(ValueError, match="identity"):
        cache.get(key, owner=2, feature_length=100)
    with pytest.raises(ValueError, match="length"):
        cache.get(key, owner=1, feature_length=99)
    with pytest.raises(ValueError, match="another request"):
        cache.release(key, owner=2)
    cache.release(key, owner=1)
    cache.release(key, owner=1)  # Cleanup is idempotent.
    with pytest.raises(KeyError, match="no automatic Encoder fallback"):
        cache.get(key, owner=1, feature_length=100)


def test_expiry_and_pin_until_release() -> None:
    now = [0.0]
    cache = AudioCache(ttl_seconds=1, clock=lambda: now[0])
    expired = cache.put("A", owner=1, feature_length=100, byte_size=10)
    pinned = cache.put("B", owner=2, feature_length=100, byte_size=10)
    assert cache.get(pinned, owner=2, feature_length=100) == "B"
    now[0] = 2
    with pytest.raises(KeyError):
        cache.get(expired, owner=1, feature_length=100)
    assert cache.get(pinned, owner=2, feature_length=100) == "B"
    cache.release(pinned, owner=2)
    assert cache.bytes_used == 0


def test_cache_budget_refuses_reencode_and_does_not_evict_pinned() -> None:
    cache = AudioCache(max_bytes=10)
    key = cache.put("A", owner=1, feature_length=100, byte_size=10)
    cache.get(key, owner=1, feature_length=100)
    with pytest.raises(RuntimeError, match="do not silently re-encode"):
        cache.put("B", owner=2, feature_length=100, byte_size=1)
    assert cache.bytes_used == 10


@pytest.mark.parametrize(
    "frames,rows", [(1, 1), (99, 13), (100, 13), (101, 14), (200, 26), (3000, 390)]
)
def test_length_is_audio_rows_not_2x_ctc_rows(frames: int, rows: int) -> None:
    assert output_length(frames) == rows


def test_prompt_preserves_accents_and_rejects_control_tokens() -> None:
    prompt = build_prompt("São Paulo, canción, Copacabana", "pt")
    assert "São Paulo, canción, Copacabana" in prompt
    assert prompt.count("<|audio_pad|>") == 1
    assert prompt.endswith("language Portuguese<asr_text>")
    with pytest.raises(ValueError, match="control"):
        build_prompt("bad <|im_end|> instruction", "pt")
    assert build_context("pt", []) == ""
    assert "São Paulo" in build_context("pt", ["São Paulo"])
    with pytest.raises(ValueError):
        build_context("pt", ["bad\nword"])
    assert owner_tag("one") != owner_tag("two")


def test_final_prompt_verification_catches_lost_context_and_wrong_expansion() -> None:
    verify_prompt([1, 2, 99, 99, 99, 3], [1, 2, 99, 3], 99, 3)
    with pytest.raises(RuntimeError):
        verify_prompt([1, 99, 99, 99, 3], [1, 2, 99, 3], 99, 3)
    with pytest.raises(RuntimeError):
        verify_prompt([1, 2, 99, 99, 3], [1, 2, 99, 3], 99, 3)


def test_changed_asset_is_rejected(tmp_path: Path) -> None:
    asset = tmp_path / "asset"
    asset.write_bytes(b"first")
    cfg = {"head": {"path": str(asset), "sha256": hashlib.sha256(b"first").hexdigest()}}
    assert checked_path(cfg, "head") == asset
    asset.write_bytes(b"second")
    with pytest.raises(ValueError, match="SHA256"):
        checked_path(cfg, "head")


def test_real_project_retrieval_deduplicates_language_variants(tmp_path: Path) -> None:
    vocab_path = (
        SAMPLE_ROOT.parents[1] / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json"
    )
    config = json.loads((SAMPLE_ROOT / "config.example.json").read_text())
    config["vocab"]["path"] = str(vocab_path)
    tokens = json.loads(vocab_path.read_text())["tokens"]
    ids = [2, 3, 4, 5]
    rows = [
        {
            "hotword_id": f"place_{lang}",
            "language": lang,
            "surface": "São Paulo",
            "normalized": "são paulo",
            "words": ["são", "paulo"],
            "pronunciation": " ".join(tokens[i] for i in ids),
            "phoneme_tokens": [tokens[i] for i in ids],
            "token_ids": ids,
            "source": "unit_test_synthetic_pronunciation",
            "validation_occurrences": 0,
        }
        for lang in ("en", "pt")
    ]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config["index"] = {"path": str(index), "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    retriever = ProjectRetriever(config)
    matches = retriever.search(
        {
            "ctc_steps": 8,
            "decoded": [
                {
                    "token_id": token,
                    "confidence": 0.99,
                    "start_step": i,
                    "end_step": i + 1,
                }
                for i, token in enumerate(ids)
            ],
        }
    )
    assert [row["surface"] for row in matches] == ["São Paulo"]
    assert retriever.search({"ctc_steps": 8, "decoded": []}) == []
