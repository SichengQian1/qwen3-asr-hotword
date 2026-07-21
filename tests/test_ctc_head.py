from __future__ import annotations

from pathlib import Path

import pytest

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.training.ctc_overfit import EpochMetrics, save_ctc_head_checkpoint


def _vocab() -> PhonemeVocab:
    tokens = ("<blank>", "a", "b")
    return PhonemeVocab(
        tokens=tokens,
        phone_tokens=("a", "b"),
        token_to_id={token: index for index, token in enumerate(tokens)},
    )


def test_linear_head_preserves_shape_lengths_and_validation() -> None:
    torch = pytest.importorskip("torch")
    from qwen_hotword.modeling.ctc_head import LinearCtcHead

    head = LinearCtcHead(4, 3)
    hidden = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3])

    assert head(hidden, input_lengths=lengths).shape == (2, 5, 3)
    assert torch.equal(head.output_lengths(lengths), lengths)
    with pytest.raises(ValueError, match="input_dimension"):
        LinearCtcHead(0, 3)
    with pytest.raises(ValueError, match="hidden dimension"):
        head(torch.randn(2, 5, 6))


def test_temporal_head_doubles_lengths_and_masks_padded_context() -> None:
    torch = pytest.importorskip("torch")
    from qwen_hotword.modeling.ctc_head import TemporalUpsampleCtcHead

    torch.manual_seed(7)
    head = TemporalUpsampleCtcHead(
        4,
        3,
        hidden_dimension=6,
        kernel_size=3,
        dropout=0.0,
        time_upsampling_factor=2,
    ).eval()
    valid = torch.randn(1, 3, 4)
    first = torch.cat((valid, torch.zeros(1, 2, 4)), dim=1)
    second = torch.cat((valid, torch.full((1, 2, 4), 1000.0)), dim=1)
    lengths = torch.tensor([3])

    first_logits = head(first, input_lengths=lengths)
    second_logits = head(second, input_lengths=lengths)

    assert first_logits.shape == (1, 10, 3)
    assert head.output_lengths(lengths).tolist() == [6]
    torch.testing.assert_close(first_logits[:, :6], second_logits[:, :6])


def test_compute_ctc_returns_effective_upsampled_lengths() -> None:
    torch = pytest.importorskip("torch")
    from qwen_hotword.modeling.ctc_head import TemporalUpsampleCtcHead, compute_ctc

    head = TemporalUpsampleCtcHead(
        4,
        3,
        hidden_dimension=6,
        kernel_size=3,
        dropout=0.0,
        time_upsampling_factor=2,
    )
    result = compute_ctc(
        head,
        torch.randn(2, 4, 4),
        torch.tensor([4, 3]),
        torch.tensor([[1, 2], [2, 0]]),
        torch.tensor([2, 1]),
    )

    assert result.logits.shape == (2, 8, 3)
    assert result.input_lengths.tolist() == [8, 6]
    assert result.loss.isfinite()


def test_checkpoint_round_trip_and_legacy_linear_compatibility(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from qwen_hotword.modeling.ctc_head import (
        LinearCtcHead,
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
        ctc_head_config,
    )

    vocab = _vocab()
    metrics = EpochMetrics(1, 1.0, 0.5, 1, 2)
    head = TemporalUpsampleCtcHead(
        1024,
        len(vocab.tokens),
        hidden_dimension=8,
        kernel_size=3,
        dropout=0.2,
        time_upsampling_factor=2,
    )
    checkpoint = tmp_path / "head.pt"
    save_ctc_head_checkpoint(checkpoint, head, vocab, metrics, seed=7)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = build_ctc_head_from_checkpoint(payload)
    restored.load_state_dict(payload["state_dict"], strict=True)

    assert ctc_head_config(restored) == ctc_head_config(head)
    assert payload["head_config"]["head_type"] == "temporal_upsample"

    legacy = {
        "head_type": "LinearCtcHead",
        "input_dimension": 1024,
        "num_classes": len(vocab.tokens),
    }
    assert isinstance(build_ctc_head_from_checkpoint(legacy), LinearCtcHead)
