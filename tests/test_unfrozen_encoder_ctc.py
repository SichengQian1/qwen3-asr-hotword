from __future__ import annotations

import pytest

from qwen_hotword.training.unfrozen_encoder_ctc import (
    _optimizer_groups,
    configure_audio_tower_unfreeze,
)


def test_configure_audio_tower_unfreezes_last_layer_and_ln_post() -> None:
    torch = pytest.importorskip("torch")

    class FakeAudioTower(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(3)]
            )
            self.ln_post = torch.nn.LayerNorm(2)
            self.proj = torch.nn.Linear(2, 2)

    tower = FakeAudioTower()
    plan = configure_audio_tower_unfreeze(
        tower,
        unfreeze_last_encoder_layers=1,
        train_ln_post=True,
    )

    assert plan.unfreeze_all_encoder is False
    assert plan.trainable_module_names == ("layers.2", "ln_post")
    assert all(
        not parameter.requires_grad for parameter in tower.layers[0].parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in tower.layers[1].parameters()
    )
    assert all(parameter.requires_grad for parameter in tower.layers[2].parameters())
    assert all(parameter.requires_grad for parameter in tower.ln_post.parameters())
    assert all(not parameter.requires_grad for parameter in tower.proj.parameters())
    assert tower.training is False
    assert tower.layers[2].training is True
    assert tower.ln_post.training is True
    assert plan.trainable_parameters > 0
    assert plan.frozen_parameters > 0


def test_configure_audio_tower_rejects_conflicting_unfreeze_modes() -> None:
    torch = pytest.importorskip("torch")

    class FakeAudioTower(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.ln_post = torch.nn.LayerNorm(2)

    with pytest.raises(ValueError, match="use either"):
        configure_audio_tower_unfreeze(
            FakeAudioTower(),
            unfreeze_last_encoder_layers=1,
            unfreeze_all_encoder=True,
        )


def test_optimizer_groups_separate_head_and_encoder_learning_rates() -> None:
    torch = pytest.importorskip("torch")

    head = torch.nn.Linear(2, 3)
    encoder = torch.nn.Linear(2, 2)
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)

    groups = _optimizer_groups(
        head,
        encoder,
        head_learning_rate=1e-3,
        encoder_learning_rate=1e-5,
        weight_decay=1e-4,
    )

    assert [group["name"] for group in groups] == ["ctc_head", "audio_encoder"]
    assert [group["lr"] for group in groups] == [1e-3, 1e-5]

