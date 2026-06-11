from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from qwen_hotword.modeling.qwen_backbone import (
    ModelInspection,
    ModelValidationError,
    _first_int,
    validate_inspection_manifest,
    validate_loaded_wrapper,
)


def test_validate_loaded_wrapper_accepts_expected_shape() -> None:
    wrapper = SimpleNamespace(
        model=SimpleNamespace(thinker=SimpleNamespace(audio_tower=object()))
    )
    validate_loaded_wrapper(wrapper)


def test_validate_loaded_wrapper_rejects_missing_audio_tower() -> None:
    wrapper = SimpleNamespace(model=SimpleNamespace(thinker=SimpleNamespace()))
    with pytest.raises(ModelValidationError, match="audio_tower"):
        validate_loaded_wrapper(wrapper)


def test_first_int_returns_first_matching_integer() -> None:
    config = SimpleNamespace(output_dim=2048, d_model=1024)
    assert _first_int(config, "output_dim", "d_model") == 2048


def test_validate_inspection_manifest_detects_mismatch(tmp_path: Path) -> None:
    inspection = ModelInspection(
        model_name="Qwen3-ASR-1.7B",
        config_class="Qwen3ASRConfig",
        model_type="qwen3_asr",
        architectures=["Qwen3ASRForConditionalGeneration"],
        audio_config_class="Qwen3ASRAudioEncoderConfig",
        audio_encoder_dimension=1024,
        audio_projected_dimension=2048,
        audio_encoder_layers=24,
        text_hidden_size=2048,
        ctc_tap_module="thinker.audio_tower.ln_post",
        ctc_tap_dimension=1024,
        total_parameters=100,
        audio_tower_parameters=20,
        weight_index_entries=10,
        audio_weight_entries=5,
        representative_weight_shapes={},
        config_sha256="config",
        weight_index_sha256="index",
    )
    manifest = {
        "architecture": {"audio_encoder_dimension": 999},
        "files": {
            "config_sha256": "config",
            "weight_index_sha256": "index",
        },
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    mismatches = validate_inspection_manifest(inspection, path)
    assert mismatches == ["audio_encoder_dimension: actual=1024, expected=999"]
