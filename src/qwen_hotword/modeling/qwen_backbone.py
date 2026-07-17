from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from qwen_hotword.config import EXPECTED_MODEL_NAME, ModelConfig

CTC_TAP_MODULE = "thinker.audio_tower.ln_post"


class ModelValidationError(RuntimeError):
    """Raised when a model is not the expected Qwen3-ASR-1.7B architecture."""


@dataclass(frozen=True)
class ModelInspection:
    model_name: str
    config_class: str
    model_type: str
    architectures: list[str]
    audio_config_class: str
    audio_encoder_dimension: int | None
    audio_projected_dimension: int | None
    audio_encoder_layers: int | None
    text_hidden_size: int | None
    ctc_tap_module: str
    ctc_tap_dimension: int | None
    total_parameters: int
    audio_tower_parameters: int
    weight_index_entries: int
    audio_weight_entries: int
    representative_weight_shapes: dict[str, list[int]]
    config_sha256: str
    weight_index_sha256: str


def inspect_local_config(model_config: ModelConfig) -> ModelInspection:
    try:
        import qwen_asr  # noqa: F401
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModel
    except ImportError as error:
        raise ModelValidationError(
            "qwen-asr, accelerate, and transformers must be installed"
        ) from error

    try:
        config = AutoConfig.from_pretrained(
            str(model_config.path),
            local_files_only=model_config.local_files_only,
            trust_remote_code=True,
        )
    except ValueError as error:
        raise ModelValidationError(
            "Qwen3-ASR config registration failed; verify qwen-asr==0.0.6 "
            "and transformers==4.57.6"
        ) from error

    with init_empty_weights():
        model: Any = AutoModel.from_config(config)
    thinker = getattr(model, "thinker", None)
    audio_tower = getattr(thinker, "audio_tower", None)
    if thinker is None or audio_tower is None:
        raise ModelValidationError("model architecture does not expose thinker.audio_tower")

    model_type = str(getattr(config, "model_type", ""))
    if "qwen3_asr" not in model_type.lower():
        raise ModelValidationError(f"unexpected model_type: {model_type!r}")

    thinker_config = getattr(config, "thinker_config", None)
    audio_config = getattr(thinker_config, "audio_config", None)
    text_config = getattr(thinker_config, "text_config", None)
    if audio_config is None or text_config is None:
        raise ModelValidationError("missing thinker audio_config or text_config")

    audio_encoder_dimension = _first_int(
        audio_config,
        "d_model",
        "hidden_size",
    )
    audio_projected_dimension = _first_int(audio_config, "output_dim")
    audio_encoder_layers = _first_int(
        audio_config,
        "encoder_layers",
        "num_hidden_layers",
    )
    text_hidden_size = _first_int(text_config, "hidden_size")
    architectures = [str(value) for value in (getattr(config, "architectures", None) or [])]
    index_summary = inspect_weight_index(model_config.path)

    return ModelInspection(
        model_name=EXPECTED_MODEL_NAME,
        config_class=type(config).__name__,
        model_type=model_type,
        architectures=architectures,
        audio_config_class=type(audio_config).__name__,
        audio_encoder_dimension=audio_encoder_dimension,
        audio_projected_dimension=audio_projected_dimension,
        audio_encoder_layers=audio_encoder_layers,
        text_hidden_size=text_hidden_size,
        ctc_tap_module=CTC_TAP_MODULE,
        ctc_tap_dimension=audio_encoder_dimension,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        audio_tower_parameters=sum(
            parameter.numel() for parameter in audio_tower.parameters()
        ),
        weight_index_entries=index_summary["weight_index_entries"],
        audio_weight_entries=index_summary["audio_weight_entries"],
        representative_weight_shapes=index_summary["representative_weight_shapes"],
        config_sha256=_sha256_file(model_config.path / "config.json"),
        weight_index_sha256=_sha256_file(
            model_config.path / "model.safetensors.index.json"
        ),
    )


def load_asr_model(model_config: ModelConfig) -> Any:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise ModelValidationError("qwen-asr and torch must be installed") from error

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[model_config.dtype]
    wrapper = Qwen3ASRModel.from_pretrained(
        str(model_config.path),
        dtype=dtype,
        device_map=model_config.device,
        local_files_only=model_config.local_files_only,
    )
    validate_loaded_wrapper(wrapper)
    return wrapper


def validate_loaded_wrapper(wrapper: Any) -> None:
    outer_model = getattr(wrapper, "model", None)
    thinker = getattr(outer_model, "thinker", None)
    audio_tower = getattr(thinker, "audio_tower", None)
    if outer_model is None or thinker is None or audio_tower is None:
        raise ModelValidationError(
            "loaded wrapper does not expose model.thinker.audio_tower"
        )


def inspection_dict(inspection: ModelInspection) -> dict[str, Any]:
    return asdict(inspection)


def validate_inspection_manifest(
    inspection: ModelInspection,
    manifest_path: str | Path,
) -> list[str]:
    path = Path(manifest_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return ["model manifest root must be a mapping"]
    architecture = raw.get("architecture")
    files = raw.get("files")
    if not isinstance(architecture, dict) or not isinstance(files, dict):
        return ["model manifest must contain architecture and files mappings"]

    actual = inspection_dict(inspection)
    mismatches: list[str] = []
    for key, expected in architecture.items():
        if key in actual and actual[key] != expected:
            mismatches.append(f"{key}: actual={actual[key]!r}, expected={expected!r}")
    expected_hashes = {
        "config_sha256": files.get("config_sha256"),
        "weight_index_sha256": files.get("weight_index_sha256"),
    }
    for key, expected in expected_hashes.items():
        if actual[key] != expected:
            mismatches.append(f"{key}: actual={actual[key]!r}, expected={expected!r}")
    return mismatches


def _first_int(obj: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int):
            return value
    return None


def inspect_weight_index(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ModelValidationError(f"missing weight index: {index_path}")
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = raw.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ModelValidationError("weight index does not contain a weight_map")

    audio_keys = [
        str(key) for key in weight_map if str(key).startswith("thinker.audio_tower.")
    ]
    required_shapes = (
        "thinker.audio_tower.layers.0.self_attn.q_proj.weight",
        "thinker.audio_tower.proj1.weight",
        "thinker.audio_tower.proj2.weight",
    )
    representative_shapes: dict[str, list[int]] = {}
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ModelValidationError("safetensors must be installed") from error

    for key in required_shapes:
        shard_name = weight_map.get(key)
        if not isinstance(shard_name, str):
            raise ModelValidationError(f"missing expected model weight: {key}")
        shard_path = model_path / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            representative_shapes[key] = list(handle.get_slice(key).get_shape())

    return {
        "weight_index_entries": len(weight_map),
        "audio_weight_entries": len(audio_keys),
        "representative_weight_shapes": representative_shapes,
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
