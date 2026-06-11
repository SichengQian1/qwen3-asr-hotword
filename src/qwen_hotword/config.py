from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_MODEL_NAME = "Qwen3-ASR-1.7B"
SUPPORTED_DTYPES = {"bfloat16", "float16"}


class ConfigError(ValueError):
    """Raised when a work-zone configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    expected_name: str
    dtype: str
    device: str
    local_files_only: bool


@dataclass(frozen=True)
class PathsConfig:
    work_root: Path
    data_root: Path
    cache_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    expected_python_major_minor: str
    expected_gpu_name: str
    minimum_gpu_memory_gib: float


@dataclass(frozen=True)
class WorkzoneConfig:
    project_name: str
    model: ModelConfig
    paths: PathsConfig
    runtime: RuntimeConfig


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _required(mapping: dict[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required field: {field}.{key}")
    return mapping[key]


def load_workzone_config(
    path: str | Path,
    *,
    require_existing_model: bool = False,
) -> WorkzoneConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "root")
    project = _mapping(_required(root, "project", "root"), "project")
    model = _mapping(_required(root, "model", "root"), "model")
    paths = _mapping(_required(root, "paths", "root"), "paths")
    runtime = _mapping(_required(root, "runtime", "root"), "runtime")

    expected_name = str(_required(model, "expected_name", "model"))
    if expected_name != EXPECTED_MODEL_NAME:
        raise ConfigError(
            f"only {EXPECTED_MODEL_NAME} is supported; got expected_name={expected_name!r}"
        )

    model_path = Path(str(_required(model, "path", "model"))).expanduser()
    if model_path.name != EXPECTED_MODEL_NAME:
        raise ConfigError(
            f"model.path must end with {EXPECTED_MODEL_NAME!r}; got {model_path}"
        )
    if require_existing_model and not model_path.is_dir():
        raise ConfigError(f"model directory does not exist: {model_path}")

    dtype = str(model.get("dtype", "bfloat16"))
    if dtype not in SUPPORTED_DTYPES:
        raise ConfigError(f"model.dtype must be one of {sorted(SUPPORTED_DTYPES)}")

    return WorkzoneConfig(
        project_name=str(project.get("name", "qwen3-asr-hotword")),
        model=ModelConfig(
            path=model_path,
            expected_name=expected_name,
            dtype=dtype,
            device=str(model.get("device", "cuda:0")),
            local_files_only=bool(model.get("local_files_only", True)),
        ),
        paths=PathsConfig(
            work_root=Path(str(_required(paths, "work_root", "paths"))).expanduser(),
            data_root=Path(str(_required(paths, "data_root", "paths"))).expanduser(),
            cache_dir=Path(str(_required(paths, "cache_dir", "paths"))).expanduser(),
            output_dir=Path(str(_required(paths, "output_dir", "paths"))).expanduser(),
        ),
        runtime=RuntimeConfig(
            expected_python_major_minor=str(
                runtime.get("expected_python_major_minor", "3.12")
            ),
            expected_gpu_name=str(runtime.get("expected_gpu_name", "NVIDIA H200")),
            minimum_gpu_memory_gib=float(runtime.get("minimum_gpu_memory_gib", 130)),
        ),
    )
