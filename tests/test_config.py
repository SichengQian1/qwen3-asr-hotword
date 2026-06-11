from pathlib import Path

import pytest
import yaml

from qwen_hotword.config import EXPECTED_MODEL_NAME, ConfigError, load_workzone_config


def valid_config() -> dict[str, object]:
    return {
        "project": {"name": "test"},
        "model": {
            "path": f"/models/{EXPECTED_MODEL_NAME}",
            "expected_name": EXPECTED_MODEL_NAME,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "local_files_only": True,
        },
        "paths": {
            "work_root": "/work",
            "data_root": "/data",
            "cache_dir": "/work/cache",
            "output_dir": "/work/runs",
        },
        "runtime": {
            "expected_python_major_minor": "3.12",
            "expected_gpu_name": "NVIDIA H200",
            "minimum_gpu_memory_gib": 130,
        },
    }


def write_config(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "workzone.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_valid_config(tmp_path: Path) -> None:
    config = load_workzone_config(write_config(tmp_path, valid_config()))
    assert config.model.expected_name == EXPECTED_MODEL_NAME
    assert config.model.path.name == EXPECTED_MODEL_NAME
    assert config.runtime.minimum_gpu_memory_gib == 130


def test_rejects_06b_model(tmp_path: Path) -> None:
    data = valid_config()
    model = data["model"]
    assert isinstance(model, dict)
    model["path"] = "/models/Qwen3-ASR-0.6B"
    model["expected_name"] = "Qwen3-ASR-0.6B"
    with pytest.raises(ConfigError, match="only Qwen3-ASR-1.7B"):
        load_workzone_config(write_config(tmp_path, data))


def test_requires_expected_model_basename(tmp_path: Path) -> None:
    data = valid_config()
    model = data["model"]
    assert isinstance(model, dict)
    model["path"] = "/models/qwen-model"
    with pytest.raises(ConfigError, match="model.path must end"):
        load_workzone_config(write_config(tmp_path, data))
