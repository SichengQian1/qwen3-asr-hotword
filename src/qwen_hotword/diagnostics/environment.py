from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwen_hotword.config import WorkzoneConfig

PACKAGE_NAMES = (
    "torch",
    "qwen-asr",
    "transformers",
    "accelerate",
    "datasets",
    "flash-attn",
    "vllm",
    "librosa",
    "soundfile",
    "PyYAML",
    "jiwer",
)


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def collect_environment(config: WorkzoneConfig) -> dict[str, Any]:
    checks: list[Check] = []
    expected_python = config.runtime.expected_python_major_minor
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(
        Check(
            "python_version",
            "pass" if actual_python == expected_python else "warn",
            f"actual={actual_python}, expected={expected_python}",
        )
    )

    gpu_data: dict[str, Any] = {
        "cuda_available": False,
        "torch_cuda_version": None,
        "devices": [],
    }
    try:
        import torch

        gpu_data["cuda_available"] = torch.cuda.is_available()
        gpu_data["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                memory_gib = round(properties.total_memory / 1024**3, 1)
                gpu_data["devices"].append(
                    {"index": index, "name": properties.name, "memory_gib": memory_gib}
                )
            selected = gpu_data["devices"][0]
            expected_name = config.runtime.expected_gpu_name
            minimum_memory = config.runtime.minimum_gpu_memory_gib
            checks.append(
                Check(
                    "gpu_type",
                    "pass" if expected_name in selected["name"] else "warn",
                    f"actual={selected['name']}, expected_contains={expected_name}",
                )
            )
            checks.append(
                Check(
                    "gpu_memory",
                    "pass" if selected["memory_gib"] >= minimum_memory else "fail",
                    f"actual={selected['memory_gib']} GiB, minimum={minimum_memory} GiB",
                )
            )
        else:
            checks.append(Check("cuda", "fail", "torch.cuda.is_available() is false"))
    except (ImportError, RuntimeError) as error:
        checks.append(Check("torch", "fail", f"{type(error).__name__}: {error}"))

    model_exists = config.model.path.is_dir()
    checks.append(
        Check(
            "model_directory",
            "pass" if model_exists else "fail",
            f"exists={model_exists}, basename={config.model.path.name}",
        )
    )

    for name, path in (
        ("work_root", config.paths.work_root),
        ("cache_dir", config.paths.cache_dir),
        ("output_dir", config.paths.output_dir),
    ):
        checks.append(
            Check(
                name,
                "pass" if path.exists() else "warn",
                f"exists={path.exists()}, basename={path.name}",
            )
        )

    return {
        "schema_version": 1,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": package_versions(),
        "gpu": gpu_data,
        "model": {
            "expected_name": config.model.expected_name,
            "configured_basename": config.model.path.name,
            "exists": model_exists,
        },
        "checks": [asdict(check) for check in checks],
    }


def overall_status(report: dict[str, Any]) -> str:
    statuses = {check["status"] for check in report["checks"]}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def write_json_report(report: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
