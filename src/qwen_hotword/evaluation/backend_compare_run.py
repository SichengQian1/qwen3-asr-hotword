"""One-command isolated vLLM 0.23.0 environment and a small matched H200 test."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.backend_comparison import (
    compare_rows,
    digest,
    read,
    sha,
    write_once,
)
from qwen_hotword.training.stage_progress import stage_progress


def command(arguments: list[str], *, label: str, log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with stage_progress(label), log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(arguments) + "\n")
        handle.flush()
        with subprocess.Popen(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            if process.wait():
                raise RuntimeError(f"{label} failed (exit {process.returncode}); see {log}")


def requirements(cfg: dict[str, Any]) -> list[str]:
    # Never install qwen-asr[vllm]: its extra pins vLLM 0.14.0.
    return [
        cfg["vllm_wheel"],
        "torch==2.11.0+cu129",
        "torchaudio==2.11.0+cu129",
        "torchvision==0.26.0+cu129",
        "transformers==4.57.6",
        "qwen-asr==0.0.6",
    ]


def memory_budget(total_mib: int, cfg: dict[str, Any]) -> dict[str, int]:
    """Plan sequential backends against total capacity, preserving shared-GPU headroom."""
    fraction = float(cfg["gpu_memory_utilization"])
    allocator_gib = float(cfg["transformers_allocator_limit_gib"])
    reserve = float(cfg["shared_memory_reserve_mib"])
    if (
        total_mib <= 0
        or not math.isfinite(fraction)
        or not 0 < fraction <= 1
        or not math.isfinite(allocator_gib)
        or not 0 < allocator_gib * 1024 <= total_mib
        or not math.isfinite(reserve)
        or reserve <= 0
    ):
        raise ValueError("invalid shared-GPU memory budget")
    vllm = math.ceil(total_mib * fraction)
    transformers = math.ceil(allocator_gib * 1024)
    return {
        "vllm_executor_budget_mib": vllm,
        "transformers_allocator_limit_mib": transformers,
        "reserve_mib": math.ceil(reserve),
        # The backend subprocesses exit in sequence; do not add their budgets.
        "required_free_mib": max(vllm, transformers) + math.ceil(reserve),
    }


def preflight(gpu: int, cfg: dict[str, Any]) -> dict[str, Any]:
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise ValueError("run from the H200 work-zone Linux x86_64 Python 3.12 environment")
    result = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=name,driver_version,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = next(csv.reader(io.StringIO(result)))
    name, driver, used, total, free = (field.strip() for field in fields)
    info = {
        "physical_gpu": gpu,
        "name": name,
        "driver": driver,
        "used_mib": int(used),
        "total_mib": int(total),
        "free_mib": int(free),
    }
    driver_tuple = tuple(map(int, driver.split(".")))
    toolkit_tuple = tuple(map(int, cfg.get("cuda_toolkit_driver", "575.51.03").split(".")))
    info["cuda_compatibility_mode"] = (
        "full_toolkit_driver"
        if driver_tuple >= toolkit_tuple
        else "CUDA12_minor_compatibility_requires_actual_kernel_probe"
    )
    if "H200" not in name:
        raise ValueError(f"expected the allocated H200: {info}")
    if tuple(map(int, driver.split("."))) < tuple(map(int, cfg["minimum_driver"].split("."))):
        raise ValueError(
            f"driver {driver} is below CUDA12 minor-compatibility minimum {cfg['minimum_driver']}; "
            "do not modify node drivers automatically. Return preflight/failure bundle first."
        )
    budget = memory_budget(int(total), cfg)
    info["memory_budget"] = budget
    if int(free) < budget["required_free_mib"]:
        raise ValueError(
            f"GPU {gpu} has {free} MiB free, needs {budget['required_free_mib']} MiB "
            f"(sequential backend budget plus {budget['reserve_mib']} MiB headroom); "
            "leave other jobs running and retry when enough memory is free"
        )
    return info


def record_preflight(gpu: int, cfg: dict[str, Any], output: Path, stage: str) -> None:
    info = preflight(gpu, cfg)
    # Free/used memory changes between stages and retries; it is observation,
    # not an immutable device identity. Keep each distinct snapshot separately.
    write_once(
        output / "device.json",
        {k: v for k, v in info.items() if k not in ("used_mib", "free_mib")},
    )
    snapshot = {"stage": stage, **info}
    write_once(output / f"memory_checks/{stage}_{digest(snapshot)[:12]}.json", snapshot)
    budget = info["memory_budget"]
    print(
        f"GPU {gpu} shared-memory check: {info['free_mib']} MiB free / "
        f"{info['total_mib']} MiB total; vLLM budget {budget['vllm_executor_budget_mib']} MiB, "
        f"Transformers allocator limit {budget['transformers_allocator_limit_mib']} MiB, "
        f"headroom {budget['reserve_mib']} MiB; backends run sequentially",
        flush=True,
    )


def environment(root: Path, cfg: dict[str, Any], output: Path, env: dict[str, str]) -> Path:
    target: Path = (root / cfg["environment"]).resolve()
    if not target.is_relative_to((root / ".venv").resolve()):
        raise ValueError("test environment must be inside this repo's ignored .venv directory")
    if (
        target == Path(sys.prefix).resolve()
        and not (target / "test_environment_contract.json").exists()
    ):
        raise ValueError("refusing to install into the invoking/production Python environment")
    identity = {
        "requirements": requirements(cfg),
        "python": "3.12",
        "torch_index": cfg["torch_index"],
    }
    marker = target / "test_environment_contract.json"
    if target.exists() and not marker.exists():
        raise ValueError("environment path already exists without this experiment's identity")
    if not target.exists():
        if shutil.disk_usage(root).free < 25 * 1024**3:
            raise ValueError("reserve at least 25 GiB for isolated packages and wheel cache")
        target.mkdir(parents=True)
        write_once(marker, identity)
    write_once(marker, identity)
    # Preserve certificate/mirror settings, but never inherit an installation
    # destination that could redirect pip out of the isolated environment.
    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONHOME"):
        env.pop(key, None)
    env["VIRTUAL_ENV"] = str(target)
    python = target / "bin/python"
    if not python.exists():
        command(
            [sys.executable, "-m", "venv", str(target)],
            label="Create isolated Python environment",
            log=output / "logs/environment.log",
            env=env,
        )
    ready = target / "test_environment_ready.json"
    if not ready.exists():
        command(
            [
                str(python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "--only-binary=:all:",
                "--extra-index-url",
                cfg["torch_index"],
                "--report",
                str(output / "pip_install_report.json"),
                *requirements(cfg),
            ],
            label="Install fixed vLLM0.23/CUDA12.9 runtime (original environment preserved)",
            log=output / "logs/environment.log",
            env=env,
        )
        command(
            [str(python), "-m", "pip", "check"],
            label="Check isolated package dependencies",
            log=output / "logs/environment.log",
            env=env,
        )
        write_once(ready, identity)
    command(
        [
            str(python),
            "-c",
            "from qwen_hotword.evaluation.backend_compare_runtime import versions; "
            "print(versions())",
        ],
        label="Verify exact installed versions",
        log=output / "logs/environment.log",
        env=env,
    )
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True)
    write_once(output / "environment.json", {"identity": identity, "pip_freeze": frozen})
    return python


def package(output: Path) -> Path:
    destination = output / "return_bundle.zip"
    temporary = output / "return_bundle.zip.pending"
    names = [p for p in output.rglob("*.json") if p.name != "pip_install_report.json"]
    names += list(output.glob("*.md")) + list(output.glob("*.tsv"))
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(names):
            archive.write(path, path.relative_to(output).as_posix())
        for log in sorted((output / "logs").glob("*.log")):
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 65536))
                archive.writestr(f"log_tails/{log.name}", handle.read())
    temporary.replace(destination)
    return destination


def summarize(output: Path) -> dict[str, Any]:
    requests = read(output / "requests.json")["requests"]
    rows = {
        backend: {path.stem: read(path) for path in (output / backend / "cells").glob("*.json")}
        for backend in ("transformers", "vllm")
    }
    details, report = compare_rows(requests, rows)
    write_once(output / "report.json", report)
    write_once(output / "details.json", {"rows": details})
    columns = [
        "language",
        "condition",
        "mode",
        "classification",
        "words",
        "reference_text",
        "transformers",
        "vllm",
        "audio_path",
    ]
    destination = output / "comparison.tsv"
    table = io.StringIO()
    writer = csv.DictWriter(
        table,
        fieldnames=columns,
        delimiter="\t",
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(details)
    if destination.exists() and destination.read_text(encoding="utf-8") != table.getvalue():
        raise ValueError("existing comparison table changed")
    if not destination.exists():
        destination.write_text(table.getvalue(), encoding="utf-8", newline="")
    totals = report["by_group"]["overall"]
    text = (
        "# H200：Transformers 与 vLLM0.23.0 转写对照\n\n"
        f"对照{totals['pairs']}组：原文完全一致{totals['exact']}组，仅格式差异"
        f"{totals['format_only']}组，词语差异{totals['word_difference']}组。\n\n"
        f"小样本参考WER：Transformers {totals['transformers_wer']}，"
        f"vLLM {totals['vllm_wer']}（小数比例）。\n\n"
        "两边FP16、同一环境、波形与实际Prompt token核对一致；未重跑CTC。\n\n"
        "逐条结果见comparison.tsv；details.json保留热词命中差异。"
        "这是功能对照，不是总体质量、310P兼容或显存/性能结论。\n"
    )
    md = output / "comparison.md"
    if md.exists() and md.read_text(encoding="utf-8") != text:
        raise ValueError("existing comparison explanation changed")
    if not md.exists():
        md.write_text(text, encoding="utf-8")
    return report


def run(root: Path, cfg: dict[str, Any], output: Path, gpu: int, script: Path) -> dict[str, Any]:
    import fcntl

    protected = [
        (root / cfg["source_run"]).resolve(),
        (root / cfg["environment"]).resolve(),
        Path(cfg["model"]).resolve(),
    ]
    if any(
        output == path or output.is_relative_to(path) or path.is_relative_to(output)
        for path in protected
    ):
        raise ValueError("output overlaps source, model or environment")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (output / "run_contract.json").exists() and any(
            p.name != "active.lock" for p in output.iterdir()
        ):
            raise ValueError("nonempty output without experiment identity; choose a new directory")
        files = [
            script,
            Path(__file__),
            Path(__file__).with_name("backend_comparison.py"),
            Path(__file__).with_name("backend_compare_runtime.py"),
        ]
        write_once(
            output / "run_contract.json",
            {
                "config": cfg,
                "physical_gpu": gpu,
                "source_sha256": {str(p): sha(p) for p in files},
            },
        )
        if (output / "report.json").exists():
            report = summarize(output)  # Validate saved cells; do not reinstall or load Qwen.
            bundle = package(output)
            return {
                **report,
                "output": str(output),
                "return_bundle": str(bundle),
                "reused_completed_run": True,
            }
        try:
            model = Path(cfg["model"])
            if model.name != "Qwen3-ASR-1.7B":
                raise ValueError("only the existing Qwen3-ASR-1.7B snapshot is allowed")
            metadata = {
                name: sha(model / name)
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "preprocessor_config.json",
                )
            }
            if (
                "Qwen3ASRForConditionalGeneration"
                not in read(model / "config.json")["architectures"]
            ):
                raise ValueError("model architecture differs")
            write_once(output / "model_metadata.json", metadata)
            with stage_progress("Check selected H200, driver and free memory before installing"):
                record_preflight(gpu, cfg, output, "before_install")
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "PYTHONPATH": str(root / "src"),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "VLLM_PLUGINS": "",
                    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            python = environment(root, cfg, output, env)
            env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            for stage in ("probe", "prepare", "transformers", "vllm"):
                if stage != "prepare":
                    record_preflight(gpu, cfg, output, stage)
                command(
                    [
                        str(python),
                        "-B",
                        str(script),
                        "_worker",
                        "--stage",
                        stage,
                        "--output-dir",
                        str(output),
                    ],
                    label=f"Backend comparison: {stage}",
                    log=output / f"logs/{stage}.log",
                    env=env,
                )
            report = summarize(output)
        except Exception as error:
            failure = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
            # Each failure is a separate immutable snapshot, not an overwritten historic result.
            write_once(output / f"failure_{digest(failure)[:12]}.json", failure)
            package(output)
            raise
        bundle = package(output)
        return {**report, "output": str(output), "return_bundle": str(bundle)}
