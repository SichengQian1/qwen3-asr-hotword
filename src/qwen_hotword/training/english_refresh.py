"""Explicit CPU preparation and the historical H200 training stages for English replacement."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path
from typing import Any

import yaml

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.training import multilingual_480_run as historical
from qwen_hotword.training.english_refresh_freeze import freeze, old_assets
from qwen_hotword.training.english_refresh_source import stage_sources
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.full_manifest import build_full_training_manifest
from qwen_hotword.training.g2p_prep import prepare_mfa_wordlist
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_capacity import _sha, _verified


def save(path: Path, obj: Any) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def verify_stage(directory: Path, name: str = "report.json") -> dict[str, Any]:
    checked: dict[str, Any] = {}
    report: dict[str, Any] = json.loads(_verified(directory, name, checked).read_text())
    for line in (directory / "sha256.txt").read_text().splitlines():
        _, rel = line.split("  ", 1)
        _verified(directory, rel, checked)
    return report


def contract(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    old = old_assets(root, config)
    inputs = {name: identity["sha256"] for name, identity in old["checked"].items()}
    for key in ("vocab", "g2p_model"):
        path = (root / config[key]).resolve()
        sha = _sha(path)
        if sha != config[f"{key}_sha256"]:
            raise ValueError(f"fixed {key} SHA differs")
        inputs[str(path)] = sha
    for country in config["countries"]:
        for split in ("train", "test"):
            path = Path(country["directory"]) / f"{split}.json"
            inputs[str(path.resolve())] = _sha(path)
    package = Path(__file__).parent
    for name in (
        "english_refresh.py",
        "english_refresh_source.py",
        "english_refresh_freeze.py",
        "full_manifest.py",
        "spanish_combined.py",
        "spanish_480_plan.py",
        "spanish_480_freeze.py",
        "g2p_prep.py",
        "noah_source_inspection.py",
        "swift_json.py",
        "multilingual_480_run.py",
        "experiment_a.py",
        "mfa_audit.py",
        "balanced_multilingual.py",
        "feature_cache.py",
        "sharded_ctc.py",
    ):
        inputs[str(package / name)] = _sha(package / name)
    inputs[str(package.parent / "phonemes/coverage.py")] = _sha(
        package.parent / "phonemes/coverage.py"
    )
    script = package.parents[2] / "scripts/run_english_refresh.py"
    inputs[str(script)] = _sha(script)
    return {"configuration": config, "input_sha256": inputs}


def unchanged(saved: dict[str, Any]) -> None:
    for name, sha in saved["input_sha256"].items():
        if _sha(Path(name)) != sha:
            raise ValueError(f"input changed: {name}")


def g2p(root: Path, config: dict[str, Any], output: Path, jobs: int) -> Path:
    marker = output / "g2p_report.json"
    if marker.exists():
        report = json.loads(marker.read_text())
        if _sha(Path(report["dictionary"])) != report["dictionary_sha256"]:
            raise ValueError("completed new MFA dictionary changed")
        return Path(report["dictionary"])
    words = output / "words"
    if words.exists():
        verify_stage(words)
    else:
        summary = prepare_mfa_wordlist(output / "source/source.tsv", words)
        if summary.status != "pass":
            raise ValueError("full English word list failed")
        save(words / "report.json", summary.to_dict())
        seal(words)
    executable = shutil.which("mfa")
    if not executable:
        raise FileNotFoundError("mfa unavailable; execute prepare in the aligner environment")
    attempt = output / ("mfa_attempt_" + uuid.uuid4().hex[:10])
    attempt.mkdir()
    dictionary = attempt / "english_us.dict"
    command = [
        executable,
        "g2p",
        "--num_jobs",
        str(jobs),
        "--num_pronunciations",
        "1",
        str(words / "words.txt"),
        str(root / config["g2p_model"]),
        str(dictionary),
    ]
    print("Fresh English US MFA for all train words: " + str(attempt), flush=True)
    with (attempt / "mfa.log").open("x", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "MFA_ROOT_DIR": str(attempt / "work")},
        )
    produced = load_mfa_dictionary(dictionary)
    requested = set((words / "words.txt").read_text().splitlines())
    report = {
        "status": "completed",
        "command": command,
        "dictionary": str(dictionary),
        "dictionary_sha256": _sha(dictionary),
        "requested_unique_words": len(requested),
        "missing_words": sorted(requested - produced.keys()),
        "extra_words": sorted(produced.keys() - requested),
        "ambiguous_words": sorted(w for w, values in produced.items() if len(set(values)) != 1),
        "unresolved_policy": "retain affected full-corpus rows in needs_review",
    }
    save(marker, report)
    return dictionary


def training_configuration(
    root: Path,
    config: dict[str, Any],
    output: Path,
    frozen: dict[str, Any],
) -> dict[str, Any]:
    baseline = old_assets(root, config)["baseline"]
    zone = yaml.safe_load((root / baseline["workzone_config"]).read_text())
    zone["paths"]["cache_dir"] = str(output / "training/cache")
    zone["paths"]["output_dir"] = str(output / "training")
    zone_path = output / "cache_workzone.yaml"
    rendered = yaml.safe_dump(zone, allow_unicode=True, sort_keys=False)
    if zone_path.exists():
        if zone_path.read_text() != rendered:
            raise ValueError("generated cache configuration changed")
    else:
        with zone_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    updated = {
        **baseline,
        "freeze_dir": frozen["output_dir"],
        "train_sha256": frozen["artifacts"]["combined"]["sha256"],
        "train_counts": {k: v["records"] for k, v in frozen["selection"].items()},
        "workzone_config": str(zone_path),
        "run_dir": str(output / "training"),
    }
    for key in ("validation_manifest", "baseline_validation_cache", "vocab"):
        updated[key] = str((root / baseline[key]).resolve())
    save(output / "training_config.json", updated)
    return updated


def prepare(
    root: Path,
    config: dict[str, Any],
    output: Path,
    *,
    resume: bool = False,
    workers: int = 16,
    jobs: int = 16,
) -> dict[str, Any]:
    if not 1 <= workers <= 32 or jobs <= 0:
        raise ValueError("workers must be 1..32 and jobs positive")
    root, output = root.resolve(), output.resolve()
    current = contract(root, config)
    for country in config["countries"]:
        directory = Path(country["directory"]).resolve()
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("output must be separate from source corpora")
    for name in current["input_sha256"]:
        if output.is_relative_to(Path(name).parent):
            raise ValueError("output must be outside input directories")
    with exclusive_feature_cache_run(output.parent / f".{output.name}.preparation"):
        if output.exists():
            if not resume:
                raise FileExistsError(
                    "output exists; explicit --resume only for identical preparation"
                )
            recorded = json.loads((output / "preparation_contract.json").read_text())
            if current != recorded:
                raise ValueError(
                    "preparation inputs/code/configuration differ; choose a new output"
                )
        else:
            output.mkdir(parents=True)
            save(output / "preparation_contract.json", current)
        if (output / "report.json").exists():
            report: dict[str, Any] = json.loads((output / "report.json").read_text())
            frozen = verify_stage(Path(report["freeze_dir"]))
            if frozen["status"] != "completed":
                raise ValueError("completed freeze required")
            return report
        source = output / "source"
        if source.exists():
            verify_stage(source)
        else:
            stage_sources(config, source)
            seal(source)
        dictionary = g2p(root, config, output, jobs)
        manifest = output / "manifest"
        if (manifest / "sha256.txt").exists():
            summary = verify_stage(manifest, "summary.json")
        else:
            built = build_full_training_manifest(
                source / "source.tsv",
                Path(config["countries"][0]["directory"]).parent,
                dictionary,
                root / config["vocab"],
                manifest,
                language="en-US",
                dataset="english_countries_refresh",
                id_prefix="en_countries_refresh_row",
                split="train",
                allow_exact_dictionary_connectors=True,
                workers=workers,
                resume=resume,
            )
            summary = built.to_dict()
            if summary["status"] != "pass":
                raise ValueError("full English manifest partition failed")
            seal(manifest)
        print(
            f"All English rows processed: {summary['source_records']}; selecting after labels",
            flush=True,
        )
        marker = output / "freeze_report.json"
        if marker.exists():
            frozen = verify_stage(Path(json.loads(marker.read_text())["output_dir"]))
        else:
            frozen = freeze(
                root, config, output, workers, output / ("frozen_" + uuid.uuid4().hex[:10])
            )
            save(marker, frozen)
        unchanged(current)
        if (output / "training_config.json").exists():
            training = json.loads((output / "training_config.json").read_text())
        else:
            training = training_configuration(root, config, output, frozen)
        report = {
            "status": "data_prepared",
            "output_dir": str(output),
            "freeze_dir": frozen["output_dir"],
            "full_english": summary,
            "available_english": frozen["available_english"],
            "selection": frozen["selection"],
            "excluded_english": frozen["excluded_english"],
            "fixed_languages_byte_identical": True,
            "fixed_validation_unchanged": True,
            "training_config": str(output / "training_config.json"),
            "training_config_sha256": _sha(output / "training_config.json"),
            "cache_config_sha256": _sha(output / "cache_workzone.yaml"),
            "training_options": training["training_options"],
            "model_loaded": False,
            "training_started": False,
            "test_labels_used": False,
        }
        save(output / "report.json", report)
        files = {
            "report.json": output / "report.json",
            "preparation_contract.json": output / "preparation_contract.json",
            "source_report.json": source / "report.json",
            "mfa_report.json": output / "g2p_report.json",
            "freeze_report.json": marker,
            "training_config.json": output / "training_config.json",
        }
        with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr(
                "sha256.txt",
                "".join(f"{_sha(path)}  {name}\n" for name, path in sorted(files.items())),
            )
        return report


def run_stage(root: Path, output: Path, stage: str, gpu: str, *, resume: bool = False) -> Any:
    output = output.resolve()
    recorded = json.loads((output / "preparation_contract.json").read_text())
    unchanged(recorded)
    report = json.loads((output / "report.json").read_text())
    if report.get("status") != "data_prepared":
        raise ValueError("complete CPU preparation before any GPU stage")
    config = json.loads((output / "training_config.json").read_text())
    if (
        _sha(output / "training_config.json") != report["training_config_sha256"]
        or _sha(output / "cache_workzone.yaml") != report["cache_config_sha256"]
    ):
        raise ValueError("generated training/cache configuration identity changed")
    if config["training_options"] != report["training_options"]:
        raise ValueError("training configuration was changed")
    os.chdir(root)
    run = output / "training"
    if stage in {"smoke", "cache"}:
        return historical.run_stage(config, run, stage, gpu)
    return historical.run_training_stage(config, run, stage, gpu, resume=resume)
