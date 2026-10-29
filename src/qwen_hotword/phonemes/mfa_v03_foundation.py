"""Versioned preparation only: no training, no Encoder/ASR, no historical label writes."""

from __future__ import annotations

import json
import os
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, normalization_key
from qwen_hotword.phonemes.mfa_v03_assets import (
    download,
    environment,
    pin,
    run_worker,
    sha,
    write_json,
)
from qwen_hotword.phonemes.mfa_v03_review import (
    build_words,
    export_review,
    portuguese_targets,
    scan_sources,
    write_rows,
)
from qwen_hotword.training.stage_progress import stage_progress


def validate_inventory(root: Path, config: dict[str, Any], assets: Path) -> dict[str, Any]:
    pin(root / config["old_vocab"], config["old_vocab_sha256"])
    pin(root / config["vocab"], config["vocab_sha256"])
    old, new = (load_phoneme_vocab(root / config[k]) for k in ("old_vocab", "vocab"))
    if new.tokens[: len(old.tokens)] != old.tokens or len(new.tokens) != 100:
        raise ValueError("v0.3 must preserve all 90 old IDs and have exactly 100 classes")
    if new.tokenization_policy != "mfa_units_v0.3":
        raise ValueError("new vocabulary requires the unit-preserving v0.3 tokenizer")
    inventory: dict[str, Any] = {}
    with stage_progress("Validate pinned model/lexicon alphabets against v0.3 units") as update:
        total = 0
        for lang, spec in config["models"].items():
            for relative, identity in spec["files"].items():
                pin(assets / lang / relative, identity["sha256"])
            folder = assets / lang / "g2p" / spec["dialect"]
            meta = json.loads((folder / "meta.json").read_text())
            if (
                meta["architecture"] != spec["architecture"]
                or meta["version"] != spec["meta_version"]
                or meta["train_date"] != spec["train_date"]
                or set(meta["phones"]) != set(spec["phones"])
            ):
                raise ValueError("pinned model metadata differs")
            symbol_path = folder / ("phones." + spec["symbol_suffix"])
            symbols = {
                line.split()[0] for line in symbol_path.read_text().splitlines() if line.strip()
            }
            symbols -= {"<eps>", "sil", "spn", "sp", "|", "_"}
            observed: Counter[str] = Counter()
            special_rows = stress_rows = pronunciation_rows = 0
            dictionary = assets / lang / "dictionary" / (spec["dialect"] + ".dict")
            with dictionary.open(encoding="utf-8-sig") as stream:
                for line in stream:
                    fields = line.split()
                    if not fields or fields[0].startswith("#"):
                        continue
                    index = 1
                    while index < len(fields):
                        try:
                            float(fields[index])
                        except ValueError:
                            break
                        index += 1
                    phones = fields[index:]
                    if not phones:
                        raise ValueError("dictionary pronunciation without phones")
                    if any(p in {"sil", "spn", "sp", "<unk>", "<eps>"} for p in phones):
                        special_rows += 1
                        continue
                    observed.update(phones)
                    stress_rows += any("ˈ" in p or "ˌ" in p for p in phones)
                    pronunciation_rows += 1
                    total += 1
                    update(total)
            all_units = set(meta["phones"]) | symbols | set(observed)
            missing = sorted(p for p in all_units if normalization_key(p) not in new.token_to_id)
            if missing:
                raise ValueError(f"{lang}: new vocabulary misses whole units: {missing}")
            inventory[lang] = {
                "meta_phone_count": len(meta["phones"]),
                "g2p_output_symbol_count": len(symbols),
                "lexicon_phone_count": len(observed),
                "lexicon_pronunciation_rows": pronunciation_rows,
                "special_non_speech_rows": special_rows,
                "explicit_stress_rows": stress_rows,
                "meta_units_not_in_g2p_symbols": sorted(set(meta["phones"]) - symbols),
                "lexicon_units_not_in_g2p_symbols": sorted(set(observed) - symbols),
                "phone_occurrences_in_lexicon_not_training_counts": dict(observed),
                "unknown_units": missing,
            }
    return {
        "status": "passed_structural_inventory_checks",
        "ctc_output_classes": len(new.tokens),
        "old_ids_preserved": True,
        "added": list(new.tokens[len(old.tokens) :]),
        "by_language": inventory,
        "training_ready": False,
        "note": "Alphabet compatibility is not pronunciation, acoustic or retrieval validation.",
    }


def prepare_contract(
    root: Path,
    config: dict[str, Any],
    output: Path,
    assets: Path,
) -> dict[str, Any]:
    for original in (root / config["source_run"], assets):
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError("foundation output must be separate from baseline and model assets")
    pin(root / config["vocab"], config["vocab_sha256"])
    pin(root / config["old_vocab"], config["old_vocab_sha256"])
    code = [
        root / "scripts/prepare_mfa_v03.py",
        root / "src/qwen_hotword/phonemes/coverage.py",
        root / "src/qwen_hotword/training/g2p_prep.py",
        *sorted((root / "src/qwen_hotword/phonemes").glob("mfa_v03_*.py")),
    ]
    contract = {
        "configuration": {k: v for k, v in config.items() if not k.startswith("_")},
        "config_sha256": sha(Path(config["_config_path"])),
        "assets_directory": str(assets),
        "source_sha256": {str(p.relative_to(root)): sha(p) for p in code},
    }
    path = output / "foundation_contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError(
                "foundation config/code changed; preserve this run and choose a new output"
            )
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("refusing unowned nonempty output directory")
        write_json(path, contract)
    return contract


def package_small(root: Path, config: dict[str, Any], output: Path) -> Path:
    target = output / "return_bundle.zip"
    names = [
        "report.json",
        "foundation_contract.json",
        "inventory_audit.json",
        "smoke_request.jsonl",
        "smoke_g2p.jsonl",
        "smoke_g2p.completed.json",
        "source_audit.json",
        "review_report.json",
        "logs/runtime_probe.json",
        "logs/runtime_probe.log",
    ]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = output / name
            if path.exists():
                archive.write(path, name)
        archive.write(root / config["vocab"], "phoneme_vocab.v0.3.json")
        archive.write(Path(config["_config_path"]), "foundation_config.json")
    return target


def run(
    root: Path,
    config: dict[str, Any],
    output: Path,
    assets: Path,
    action: str,
    *,
    mfa_python: str | None = None,
    asset_archive: Path | None = None,
) -> dict[str, Any]:
    prepare_contract(root, config, output, assets)
    stages = ["download", "setup", "smoke", "audit", "review"] if action == "all" else [action]
    prefix = [mfa_python] if mfa_python else None
    report: dict[str, Any] = {
        "status": "running",
        "scope": "mfa_v03_foundation_no_training",
        "output": str(output),
        "assets": str(assets),
        "stages": {},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "gpu_used": False,
        "encoder_loaded": False,
        "test_set_used": False,
        "old_labels_modified": False,
        "environment_modified": False,
        "human_review": "pending",
        "training_started": False,
    }
    try:
        for stage in stages:
            if stage == "download":
                receipt = download(config, assets, archive=asset_archive)
                report["stages"][stage] = {
                    "status": receipt["status"],
                    "files": len(receipt["files"]),
                }
                result = validate_inventory(root, config, assets)
                write_json(output / "inventory_audit.json", result)
                report["inventory"] = result
            elif stage == "setup":
                if not (assets / "verified.json").is_file():
                    raise ValueError("run download first to initialize the owned asset directory")
                if prefix is None:
                    prefix = environment(config, assets, output / "logs")
                report["stages"][stage] = {
                    "status": "runtime_selected_pending_fst_smoke",
                    "python_command": prefix,
                    "environment_modified": False,
                }
            elif stage == "smoke":
                validate_inventory(root, config, assets)
                if prefix is None:
                    prefix = environment(config, assets, output / "logs")
                request = output / "smoke_request.jsonl"
                write_rows(
                    request,
                    [
                        {"language": lang, "word": word}
                        for lang, words in config["smoke_words"].items()
                        for word in words
                    ],
                )
                result = run_worker(
                    root, config, assets, prefix, request, output / "smoke_g2p.jsonl"
                )
                if any(result["counts"].get(lang + ":ok", 0) < 1 for lang in ("en", "es", "pt")):
                    raise ValueError("one or more real G2P models produced no usable smoke word")
                report["stages"][stage] = result
                report["native_smoke_has_unresolved"] = any(
                    not key.endswith(":ok") for key in result["counts"]
                )
            elif stage in {"audit", "review"}:
                targets = portuguese_targets(root, config)
                selected = scan_sources(root, config, output, targets)
                report["stages"]["audit"] = json.loads((output / "source_audit.json").read_text())
                if stage == "review":
                    if prefix is None:
                        prefix = environment(config, assets, output / "logs")
                    marker = output / "review.completed.json"
                    if marker.exists():
                        saved = json.loads(marker.read_text())
                        for name, digest in saved["files"].items():
                            pin(output / name, digest)
                        result = saved["report"]
                    else:
                        combined = build_words(
                            root, config, assets, output, prefix, selected, targets
                        )
                        result = export_review(root, config, output, selected, combined)
                        if result["audio_errors"]:
                            raise ValueError(
                                "original audio copy incomplete; see review_report.json"
                            )
                        files = [
                            "review_audio.zip",
                            "review_report.json",
                            "word_pronunciations.jsonl",
                            "business_hotword_review.jsonl",
                            "review_package/review.html",
                            "review_package/review.jsonl",
                        ]
                        write_json(
                            marker,
                            {
                                "report": result,
                                "files": {name: sha(output / name) for name in files},
                            },
                        )
                    report["stages"][stage] = result
            write_json(output / "report.json", report)
        report["status"] = (
            "prepared_pending_human_review" if action in {"all", "review"} else "completed"
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
    write_json(output / "report.json", report)
    bundle = package_small(root, config, output)
    report["return_bundle"] = str(bundle)
    write_json(output / "report.json", report)
    package_small(root, config, output)
    return report


def acquire_lock(output: Path) -> tuple[int, Path]:
    """External exclusive lock; never delete another process's or an interrupted lock."""
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.with_name(output.name + ".lock")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    return descriptor, path
