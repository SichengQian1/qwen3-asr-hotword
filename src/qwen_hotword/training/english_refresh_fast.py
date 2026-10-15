"""Reuse sealed labels and select by path, explicitly omitting WAV byte fingerprinting."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.training import english_refresh as original
from qwen_hotword.training import multilingual_480_run as historical
from qwen_hotword.training.balanced_multilingual import _write_interleaved_manifest
from qwen_hotword.training.english_refresh_freeze import old_assets, release_pool
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.spanish_480_plan import _bucket, _rows, metrics, stratified_select
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress

POLICY = "path_only_no_audio_byte_audit_user_requested"
VERSION = "en-countries-path-only-es-pt-fixed-v1"


@contextmanager
def activity() -> Iterator[None]:
    stopped = threading.Event()
    started = time.monotonic()

    def show() -> None:
        while not stopped.wait(10):
            print(
                f"Fast preparation active: {time.monotonic() - started:.0f}s; metadata only",
                flush=True,
            )

    thread = threading.Thread(target=show, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def path_key(value: str) -> str:
    # Staged paths were already resolved. Do not stat/readlink hundreds of thousands
    # of shared-storage WAVs again merely to compare strings.
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("canonical absolute staged audio path required")
    return os.path.normpath(value)


def light_candidate(row: dict[str, Any]) -> dict[str, Any]:
    frames, tokens = row["estimated_ctc_input_length"], row["phoneme_token_ids"]
    seconds = float(row["duration_seconds"])
    minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:], strict=False))
    if (
        minimum != row["ctc_minimum_input_length"]
        or type(frames) is not int
        or frames <= 0
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("staged CTC lengths differ")
    release = row["release_source"]
    if release == "original_ready":
        valid = minimum <= frames
    else:
        valid = release == "temporal_2x_recovery" and frames < minimum <= 1.8 * frames
    if not valid or not tokens or any(type(t) is not int or not 0 < t < 90 for t in tokens):
        raise ValueError("invalid staged release or phoneme labels")
    density, ratio = len(tokens) / (2 * frames), minimum / (2 * frames)
    source = row["source_corpus"]
    return {
        "id": row["id"],
        "audio_path": path_key(row["audio_path"]),
        "duration_seconds": seconds,
        "source": source,
        "release_source": release,
        "speaker_id": None,
        "stratum": json.dumps(
            [
                source,
                release,
                _bucket(seconds, (3, 6, 10, 20)),
                _bucket(density, (0.2, 0.3, 0.4, 0.5, 0.6)),
                _bucket(ratio, (0.5, 0.75, 0.9)),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "reference_tokens_per_effective_frame": density,
        "effective_ctc_ratio": ratio,
    }


def path_candidates(
    pool: Path, forbidden: set[str], seed: int, total: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    started = time.monotonic()
    for number, row in enumerate(_rows(pool), 1):
        c = light_candidate(row)
        key = c["audio_path"]
        if key in forbidden:
            excluded["fixed_train_or_heldout_path_overlap"] += 1
        else:
            signature = hashlib.sha256(
                json.dumps(
                    [
                        row["text"],
                        row["phoneme_token_ids"],
                        row["duration_seconds"],
                        row["estimated_ctc_input_length"],
                    ],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            priority = hashlib.sha256(f"{seed}\0{c['id']}".encode()).hexdigest()
            if key not in groups:
                groups[key] = {
                    "candidate": c,
                    "signature": signature,
                    "priority": priority,
                    "count": 1,
                    "conflict": False,
                }
            else:
                group = groups[key]
                group["count"] += 1
                group["conflict"] |= signature != group["signature"]
                if priority < group["priority"]:
                    group.update(candidate=c, priority=priority)
        if number % 5120 == 0 or number == total:
            print(
                f"Path-only metadata: {number}/{total}, elapsed {time.monotonic() - started:.1f}s",
                flush=True,
            )
    result = []
    for group in groups.values():
        if group["conflict"]:
            excluded["same_path_conflicting_labels"] += group["count"]
        else:
            result.append(group["candidate"])
            excluded["duplicate_consistent_path"] += group["count"] - 1
    return result, dict(excluded)


def _frozen(root: Path, cfg: dict[str, Any], prepared: Path, output: Path) -> dict[str, Any]:
    old = old_assets(root, cfg)
    print("[2/5] Read protected PATHS and old fixed manifests; no WAV files opened", flush=True)
    fixed_paths = {path_key(r["audio_path"]) for p in old["paths"].values() for r in _rows(p)}
    protected = {
        path_key(r["audio_path"])
        for r in _rows(old["fingerprints"])
        if set(r["roles"]) & {"validation", "test"}
    }
    protected |= {path_key(r["audio_path"]) for r in _rows(old["validation"])}
    protected |= {
        path_key(r["audio_path"]) for r in _rows(prepared / "source/test_audio_paths.jsonl")
    }
    output.mkdir()
    print("[3/5] Release previously completed labels; select 480h using metadata only", flush=True)
    count = release_pool(output, prepared / "source", prepared / "manifest")
    candidates, excluded = path_candidates(
        output / "released_english_pool.jsonl",
        fixed_paths | protected,
        cfg["selection_seed"],
        count,
    )
    capacity = metrics(candidates)
    original.save(
        output / "capacity_report.json",
        {
            "released_records": count,
            "available": capacity,
            "excluded": excluded,
            "isolation_policy": POLICY,
        },
    )
    chosen, quotas = stratified_select(
        candidates, cfg["target_hours"] * 3600, cfg["selection_seed"]
    )
    wanted = {r["id"] for r in chosen}
    paths = {lang: output / f"full_ctc_train_{lang}.jsonl" for lang in ("en", "es", "pt")}
    with paths["en"].open("x", encoding="utf-8") as handle:
        for row in _rows(output / "released_english_pool.jsonl"):
            if row["id"] in wanted:
                wanted.remove(row["id"])
                row.update(dataset_version=VERSION, audio_isolation_policy=POLICY)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if wanted:
        raise ValueError("selected IDs missing from completed labels")
    selection = {"en": metrics(chosen)}
    staged = {}
    print(
        "[4/5] Copy fixed ES/PT manifests and combine; check IDs/paths, skip WAV byte audit",
        flush=True,
    )
    for lang in ("es", "pt"):
        shutil.copyfile(old["paths"][lang], paths[lang])
        if _sha(paths[lang]) != cfg["fixed_languages"][lang]["sha256"]:
            raise ValueError("fixed manifest bytes changed")
        n, seconds = 0, 0.0
        for row in _rows(paths[lang]):
            n += 1
            seconds += row["duration_seconds"]
        if n != cfg["fixed_languages"][lang]["records"] or seconds / 3600 < cfg["target_hours"]:
            raise ValueError("fixed language count/hours differ")
        selection[lang] = {"records": n, "hours": seconds / 3600}
    for lang, path in paths.items():
        staged[lang] = output / f"combined_input_{lang}.jsonl"
        with staged[lang].open("x", encoding="utf-8") as handle:
            for number, row in enumerate(_rows(path), 1):
                row.update(dataset_version=VERSION, audio_isolation_policy=POLICY)
                row["audio_path"] = path_key(row["audio_path"])
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if number % 51200 == 0:
                    print(f"Combine metadata {lang}: {number} records", flush=True)
    combined = output / "full_ctc_train.jsonl"
    combined_metrics = _write_interleaved_manifest(
        combined,
        staged,
        dataset_version=VERSION,
        forbidden_audio_paths=protected,
    )
    report = {
        "status": "completed",
        "output_dir": str(output),
        "training_manifest_ready": True,
        "selection": selection,
        "available_english": capacity,
        "excluded_english": excluded,
        "stratum_quotas": quotas,
        "audio_isolation_policy": POLICY,
        "audio_identity_audit": {
            "status": "not_performed",
            "performed": False,
            "byte_isolation_verified": False,
            "conflict_groups_by_reason": {},
            "note": "No WAV content read; empty map is NOT a byte-audit pass.",
        },
        "path_identity_audit": {
            "status": "pass",
            "protected_paths": len(protected),
            "cross_split_path_overlaps": 0,
        },
        "artifacts": {
            **{lang: {"path": str(p), "sha256": _sha(p)} for lang, p in paths.items()},
            "combined": {"path": str(combined), "sha256": _sha(combined), **combined_metrics},
        },
        "fixed_languages_byte_identical": True,
        "fixed_validation_unchanged": True,
        "test_labels_used": False,
        "model_loaded": False,
        "training_started": False,
        "limitations": [
            "Different paths containing identical audio are NOT detected.",
            "480h is manifest duration, not certified unique-recording duration.",
            "No speaker-disjoint or byte-disjoint claim; label/release policies unchanged.",
        ],
    }
    original.save(output / "report.json", report)
    seal(output)
    return report


def prepare(root: Path, cfg: dict[str, Any], prepared: Path, output: Path) -> dict[str, Any]:
    with activity():
        return _prepare(root, cfg, prepared, output)


def _prepare(root: Path, cfg: dict[str, Any], prepared: Path, output: Path) -> dict[str, Any]:
    root, prepared, output = (p.resolve() for p in (root, prepared, output))
    if output.exists():
        raise FileExistsError(
            "fast preparation requires a NEW directory; preserve previous attempts"
        )
    if output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError("fast output must be separate from completed source preparation")
    print(
        "[1/5] Verify completed source/labels and original inputs; no MFA or WAV scan", flush=True
    )
    saved = json.loads((prepared / "preparation_contract.json").read_text())
    if saved["configuration"] != cfg:
        raise ValueError("source preparation configuration differs")
    original.unchanged(saved)
    if any(output.is_relative_to(Path(p).parent) for p in saved["input_sha256"]):
        raise ValueError("new fast output must be outside input directories")
    source = original.verify_stage(prepared / "source")
    summary = original.verify_stage(prepared / "manifest", "summary.json")
    g2p = json.loads((prepared / "g2p_report.json").read_text())
    if g2p["status"] != "completed" or _sha(Path(g2p["dictionary"])) != g2p["dictionary_sha256"]:
        raise ValueError("completed dictionary identity differs")
    if summary["status"] != "pass" or summary["source_records"] != source["source_records"]:
        raise ValueError("full labels must be completed before fast selection")
    inputs = dict(saved["input_sha256"])
    for folder in (prepared / "source", prepared / "manifest"):
        for line in (folder / "sha256.txt").read_text().splitlines():
            sha, relative = line.split("  ", 1)
            inputs[str(folder / relative)] = sha
    for p in (
        prepared / "g2p_report.json",
        Path(g2p["dictionary"]),
        Path(__file__),
        Path(__file__).parents[3] / "scripts/run_english_refresh_fast.py",
    ):
        inputs[str(p)] = _sha(p)
    with exclusive_feature_cache_run(output.parent / f".{output.name}.preparation"):
        output.mkdir(parents=True)
        original.save(
            output / "preparation_contract.json",
            {
                "configuration": cfg,
                "input_sha256": inputs,
                "audio_isolation_policy": POLICY,
                "source_preparation": str(prepared),
            },
        )
        frozen = _frozen(root, cfg, prepared, output / "frozen")
        print("[5/5] Publish training configuration and return bundle", flush=True)
        training = original.training_configuration(root, cfg, output, frozen)
        training["audio_isolation_policy"] = POLICY
        # training_configuration creates a new file; preserve it and publish the explicit policy
        # through a separate config used by the existing, unchanged GPU orchestration.
        policy_path = output / "training_config_path_only.json"
        original.save(policy_path, training)
        report = {
            "status": "data_prepared",
            "output_dir": str(output),
            "freeze_dir": frozen["output_dir"],
            "source_preparation": str(prepared),
            "full_english": summary,
            "available_english": frozen["available_english"],
            "selection": frozen["selection"],
            "excluded_english": frozen["excluded_english"],
            "audio_isolation_policy": POLICY,
            "audio_byte_audit_performed": False,
            "fixed_languages_byte_identical": True,
            "fixed_validation_unchanged": True,
            "model_loaded": False,
            "training_started": False,
            "training_config": str(policy_path),
            "training_config_sha256": _sha(policy_path),
            "cache_config_sha256": _sha(output / "cache_workzone.yaml"),
            "training_options": training["training_options"],
        }
        original.unchanged({"input_sha256": inputs})
        original.save(output / "report.json", report)
        seal(output)
        files = {
            "report.json": output / "report.json",
            "freeze_report.json": output / "frozen/report.json",
            "training_config.json": policy_path,
            "preparation_contract.json": output / "preparation_contract.json",
        }
        with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as z:
            for name, path in files.items():
                z.write(path, name)
            z.writestr("sha256.txt", "".join(f"{_sha(p)}  {n}\n" for n, p in sorted(files.items())))
        return report


def run_stage(root: Path, output: Path, stage: str, gpu: str, *, resume: bool = False) -> Any:
    report = json.loads((output / "report.json").read_text())
    if report.get("status") != "data_prepared" or report.get("audio_isolation_policy") != POLICY:
        raise ValueError("completed explicit path-only preparation required")
    with stage_progress("Verify fixed preparation inputs (metadata only)"):
        original.unchanged(json.loads((output / "preparation_contract.json").read_text()))
    path = output / "training_config_path_only.json"
    if (
        _sha(path) != report["training_config_sha256"]
        or _sha(output / "cache_workzone.yaml") != report["cache_config_sha256"]
    ):
        raise ValueError("training configuration changed")
    config = json.loads(path.read_text())
    if (
        config["audio_isolation_policy"] != POLICY
        or config["training_options"] != report["training_options"]
    ):
        raise ValueError("path-only policy or training options changed")
    os.chdir(root)
    if stage in {"smoke", "cache"}:
        from qwen_hotword.training.english_refresh_gpu import run_stage as run_gpu_stage

        return run_gpu_stage(config, output / "training", stage, gpu)
    with stage_progress(f"{stage} Head training: see validation/shard/epoch logs"):
        return historical.run_training_stage(config, output / "training", stage, gpu, resume=resume)
