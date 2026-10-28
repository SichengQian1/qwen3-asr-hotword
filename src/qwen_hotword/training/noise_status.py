"""Read-only progress and bounded CPU diagnostics for an existing noise experiment.

Never calls the training orchestrator or rewrites its sealed plan/cache/state. Tensor
inspection uses CPU mmap and reads offsets/labels, not full Encoder feature arrays.
"""

from __future__ import annotations

import json
import random
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qwen_hotword.training.english_newval import save
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _head(folder: Path, epochs: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_started",
        "requested_epochs": epochs,
        "persisted_epoch": None,
        "process_liveness": "not_checked; file presence does not establish a running process",
    }
    state = folder / "training_state_latest.pt"
    if state.exists():
        import torch

        payload = torch.load(state, map_location="cpu", weights_only=True)
        epoch = payload["completed_epoch"]
        result.update(
            status="initialized_no_completed_epoch" if epoch == 0 else "epochs_persisted",
            persisted_epoch=epoch,
            best_epoch=payload.get("best_epoch"),
            saved_training_seconds=payload.get("training_seconds"),
            optimizer_has_saved_updates=bool(payload.get("optimizer_state_dict", {}).get("state")),
            initial_validation=payload.get("initial_validation"),
            initial_validation_by_group=payload.get("initial_validation_by_group"),
            best_validation=payload.get("best_validation"),
            best_validation_by_group=payload.get("best_validation_by_group"),
            last_epoch=payload.get("history", [])[-1] if payload.get("history") else None,
        )
    elif folder.exists() and any(p.name != ".ctc_training.lock" for p in folder.iterdir()):
        result["status"] = "partial_files_without_training_state"
    if (folder / "report.json").exists():
        report = _object(folder / "report.json")
        result["reported"] = {
            k: report.get(k)
            for k in (
                "status",
                "epochs_completed",
                "best_epoch",
                "initial_validation_by_group",
                "best_validation_by_group",
                "best_validation_macro_phoneme_error_rate",
                "training_seconds",
                "best_checkpoint_path",
            )
        }
    marker = folder / "completed.json"
    if marker.exists():
        identity = _object(marker)
        matches = {
            "report": _sha(folder / "report.json") == identity["report_sha256"],
            "checkpoint": _sha(folder / "ctc_head_best.pt") == identity["checkpoint_sha256"],
        }
        result.update(
            status="completed_verified" if all(matches.values()) else "identity_mismatch",
            completion_checks=matches,
        )
    return result


def _cache_progress(run: Path, plan: dict[str, Any]) -> dict[str, Any]:
    marker_path = run / "cache_completed.json"
    marker = _object(marker_path) if marker_path.exists() else {}
    groups: dict[str, Any] = {}
    invalid = []
    with stage_progress(
        "Read saved feature progress (metadata only)", total=len(plan["parts"]), every=250
    ) as tick:
        for n, part in enumerate(plan["parts"], 1):
            group = groups.setdefault(
                part["split"],
                {
                    "expected_parts": 0,
                    "completed_parts_observed": 0,
                    "records": 0,
                    "feature_bytes": 0,
                    "extraction_seconds_sum": 0.0,
                },
            )
            group["expected_parts"] += 1
            folder = (run / part["manifest"]).parent / "features"
            summary = folder / "cache_summary.json"
            if summary.exists():
                try:
                    info = _object(summary)
                    shard = folder / "shards/shard-000000.pt"
                    metadata = _object(folder / "shards/shard-000000.json")
                    if (
                        info["status"] != "pass"
                        or info["sample_count"] != part["records"]
                        or not shard.is_file()
                        or shard.stat().st_size != metadata["feature_bytes"]
                    ):
                        raise ValueError("feature completion/count/size mismatch")
                    group["completed_parts_observed"] += 1
                    group["records"] += info["sample_count"]
                    group["feature_bytes"] += info["feature_bytes"]
                    group["extraction_seconds_sum"] += info["extraction_seconds"]
                except (OSError, KeyError, TypeError, ValueError) as error:
                    invalid.append({"manifest": part["manifest"], "error": str(error)})
            tick(n)
    observed = sum(g["completed_parts_observed"] for g in groups.values())
    return {
        "completion_marker_present": bool(marker),
        "marker_status": marker.get("status"),
        "expected_parts": len(plan["parts"]),
        "completed_parts_observed": observed,
        "by_split": groups,
        "invalid_part_examples": invalid[:20],
        "invalid_parts": len(invalid),
        "mixing": marker.get("mixing"),
        "verification_scope": (
            "saved summaries, tensor existence/size; not full tensor SHA or values"
        ),
    }


def _tensor_index(
    path: Path, metadata: dict[str, Any]
) -> dict[str, tuple[tuple[int, int], tuple[int, ...]]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    hidden = payload["hidden_states"]
    offsets = payload["hidden_offsets"].tolist()
    token_offsets = payload["token_offsets"].tolist()
    tokens = payload["token_ids"]
    ids = metadata["sample_ids"]
    if (
        hidden.ndim != 2
        or hidden.shape[1] != 1024
        or hidden.dtype != torch.bfloat16
        or len(offsets) != len(ids) + 1
        or len(token_offsets) != len(ids) + 1
        or len(set(ids)) != len(ids)
        or offsets[0] != 0
        or token_offsets[0] != 0
        or offsets[-1] != hidden.shape[0]
        or token_offsets[-1] != len(tokens)
        or any(a > b for a, b in zip(offsets, offsets[1:], strict=False))
        or any(a > b for a, b in zip(token_offsets, token_offsets[1:], strict=False))
    ):
        raise ValueError(f"invalid cached tensor structure: {path}")
    return {
        sample_id: (
            (offsets[i + 1] - offsets[i], int(hidden.shape[1])),
            tuple(tokens[token_offsets[i] : token_offsets[i + 1]].tolist()),
        )
        for i, sample_id in enumerate(ids)
    }


def inspect_replacements(
    run: Path,
    plan: dict[str, Any],
    *,
    ratio: int,
    epoch: int,
    max_shards: int,
    max_examples: int = 10,
) -> dict[str, Any]:
    root = Path(plan["train_cache"])
    count = _object(root / "cache_index.json")["shard_count"]
    order = list(range(count))
    random.Random(plan["configuration"]["seed"] + epoch).shuffle(order)
    if max_shards:
        order = order[:max_shards]
    parts = {p["base_shard"]: p for p in plan["parts"] if p["split"] == "train"}
    result: dict[str, Any] = {
        "ratio": ratio,
        "epoch_order": epoch,
        "total_training_shards": count,
        "shard_limit": max_shards,
        "checked_shards": 0,
        "checked_replacements": 0,
        "label_mismatches": 0,
        "frame_mismatches": 0,
        "manifest_label_mismatches": 0,
        "examples": [],
        "stopped_on_first_mismatching_shard": False,
        "gpu_used": False,
        "encoder_loaded": False,
        "tensor_access": "CPU mmap; offsets/labels only",
        "full_tensor_sha_verified": False,
    }
    with stage_progress(
        "Locate replacement mismatch in training order", total=len(order), every=25
    ) as tick:
        for position, index in enumerate(order, 1):
            result["checked_shards"] += 1
            tick(position)
            if index not in parts:
                continue
            part = parts[index]
            manifest = run / part["manifest"]
            if _sha(manifest) != part["sha256"]:
                raise ValueError(f"frozen noise manifest identity mismatch: {manifest}")
            rows = [r for r in _rows(manifest) if ratio == 50 or r["selected_30"]]
            if not rows:
                continue
            clean_folder = root / "shards"
            noisy_folder = manifest.parent / "features/shards"
            original = _tensor_index(
                clean_folder / f"shard-{index:06d}.pt",
                _object(clean_folder / f"shard-{index:06d}.json"),
            )
            noisy = _tensor_index(
                noisy_folder / "shard-000000.pt", _object(noisy_folder / "shard-000000.json")
            )
            audits = {
                a["id"]: a for a in json.loads((manifest.parent / "mix_audit.json").read_text())
            }
            found = False
            for row in rows:
                sample_id = row["id"]
                a, b = original[sample_id], noisy[sample_id]
                labels_differ = a[1] != b[1]
                frames_differ = a[0] != b[0]
                manifest_differ = tuple(row["phoneme_token_ids"]) != b[1]
                result["checked_replacements"] += 1
                result["label_mismatches"] += int(labels_differ)
                result["frame_mismatches"] += int(frames_differ)
                result["manifest_label_mismatches"] += int(manifest_differ)
                if labels_differ or frames_differ or manifest_differ:
                    found = True
                    if len(result["examples"]) < max_examples:
                        result["examples"].append(
                            {
                                "sample_id": sample_id,
                                "base_shard": index,
                                "language": row["language"],
                                "audio_path": row["audio_path"],
                                "original_shape": list(a[0]),
                                "noise_shape": list(b[0]),
                                "original_label_ids": list(a[1]),
                                "noise_label_ids": list(b[1]),
                                "manifest_label_ids": row["phoneme_token_ids"],
                                "label_mismatch": labels_differ,
                                "frame_mismatch": frames_differ,
                                "mix_audit": audits.get(sample_id),
                                "original_feature": str(clean_folder / f"shard-{index:06d}.pt"),
                                "noise_feature": str(noisy_folder / "shard-000000.pt"),
                                "original_feature_sha256_recorded": _object(
                                    clean_folder / f"shard-{index:06d}.json"
                                )["feature_sha256"],
                                "noise_feature_sha256_recorded": _object(
                                    noisy_folder / "shard-000000.json"
                                )["feature_sha256"],
                            }
                        )
            if found:
                result["stopped_on_first_mismatching_shard"] = True
                break
    result["all_training_shards_checked"] = result["checked_shards"] == count
    result["conclusion"] = (
        "replacement mismatch reproduced from saved caches"
        if result["stopped_on_first_mismatching_shard"]
        else "all training replacements passed shape/label checks; tensor values were not audited"
        if result["all_training_shards_checked"]
        else "no mismatch in inspected subset; not evidence that unchecked shards are valid"
    )
    return result


def inspect(
    run: Path, *, check_replacements: bool = False, ratio: int = 30, max_shards: int = 16
) -> dict[str, Any]:
    plan = _object(run / "plan.json")
    prepared = _object(run / "prepared.json")
    plan_matches = _sha(run / "plan.json") == prepared["plan_sha256"]
    if not plan_matches:
        raise ValueError("prepared plan SHA mismatch; do not diagnose using an altered plan")
    result: dict[str, Any] = {
        "scope": "saved_noise_experiment_status_no_inference_no_training",
        "diagnostic_source_sha256": _sha(Path(__file__)),
        "run": str(run),
        "plan_sha256": prepared["plan_sha256"],
        "preparation": prepared,
        "cache": _cache_progress(run, plan),
        "training": {},
        "source_code_drift": [],
        "snapshot_consistency": "files read while an external process might still be running",
    }
    for name, expected in plan["input_sha256"].items():
        path = Path(name)
        if path.suffix == ".py":
            actual = _sha(path) if path.is_file() else None
            if actual != expected:
                result["source_code_drift"].append(
                    {"path": name, "expected": expected, "actual": actual}
                )
    for percent in (30, 50):
        try:
            result["training"][str(percent)] = _head(
                run / f"head_{percent}", plan["configuration"]["training"]["epochs"]
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result["training"][str(percent)] = {"status": "inspection_error", "error": str(error)}
    evaluation = run / "evaluation/report.json"
    result["evaluation"] = {
        "status": "report_present"
        if evaluation.exists()
        else "partial_files"
        if (run / "evaluation").exists()
        else "not_started",
        "report": _object(evaluation) if evaluation.exists() else None,
    }
    if check_replacements:
        completed = result["training"][str(ratio)].get("persisted_epoch") or 0
        try:
            result["replacement_check"] = inspect_replacements(
                run, plan, ratio=ratio, epoch=completed + 1, max_shards=max_shards
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            result["replacement_check"] = {
                "status": "inspection_error",
                "error": str(error),
                "examples": [],
                "all_training_shards_checked": False,
            }
    result["interpretation"] = [
        "A completed feature cache is not completed Head training or measured retrieval quality.",
        "Persisted epoch 0 proves initial validation finished; "
        "unsaved optimizer steps might still have run.",
        "The current replacement guard belongs to training; validation views have no replacements.",
        "A shape difference is not permission to crop/pad or bypass the label/frame guard.",
    ]
    return result


def terminal_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Keep terminal output compact; detailed token arrays stay in the result bundle."""
    cache = report["cache"]
    heads = {}
    for ratio, head in report["training"].items():
        heads[ratio] = {
            "status": head["status"],
            "persisted_epoch": head.get("persisted_epoch"),
            "requested_epochs": head.get("requested_epochs"),
            "best_epoch": head.get("best_epoch"),
            "initial_per_by_group": {
                group: value.get("phoneme_error_rate")
                for group, value in (head.get("initial_validation_by_group") or {}).items()
            },
            "best_per_by_group": {
                group: value.get("phoneme_error_rate")
                for group, value in (head.get("best_validation_by_group") or {}).items()
            },
        }
        if "error" in head:
            heads[ratio]["error"] = head["error"]
    result: dict[str, Any] = {
        "status": "inspection_completed",
        "run": report["run"],
        "preparation": report["preparation"],
        "cache": {
            key: cache[key]
            for key in (
                "marker_status",
                "expected_parts",
                "completed_parts_observed",
                "by_split",
                "invalid_parts",
            )
        },
        "training": heads,
        "evaluation": report["evaluation"]["status"],
        "source_code_drift_count": len(report["source_code_drift"]),
        "return_bundle": report["return_bundle"],
    }
    if "replacement_check" in report:
        check = report["replacement_check"]
        result["replacement_check"] = {
            key: value for key, value in check.items() if key != "examples"
        }
        result["replacement_check"]["first_example"] = {
            key: value
            for key, value in (check["examples"][:1] or [{}])[0].items()
            if key
            in {
                "sample_id",
                "base_shard",
                "language",
                "original_shape",
                "noise_shape",
                "label_mismatch",
                "frame_mismatch",
            }
        }
    return result


def write_diagnostic(report: dict[str, Any], destination: Path) -> dict[str, Any]:
    run = Path(report["run"])
    if destination.exists() or destination.is_relative_to(run) or run.is_relative_to(destination):
        raise ValueError("diagnostic output must be a NEW directory outside the training run")
    destination.mkdir(parents=True)
    report = {**report, "inspected_at": datetime.now(timezone(timedelta(hours=8))).isoformat()}
    save(destination / "report.json", report)
    digest = _sha(destination / "report.json")
    with zipfile.ZipFile(destination / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report.json", (destination / "report.json").read_bytes())
        archive.writestr("sha256.txt", f"{digest}  report.json\n")
    return {
        **report,
        "diagnostic_output": str(destination),
        "return_bundle": str(destination / "return_bundle.zip"),
    }
