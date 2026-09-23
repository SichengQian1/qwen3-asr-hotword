"""Freeze a reviewed ID plan only after mechanical file-identity isolation checks."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_480_plan import _rows, candidate, metrics
from qwen_hotword.training.spanish_capacity import _sha, _tsv, _verified


def protected_audio(pool: Path, identities: dict[str, Any]) -> dict[str, set[str]]:
    """Derive a conservative holdout superset without opening the sealed test manifest."""
    config = json.loads(_verified(pool, "split_config.json", identities).read_text())
    assignment_path = _verified(pool, "speaker_split_assignments.tsv", identities)
    assignments: dict[str, str] = {}
    for row in _tsv(assignment_path, {"speaker_id", "split"}):
        if (
            not row["speaker_id"]
            or row["speaker_id"] in assignments
            or row["split"] not in {"train", "validation", "test"}
        ):
            raise ValueError("invalid speaker assignment metadata")
        assignments[row["speaker_id"]] = row["split"]
    protected: dict[str, set[str]] = defaultdict(set)
    for corpus in config["inputs"].values():
        identity = corpus["source_tsv"]
        source = Path(identity["path"])
        digest = _sha(source)
        if digest != identity["sha256"]:
            raise ValueError(f"source metadata identity mismatch: {source}")
        identities[str(source)] = {"sha256": digest}
        # Source TSV includes transcripts, but only audio/speaker/split fields are used here.
        for row in _tsv(source, {"audio", "speaker_id", "source_split"}):
            speaker, explicit = row["speaker_id"], row["source_split"]
            if not speaker or explicit not in {"train", "validation", "test", "unsplit"}:
                raise ValueError("invalid source split metadata")
            assigned = assignments.get(speaker)
            if assigned and explicit != "unsplit" and explicit != assigned:
                raise ValueError("source/assignment split conflict")
            split = assigned or explicit
            if split in {"validation", "test"}:
                path = Path(row["audio"])
                if not path.is_absolute():
                    raise ValueError("holdout metadata audio path must be absolute")
                protected[str(path.resolve())].add(split)
    if not protected:
        raise ValueError("empty holdout protection registry")
    return dict(protected)


def _fingerprint(filename: str) -> tuple[str, str, int]:
    path = Path(filename)
    before = path.stat()
    digest = _sha(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"audio changed during hashing: {path}")
    return filename, digest, after.st_size


def audit_audio(
    selected: dict[str, dict[str, Any]],
    protected: dict[str, set[str]],
    output: Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    if not 1 <= workers <= 32:
        raise ValueError("workers must be within 1..32")
    files = iter(sorted(set(selected) | set(protected)))
    digest_members: dict[str, list[str]] = defaultdict(list)
    checked, total_bytes = 0, 0
    with (
        (output / "audio_fingerprints.jsonl").open("x") as handle,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        while batch := list(islice(files, 256)):
            for path, digest, size in executor.map(_fingerprint, batch):
                if path in selected and selected[path].get("audio_file_sha256") not in (
                    None,
                    digest,
                ):
                    raise ValueError(f"selected Noah audio differs from staging SHA: {path}")
                digest_members[digest].append(path)
                roles = sorted(
                    protected.get(path, set()) | ({"train"} if path in selected else set())
                )
                handle.write(
                    json.dumps(
                        {
                            "audio_path": path,
                            "sha256": digest,
                            "size_bytes": size,
                            "roles": roles,
                        }
                    )
                    + "\n"
                )
                checked += 1
                total_bytes += size
            if checked % 5120 == 0:
                print(f"Audio file identity checked: {checked}", flush=True)
    conflicts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for digest, paths in digest_members.items():
        train_paths = [p for p in paths if p in selected]
        heldout_roles = set().union(*(protected.get(p, set()) for p in paths))
        reasons = []
        if len(train_paths) > 1:
            reasons.append("duplicate_train_file_bytes")
        if train_paths and heldout_roles:
            reasons.append("train_holdout_file_overlap")
        if {"validation", "test"} <= heldout_roles:
            reasons.append("validation_test_protected_file_overlap")
        conflicts.update(reasons)
        if reasons and len(examples) < 10:
            examples.append({"sha256": digest, "reasons": reasons, "paths": paths[:5]})
    return {
        "checked_files": checked,
        "bytes_read": total_bytes,
        "unique_train_file_hashes": sum(
            any(p in selected for p in ps) for ps in digest_members.values()
        ),
        "conflict_groups_by_reason": dict(conflicts),
        "conflict_examples": examples,
        "train_holdout_path_overlaps": len(set(selected) & set(protected)),
        "protected_paths_by_split": {
            s: sum(s in roles for roles in protected.values()) for s in ("validation", "test")
        },
    }


def write_draft(
    choices: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    protected: dict[str, set[str]],
) -> list[dict[str, Any]]:
    wanted = {(r["origin"], r["id"]): r for r in choices}
    if len(wanted) != len(choices) or len({r["id"] for r in choices}) != len(choices):
        raise ValueError("duplicate chosen ID")
    if len({r["audio_path"] for r in choices}) != len(choices):
        raise ValueError("duplicate chosen path")
    sources = [("old_train", Path(config["pool"]) / "full_ctc_train.jsonl")]
    sources += [
        ("noah", Path(config["manifest"]) / name)
        for name in ("train_ready.jsonl", "needs_review.jsonl")
    ]
    bound = []
    with (output / "train.pending.jsonl").open("x") as handle:
        for origin, path in sources:
            for raw in _rows(path):
                chosen = wanted.pop((origin, raw["id"]), None)
                if chosen is None:
                    continue
                release = chosen["release_source"]
                actual = candidate(raw, chosen["source"], release)
                for field in (
                    "id",
                    "audio_path",
                    "duration_seconds",
                    "effective_ctc_ratio",
                    "reference_tokens_per_effective_frame",
                    "stratum",
                ):
                    if actual[field] != chosen[field]:
                        raise ValueError(f"chosen/source {field} mismatch")
                if actual["audio_path"] in protected:
                    raise ValueError("selected path is protected holdout audio")
                if origin == "old_train":
                    if raw["split"] != "train" or raw["language"] != "es":
                        raise ValueError("old record is not Spanish train")
                    if raw["ctc_time_upsampling_factor"] != 2:
                        raise ValueError("old temporal policy differs")
                    record = dict(raw)
                    record["original_release_source"] = raw["release_source"]
                    record["release_source"] = release
                else:
                    if raw["split"] != "unsplit" or raw["language"] != "es-419":
                        raise ValueError("Noah record is not an unsplit es-419 candidate")
                    reasons = {i["reason"] for i in raw["issues"]}
                    if release == "original_ready":
                        if reasons or raw["training_ready"] is not True:
                            raise ValueError("invalid Noah original-ready record")
                    elif reasons != {"ctc_length_infeasible"} or raw["training_ready"] is not False:
                        raise ValueError("Noah recovery has non-temporal issues")
                    record = {
                        k: raw[k]
                        for k in (
                            "id",
                            "audio_path",
                            "text",
                            "phoneme_token_ids",
                            "label_length",
                            "ctc_minimum_input_length",
                            "estimated_ctc_input_length",
                            "duration_seconds",
                            "source_tsv",
                            "split_hash",
                        )
                    }
                    record.update(
                        schema_version=1,
                        dataset_version="es-480h-noah-expansion-v1",
                        split="train",
                        language="es",
                        source_language="es-419",
                        source_corpus="noah_es_mobile",
                        source_batch=chosen["source"],
                        source_id=chosen["source_id"],
                        speaker_id=chosen.get("speaker_id"),
                        directory_group_hint=chosen.get("directory_group_hint"),
                        source_split="unsplit",
                        source_row_number=raw["row_number"],
                        split_assignment_source="fixed_noah_expansion_plan_not_speaker_split",
                        ctc_time_upsampling_factor=2,
                        effective_ctc_input_length=2 * raw["estimated_ctc_input_length"],
                        effective_ctc_target_ratio=actual["effective_ctc_ratio"],
                        release_source=release,
                        audio_file_sha256=chosen["audio_file_sha256"],
                    )
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                bound.append(chosen)
    if wanted:
        raise ValueError("chosen IDs not found in input manifests")
    return bound


def freeze_plan(plan: Path, output: Path, *, workers: int = 8) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if (plan / "FAILED.txt").exists():
        raise ValueError("plan is marked failed")
    identities: dict[str, Any] = {}
    report = json.loads(_verified(plan, "report.json", identities).read_text())
    config = json.loads(_verified(plan, "config.json", identities).read_text())
    ids_path = _verified(plan, "proposed_ids.jsonl", identities)
    if report["status"] != "plan_completed" or report["training_ready"] is not False:
        raise ValueError("input is not a completed provisional plan")
    if report["target_hours"] != config["target_hours"] or config["target_hours"] != 480:
        raise ValueError("unexpected target")
    identities.update(report["inputs"])
    for name, identity in identities.items():
        if _sha(Path(name)) != identity["sha256"]:
            raise ValueError(f"input identity changed: {name}")
    pool = Path(config["pool"])
    old = json.loads(_verified(pool, "split_summary.json", identities).read_text())
    if (
        old["status"] != "pass"
        or old["test_set_used"] is not False
        or old["test_set_sealed"] is not True
    ):
        raise ValueError("old pool is not sealed/pass")
    protected = protected_audio(pool, identities)
    for row in _rows(_verified(pool, "full_ctc_validation.jsonl", identities)):
        if "validation" not in protected.get(str(Path(row["audio_path"]).resolve()), set()):
            raise ValueError("validation path missing from protected registry")
    for split in ("validation", "test"):
        if sum(split in roles for roles in protected.values()) < old["split_records"][split]:
            raise ValueError("source-derived protection is smaller than sealed split")
    choices = list(_rows(ids_path))
    if len(choices) != report["combined"]["records"] or not math.isclose(
        sum(r["duration_seconds"] for r in choices) / 3600,
        report["combined"]["hours"],
        abs_tol=1e-8,
    ):
        raise ValueError("plan counts/hours mismatch")
    output.mkdir(parents=True)
    try:
        bound = write_draft(choices, config, output, protected)
        selected = {r["audio_path"]: r for r in bound}
        audit = audit_audio(selected, protected, output, workers=workers)
        for name, identity in identities.items():
            if _sha(Path(name)) != identity["sha256"]:
                raise ValueError(f"input changed during freeze: {name}")
        blocked = bool(audit["conflict_groups_by_reason"])
        final = output / "full_ctc_train.jsonl"
        if not blocked:
            (output / "train.pending.jsonl").rename(final)
            summary = {
                "status": "pass",
                "dataset_version": "es-480h-noah-expansion-v1",
                "manifest_paths": dict(old["manifest_paths"], train=str(final.resolve())),
                "manifest_sha256": dict(old["manifest_sha256"], train=_sha(final)),
                "split_records": dict(old["split_records"], train=len(bound)),
                "split_audio_hours": dict(
                    old["split_audio_hours"], train=report["combined"]["hours"]
                ),
                "test_set_sealed": True,
                "test_set_used": False,
                "test_audio_bytes_read_for_identity_only": True,
                "test_manifest_content_read": False,
                "speaker_disjoint_claimed": False,
                "validation_test_manifests_modified": False,
            }
            # Validation/test remain references to original files; no copy or re-splitting.
            (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        result = {
            "status": "blocked_audio_conflicts" if blocked else "completed",
            "training_manifest_ready": not blocked,
            "output_dir": str(output),
            "train_manifest": None if blocked else str(final),
            "train_manifest_sha256": None if blocked else _sha(final),
            "selection": metrics(bound),
            "audio_identity_audit": audit,
            "inputs": identities,
            "validation_reference": report["validation_reference"],
            "test_reference": report["test_reference"],
            "model_loaded": False,
            "training_started": False,
            "test_manifest_content_read": False,
            "test_evaluation_performed": False,
            "test_audio_bytes_read_for_identity_only": True,
            "limitations": [
                "File SHA detects identical bytes, not re-encoding or overlapping segments.",
                "Noah speaker identity is unknown; no cross-source speaker-disjoint guarantee.",
                "Holdout source registry is a conservative superset, including unreleased rows.",
                "Label correctness and decoded waveform integrity are not newly certified.",
                "Encoder feature caches and training remain separate work.",
            ],
        }
        (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file())
        )
        return {k: v for k, v in result.items() if k not in {"inputs", "selection"}} | {
            "selection": {
                "records": len(bound),
                "hours": result["selection"]["hours"],
                "by_source": result["selection"]["by_dimension"]["source"],
                "by_release": result["selection"]["by_dimension"]["release"],
            },
            "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
        }
    except Exception as error:
        (output / "FAILED.txt").write_text(str(error))
        raise
