"""Apply reviewed train-only exclusions, refill the same strata, and re-audit before release."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.training.audio_conflict_audit import inspect_conflicts
from qwen_hotword.training.balanced_multilingual import _write_interleaved_manifest
from qwen_hotword.training.en_pt_480_plan import _scan
from qwen_hotword.training.multilingual_480_freeze import _track, _unchanged
from qwen_hotword.training.spanish_480_freeze import _fingerprint, audit_audio
from qwen_hotword.training.spanish_480_plan import _rows, candidate
from qwen_hotword.training.spanish_capacity import _sha, _verified

VERSION = "en-es-pt-480h-content-dedup-v2"


def _add(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["records"] += 1
    stats["hours"] += row["duration_seconds"] / 3600
    bins = json.loads(row["stratum"])
    for dimension, key in (
        ("source", row["source"]),
        ("release", row["release_source"]),
        ("duration_bucket", str(bins[2])),
        ("density_bucket", str(bins[3])),
        ("ctc_ratio_bucket", str(bins[4])),
    ):
        value = (
            stats["by_dimension"]
            .setdefault(dimension, {})
            .setdefault(key, {"records": 0, "hours": 0.0})
        )
        value["records"] += 1
        value["hours"] += row["duration_seconds"] / 3600


def choose_replacements(
    candidates: list[dict[str, Any]],
    removed_seconds: dict[str, float],
    needed_seconds: float,
    forbidden_hashes: set[str],
    forbidden_paths: set[str],
    excluded_ids: set[str],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replace only removed strata; inspect real file bytes before accepting each new clip."""
    if needed_seconds <= 1e-8:
        return [], {"records": 0, "hours": 0.0, "audio_files_checked": 0}
    if not removed_seconds or sum(removed_seconds.values()) < needed_seconds - 1e-6:
        raise ValueError("refill exceeds removed hours")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = row["stratum"]
        if (
            key in removed_seconds
            and row["id"] not in excluded_ids
            and row["audio_path"] not in forbidden_paths
        ):
            groups[key].append(row)
    quotas = {
        key: needed_seconds * value / sum(removed_seconds.values())
        for key, value in removed_seconds.items()
    }
    for rows in groups.values():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}\0{r['id']}".encode()).hexdigest())
    used = dict.fromkeys(quotas, 0.0)
    indices = dict.fromkeys(quotas, 0)
    chosen: list[dict[str, Any]] = []
    seconds, checked = 0.0, 0
    rejected: Counter[str] = Counter()
    while seconds < needed_seconds - 1e-8:
        available = [key for key in quotas if indices[key] < len(groups[key])]
        if not available:
            raise ValueError("insufficient content-unique replacement capacity in removed strata")
        key = min(available, key=lambda k: (-(quotas[k] - used[k]), k))
        row = groups[key][indices[key]]
        indices[key] += 1
        _, digest, size = _fingerprint(row["audio_path"])
        checked += 1
        if digest in forbidden_hashes:
            rejected["known_or_new_duplicate_file_hash"] += 1
            continue
        forbidden_hashes.add(digest)
        forbidden_paths.add(row["audio_path"])
        excluded_ids.add(row["id"])
        chosen.append(dict(row, audio_file_sha256=digest, audio_size_bytes=size))
        used[key] += row["duration_seconds"]
        seconds += row["duration_seconds"]
    return chosen, {
        "records": len(chosen),
        "hours": seconds / 3600,
        "audio_files_checked": checked,
        "rejected": dict(rejected),
        "strata": {
            key: {"quota_hours": quotas[key] / 3600, "added_hours": used[key] / 3600}
            for key in sorted(quotas)
        },
    }


def repair_training(config: dict[str, Any], output: Path, *, workers: int = 8) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be within 1..32")
    root = Path(config["freeze_dir"])
    identities: dict[str, Any] = {}
    for name, digest in config["expected_sha256"].items():
        _track(root / name, identities, digest)
    print("Reconstructing the reviewed exclusions from saved fingerprints", flush=True)
    diagnosis, removals = inspect_conflicts(root, target_hours=config["target_hours"])
    if diagnosis["proposed_removed_ids_sha256"] != config["removed_ids_sha256"]:
        raise ValueError("exclusion set differs from reviewed diagnosis")
    if diagnosis["proposed_action_groups"].get("holdout_internal_conflict_requires_review"):
        raise ValueError("holdout-internal conflicts require a separate decision")
    if any(row["language"] == "es" for row in removals):
        raise ValueError("this repair must preserve the complete Spanish selection")
    identities.update(diagnosis["inputs"])
    blocked = json.loads((root / "report.json").read_text())
    plan = Path(blocked["plan_dir"])
    original = json.loads(_verified(plan, "config.json", identities).read_text())
    if original["target_hours"] != config["target_hours"]:
        raise ValueError("target differs from original plan")
    for lang in ("en", "pt"):
        if original["pools"][lang]["manifest_sha256"]["train"] != config["train_pool_sha256"][lang]:
            raise ValueError("replacement train pool differs from reviewed inventory")
    old_hashes: dict[str, str] = {}
    forbidden_hashes, forbidden_paths = set(), set()
    protected: dict[str, set[str]] = {}
    for row in _rows(root / "audio_fingerprints.jsonl"):
        path, digest = row["audio_path"], row["sha256"]
        forbidden_hashes.add(digest)
        forbidden_paths.add(path)
        if "train" in row["roles"]:
            old_hashes[path] = digest
        roles = set(row["roles"]) - {"train"}
        if roles:
            protected[path] = roles
    remove_ids = {row["id"] for row in removals}
    all_old_ids: set[str] = set()
    removed_strata: dict[str, dict[str, float]] = {
        lang: defaultdict(float) for lang in ("en", "pt")
    }
    removed_found: set[str] = set()
    pending = {lang: output / f"{lang}.pending.jsonl" for lang in ("en", "es", "pt")}
    selected: dict[str, dict[str, Any]] = {}
    stats: dict[str, Any] = {
        lang: {"records": 0, "hours": 0.0, "by_dimension": {}} for lang in pending
    }
    output.mkdir(parents=True)
    try:
        with (output / "excluded_rows.jsonl").open("x") as handle:
            for row in removals:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for lang, path in pending.items():
            print(f"Keeping fixed {lang} rows except reviewed exclusions", flush=True)
            with path.open("x", encoding="utf-8") as handle:
                for raw in _rows(root / f"{lang}.pending.jsonl"):
                    if raw["id"] in all_old_ids:
                        raise ValueError("duplicate original selected ID")
                    all_old_ids.add(raw["id"])
                    release = raw["release_source"].replace(
                        "temporal_2x_recovered", "temporal_2x_recovery"
                    )
                    item = candidate(raw, raw["source_corpus"], release)
                    if raw["id"] in remove_ids:
                        removed_found.add(raw["id"])
                        removed_strata[lang][item["stratum"]] += item["duration_seconds"]
                        continue
                    audio = item["audio_path"]
                    if audio in selected or audio in protected or audio not in old_hashes:
                        raise ValueError("retained identity conflict or missing saved fingerprint")
                    selected[audio] = {"audio_file_sha256": old_hashes[audio]}
                    raw["dataset_version"] = VERSION
                    handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    _add(stats[lang], item)
            expected = diagnosis["projected_remaining"][lang]
            if stats[lang]["records"] != expected["records"] or not math.isclose(
                stats[lang]["hours"], expected["hours"], abs_tol=1e-6
            ):
                raise ValueError("retained counts/hours differ from diagnosis")
        if removed_found != remove_ids:
            raise ValueError("reviewed exclusions were not applied exactly")
        replacement_summary = {}
        with (output / "replacement_ids.jsonl").open("x", encoding="utf-8") as replacement_handle:
            for lang in ("en", "pt"):
                print(f"Selecting content-unique {lang} replacements in removed strata", flush=True)
                candidates, _ = _scan(original["pools"][lang], lang, identities)
                replacements, replacement_summary[lang] = choose_replacements(
                    candidates,
                    removed_strata[lang],
                    (config["target_hours"] - stats[lang]["hours"]) * 3600,
                    forbidden_hashes,
                    forbidden_paths,
                    all_old_ids,
                    seed=config["seed"],
                )
                wanted = {row["id"]: row for row in replacements}
                train = Path(original["pools"][lang]["pool"]) / "full_ctc_train.jsonl"
                with pending[lang].open("a", encoding="utf-8") as handle:
                    for raw in _rows(train):
                        chosen = wanted.pop(raw["id"], None)
                        if chosen is None:
                            continue
                        item = candidate(raw, raw["source_corpus"], chosen["release_source"])
                        if any(chosen[k] != v for k, v in item.items()):
                            raise ValueError("replacement metadata changed")
                        selected[item["audio_path"]] = {
                            "audio_file_sha256": chosen["audio_file_sha256"]
                        }
                        raw.update(
                            experiment="full-ctc-v1",
                            source_dataset_version=raw.get("dataset_version"),
                            dataset_version=VERSION,
                            balanced_language_bucket=lang,
                            original_release_source=raw["release_source"],
                            release_source=chosen["release_source"],
                        )
                        handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
                        replacement_handle.write(
                            json.dumps(dict(chosen, language=lang), ensure_ascii=False) + "\n"
                        )
                        _add(stats[lang], item)
                if wanted or stats[lang]["hours"] < config["target_hours"] - 1e-8:
                    raise ValueError("replacement rows missing or target not reached")
        print(
            "Final full audio identity verification of repaired train and protected holdouts",
            flush=True,
        )
        audit = audit_audio(selected, protected, output, workers=workers)
        (output / "audio_identity_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
        )
        _unchanged(identities)
        if audit["conflict_groups_by_reason"] or audit["unique_train_file_hashes"] != len(selected):
            raise ValueError("repaired audio audit still has conflicts; no final manifest released")
        combined = output / "combined.pending.jsonl"
        combined_stats = _write_interleaved_manifest(combined, pending, dataset_version=VERSION)
        if combined_stats["records"] != len(selected):
            raise ValueError("combined count differs")
        artifacts: dict[str, Any] = {}
        for lang, path in pending.items():
            final = output / f"full_ctc_train_{lang}.jsonl"
            path.rename(final)
            artifacts[lang] = {"path": str(final), "sha256": _sha(final)}
        final = output / "full_ctc_train.jsonl"
        combined.rename(final)
        artifacts["combined"] = {"path": str(final), "sha256": _sha(final), **combined_stats}
        result = {
            "status": "completed",
            "training_manifest_ready": True,
            "output_dir": str(output),
            "source_freeze_dir": str(root),
            "excluded": diagnosis["proposed_removals"],
            "excluded_ids_sha256": diagnosis["proposed_removed_ids_sha256"],
            "replacements": replacement_summary,
            "selection": stats,
            "artifacts": artifacts,
            "audio_identity_audit": audit,
            "retained_audio_rehashed": True,
            "heldout_references": blocked["heldout_references"],
            "frozen_es_reference": blocked["frozen_es_reference"],
            "test_audio_bytes_read_for_identity_only": True,
            "test_manifest_content_read": False,
            "model_loaded": False,
            "training_started": False,
            "limitations": [
                "Exclusions apply only to the new training version; original outputs stay intact.",
                "Label disagreements were quarantined, not manually adjudicated or corrected.",
                "File SHA does not identify re-encoded copies or overlapping recordings.",
                "Missing speaker IDs still prevent cross-source speaker-disjoint guarantees.",
                "No new pronunciation or waveform decoding quality certification.",
            ],
            "return_files": [str(output / name) for name in ("report.json", "sha256.txt")],
        }
        for name, value in (
            ("report.json", result),
            ("config.json", config),
            ("input_identities.json", identities),
        ):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(output.iterdir()))
        )
        return result
    except Exception as error:
        (output / "FAILED.txt").write_text(str(error))
        raise
