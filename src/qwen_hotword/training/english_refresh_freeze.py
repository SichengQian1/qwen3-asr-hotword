"""Release all new English labels, deduplicate, select 480h and retain fixed ES/PT."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.training.balanced_multilingual import _write_interleaved_manifest
from qwen_hotword.training.g2p_prep import normalize_training_text
from qwen_hotword.training.spanish_480_freeze import _fingerprint, audit_audio
from qwen_hotword.training.spanish_480_plan import _rows, candidate, metrics, stratified_select
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.spanish_combined import (
    SpanishCorpusInput,
    _iter_manifest_rows,
    _release_record,
)

VERSION = "en-countries-refresh-es-pt-fixed-480h-v1"


def old_assets(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    baseline_path = root / config["baseline_training_config"]
    baseline = json.loads(baseline_path.read_text())
    freeze = (root / config["fixed_freeze"]).resolve()
    if (root / baseline["freeze_dir"]).resolve() != freeze:
        raise ValueError("fixed ES/PT freeze differs from historical training configuration")
    checked: dict[str, Any] = {}
    report = json.loads(_verified(freeze, "report.json", checked).read_text())
    if (
        report.get("status") != "completed"
        or not report.get("training_manifest_ready")
        or report.get("audio_identity_audit", {}).get("conflict_groups_by_reason")
        or report["artifacts"]["combined"]["sha256"] != baseline["train_sha256"]
    ):
        raise ValueError("completed original 480h freeze required")
    paths = {}
    for lang in ("es", "pt"):
        path = _verified(freeze, f"full_ctc_train_{lang}.jsonl", checked)
        if _sha(path) != config["fixed_languages"][lang]["sha256"]:
            raise ValueError(f"original {lang} manifest identity differs")
        paths[lang] = path
    fingerprint = _verified(freeze, "audio_fingerprints.jsonl", checked)
    validation = (root / baseline["validation_manifest"]).resolve()
    if _sha(validation) != baseline["validation_sha256"]:
        raise ValueError("fixed historical validation identity differs")
    checked[str(validation)] = {"sha256": _sha(validation)}
    checked[str(baseline_path.resolve())] = {"sha256": _sha(baseline_path)}
    return {
        "baseline": baseline,
        "paths": paths,
        "fingerprints": fingerprint,
        "checked": checked,
        "validation": validation,
    }


def protected_registry(old: dict[str, Any], source: Path, workers: int) -> tuple[Any, Any, Any]:
    fixed = {
        str(Path(row["audio_path"]).resolve()) for p in old["paths"].values() for row in _rows(p)
    }
    protected: dict[str, set[str]] = defaultdict(set)
    hashes, expected_fixed = set(), {}
    for row in _rows(old["fingerprints"]):
        path = str(Path(row["audio_path"]).resolve())
        roles = set(row["roles"]) & {"validation", "test"}
        if roles:
            protected[path].update(roles)
            hashes.add(row["sha256"])
        if path in fixed:
            hashes.add(row["sha256"])
            expected_fixed[path] = row["sha256"]
    if set(expected_fixed) != fixed:
        raise ValueError("fixed ES/PT train is absent from original audited fingerprints")
    for row in _rows(old["validation"]):
        if "validation" not in protected.get(str(Path(row["audio_path"]).resolve()), set()):
            raise ValueError("fixed validation missing from original protection registry")
    tests = sorted({r["audio_path"] for r in _rows(source / "test_audio_paths.jsonl")})
    with ThreadPoolExecutor(max_workers=workers) as executor:
        test_paths = iter(tests)
        checked = 0
        while batch := list(islice(test_paths, 256)):
            for path, sha, _ in executor.map(_fingerprint, batch):
                protected[path].add("test")
                hashes.add(sha)
            checked += len(batch)
            if checked % 5120 == 0:
                print(f"New test paths protected by audio bytes: {checked}", flush=True)
    return dict(protected), hashes, expected_fixed


def release_pool(output: Path, source: Path, manifest: Path) -> int:
    ranges = json.loads((source / "report.json").read_text())["countries"]
    count = 0
    with (output / "released_english_pool.jsonl").open("x", encoding="utf-8") as handle:
        for raw, release, path, number in _iter_manifest_rows(
            SpanishCorpusInput("new_english", manifest, source / "source.tsv"),
            time_upsampling_factor=2,
            release_max_effective_ratio=0.90,
            include_not_released=False,
        ):
            country = next(
                (
                    k
                    for k, v in ranges.items()
                    if v["first_tsv_row"] <= raw["row_number"] <= v["last_tsv_row"]
                ),
                None,
            )
            if country is None:
                raise ValueError("full manifest row is outside source country provenance ranges")
            record = _release_record(
                raw,
                corpus_name=f"new_english_{country}",
                release_source=release,
                split="train",
                speaker_id="",
                source_split="train",
                time_upsampling_factor=2,
                dataset_version=VERSION,
                path=path,
                line_number=number,
            )
            record.update(
                country=country,
                speaker_id=None,
                experiment="full-ctc-v1",
                balanced_language_bucket="en",
                release_source=release.replace("recovered", "recovery"),
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def unique_candidates(
    pool: Path,
    output: Path,
    forbidden_hashes: set[str],
    forbidden_paths: set[str],
    workers: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Byte-identical conflicting labels are quarantined as a whole group."""
    groups: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    seen_paths: dict[str, str] = {}
    rows = iter(_rows(pool))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        checked = 0
        while batch := list(islice(rows, 256)):
            prints = list(executor.map(_fingerprint, [r["audio_path"] for r in batch]))
            for row, (path, sha, _) in zip(batch, prints, strict=True):
                if path in seen_paths and seen_paths[path] != sha:
                    raise ValueError("same source audio changed during fingerprint scan")
                seen_paths[path] = sha
                signature = hashlib.sha256(
                    json.dumps(
                        [
                            normalize_training_text(row["text"]),
                            row["phoneme_token_ids"],
                            row["duration_seconds"],
                            row["estimated_ctc_input_length"],
                        ],
                        ensure_ascii=False,
                    ).encode()
                ).hexdigest()
                c = candidate(row, row["source_corpus"], row["release_source"])
                c.update(audio_file_sha256=sha, country=row["country"])
                groups[sha].append((c, signature))
            checked += len(batch)
            if checked % 5120 == 0:
                print(f"English eligible audio fingerprinted: {checked}", flush=True)
    eligible = []
    with (output / "excluded_english.jsonl").open("x", encoding="utf-8") as handle:
        for sha, group in sorted(groups.items()):
            reason = None
            if sha in forbidden_hashes or any(c["audio_path"] in forbidden_paths for c, _ in group):
                reason = "fixed_es_pt_or_heldout_audio_overlap"
            elif len({signature for _, signature in group}) > 1:
                reason = "same_file_bytes_conflicting_labels_or_metadata"
            ordered = sorted(
                group,
                key=lambda item: hashlib.sha256(f"{seed}\0{item[0]['id']}".encode()).hexdigest(),
            )
            for n, (c, _) in enumerate(ordered):
                why = reason or ("duplicate_consistent_file_bytes" if n else None)
                if why:
                    reasons[why] += 1
                    handle.write(json.dumps({**c, "reason": why}, ensure_ascii=False) + "\n")
                else:
                    eligible.append(c)
    return eligible, dict(reasons)


def freeze(
    root: Path,
    config: dict[str, Any],
    work: Path,
    workers: int,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("freeze requires a new stage directory; preserve failed outputs")
    old = old_assets(root, config)
    protected, forbidden_hashes, expected_fixed = protected_registry(old, work / "source", workers)
    output.mkdir(parents=True)
    released_count = release_pool(output, work / "source", work / "manifest")
    eligible, excluded = unique_candidates(
        output / "released_english_pool.jsonl",
        output,
        forbidden_hashes,
        set(expected_fixed) | set(protected),
        workers,
        config["selection_seed"],
    )
    available = metrics(eligible)
    (output / "capacity_report.json").write_text(
        json.dumps(
            {
                "released_records": released_count,
                "available": available,
                "excluded": excluded,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "available_unique_english_hours": available["hours"],
                "eligible_records": available["records"],
                "excluded": excluded,
            }
        ),
        flush=True,
    )
    chosen, quotas = stratified_select(
        eligible, config["target_hours"] * 3600, config["selection_seed"]
    )
    wanted = {r["id"]: r for r in chosen}
    selected = {p: {"audio_file_sha256": sha} for p, sha in expected_fixed.items()}
    paths = {lang: output / f"full_ctc_train_{lang}.jsonl" for lang in ("en", "es", "pt")}
    with paths["en"].open("x", encoding="utf-8") as handle:
        for row in _rows(output / "released_english_pool.jsonl"):
            choice = wanted.pop(row["id"], None)
            if choice is None:
                continue
            row["audio_file_sha256"] = choice["audio_file_sha256"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            selected[row["audio_path"]] = {"audio_file_sha256": choice["audio_file_sha256"]}
    if wanted:
        raise ValueError("chosen IDs missing from full released pool")
    selection = {"en": metrics(chosen)}
    for lang in ("es", "pt"):
        shutil.copyfile(old["paths"][lang], paths[lang])
        count, seconds = 0, 0.0
        for row in _rows(paths[lang]):
            count += 1
            seconds += row["duration_seconds"]
        if count != config["fixed_languages"][lang]["records"]:
            raise ValueError("fixed language record count differs")
        selection[lang] = {"records": count, "hours": seconds / 3600}
        if selection[lang]["hours"] < config["target_hours"]:
            raise ValueError("fixed language does not reach requested hours")
        if _sha(paths[lang]) != config["fixed_languages"][lang]["sha256"]:
            raise ValueError("fixed language bytes changed while copying")
    audit = audit_audio(selected, protected, output, workers=workers)
    if audit["conflict_groups_by_reason"]:
        (output / "audio_identity_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        raise ValueError(
            "final audio identity audit has conflicts; no training configuration published"
        )
    # Individual ES/PT manifests remain byte-identical. Combined rows get only new version metadata.
    staged = {}
    for lang, path in paths.items():
        staged[lang] = output / f"combined_input_{lang}.jsonl"
        with staged[lang].open("x", encoding="utf-8") as handle:
            for row in _rows(path):
                row["dataset_version"] = VERSION
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    combined = output / "full_ctc_train.jsonl"
    combined_metrics = _write_interleaved_manifest(combined, staged, dataset_version=VERSION)
    for name, identity in old["checked"].items():
        if _sha(Path(name)) != identity["sha256"]:
            raise ValueError("fixed asset changed during freeze")
    report = {
        "status": "completed",
        "output_dir": str(output),
        "training_manifest_ready": True,
        "selection": selection,
        "available_english": available,
        "excluded_english": excluded,
        "stratum_quotas": quotas,
        "audio_identity_audit": audit,
        "artifacts": {
            **{lang: {"path": str(p), "sha256": _sha(p)} for lang, p in paths.items()},
            "combined": {"path": str(combined), "sha256": _sha(combined), **combined_metrics},
        },
        "fixed_languages_byte_identical": True,
        "fixed_validation_unchanged": True,
        "speaker_identity_available": False,
        "speaker_disjoint_claimed": False,
        "test_labels_used": False,
        "test_evaluation_performed": False,
        "model_loaded": False,
        "training_started": False,
        "limitations": [
            "Byte SHA does not detect re-encoding or overlapping segments.",
            "English US G2P plus existing IPA normalization is not pronunciation certification.",
            "Country directories are corpus provenance, not verified speaker accents.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    seal(output)
    return report
