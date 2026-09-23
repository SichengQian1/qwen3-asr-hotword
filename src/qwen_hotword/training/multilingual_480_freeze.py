"""Materialize a fixed three-language plan after audio identity checks, without training."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.training.balanced_multilingual import (
    EXPECTED_LANGUAGE_TAGS,
    _write_interleaved_manifest,
)
from qwen_hotword.training.combined_training import (
    _is_recommended_temporal_recovery,
    _ready_duration,
)
from qwen_hotword.training.full_training import assign_full_training_split
from qwen_hotword.training.spanish_480_freeze import audit_audio, protected_audio
from qwen_hotword.training.spanish_480_plan import _rows, candidate, metrics
from qwen_hotword.training.spanish_capacity import _sha, _verified

VERSION = "en-es-pt-480h-fixed-plan-v1"


def _track(path: Path, identities: dict[str, Any], expected: str | None = None) -> Path:
    actual = _sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"input SHA256 differs: {path}")
    previous = identities.get(str(path))
    if previous and previous["sha256"] != actual:
        raise ValueError(f"input changed: {path}")
    identities[str(path)] = {"sha256": actual, "size_bytes": path.stat().st_size}
    return path


def _unchanged(identities: dict[str, Any]) -> None:
    for name, identity in identities.items():
        if _sha(Path(name)) != identity["sha256"]:
            raise ValueError(f"input changed during freeze: {name}")


def portuguese_protected(
    pool: Path, summary: dict[str, Any], identities: dict[str, Any]
) -> dict[str, set[str]]:
    """Rebuild holdout identities from source split metadata, not sealed test answers."""
    config = json.loads(_track(pool / "split_config.json", identities).read_text())
    if (
        config["split_strategy"] != "existing_stable_split_hash"
        or config["split_fractions"] != summary["split_fractions"]
        or config["release_policy"]["time_upsampling_factor"] != 2
        or config["release_policy"]["maximum_effective_ratio"] != 0.9
        or set(config["inputs"]) != set(summary["corpus_metrics"])
    ):
        raise ValueError("unsupported Portuguese split/release policy")
    fractions = config["split_fractions"]
    if (
        set(fractions) != {"train", "validation", "test"}
        or any(not 0 < v < 1 for v in fractions.values())
        or not math.isclose(sum(fractions.values()), 1)
    ):
        raise ValueError("invalid split fractions")
    protected: dict[str, set[str]] = defaultdict(set)
    totals = {split: [0, 0.0] for split in fractions}
    ids, paths = set(), set()
    for source, inputs in config["inputs"].items():
        counts = {split: [0, 0.0] for split in fractions}
        for kind in ("ready_manifest", "review_manifest"):
            identity = inputs[kind]
            path = _track(Path(identity["path"]), identities, identity["sha256"])
            for number, raw in enumerate(_rows(path), 1):
                # Only issue/length/duration/split/identity metadata is consulted.
                # Source JSON rows contain text and phones, but neither is used here.
                if kind == "ready_manifest":
                    seconds = _ready_duration(raw, path, number)
                else:
                    if not _is_recommended_temporal_recovery(
                        raw,
                        time_upsampling_factor=2,
                        release_max_effective_ratio=0.9,
                        path=path,
                        line_number=number,
                    ):
                        continue
                    seconds = float(raw["duration_seconds"])
                split_hash = float(raw["split_hash"])
                if not 0 <= split_hash < 1 or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("invalid source identity metadata")
                split = assign_full_training_split(split_hash, fractions)
                audio = str(Path(raw["audio_path"]).resolve())
                if raw["id"] in ids or audio in paths:
                    raise ValueError("duplicate Portuguese released source ID/path")
                ids.add(raw["id"])
                paths.add(audio)
                counts[split][0] += 1
                counts[split][1] += seconds / 3600
                if split != "train":
                    protected[audio].add(split)
        for split, (count, hours) in counts.items():
            expected = summary["corpus_metrics"][source][f"split_{split}"]
            if count != expected["records"] or not math.isclose(
                hours, expected["hours"], abs_tol=1e-6
            ):
                raise ValueError(f"source-derived Portuguese {source}/{split} differs")
            totals[split][0] += count
            totals[split][1] += hours
        print(f"Portuguese source identities reconstructed: {source}", flush=True)
    for split, (count, hours) in totals.items():
        if count != summary["split_records"][split] or not math.isclose(
            hours, summary["split_audio_hours"][split], abs_tol=1e-6
        ):
            raise ValueError("source-derived Portuguese split totals differ")
    return dict(protected)


def _validate_protection(
    pool: Path,
    summary: dict[str, Any],
    protected: dict[str, set[str]],
    identities: dict[str, Any],
) -> None:
    path = _track(
        pool / "full_ctc_validation.jsonl", identities, summary["manifest_sha256"]["validation"]
    )
    count = 0
    for raw in _rows(path):
        if raw["split"] != "validation" or "validation" not in protected.get(
            str(Path(raw["audio_path"]).resolve()), set()
        ):
            raise ValueError("validation identity absent from protection registry")
        count += 1
    if count != summary["split_records"]["validation"]:
        raise ValueError("validation count differs")
    for split in ("validation", "test"):
        if sum(split in roles for roles in protected.values()) < summary["split_records"][split]:
            raise ValueError("protection registry smaller than heldout split")


def _write_language(
    path: Path,
    destination: Path,
    language: str,
    choices: list[dict[str, Any]] | None,
    selected: dict[str, dict[str, Any]],
    seen_ids: set[str],
    protected: dict[str, set[str]],
) -> dict[str, Any]:
    wanted = {row["id"]: row for row in choices} if choices is not None else None
    if wanted is not None and len(wanted) != len(choices or []):
        raise ValueError("duplicate chosen ID")
    count, seconds = 0, 0.0
    with destination.open("x", encoding="utf-8") as handle:
        for raw in _rows(path):
            choice = wanted.pop(raw["id"], None) if wanted is not None else None
            if wanted is not None and choice is None:
                continue
            if raw["split"] != "train" or raw["language"] not in EXPECTED_LANGUAGE_TAGS[language]:
                raise ValueError("selected source split/language mismatch")
            release = raw["release_source"].replace("temporal_2x_recovered", "temporal_2x_recovery")
            actual = candidate(raw, raw["source_corpus"], release)
            if choice is not None and actual != choice:
                raise ValueError("selected ID does not match source metadata")
            audio = actual["audio_path"]
            if raw["id"] in seen_ids or audio in selected or audio in protected:
                raise ValueError("duplicate chosen identity or protected path overlap")
            seen_ids.add(raw["id"])
            selected[audio] = {"audio_file_sha256": raw.get("audio_file_sha256")}
            record = dict(raw)
            record.update(
                experiment="full-ctc-v1",
                dataset_version=VERSION,
                source_dataset_version=raw.get("dataset_version"),
                balanced_language_bucket=language,
                original_release_source=raw["release_source"],
                release_source=release,
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            seconds += raw["duration_seconds"]
    if wanted:
        raise ValueError("selected IDs missing from source")
    return {"records": count, "hours": seconds / 3600}


def freeze_plan(config: dict[str, Any], output: Path, *, workers: int = 8) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if not 1 <= workers <= 32:
        raise ValueError("workers must be within 1..32")
    plan = Path(config["plan_dir"])
    if (plan / "FAILED.txt").exists():
        raise ValueError("plan marked failed")
    identities: dict[str, Any] = {}
    report = json.loads(_verified(plan, "report.json", identities).read_text())
    original = json.loads(_verified(plan, "config.json", identities).read_text())
    if (
        report["status"] != "plan_completed"
        or report["training_ready"] is not False
        or report["target_hours_per_selected_language"] != original["target_hours"]
        or original["target_hours"] != config["target_hours"]
        or report["frozen_es_reference"] != original["frozen_es"]
    ):
        raise ValueError("invalid completed plan/config")
    identities.update(report["inputs"])
    _unchanged(identities)
    choices = {}
    for language in ("en", "pt"):
        path = _verified(plan, f"proposed_ids_{language}.jsonl", identities)
        digest = _sha(path)
        if (
            digest != config["proposed_ids_sha256"][language]
            or digest != report["languages"][language]["proposed_ids_sha256"]
        ):
            raise ValueError("selected ID plan differs from reviewed result")
        choices[language] = list(_rows(path))
        if metrics(choices[language]) != report["languages"][language]["selected"]:
            raise ValueError("selected plan distributions differ")
    pools = {lang: Path(spec["pool"]) for lang, spec in original["pools"].items()}
    pools["es"] = Path(config["es_holdout_pool"])
    summaries, protection_counts = {}, {}
    protected: dict[str, set[str]] = defaultdict(set)
    for language, pool in pools.items():
        print(f"Building {language} holdout identity protection (no model)", flush=True)
        path = pool / "split_summary.json"
        if language == "es":
            _verified(pool, path.name, identities)
        else:
            _track(path, identities)
        summary = json.loads(path.read_text())
        if (
            summary["status"] != "pass"
            or summary["test_set_sealed"] is not True
            or summary["test_set_used"] is not False
        ):
            raise ValueError("source pool not sealed/pass")
        if (
            language != "es"
            and summary["manifest_sha256"] != original["pools"][language]["manifest_sha256"]
        ):
            raise ValueError("pool identity differs from reviewed plan")
        registry = (
            portuguese_protected(pool, summary, identities)
            if language == "pt"
            else (protected_audio(pool, identities))
        )
        _validate_protection(pool, summary, registry, identities)
        for audio, roles in registry.items():
            protected[audio].update(roles)
        protection_counts[language] = {
            split: sum(split in roles for roles in registry.values())
            for split in ("validation", "test")
        }
        summaries[language] = summary
    es = original["frozen_es"]
    es_path = _track(Path(es["train_manifest"]), identities, es["sha256"])
    es_summary = json.loads(_verified(es_path.parent, "split_summary.json", identities).read_text())
    for split in ("validation", "test"):
        if es_summary["manifest_sha256"][split] != summaries["es"]["manifest_sha256"][split]:
            raise ValueError("Spanish frozen/old heldout identities differ")
    output.mkdir(parents=True)
    try:
        selected: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        selection = {}
        pending = {lang: output / f"{lang}.pending.jsonl" for lang in ("en", "es", "pt")}
        for lang in pending:
            path = es_path if lang == "es" else pools[lang] / "full_ctc_train.jsonl"
            _track(
                path,
                identities,
                es["sha256"]
                if lang == "es"
                else original["pools"][lang]["manifest_sha256"]["train"],
            )
            selection[lang] = _write_language(
                path, pending[lang], lang, choices.get(lang), selected, seen, dict(protected)
            )
            expected = es if lang == "es" else report["languages"][lang]["selected"]
            if selection[lang]["records"] != expected["records"] or not math.isclose(
                selection[lang]["hours"], expected["hours"], abs_tol=1e-8
            ):
                raise ValueError("materialized language selection counts/hours differ")
        audit = audit_audio(selected, dict(protected), output, workers=workers)
        _unchanged(identities)
        blocked = bool(audit["conflict_groups_by_reason"])
        artifacts: dict[str, Any] = {}
        if not blocked:
            combined = output / "combined.pending.jsonl"
            combined_metrics = _write_interleaved_manifest(
                combined, pending, dataset_version=VERSION
            )
            if combined_metrics["records"] != len(selected):
                raise ValueError("combined record count differs")
            for lang, path in pending.items():
                final = output / f"full_ctc_train_{lang}.jsonl"
                path.rename(final)
                artifacts[lang] = {"path": str(final), "sha256": _sha(final)}
            final = output / "full_ctc_train.jsonl"
            combined.rename(final)
            artifacts["combined"] = {"path": str(final), "sha256": _sha(final), **combined_metrics}
        result = {
            "status": "blocked_audio_conflicts" if blocked else "completed",
            "training_manifest_ready": not blocked,
            "output_dir": str(output),
            "selection": selection,
            "artifacts": artifacts,
            "frozen_es_reference": es,
            "plan_dir": str(plan),
            "audio_identity_audit": audit,
            "protection_counts_by_language": protection_counts,
            "heldout_references": {
                lang: {
                    split: {
                        "path": summary["manifest_paths"][split],
                        "sha256": summary["manifest_sha256"][split],
                        "records": summary["split_records"][split],
                        "hours": summary["split_audio_hours"][split],
                    }
                    for split in ("validation", "test")
                }
                for lang, summary in summaries.items()
            },
            "model_loaded": False,
            "training_started": False,
            "test_manifest_content_read": False,
            "test_evaluation_performed": False,
            "test_audio_bytes_read_for_identity_only": True,
            "source_rows_parsed_for_split_identity_only": True,
            "limitations": [
                "File SHA detects identical bytes, not re-encoding or overlapping segments.",
                "PT manifest speaker IDs are absent; no cross-source speaker-disjoint claim.",
                "EN/ES protection may include unreleased rows; conflicts block without resampling.",
                "Source metadata is parsed to reconstruct heldout identities, not evaluate models.",
                "Label accuracy and waveform decoding are not newly certified.",
                "Encoder caches and training remain separate work.",
            ],
            "return_files": [str(output / name) for name in ("report.json", "sha256.txt")],
        }
        for name, value in (
            ("report.json", result),
            ("input_config.json", config),
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
