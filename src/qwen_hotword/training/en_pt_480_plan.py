"""Select provisional EN/PT IDs from fixed train pools, preserving frozen ES."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.training.balanced_multilingual import (
    EXPECTED_LANGUAGE_TAGS,
    LanguagePool,
    _validate_pool_summary,
)
from qwen_hotword.training.spanish_480_plan import _rows, candidate, metrics, stratified_select
from qwen_hotword.training.spanish_capacity import _sha


def _identity(path: Path) -> dict[str, Any]:
    return {"sha256": _sha(path), "size_bytes": path.stat().st_size}


def _speakers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hours: dict[tuple[str, str], float] = defaultdict(float)
    missing = 0
    for row in rows:
        speaker = row.get("speaker_id")
        if isinstance(speaker, str) and speaker.strip():
            hours[row["source"], speaker.strip()] += row["duration_seconds"] / 3600
        else:
            missing += 1
    return {
        "source_scoped_known_speakers": len(hours),
        "missing_speaker_records": missing,
        "top5_known_speaker_hours": sorted(hours.values(), reverse=True)[:5],
        "cross_source_speaker_identity_verified": False,
    }


def _scan(
    spec: dict[str, Any], language: str, identities: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = Path(spec["pool"]).resolve()
    summary_path, train = pool / "split_summary.json", pool / "full_ctc_train.jsonl"
    identities[str(summary_path)] = _identity(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["manifest_sha256"] != spec["manifest_sha256"]:
        raise ValueError(f"pinned split identities differ: {language}")
    actual = _validate_pool_summary(summary, LanguagePool(language, pool), train)
    identities[str(train)] = {"sha256": actual, "size_bytes": train.stat().st_size}
    candidates, ids, paths = [], set(), set()
    for raw in _rows(train):
        if (
            raw["split"] != "train"
            or raw["language"] not in EXPECTED_LANGUAGE_TAGS[language]
            or raw["ctc_time_upsampling_factor"] != 2
            or raw["source_corpus"] not in spec["sources"]
        ):
            raise ValueError(f"unexpected language/source/split/time factor: {language}")
        for key in ("id", "audio_path", "source_corpus"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise ValueError(f"invalid {key}")
        release = raw["release_source"].replace("temporal_2x_recovered", "temporal_2x_recovery")
        row = candidate(raw, raw["source_corpus"], release)
        if (
            raw["label_length"] != len(raw["phoneme_token_ids"])
            or raw["effective_ctc_input_length"] != 2 * raw["estimated_ctc_input_length"]
            or not math.isclose(
                raw["effective_ctc_target_ratio"], row["effective_ctc_ratio"], abs_tol=1e-9
            )
        ):
            raise ValueError("inconsistent saved label length, frames or CTC ratio")
        if row["id"] in ids or row["audio_path"] in paths:
            raise ValueError(f"duplicate train ID/path: {language}")
        ids.add(row["id"])
        paths.add(row["audio_path"])
        candidates.append(row)
    measured = metrics(candidates)
    if (
        measured["records"] != summary["split_records"]["train"]
        or measured["records"] != spec["records"]
        or not math.isclose(measured["hours"], summary["split_audio_hours"]["train"], abs_tol=1e-6)
        or not math.isclose(measured["hours"], spec["hours"], abs_tol=1e-6)
    ):
        raise ValueError(f"train counts/hours differ: {language}")
    sources = measured["by_dimension"]["source"]
    if set(sources) != set(spec["sources"]):
        raise ValueError(f"missing expected source: {language}")
    for source, value in sources.items():
        recorded = summary["corpus_metrics"][source]["split_train"]
        if value["records"] != recorded["records"] or not math.isclose(
            value["hours"], recorded["hours"], abs_tol=1e-6
        ):
            raise ValueError(f"source metrics differ: {language}/{source}")
    return candidates, summary


def prepare_plan(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Write ID proposals only; no audio, validation/test content, models or labels changed."""
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if set(config["pools"]) != {"en", "pt"}:
        raise ValueError("exactly en and pt pools required")
    target = float(config["target_hours"])
    if not math.isfinite(target) or target <= 0:
        raise ValueError("positive finite target required")
    es = config["frozen_es"]
    es_path = Path(es["train_manifest"]).resolve()
    identities = {str(es_path): _identity(es_path)}
    if identities[str(es_path)]["sha256"] != es["sha256"]:
        raise ValueError("frozen Spanish manifest SHA256 differs")
    output.mkdir(parents=True)
    try:
        languages: dict[str, Any] = {}
        selected_ids: set[str] = set()
        selected_paths: set[str] = set()
        for language in ("en", "pt"):
            print(f"Scanning fixed {language} train pool (metadata only)", flush=True)
            rows, summary = _scan(config["pools"][language], language, identities)
            chosen, quotas = stratified_select(rows, target * 3600, int(config["seed"]))
            for row in chosen:
                if row["id"] in selected_ids or row["audio_path"] in selected_paths:
                    raise ValueError("cross-language selected ID/path overlap")
                selected_ids.add(row["id"])
                selected_paths.add(row["audio_path"])
            path = output / f"proposed_ids_{language}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in chosen:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            languages[language] = {
                "available": metrics(rows),
                "selected": metrics(chosen),
                "available_speakers": _speakers(rows),
                "selected_speakers": _speakers(chosen),
                "stratum_quotas": quotas,
                "proposed_ids_sha256": _sha(path),
                "heldout_references_not_opened": {
                    split: {
                        "sha256": summary["manifest_sha256"][split],
                        "records": summary["split_records"][split],
                        "hours": summary["split_audio_hours"][split],
                    }
                    for split in ("validation", "test")
                },
            }
        # Preserve every ES row; check only selected identities, without re-selection.
        es_count, es_seconds = 0, 0.0
        for raw in _rows(es_path):
            audio = str(Path(raw["audio_path"]).resolve())
            if raw["split"] != "train" or raw["language"] != "es":
                raise ValueError("frozen ES split/language differs")
            if raw["id"] in selected_ids or audio in selected_paths:
                raise ValueError("selected identity overlaps frozen Spanish or duplicate ES")
            selected_ids.add(raw["id"])
            selected_paths.add(audio)
            es_count += 1
            es_seconds += raw["duration_seconds"]
        if es_count != es["records"] or not math.isclose(
            es_seconds / 3600, es["hours"], abs_tol=1e-6
        ):
            raise ValueError("frozen ES counts/hours differ")
        for filename, identity in identities.items():
            if _sha(Path(filename)) != identity["sha256"]:
                raise ValueError("input changed during planning")
        report = {
            "status": "plan_completed",
            "training_ready": False,
            "output_dir": str(output),
            "target_hours_per_selected_language": target,
            "seed": config["seed"],
            "policy": "proportional_hours_by_source_release_duration_density_ratio",
            "frozen_es_reference": es,
            "languages": languages,
            "inputs": identities,
            "definitions": {
                "density": "L/(2*T_est); manifest estimates, not feature cache frames",
                "ctc_ratio": "(L+adjacent_repeats)/(2*T_est)",
                "bin_policy": "upper-inclusive",
                "duration_edges": [3, 6, 10, 20],
                "density_edges": [0.2, 0.3, 0.4, 0.5, 0.6],
                "ratio_edges": [0.5, 0.75, 0.9],
            },
            "limitations": [
                "Proportional composition is a controlled starting point, not an optimum.",
                "Selected IDs/paths are unique; audio byte/content deduplication is pending.",
                "Inherited heldout splits are referenced, not newly audited in this plan.",
                "Speaker metadata may be missing; no cross-source speaker-disjoint claim.",
                "No new certification of transcription, pronunciation or audio quality.",
            ],
            "audio_read": False,
            "validation_manifest_content_read": False,
            "test_manifest_content_read": False,
            "model_loaded": False,
            "training_started": False,
            "final_training_manifest_written": False,
        }
        for filename, data in (("report.json", report), ("config.json", config)):
            (output / filename).write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(output.iterdir())), encoding="utf-8"
        )
        return report
    except Exception as error:
        (output / "FAILED.txt").write_text(str(error), encoding="utf-8")
        raise


def compact_report(report: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in report.items() if key not in {"inputs", "languages"}}
    result["languages"] = {}
    for language, data in report["languages"].items():
        compact = {k: v for k, v in data.items() if k != "stratum_quotas"}
        for name in ("available", "selected"):
            compact[name] = {
                **data[name],
                "by_dimension": {
                    k: v for k, v in data[name]["by_dimension"].items() if k != "stratum"
                },
            }
        result["languages"][language] = compact
    result["return_files"] = [
        str(Path(report["output_dir"]) / name) for name in ("report.json", "sha256.txt")
    ]
    return result
