"""Plan an expansion without claiming holdout/content isolation is already complete."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_capacity import _sha, _verified


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    return sum(value > edge for edge in edges)


def candidate(row: dict[str, Any], source: str, release: str) -> dict[str, Any]:
    seconds = float(row["duration_seconds"])
    frames = row["estimated_ctc_input_length"]
    tokens = row["phoneme_token_ids"]
    if not math.isfinite(seconds) or seconds <= 0 or type(frames) is not int or frames <= 0:
        raise ValueError("invalid duration or estimated frames")
    if not tokens or any(type(t) is not int or not 0 < t < 90 for t in tokens):
        raise ValueError("invalid phoneme target")
    minimum = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:], strict=False))
    if row["ctc_minimum_input_length"] != minimum:
        raise ValueError("CTC minimum differs from actual token sequence")
    if release == "original_ready":
        if minimum > frames:
            raise ValueError("original-ready is infeasible at 1x")
    elif release == "temporal_2x_recovery":
        if not frames < minimum <= 1.8 * frames:
            raise ValueError("recovery violates existing 2x/0.90 rule")
    else:
        raise ValueError("unknown release source")
    density, ratio = len(tokens) / (2 * frames), minimum / (2 * frames)
    stratum = json.dumps(
        [
            source,
            release,
            _bucket(seconds, (3, 6, 10, 20)),
            _bucket(density, (0.2, 0.3, 0.4, 0.5, 0.6)),
            _bucket(ratio, (0.5, 0.75, 0.9)),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "id": row["id"],
        "audio_path": str(Path(row["audio_path"]).resolve()),
        "duration_seconds": seconds,
        "source": source,
        "release_source": release,
        "speaker_id": row.get("speaker_id"),
        "stratum": stratum,
        "reference_tokens_per_effective_frame": density,
        "effective_ctc_ratio": ratio,
    }


def stratified_select(
    rows: list[dict[str, Any]],
    target_seconds: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Proportional hours, SHA-shuffled within bins; overshoot is less than one clip."""
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("additional target must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError("duplicate candidate ID")
        seen.add(row["id"])
        groups[row["stratum"]].append(row)
    total = sum(r["duration_seconds"] for r in rows)
    if total < target_seconds:
        raise ValueError("insufficient eligible capacity")
    selected: list[dict[str, Any]] = []
    quotas, used, indices = {}, {}, {}
    for key in sorted(groups):
        group = groups[key]
        group.sort(key=lambda r: hashlib.sha256(f"{seed}\0{r['id']}".encode()).hexdigest())
        quotas[key] = target_seconds * sum(r["duration_seconds"] for r in group) / total
        used[key], indices[key] = 0.0, 0
        for row in group:
            if used[key] + row["duration_seconds"] > quotas[key]:
                break
            selected.append(row)
            used[key] += row["duration_seconds"]
            indices[key] += 1
    selected_seconds = sum(r["duration_seconds"] for r in selected)
    while selected_seconds < target_seconds - 1e-8:
        available = [k for k in groups if indices[k] < len(groups[k])]
        key = min(available, key=lambda k: (-(quotas[k] - used[k]), k))
        row = groups[key][indices[key]]
        indices[key] += 1
        selected.append(row)
        used[key] += row["duration_seconds"]
        selected_seconds += row["duration_seconds"]
    return sorted(selected, key=lambda r: r["id"]), {
        key: {"quota_hours": quotas[key] / 3600, "selected_hours": used[key] / 3600}
        for key in sorted(groups)
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        for dimension, key in (
            ("source", row["source"]),
            ("release", row["release_source"]),
            ("stratum", row["stratum"]),
            ("duration_bucket", str(_bucket(row["duration_seconds"], (3, 6, 10, 20)))),
            (
                "density_bucket",
                str(
                    _bucket(row["reference_tokens_per_effective_frame"], (0.2, 0.3, 0.4, 0.5, 0.6))
                ),
            ),
            ("ctc_ratio_bucket", str(_bucket(row["effective_ctc_ratio"], (0.5, 0.75, 0.9)))),
        ):
            g = groups[dimension].setdefault(key, {"records": 0, "hours": 0.0})
            g["records"] += 1
            g["hours"] += row["duration_seconds"] / 3600
    return {
        "records": len(rows),
        "hours": sum(r["duration_seconds"] for r in rows) / 3600,
        "by_dimension": dict(groups),
    }


def prepare_plan(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    identities: dict[str, Any] = {}
    pool, staging, manifest = (Path(config[k]) for k in ("pool", "staging", "manifest"))
    paths = {
        "old_summary": _verified(pool, "split_summary.json", identities),
        "old_train": _verified(pool, "full_ctc_train.jsonl", identities),
        "old_validation": _verified(pool, "full_ctc_validation.jsonl", identities),
        "source_report": _verified(staging, "report.json", identities),
        "candidates": _verified(staging, "candidates.jsonl", identities),
        "source_tsv": _verified(staging, "source.tsv", identities),
        "summary": manifest / "summary.json",
        "build_config": manifest / "build_config.json",
        "ready": manifest / "train_ready.jsonl",
        "review": manifest / "needs_review.jsonl",
    }
    for name, path in paths.items():
        actual = _sha(path)
        identities[str(path)] = {"sha256": actual, "size_bytes": path.stat().st_size}
        if name in config["expected_sha256"] and actual != config["expected_sha256"][name]:
            raise ValueError(f"pinned identity mismatch: {name}")
    old = json.loads(paths["old_summary"].read_text())
    summary = json.loads(paths["summary"].read_text())
    source_report = json.loads(paths["source_report"].read_text())
    build = json.loads(paths["build_config"].read_text())
    if (
        old["status"] != "pass"
        or old["test_set_used"] is not False
        or summary["status"] != "pass"
        or summary["split"] != "unsplit"
        or source_report["status"] != "completed"
    ):
        raise ValueError("input stage not complete or inappropriate split")
    if source_report["source_json_sha256"] != config["source_json_sha256"]:
        raise ValueError("source JSON identity mismatch")
    if Path(build["tsv"]["path"]).resolve() != paths["source_tsv"].resolve():
        raise ValueError("manifest is not linked to this staging TSV")
    for key in ("dictionary", "vocab"):
        path = Path(build[key]["path"])
        if _sha(path) != config["expected_sha256"][key]:
            raise ValueError(f"label dependency mismatch: {key}")
        identities[str(path)] = {"sha256": _sha(path), "size_bytes": path.stat().st_size}

    old_train, known_paths, known_ids = [], set(), set()
    for split in ("train", "validation"):
        path = paths[f"old_{split}"]
        if _sha(path) != old["manifest_sha256"][split]:
            raise ValueError("old manifest/summary identity mismatch")
        count, seconds = 0, 0.0
        for row in _rows(path):
            if row["split"] != split or row["language"] != "es":
                raise ValueError("old split/language mismatch")
            audio = str(Path(row["audio_path"]).resolve())
            if audio in known_paths or row["id"] in known_ids:
                raise ValueError("duplicate/overlapping old audio or ID")
            known_paths.add(audio)
            known_ids.add(row["id"])
            count += 1
            seconds += row["duration_seconds"]
            if split == "train":
                release = row["release_source"].replace(
                    "temporal_2x_recovered", "temporal_2x_recovery"
                )
                old_train.append(candidate(row, row["source_corpus"], release))
        if count != old["split_records"][split] or not math.isclose(
            seconds / 3600,
            old["split_audio_hours"][split],
            abs_tol=1e-6,
        ):
            raise ValueError("old counts or hours mismatch")

    sidecar, hashes, original_ids = {}, set(), set()
    for row in _rows(paths["candidates"]):
        audio = str(Path(row["audio_path"]).resolve())
        digest = row["audio_file_sha256"]
        if audio in sidecar or digest in hashes or row["id"] in original_ids:
            raise ValueError("duplicate staging path, file hash or source ID")
        if row["status"] != "audio_text_candidate" or row["split"] != "unassigned":
            raise ValueError("invalid staging candidate")
        hashes.add(digest)
        original_ids.add(row["id"])
        sidecar[audio] = {
            k: row[k]
            for k in (
                "id",
                "text",
                "duration_seconds",
                "source_batch",
                "audio_file_sha256",
                "directory_group_hint",
                "speaker_id",
            )
        }
    eligible, seen_paths, seen_ids = [], set(), set()
    counts = {"ready": 0, "review": 0}
    hours = {"ready": 0.0, "review": 0.0}
    for kind in counts:
        for row in _rows(paths[kind]):
            audio = str(Path(row["audio_path"]).resolve())
            if audio in seen_paths or row["id"] in seen_ids or row["id"] in known_ids:
                raise ValueError("duplicate or colliding new manifest ID/path")
            seen_paths.add(audio)
            seen_ids.add(row["id"])
            src = sidecar.get(audio)
            if (
                src is None
                or row["text"] != src["text"]
                or not math.isclose(
                    row["duration_seconds"],
                    src["duration_seconds"],
                    abs_tol=1e-6,
                )
            ):
                raise ValueError("new manifest/staging join mismatch")
            if row["split"] != "unsplit" or row["language"] != "es-419":
                raise ValueError("new split/language mismatch")
            if audio in known_paths:
                raise ValueError("new candidate overlaps old train/validation path")
            counts[kind] += 1
            hours[kind] += row["duration_seconds"] / 3600
            reasons = {i["reason"] for i in row["issues"]}
            if kind == "ready":
                if reasons or row["training_ready"] is not True:
                    raise ValueError("invalid ready record")
                release = "original_ready"
            else:
                if not reasons or row["training_ready"] is not False:
                    raise ValueError("invalid review record")
                if reasons != {"ctc_length_infeasible"}:
                    continue
                if row["ctc_minimum_input_length"] > 1.8 * row["estimated_ctc_input_length"]:
                    continue
                release = "temporal_2x_recovery"
            item = candidate(row, src["source_batch"], release)
            item.update(
                source_id=src["id"],
                audio_file_sha256=src["audio_file_sha256"],
                directory_group_hint=src["directory_group_hint"],
                speaker_id=src["speaker_id"],
            )
            eligible.append(item)
    if seen_paths != set(sidecar) or len(sidecar) != source_report["candidate_records"]:
        raise ValueError("new manifest does not partition all staging candidates")
    if (
        counts["ready"] != summary["ready_records"]
        or counts["review"] != summary["review_records"]
        or sum(counts.values()) != summary["source_records"]
        or not math.isclose(sum(hours.values()), summary["total_audio_hours"], abs_tol=1e-6)
        or not math.isclose(hours["ready"], summary["ready_audio_hours"], abs_tol=1e-6)
    ):
        raise ValueError("new manifest summary mismatch")
    old_seconds = sum(r["duration_seconds"] for r in old_train)
    selected, quotas = stratified_select(
        eligible,
        config["target_hours"] * 3600 - old_seconds,
        config["seed"],
    )
    # No output is created until input checks and selection complete. Never write training labels.
    output.mkdir(parents=True)
    try:
        chosen = output / "proposed_ids.jsonl"
        with chosen.open("w") as handle:
            for origin, rows in (("old_train", old_train), ("noah", selected)):
                for row in sorted(rows, key=lambda r: r["id"]):
                    handle.write(json.dumps({"origin": origin, **row}, ensure_ascii=False) + "\n")
        for filename, identity in identities.items():
            if _sha(Path(filename)) != identity["sha256"]:
                raise ValueError("input changed during planning")
        report = {
            "status": "plan_completed",
            "training_ready": False,
            "target_hours": config["target_hours"],
            "seed": config["seed"],
            "old_train": metrics(old_train),
            "noah_available": metrics(eligible),
            "noah_selected": metrics(selected),
            "combined": metrics(old_train + selected),
            "stratum_quotas": quotas,
            "inputs": identities,
            "validation_reference": {
                "sha256": old["manifest_sha256"]["validation"],
                "hours": old["split_audio_hours"]["validation"],
            },
            "test_reference": {"sha256": old["manifest_sha256"]["test"]},
            "definitions": {
                "density": "L/(2*T_est)",
                "ctc_ratio": "(L+adjacent_repeats)/(2*T_est)",
                "bin_policy": "upper-inclusive",
                "duration_edges": [3, 6, 10, 20],
                "density_edges": [0.2, 0.3, 0.4, 0.5, 0.6],
                "ratio_edges": [0.5, 0.75, 0.9],
            },
            "pending": [
                "cross-pool audio content duplicates",
                "sealed holdout identity protection",
                "Noah speaker identity unknown; no speaker-disjoint claim",
            ],
            "model_loaded": False,
            "audio_read": False,
            "test_manifest_content_read": False,
            "final_training_manifest_written": False,
            "output_dir": str(output),
        }
        for name, value in (("report.json", report), ("config.json", config)):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file())
        )
        # Full distributions remain in report.json; stdout is deliberately compact.
        return (
            {k: v for k, v in report.items() if k not in {"stratum_quotas", "inputs"}}
            | {
                key: {
                    **report[key],
                    "by_dimension": {
                        d: g for d, g in report[key]["by_dimension"].items() if d != "stratum"
                    },
                }
                for key in ("old_train", "noah_available", "noah_selected", "combined")
            }
            | {"return_files": [str(output / "report.json"), str(output / "sha256.txt")]}
        )
    except Exception as error:
        (output / "FAILED.txt").write_text(str(error))
        raise
