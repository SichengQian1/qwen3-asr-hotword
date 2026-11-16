"""Review delivery and source-stratified, recording-disjoint Arabic manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.experiment_a import read_audio_metadata
from qwen_hotword.training.stage_progress import stage_progress


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rank(value: str, seed: int = 20261116) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def recording_key(audio: str) -> str:
    """Known source naming contracts; unknown layouts require an explicit map, not guessing."""
    stem = Path(audio).stem
    if "__seg" in stem:
        return "video:" + stem.split("__", 1)[0]
    if "_spk-" in stem and "_seg-" in stem:
        return "recording:" + stem.split("_spk-", 1)[0]
    if match := re.match(r"(spk\d+)-", stem):
        return "biaobei-speaker:" + match[1]
    if match := re.match(r"(.+_ARB)_\d+(?:\.\d+)?_\d+(?:\.\d+)?$", stem):
        return "broadcast:" + match[1]
    if match := re.match(r"(.+_\d{8}_\d{6})_\d+_\d+$", stem):
        return "broadcast:" + match[1]
    raise ValueError("unknown_recording_key:" + audio)


def assign_split(key: str, config: dict[str, Any]) -> str:
    bucket = int(rank(key, config["seed"])[:12], 16) % 100
    if bucket < config["test_percent"]:
        return "test"
    if bucket < config["test_percent"] + config["validation_percent"]:
        return "validation"
    return "train"


def waterfill(capacity: dict[str, float], goal: float) -> dict[str, float]:
    """Equal hours per source, redistributing only exhausted sources' share."""
    if not capacity or sum(capacity.values()) + 1e-7 < goal:
        raise ValueError(f"insufficient source capacity: {sum(capacity.values()) / 3600:.3f}h")
    targets = dict.fromkeys(capacity, 0.0)
    remaining = set(capacity)
    while remaining:
        share = (goal - sum(targets.values())) / len(remaining)
        exhausted = {s for s in remaining if capacity[s] <= share}
        if not exhausted:
            for source in remaining:
                targets[source] = share
            break
        for source in exhausted:
            targets[source] = capacity[source]
        remaining -= exhausted
    return targets


def fingerprint_audio(row: dict[str, Any]) -> tuple[str, int, int, str]:
    path = Path(row["audio_path"])
    before = path.stat()
    digest = sha(path)
    header = read_audio_metadata(path)
    if not math.isclose(header.duration_seconds, row["duration_seconds"], abs_tol=1e-8):
        raise ValueError(f"audio duration changed since source audit: {path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"audio changed while fingerprinting: {path}")
    return str(path), after.st_size, after.st_mtime_ns, digest


def prefetch_hashes(
    rows: list[dict[str, Any]], db: sqlite3.Connection, pool: ThreadPoolExecutor
) -> None:
    jobs = []
    for row in rows:
        path = Path(row["audio_path"])
        stat = path.stat()
        old = db.execute("SELECT size,mtime FROM hashes WHERE path=?", (str(path),)).fetchone()
        if old != (stat.st_size, stat.st_mtime_ns):
            jobs.append(pool.submit(fingerprint_audio, row))
    for future in as_completed(jobs):
        db.execute("INSERT OR REPLACE INTO hashes VALUES (?,?,?,?)", future.result())
        db.commit()


def export_review(labels: Path) -> dict[str, Any]:
    report = read(labels / "report.json")
    pin(labels / "labels.jsonl", report["labels_sha256"])
    final = labels / "review_delivery"
    if final.exists():
        existing = read(final / "summary.json")
        if existing["labels_sha256"] != report["labels_sha256"]:
            raise ValueError("review delivery belongs to other labels")
        # Human annotations in the exported files must never be overwritten by re-export.
        return dict(existing)
    output = Path(tempfile.mkdtemp(prefix=".review-", dir=labels))
    words: Counter[tuple[str, str]] = Counter()
    variants: dict[tuple[str, str], dict[str, Any]] = {}
    rules: Counter[str] = Counter()
    samples: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    representative: dict[tuple[str, str], dict[str, Any]] = {}
    changed, total = 0, 0
    differences = output / "word_differences.jsonl"
    with (
        differences.with_suffix(".pending").open("w", encoding="utf-8") as dst,
        stage_progress("Export pronunciation differences and deterministic listening set") as tick,
    ):
        for total, row in enumerate(iter_jsonl(labels / "labels.jsonl"), 1):
            sample_changed = False
            for trace in row["word_traces"]:
                key = row["group"], trace["surface"]
                words[key] += 1
                variant = json.dumps(trace["phones"], ensure_ascii=False)
                item = variants.setdefault(key, {"counts": Counter(), "examples": {}})
                item["counts"][variant] += 1
                item["examples"].setdefault(
                    variant,
                    {
                        "trace": trace,
                        "source": row["source"],
                        "row": row["row"],
                        "audio_path": row["audio_path"],
                        "reference_text": row["text"],
                    },
                )
                for rule in trace["rules"]:
                    rules[row["group"] + "/" + rule] += 1
                    representative.setdefault((row["group"], rule), row)
                for alternative in trace.get("rule_alternatives", []):
                    name = "candidate/" + alternative["rule"]
                    rules[row["group"] + "/" + name] += 1
                    representative.setdefault((row["group"], name), row)
                if trace["baseline_issue"] or trace["phones"] != trace["baseline_phones"]:
                    sample_changed = True
                    dst.write(
                        json.dumps(
                            {
                                "source": row["source"],
                                "row": row["row"],
                                "audio_path": row["audio_path"],
                                "group": row["group"],
                                **trace,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            changed += sample_changed
            group = row["group"]
            # 50 unconditioned + 50 changed-label cases per group; not a PER-tail sample.
            for category in ("random", "changed") if sample_changed else ("random",):
                pool = samples[group + "/" + category]
                pool.append((rank(row["source"] + ":" + str(row["row"])), row))
                pool.sort(key=lambda x: x[0])
                del pool[50:]
            tick(total)
    differences.with_suffix(".pending").replace(differences)
    for group in ("ar_egy", "ar_msa"):
        top = sorted((key for key in words if key[0] == group), key=lambda k: (-words[k], k))[:300]
        write_json(
            output / f"{group}_top300.json",
            [
                {
                    "surface": k[1],
                    "word_occurrences": words[k],
                    "pronunciations": [
                        {
                            "phones": json.loads(v),
                            "occurrences": n,
                            **variants[k]["examples"][v],
                        }
                        for v, n in variants[k]["counts"].most_common()
                    ],
                    "review_status": "pending",
                    "reviewer": None,
                }
                for k in top
            ],
        )
    chosen = {(r["source"], r["row"]): r for pool in samples.values() for _, r in pool}
    chosen.update({(r["source"], r["row"]): r for r in representative.values()})
    write_json(
        output / "listening.json",
        [
            {
                **r,
                "transcript_matches_audio": None,
                "phones_match_audio": None,
                "corrected_phones": None,
                "reviewer": None,
            }
            for _, r in sorted(chosen.items())
        ],
    )
    result = {
        "status": "review_prepared",
        "labeled_rows": total,
        "changed_rows_vs_legacy_rules": changed,
        "rule_word_counts": dict(rules),
        "listening_samples": len(chosen),
        "sampling": "random_and_changed_not_PER_tail",
        "human_pronunciation_review_completed": False,
        "labels_sha256": report["labels_sha256"],
    }
    blocking: dict[str, Counter[str]] = defaultdict(Counter)
    with stage_progress("Summarize remaining review blockers (no model rerun)") as tick:
        n = 0
        for path in sorted((labels / "parts").glob("*/*/review.jsonl")):
            for row in iter_jsonl(path):
                blocking[row["group"]][row["label_issue"]] += 1
                n += 1
                tick(n)
    write_json(
        output / "blocking_issues.json",
        {
            group: {
                "rows": sum(counts.values()),
                "top100_first_blocking_reasons": counts.most_common(100),
            }
            for group, counts in blocking.items()
        },
    )
    write_json(output / "summary.json", result)
    output.rename(final)
    return result


def verified_labels(labels: Path) -> dict[str, Any]:
    report = read(labels / "report.json")
    if report["status"] != "automatic_labels_completed" or report["num_classes"] != 167:
        raise ValueError("incomplete or incompatible Arabic labels")
    if read(labels / "contract.json") != report["identity"]:
        raise ValueError("label contract/report mismatch")
    for path, digest in report["identity"].items():
        pin(Path(path), digest)
    pin(labels / "labels.jsonl", report["labels_sha256"])
    return dict(report)


def freeze(
    root: Path, config_path: Path, labels: Path, output: Path, resume: bool
) -> dict[str, Any]:
    # Runtime import keeps this module usable for CPU-only rule tests.
    from scripts.arabic_diagnostic_common import completed_run, recheck

    config = read(config_path)
    if (
        not 1 <= config["metadata_workers"] <= 64
        or config["train_hours"] <= 0
        or min(config["heldout_hours_per_group"], config["test_hours_per_group"]) <= 0
        or min(config["validation_percent"], config["test_percent"]) <= 0
        or not 0 < config["validation_percent"] + config["test_percent"] < 100
    ):
        raise ValueError("invalid split targets or worker count")
    source = (root / config["source_run"]).resolve()
    if any(output == p or output in p.parents or p in output.parents for p in (source, labels)):
        raise ValueError("new output must be separate from source run and labels")
    run = completed_run(source)
    old_checkpoint = Path(run["head"]["best_checkpoint_path"])
    report = verified_labels(labels)
    if report["vocab_sha256"] != run["plan"]["vocabularies"]["ar"]["sha256"]:
        raise ValueError("repaired labels must retain the frozen 167-class vocabulary identity")
    exclusions = root / config["quality_exclusions"] if config["quality_exclusions"] else None
    excluded = set()
    if exclusions:
        for item in iter_jsonl(exclusions):
            if not item.get("reason") or not item.get("reviewer"):
                raise ValueError("quality exclusions require reason and reviewer")
            excluded.add((item["source"], item["row"]))
    contract = {
        "source_identity": run["identity"],
        "old_checkpoint": {"path": str(old_checkpoint), "sha256": sha(old_checkpoint)},
        "config_path": str(config_path),
        "config_sha256": sha(config_path),
        "labels_report_sha256": sha(labels / "report.json"),
        "labels_sha256": report["labels_sha256"],
        "vocab_sha256": report["vocab_sha256"],
        "code_sha256": sha(Path(__file__)),
        "quality_exclusions_sha256": sha(exclusions) if exclusions else None,
        "quality_exclusions_path": str(exclusions) if exclusions else None,
        "review_files": {
            str(path): sha(path) for path in (labels / "review_delivery").glob("*.json")
        },
        "listening_sha256": sha(labels / "review_delivery/listening.json")
        if (labels / "review_delivery/listening.json").exists()
        else None,
    }
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if read(output / "contract.json") != contract:
            raise ValueError("freeze identity changed; use a new directory")
        if (output / "prepare_report.json").exists():
            result = read(output / "prepare_report.json")
            for item in result["manifests"].values():
                pin(Path(item["path"]), item["sha256"])
            return dict(result)
    else:
        write_json(output / "contract.json", contract)
    protected = set()
    feature_order: dict[str, int] = {}
    old_validation = {}
    old_validation_recordings = set()
    paired_validation = []
    for lang in ("en", "ar"):
        for split in ("train", "validation"):
            for row in iter_jsonl(Path(run["plan"]["manifests"][f"{lang}/{split}"]["path"])):
                feature_order[row["audio_sha256"]] = len(feature_order)
                if lang == "en" or split == "validation":
                    protected.add(row["audio_sha256"])
                if lang == "ar" and split == "validation":
                    old_validation[row["source"], row["source_row"]] = row
                    old_validation_recordings.add(recording_key(row["audio_path"]))
    listened_groups = set()
    listening = labels / "review_delivery/listening.json"
    if listening.exists():
        for row in read(listening):
            try:
                listened_groups.add(recording_key(row["audio_path"]))
            except ValueError:
                continue
    for path in (labels / "review_delivery").glob("*_top300.json"):
        for item in read(path):
            for example in item["pronunciations"]:
                try:
                    listened_groups.add(recording_key(example["audio_path"]))
                except ValueError:
                    continue
    pools: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    with stage_progress("Partition Arabic by original recording and source (no WAV reads)") as tick:
        for i, row in enumerate(iter_jsonl(labels / "labels.jsonl"), 1):
            tick(i)
            if (row["source"], row["row"]) in excluded:
                rejected["reviewed_quality_exclusion"] += 1
                continue
            if any(t["exception_candidates"] and not t["exception"] for t in row["word_traces"]):
                rejected["lexical_exception_needs_review"] += 1
                continue
            old = old_validation.get((row["source"], row["row"]))
            if old is not None:
                if old["text"] != row["text"] or old["audio_path"] != row["audio_path"]:
                    raise ValueError("paired validation text/audio identity differs")
                paired_validation.append(
                    {
                        **old,
                        "phoneme_token_ids": row["ids"],
                        "label_length": len(row["ids"]),
                        "ctc_minimum_input_length": row["ctc_minimum_input_length"],
                    }
                )
            try:
                recording = recording_key(row["audio_path"])
            except ValueError:
                rejected["unknown_recording_key"] += 1
                continue
            split = assign_split(recording, config)
            if recording in old_validation_recordings:
                split = "validation"
            if split == "test" and recording in listened_groups:
                split = "validation"
            slim = {
                k: row[k]
                for k in (
                    "group",
                    "source",
                    "row",
                    "text",
                    "audio_path",
                    "duration_seconds",
                    "ids",
                    "ctc_minimum_input_length",
                )
            }
            slim["recording_key"] = recording
            pools[split, row["group"], row["source"]].append(slim)
    if not paired_validation:
        raise ValueError("no repaired old validation samples for paired comparison")
    capacity = {k: sum(r["duration_seconds"] for r in rows) for k, rows in pools.items()}
    targets: dict[tuple[str, str, str], float] = {}
    sources_by_group = {g: {s for _, group, s in pools if group == g} for g in ("ar_egy", "ar_msa")}
    if any(len(s) != 4 for s in sources_by_group.values()):
        raise ValueError("expected four qualified sources in each Arabic group")
    for split, field in (
        ("validation", "heldout_hours_per_group"),
        ("test", "test_hours_per_group"),
    ):
        for group, sources in sources_by_group.items():
            goal = config[field] * 3600 / len(sources)
            for source_name in sources:
                k = split, group, source_name
                if capacity.get(k, 0) < goal:
                    raise ValueError(f"insufficient grouped holdout capacity: {k}")
                targets[k] = goal
    egyptian_hours = sum(v / 3600 for k, v in capacity.items() if k[:2] == ("train", "ar_egy"))
    egyptian_goal = min(egyptian_hours, config["train_hours"])
    for group, hours in (
        ("ar_egy", egyptian_goal),
        ("ar_msa", config["train_hours"] - egyptian_goal),
    ):
        caps = {s: capacity.get(("train", group, s), 0) for s in sources_by_group[group]}
        goals = waterfill(caps, hours * 3600)
        targets.update({("train", group, s): v for s, v in goals.items()})
    for rows in pools.values():
        rows.sort(key=lambda r: rank(r["audio_path"], config["seed"]))
    # Persist selected-candidate fingerprints immediately. Resume avoids repeating full WAV reads.
    db = sqlite3.connect(output / "audio_fingerprints.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS hashes("
        "path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha TEXT)"
    )
    selected: dict[str, tuple[tuple[str, str, str], dict[str, Any]]] = {}
    bad_hashes: set[str] = set()
    seconds: Counter[tuple[str, str, str]] = Counter()
    positions: Counter[tuple[str, str, str]] = Counter()
    priority = {"test": 0, "validation": 1, "train": 2}
    keys = sorted(targets, key=lambda k: (priority[k[0]], k[1:]))
    try:
        with (
            ThreadPoolExecutor(max_workers=config["metadata_workers"]) as hash_pool,
            stage_progress(
                "Fingerprint selected candidates; persisted per file; full WAV byte reads"
            ) as tick,
        ):
            checked = 0
            while True:
                progressed = False
                for key in keys:
                    rows = pools[key]
                    while seconds[key] + 1e-7 < targets[key] and positions[key] < len(rows):
                        if positions[key] % 128 == 0:
                            prefetch_hashes(
                                rows[positions[key] : positions[key] + 128], db, hash_pool
                            )
                        row = rows[positions[key]]
                        positions[key] += 1
                        progressed = True
                        path = Path(row["audio_path"])
                        stat = path.stat()
                        cached = db.execute(
                            "SELECT size,mtime,sha FROM hashes WHERE path=?", (str(path),)
                        ).fetchone()
                        if cached and cached[:2] == (stat.st_size, stat.st_mtime_ns):
                            digest = cached[2]
                        else:
                            fingerprint = fingerprint_audio(row)
                            digest = fingerprint[3]
                            db.execute(
                                "INSERT OR REPLACE INTO hashes VALUES (?,?,?,?)",
                                fingerprint,
                            )
                            db.commit()
                        checked += 1
                        tick(checked)
                        if digest in protected or digest in bad_hashes:
                            rejected["protected_or_conflicting_audio"] += 1
                            continue
                        if digest in selected:
                            old_key, old_row = selected[digest]
                            if old_row["text"] != row["text"] or old_row["ids"] != row["ids"]:
                                bad_hashes.add(digest)
                                seconds[old_key] -= old_row["duration_seconds"]
                                del selected[digest]
                                rejected["conflicting_audio_labels"] += 2
                            else:
                                rejected["duplicate_audio"] += 1
                            continue
                        selected[digest] = (key, row)
                        seconds[key] += row["duration_seconds"]
                if not progressed:
                    # Keep every remaining qualified EGY training recording. Byte dedup and
                    # old-validation protection can reduce the metadata-only EGY estimate.
                    for k in keys:
                        if k[:2] == ("train", "ar_egy"):
                            targets[k] = seconds[k]
                    msa_goal = max(
                        0.0,
                        config["train_hours"] * 3600
                        - sum(seconds[k] for k in keys if k[:2] == ("train", "ar_egy")),
                    )
                    msa_caps = {
                        s: (
                            seconds[("train", "ar_msa", s)]
                            if positions[("train", "ar_msa", s)] >= len(pools["train", "ar_msa", s])
                            else capacity["train", "ar_msa", s]
                        )
                        for s in sources_by_group["ar_msa"]
                    }
                    redistributed = waterfill(msa_caps, msa_goal)
                    for s, goal in redistributed.items():
                        targets["train", "ar_msa", s] = goal
                    if all(
                        seconds[k] + 1e-7 >= targets[k] or positions[k] >= len(pools[k])
                        for k in keys
                    ):
                        break
    finally:
        db.close()
    # Exhausted EGY or MSA sources may lose hours to byte duplicates. Preserve source quotas;
    # fail explicitly rather than silently making a smaller or differently balanced dataset.
    shortages = {
        "/".join(k): (targets[k] - seconds[k]) / 3600
        for k in keys
        if seconds[k] + 1e-7 < targets[k]
    }
    if shortages:
        write_json(
            output / "selection_shortfall.json",
            {
                "shortfalls_hours": shortages,
                "rejections": dict(rejected),
                "fingerprints_saved": True,
            },
        )
        raise ValueError(
            f"post-deduplication source shortfall; see {output}/selection_shortfall.json"
        )
    manifests = {}
    paired_path = output / "paired_ar_validation.jsonl"
    with paired_path.with_suffix(".pending").open("w", encoding="utf-8") as stream:
        for row in sorted(paired_validation, key=lambda r: feature_order[r["audio_sha256"]]):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    paired_path.with_suffix(".pending").replace(paired_path)
    manifests["ar/paired_validation"] = {
        "path": str(paired_path),
        "sha256": sha(paired_path),
        "samples": len(paired_validation),
        "original_samples": len(old_validation),
        "missing_or_review_samples": len(old_validation) - len(paired_validation),
    }
    for split in ("train", "validation", "test"):
        path = output / f"{split}.jsonl"
        counts: Counter[str] = Counter()
        selected_hours: Counter[str] = Counter()
        with path.with_suffix(".pending").open("w", encoding="utf-8") as stream:
            for digest, (key, row) in sorted(
                selected.items(),
                key=lambda pair: (
                    feature_order.get(pair[0], len(feature_order)),
                    pair[1][1]["audio_path"],
                ),
            ):
                if key[0] != split:
                    continue
                item = {
                    "id": "en-ar-v1-" + digest,
                    "audio_sha256": digest,
                    "audio_path": row["audio_path"],
                    "text": row["text"],
                    "language": "ar",
                    "source_group": row["group"],
                    "source": row["source"],
                    "source_row": row["row"],
                    "recording_key": row["recording_key"],
                    "duration_seconds": row["duration_seconds"],
                    "phoneme_token_ids": row["ids"],
                    "label_length": len(row["ids"]),
                    "ctc_minimum_input_length": row["ctc_minimum_input_length"],
                    "ctc_time_upsampling_factor": 2,
                    "split": split,
                    "experiment": "full-ctc-v1",
                }
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[row["group"]] += 1
                selected_hours[row["group"]] += row["duration_seconds"] / 3600
        path.with_suffix(".pending").replace(path)
        manifests[f"ar/{split}"] = {
            "path": str(path),
            "sha256": sha(path),
            "samples": sum(counts.values()),
            "counts": dict(counts),
            "hours": dict(selected_hours),
        }
    (output / "english").mkdir(exist_ok=True)
    for split in ("train", "validation"):
        old = run["plan"]["manifests"][f"en/{split}"]
        dest = output / "english" / f"{split}.jsonl"
        shutil.copyfile(old["path"], dest)
        pin(dest, old["sha256"])
        manifests[f"en/{split}"] = {**old, "path": str(dest)}
    recheck(run)
    pin(labels / "labels.jsonl", report["labels_sha256"])
    pin(config_path, contract["config_sha256"])
    pin(old_checkpoint, contract["old_checkpoint"]["sha256"])
    result = {
        "status": "prepared",
        "source_run": str(source),
        "labels_dir": str(labels),
        "contract": contract,
        "manifests": manifests,
        "config": config,
        "source_targets_hours": {"/".join(k): v / 3600 for k, v in targets.items()},
        "selected_hours": {"/".join(k): v / 3600 for k, v in seconds.items()},
        "rejections": dict(rejected),
        "recording_disjoint": True,
        "speaker_disjoint_verified": False,
        "old_ar_validation_excluded": True,
        "old_ar_validation_recordings_excluded_from_new_train": True,
        "review_delivery_recordings_excluded_from_test": True,
        "test_set_used_for_training": False,
        "human_pronunciation_review_completed": False,
    }
    write_json(output / "prepare_report.json", result)
    return result
