"""Freeze full-corpus nested 30/50 percent noise views without rewriting source labels."""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.training.english_newval import pin, save
from qwen_hotword.training.noise_audio import audit_noise, rank
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, load_disk_feature_cache
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def language(row: dict[str, Any]) -> str:
    value = row.get("balanced_language_bucket", row["language"])
    return "pt" if value in {"pt-BR", "pt"} else str(value)


def nested_selection(items: list[tuple[str, float, str]], seed: int) -> tuple[set[str], set[str]]:
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for sample_id, seconds, group in items:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid source duration")
        groups[group].append((sample_id, seconds))
    selections: list[set[str]] = [set(), set()]
    for stratum in groups.values():
        ordered = sorted(stratum, key=lambda row: rank(seed, row[0]))
        cumulative = [0.0]
        for _, seconds in ordered:
            cumulative.append(cumulative[-1] + seconds)
        for selected, ratio in zip(selections, (0.3, 0.5), strict=True):
            cut = min(
                range(len(cumulative)), key=lambda i: abs(cumulative[i] - ratio * cumulative[-1])
            )
            selected.update(sample_id for sample_id, _ in ordered[:cut])
    if not selections[0] <= selections[1]:
        raise ValueError("30 percent selection is not nested in 50 percent")
    return selections[0], selections[1]


def source_cache(plan: dict[str, Any], split: str, *, verify: bool) -> DiskFeatureCache:
    return load_disk_feature_cache(
        plan[f"{split}_cache"],
        expected_split=split,
        source_manifest_path=plan[f"{split}_manifest"],
        vocab_path=plan["vocab"],
        verify_sha256=verify,
    )


def inspect_manifest(
    path: Path, cache: DiskFeatureCache
) -> tuple[list[tuple[str, float, str]], set[str], dict[str, Any]]:
    items, paths, seen = [], set(), set()
    totals: dict[str, Any] = {}
    expected = itertools.chain.from_iterable(s.sample_ids for s in cache.shards)
    with stage_progress(
        f"Check frozen {cache.split} IDs/paths/labels", total=cache.sample_count
    ) as tick:
        for n, (row, sample_id) in enumerate(itertools.zip_longest(_rows(path), expected), 1):
            if row is None or row["id"] != sample_id or sample_id in seen:
                raise ValueError("manifest/cache order, length or ID uniqueness mismatch")
            seen.add(sample_id)
            audio = Path(row["audio_path"])
            if not audio.is_absolute() or ".." in audio.parts or str(audio) in paths:
                raise ValueError("noncanonical or duplicate source audio path")
            paths.add(str(audio))
            phones = row["phoneme_token_ids"]
            if (
                not phones
                or any(type(t) is not int or not 0 < t < 90 for t in phones)
                or row["label_length"] != len(phones)
                or row.get("ctc_time_upsampling_factor", 1) != 2
                or row.get("experiment") != "full-ctc-v1"
                or row.get("split") != cache.split
            ):
                raise ValueError("source label/split/Head-time contract differs")
            lang, seconds = language(row), float(row["duration_seconds"])
            if lang not in {"en", "es", "pt"} or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid language or duration")
            bucket = next((b for b in (4, 8, 16) if seconds <= b), 32)
            group = f"{lang}:{row.get('source', row.get('source_corpus', 'unknown'))}:{bucket}"
            items.append((sample_id, seconds, group))
            stats = totals.setdefault(lang, {"records": 0, "hours": 0.0})
            stats["records"] += 1
            stats["hours"] += seconds / 3600
            tick(n)
    return items, paths, totals


def verify_plan(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads((output / "plan.json").read_text())
    if plan["configuration"] != config:
        raise ValueError("configuration differs; use a new experiment directory")
    pin(output / "plan.json", {}, json.loads((output / "prepared.json").read_text())["plan_sha256"])
    with stage_progress("Verify experiment sources, noise files, code and frozen manifests"):
        pin(output / "noise_inventory.json", {}, plan["noise_inventory_sha256"])
        for name, digest in plan["input_sha256"].items():
            pin(Path(name), {}, digest)
        for part in plan["parts"]:
            pin(output / part["manifest"], {}, part["sha256"])
    return plan


def prepare(root: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(
            "prepare needs a NEW directory; --resume verifies completed preparation"
        )
    source = (root / config["source_run"]).resolve()
    noise_folder = Path(config["noise_root"]).resolve()
    if any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output)
        for p in (source, noise_folder)
    ):
        raise ValueError("experiment output overlaps original data/run")
    identities: dict[str, str] = {}
    pin(source / "plan.json", identities, config["source_plan_sha256"])
    pin(source / "head/report.json", identities, config["source_report_sha256"])
    old = json.loads((source / "plan.json").read_text())
    report = json.loads((source / "head/report.json").read_text())
    if report["status"] != "completed" or report["best_epoch"] != 27:
        raise ValueError("expected completed epoch-27 baseline")
    head = source / "head/ctc_head_best.pt"
    pin(head, identities, config["initial_head_sha256"])
    vocab = (root / config["vocab"]).resolve()
    pin(vocab, identities, config["vocab_sha256"])
    if Path(old["vocab"]).resolve() != vocab:
        raise ValueError("baseline vocabulary path differs")
    plan = {
        "configuration": config,
        "source_run": str(source),
        "initial_head": str(head),
        "vocab": str(vocab),
        "train_manifest": old["train_manifest"],
        "train_cache": old["train_cache"],
        "validation_manifest": str(source / "full_ctc_validation.jsonl"),
        "validation_cache": str(source / "cache/validation"),
        "model": old["model"],
        "workzone_config": str((root / old["workzone_config"]).resolve()),
        "encoder_batch_size": old["encoder_batch_size"],
        "head_config": report["head_config"],
        "input_sha256": identities,
        "parts": [],
        "test_set_used": False,
    }
    for protected in (
        Path(plan["train_manifest"]).parent,
        Path(plan["train_cache"]),
        Path(plan["model"]["path"]),
    ):
        protected = protected.resolve()
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("noise output overlaps a protected training/model input")
    # The existing release already protects its heldout paths. Retain its provenance;
    # do not claim an acoustic/speaker content audit from a path-only check.
    for name, digest in old["manifest_sha256"].items():
        pin(source / name, identities, digest)
    upgraded = {
        (Path(__file__).parent / name).resolve() for name in ("sharded_ctc.py", "ctc_overfit.py")
    }
    historical_code: dict[str, str] = {}
    for name, digest in old["input_sha256"].items():
        if Path(name).resolve() in upgraded:
            # Reuse past DATA identities, not an obsolete training implementation.
            # Only these two intentional API additions are exempted; current code
            # is pinned below in this new experiment, never in the old run.
            historical_code[name] = digest
            continue
        pin(Path(name), identities, digest)
    plan["historical_training_code_sha256"] = historical_code
    caches = {}
    for split in ("train", "validation"):
        pin(Path(plan[f"{split}_manifest"]), identities)
        caches[split] = source_cache(plan, split, verify=False)
        for name in ("cache_config.json", "cache_index.json", "cache_summary.json"):
            pin(Path(plan[f"{split}_cache"]) / name, identities)
    items, train_paths, train_stats = inspect_manifest(
        Path(plan["train_manifest"]), caches["train"]
    )
    val_items, val_paths, val_stats = inspect_manifest(
        Path(plan["validation_manifest"]), caches["validation"]
    )
    if train_paths & val_paths or {r[0] for r in items} & {r[0] for r in val_items}:
        raise ValueError("training/validation overlap")
    if (
        caches["train"].sample_count != report["train_sample_count"]
        or caches["validation"].sample_count != report["validation_sample_count"]
        or {k: v["records"] for k, v in train_stats.items()} != config["train_counts"]
        or any(abs(v["hours"] - 480) > 0.1 for v in train_stats.values())
    ):
        raise ValueError("full three-language 480-hour baseline differs")
    selections = nested_selection(items, config["seed"])
    noise = audit_noise(noise_folder, config["seed"])
    for entry in noise["entries"]:
        identities[entry["path"]] = entry["sha256"]
    identities.update(noise["metadata_sha256"])
    package = Path(__file__).parents[1]
    for relative in (
        "training/noise_audio.py",
        "training/noise_prepare.py",
        "training/noise_cache.py",
        "training/noise_run.py",
        "training/noise_evaluation.py",
        "training/sharded_ctc.py",
        "training/ctc_overfit.py",
        "training/feature_cache.py",
        "modeling/audio_encoder.py",
        "modeling/ctc_head.py",
    ):
        pin(package / relative, identities)
    pin(root / "scripts/run_noise_finetune.py", identities)
    output.mkdir(parents=True)
    save(output / "noise_inventory.json", noise)
    pools = {
        split: [e for e in noise["entries"] if e["split"] == split]
        for split in ("train", "validation")
    }
    selected_hours: dict[str, Any] = {"30": {}, "50": {}}
    for split, cache in caches.items():
        iterator = iter(_rows(Path(plan[f"{split}_manifest"])))
        with stage_progress(
            f"Freeze {split} noise manifests", total=cache.shard_count, every=50
        ) as tick:
            for shard in cache.shards:
                rows = list(itertools.islice(iterator, shard.record_count))
                chosen = []
                for row in rows:
                    base_id = row["id"]
                    if split == "train" and base_id not in selections[1]:
                        continue
                    index = int(rank(config["seed"], f"noise:{split}:{base_id}"), 16)
                    entry = pools[split][index % len(pools[split])]
                    offset = (
                        int(rank(config["seed"], f"offset:{base_id}"), 16) % entry["samples_16k"]
                    )
                    aug = {
                        **row,
                        "selected_30": base_id in selections[0],
                        "noise10db": {
                            "path": entry["path"],
                            "sha256": entry["sha256"],
                            "offset": offset,
                        },
                    }
                    if split == "validation":
                        aug["id"] = "noise10db::" + base_id
                    else:
                        for ratio, selected in zip(("30", "50"), selections, strict=True):
                            if base_id in selected:
                                lang = language(row)
                                selected_hours[ratio][lang] = (
                                    selected_hours[ratio].get(lang, 0)
                                    + row["duration_seconds"] / 3600
                                )
                    chosen.append(aug)
                if chosen:
                    folder = output / "noise" / split / f"part-{shard.shard_index:06d}"
                    folder.mkdir(parents=True)
                    manifest = folder / "manifest.jsonl"
                    with manifest.open("x", encoding="utf-8") as handle:
                        for row in chosen:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    plan["parts"].append(
                        {
                            "split": split,
                            "base_shard": shard.shard_index,
                            "records": len(chosen),
                            "manifest": manifest.relative_to(output).as_posix(),
                            "sha256": _sha(manifest),
                        }
                    )
                tick(shard.shard_index + 1)
    plan.update(
        train=train_stats,
        validation=val_stats,
        noisy_hours=selected_hours,
        noise_inventory_sha256=_sha(output / "noise_inventory.json"),
        inherited_audio_isolation_policy=old["audio_isolation_policy"],
        new_speech_content_deduplication=False,
        existing_business_noise_source_verified=False,
    )
    save(output / "plan.json", plan)
    save(
        output / "prepared.json",
        {
            "status": "prepared",
            "plan_sha256": _sha(output / "plan.json"),
            "train": train_stats,
            "noisy_hours": selected_hours,
            "noise_recordings": len(noise["entries"]),
            "excluded_noise": len(noise["excluded"]),
            "test_set_used": False,
        },
    )
    return plan
