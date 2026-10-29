"""Full train/validation relabeling with pinned new G2P Top1, never old phone IDs."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import environment, pin, run_worker, sha, write_json
from qwen_hotword.phonemes.mfa_v03_foundation import validate_inventory
from qwen_hotword.phonemes.mfa_v03_review import rows, write_rows
from qwen_hotword.training.mfa_round_common import (
    digest,
    old_noise_part,
    read,
    select_noise,
    sources,
    words,
)
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.noise_prepare import language
from qwen_hotword.training.sharded_ctc import FeatureShardDescriptor
from qwen_hotword.training.stage_progress import stage_progress


def offsets(shard: FeatureShardDescriptor) -> dict[str, int]:
    import torch

    pin(shard.feature_path, shard.feature_sha256)
    payload = torch.load(shard.feature_path, mmap=True, map_location="cpu", weights_only=True)
    hidden = payload["hidden_states"]
    values = payload["hidden_offsets"].tolist()
    if (
        hidden.dtype != torch.bfloat16
        or list(hidden.shape[1:]) != [1024]
        or len(values) != len(shard.sample_ids) + 1
        or values[0] != 0
        or values[-1] != hidden.shape[0]
        or any(a >= b for a, b in zip(values, values[1:], strict=False))
    ):
        raise ValueError("invalid inherited feature offsets")
    return dict(
        zip(
            shard.sample_ids, (b - a for a, b in zip(values, values[1:], strict=False)), strict=True
        )
    )


def top1_tokens(
    word_list: list[str], generated: dict[str, dict[str, Any]]
) -> tuple[list[int], list[str]]:
    ids, issues = [], []
    for word in word_list:
        record = generated.get(word)
        if record is None or record["status"] != "ok" or len(record["candidates"]) != 1:
            issues.append("g2p_unresolved:" + word)
            continue
        candidate = record["candidates"][0]
        if not candidate["usable"] or candidate["oov_units"]:
            issues.append("phone_oov:" + word)
        else:
            ids.extend(candidate["token_ids"])
    return ([] if issues else ids), issues


def relabel(
    row: dict[str, Any], generated: dict[str, dict[str, Any]], frames: int
) -> tuple[dict[str, Any] | None, list[str]]:
    word_list, issues = words(row["text"])
    ids, missing = top1_tokens(word_list, generated)
    issues += missing
    minimum = len(ids) + sum(a == b for a, b in zip(ids, ids[1:], strict=False))
    if not ids:
        issues.append("empty_complete_label")
    if minimum > 2 * frames:
        issues.append("new_label_ctc_infeasible")
    if issues:
        return None, sorted(set(issues))
    stale = {
        "phoneme_token_ids",
        "phoneme_tokens",
        "phonemes",
        "ipa",
        "pronunciation",
        "pronunciations",
        "label_length",
        "ctc_minimum_input_length",
        "ctc_minimum_input_ratio",
        "g2p_model",
        "g2p_dictionary",
        "dictionary_path",
        "vocab_path",
    }
    new = {k: v for k, v in row.items() if k not in stale}
    new.update(
        words=word_list,
        phoneme_token_ids=ids,
        label_length=len(ids),
        ctc_minimum_input_length=minimum,
        ctc_time_upsampling_factor=2,
        estimated_ctc_input_length=frames,
        effective_ctc_input_length=2 * frames,
        ctc_minimum_input_ratio=minimum / (2 * frames),
        experiment="full-ctc-v1",
        legacy_label_sha256=digest(row["phoneme_token_ids"]),
        label_policy="pinned_v03_G2P_Top1_no_legacy_phone_fallback",
    )
    return new, []


def generate(
    root: Path, cfg: dict[str, Any], output: Path, word_sets: dict[str, set[str]]
) -> dict[str, dict[str, dict[str, Any]]]:
    foundation = read(root / cfg["foundation_config"])
    worker = {**foundation, "g2p_num_pronunciations": 1}
    worker_path = output / "g2p_worker_config.json"
    write_json(worker_path, worker)
    worker["_config_path"] = str(worker_path)
    prefix = environment(worker, root / cfg["assets_dir"], output / "logs")
    jobs = []
    for lang, values in sorted(word_sets.items()):
        ordered = sorted(values)
        size = cfg["g2p_words_per_part"]
        for index in range(0, len(ordered), size):
            folder = output / "g2p" / lang / f"part-{index // size:05d}"
            folder.mkdir(parents=True, exist_ok=True)
            request = folder / "request.jsonl"
            write_rows(
                request, [{"language": lang, "word": w} for w in ordered[index : index + size]]
            )
            jobs.append((request, folder / "predictions.jsonl"))

    def task(job: tuple[Path, Path]) -> None:
        run_worker(root, worker, root / cfg["assets_dir"], prefix, *job)

    with (
        stage_progress(
            "Generate NEW Top1 pronunciations (resumable word parts)", total=len(jobs), every=1
        ) as tick,
        ThreadPoolExecutor(max_workers=cfg["g2p_workers"]) as pool,
    ):
        for index, _ in enumerate(pool.map(task, jobs), 1):
            tick(index)
    result: dict[str, dict[str, dict[str, Any]]] = {lang: {} for lang in word_sets}
    for _, path in jobs:
        for row in rows(path):
            result[row["language"]][row["word"]] = row
    if any(set(result[lang]) != values for lang, values in word_sets.items()):
        raise ValueError("G2P output does not account for all requested words")
    return result


def build_hotwords(
    table: dict[str, Any],
    generated: dict[str, dict[str, dict[str, Any]]],
    vocab_path: Path,
    output: Path,
) -> dict[str, Any]:
    vocab = load_phoneme_vocab(vocab_path)
    variants, failures, seen = [], [], set()
    for entry in table["entries"]:
        for variant in entry["variants"]:
            lang = "pt" if variant["language"] in {"pt", "pt-BR"} else variant["language"]
            ws, issues = words(variant["surface"])
            ids, missing = top1_tokens(ws, generated[lang])
            issues += missing
            if any(x in variant["surface"] for x in ("\n", "\r", "<|", "|>")):
                issues.append("unsafe_prompt_surface")
            key = (entry["normalized"], lang, tuple(ids))
            if issues or not ids:
                failures.append({"surface": variant["surface"], "language": lang, "issues": issues})
            elif key not in seen:
                seen.add(key)
                tokens = tuple(vocab.tokens[i] for i in ids)
                value = HotwordEntry(
                    hotword_id="v03_" + digest(key)[:24],
                    language="pt-BR" if lang == "pt" else lang,
                    surface=variant["surface"],
                    normalized=entry["normalized"],
                    words=tuple(ws),
                    pronunciation=" ".join(tokens),
                    phoneme_tokens=tokens,
                    token_ids=tuple(ids),
                    source="pinned_new_G2P_Top1",
                    validation_occurrences=0,
                )
                variants.append(value.to_dict())
    active = {v["normalized"] for v in variants}
    inactive = [e["surface"] for e in table["entries"] if e["normalized"] not in active]
    write_rows(output / "indexed_pronunciations.v03.jsonl", variants)
    write_rows(output / "hotword_unresolved.jsonl", failures)
    report = {
        "declared_surfaces": len(table["entries"]),
        "active_surfaces": len(active),
        "active_variants": len(variants),
        "inactive_surfaces": inactive,
        "denominator_policy": "all original surfaces retained in evaluation, including inactive",
        "pronunciation_policy": "same new Top1 as training, at most one per surface/language",
    }
    write_json(output / "hotword_report.json", report)
    if not variants:
        raise ValueError("no usable new hotword pronunciations")
    return report


def prepare(root: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    marker = output / "prepared.json"
    if marker.exists():
        saved = read(marker)
        for name, checksum in saved["files"].items():
            pin(output / name, checksum)
        return saved
    pin(root / cfg["foundation_config"], cfg["foundation_config_sha256"])
    foundation = read(root / cfg["foundation_config"])
    validate_inventory(root, foundation, root / cfg["assets_dir"])
    previous = root / cfg["foundation_output"]
    completed = read(previous / "report.json")
    old_contract = read(previous / "foundation_contract.json")
    if (
        completed["status"] != "prepared_pending_human_review"
        or old_contract["configuration"] != foundation
        or completed.get("stages", {}).get("review", {}).get("audio_copied") != 200
    ):
        # Source records are evidence of preparation, not a human pronunciation approval.
        raise ValueError("expected completed v0.3 foundation preparation")
    old, caches = sources(root, cfg)
    table_path = root / foundation["hotword_table"]
    pin(table_path, foundation["hotword_table_sha256"])
    table = read(table_path)
    if len(table["entries"]) != 4000:
        raise ValueError("expected unchanged 4000 surface table")
    word_sets: dict[str, set[str]] = {lang: set() for lang in ("en", "es", "pt")}
    for split in ("train", "validation"):
        with stage_progress(
            f"Collect {split} words for NEW labels", total=caches[split].sample_count
        ) as tick:
            for number, row in enumerate(rows(Path(old[f"{split}_manifest"])), 1):
                ws, issues = words(row["text"])
                if not issues:
                    word_sets[language(row)].update(ws)
                tick(number)
    for entry in table["entries"]:
        for variant in entry["variants"]:
            lang = "pt" if variant["language"] in {"pt", "pt-BR"} else variant["language"]
            ws, issues = words(variant["surface"])
            if not issues:
                word_sets[lang].update(ws)
    generated = generate(root, cfg, output, word_sets)
    hotwords = build_hotwords(table, generated, root / cfg["vocab"], output)
    noise = read(root / cfg["noise_source"] / "noise_inventory.json")
    pools = {s: [r for r in noise["entries"] if r["split"] == s] for s in ("train", "validation")}
    if not all(pools.values()) or (
        {r["pcm_sha256"] for r in pools["train"]} & {r["pcm_sha256"] for r in pools["validation"]}
    ):
        raise ValueError("noise recording split overlap/empty pool")
    previous_parts = {(p["split"], p["base_shard"]): p for p in old["parts"]}
    items: list[dict[str, Any]] = []
    label_parts: list[dict[str, Any]] = []
    old_noisy_ids: set[str] = set()
    totals: dict[str, Any] = {}
    frequencies: dict[str, Counter[int]] = defaultdict(Counter)
    paths: dict[str, set[str]] = {}
    source_ids: dict[str, set[str]] = {}
    for split, cache in caches.items():
        iterator = iter(rows(Path(old[f"{split}_manifest"])))
        paths[split], source_ids[split] = set(), set()
        with stage_progress(
            f"Relabel ALL {split} records; verify actual CTC frames",
            total=cache.shard_count,
            every=25,
        ) as tick:
            for shard in cache.shards:
                original = list(itertools.islice(iterator, shard.record_count))
                if [r["id"] for r in original] != list(shard.sample_ids):
                    raise ValueError("old manifest and cache sample order differ")
                lengths = offsets(shard)
                prior = previous_parts.get((split, shard.shard_index))
                old_frames: dict[str, int] = {}
                if prior:
                    cached = old_noise_part(root, cfg, old, prior)
                    old_frames = offsets(cached.shards[0])
                chosen, rejected = [], []
                for row in original:
                    lang = language(row)
                    sample_id = row["id"]
                    if (
                        row["split"] != split
                        or sample_id in source_ids[split]
                        or row["audio_path"] in paths[split]
                    ):
                        raise ValueError("invalid split or duplicate source ID/audio path")
                    source_ids[split].add(sample_id)
                    paths[split].add(row["audio_path"])
                    stats = totals.setdefault(
                        f"{split}/{lang}",
                        {
                            "source_records": 0,
                            "source_hours": 0.0,
                            "ready_records": 0,
                            "ready_hours": 0.0,
                            "failure_reasons": Counter(),
                        },
                    )
                    stats["source_records"] += 1
                    stats["source_hours"] += row["duration_seconds"] / 3600
                    prior_id = sample_id if split == "train" else "noise10db::" + sample_id
                    minimum_frames = min(
                        lengths[sample_id], old_frames.get(prior_id, lengths[sample_id])
                    )
                    new, issues = relabel(row, generated[lang], minimum_frames)
                    if new is None:
                        rejected.append({"source_record": row, "issues": issues})
                        stats["failure_reasons"].update(issues)
                        continue
                    chosen.append(new)
                    stats["ready_records"] += 1
                    stats["ready_hours"] += row["duration_seconds"] / 3600
                    frequencies[f"{split}/{lang}"].update(new["phoneme_token_ids"])
                    if split == "train":
                        if prior_id in old_frames:
                            old_noisy_ids.add(sample_id)
                        items.append({"id": sample_id, "language": row["language"]})
                folder = output / "labels" / split / f"part-{shard.shard_index:06d}"
                folder.mkdir(parents=True, exist_ok=True)
                write_rows(folder / "ready.jsonl", chosen)
                write_rows(folder / "needs_review.jsonl", rejected)
                label_parts.append(
                    {
                        "split": split,
                        "base_shard": shard.shard_index,
                        "path": str(folder / "ready.jsonl"),
                        "old_part": prior,
                    }
                )
                tick(shard.shard_index + 1)
        if next(iterator, None) is not None:
            raise ValueError("source manifest has more records than cache")
    if paths["train"] & paths["validation"] or source_ids["train"] & source_ids["validation"]:
        raise ValueError("source training/validation overlap")
    if {lang: totals[f"train/{lang}"]["source_records"] for lang in ("en", "es", "pt")} != cfg[
        "train_counts"
    ]:
        raise ValueError("baseline three-language training inventory changed")
    write_json(
        output / "label_report.json",
        {
            "by_split_language": totals,
            "unique_words": {lang: len(ws) for lang, ws in word_sets.items()},
            "token_counts": {key: dict(value) for key, value in frequencies.items()},
            "unused_training_tokens": [
                token
                for i, token in enumerate(load_phoneme_vocab(root / cfg["vocab"]).tokens)
                if i >= 2
                and not any(frequencies[f"train/{lang}"][i] for lang in ("en", "es", "pt"))
            ],
            "human_verified": False,
            "test_set_used_for_training": False,
            "cross_split_exact_path_overlap": 0,
            "speaker_and_acoustic_near_duplicate_isolation_verified": False,
        },
    )
    for key, stats in totals.items():
        fraction = 1 - stats["ready_records"] / stats["source_records"]
        if fraction > cfg["maximum_quarantine_fraction"]:
            raise ValueError(
                f"{key}: {fraction:.2%} quarantined; see label_report; "
                "do not train a silently reduced corpus"
            )
    selected = select_noise(items, old_noisy_ids, cfg["seed"])
    parts: list[dict[str, Any]] = []
    noise_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"records": 0, "hours": 0.0, "total_records": 0}
    )
    for item in label_parts:
        split, index = item["split"], item["base_shard"]
        ready = list(rows(Path(item["path"])))
        for condition in ("mixed",) if split == "train" else ("original", "noise10db"):
            final = []
            for row in ready:
                use_noise = row["id"] in selected if split == "train" else condition == "noise10db"
                base_id = row["id"]
                new = {**row, "base_sample_id": base_id, "noise_selected": use_noise}
                if split == "train":
                    stat = noise_stats[language(row)]
                    stat["total_records"] += 1
                    stat["records"] += int(use_noise)
                    stat["hours"] += row["duration_seconds"] / 3600 if use_noise else 0
                if use_noise:
                    pool = pools[split]
                    noise_entry = pool[
                        int(rank(cfg["seed"], f"noise:{split}:{base_id}"), 16) % len(pool)
                    ]
                    new["noise10db"] = {
                        "path": noise_entry["path"],
                        "sha256": noise_entry["sha256"],
                        "offset": int(rank(cfg["seed"], f"offset:{base_id}"), 16)
                        % noise_entry["samples_16k"],
                    }
                    if split == "validation":
                        new["id"] = "noise10db::" + base_id
                final.append(new)
            if not final:
                continue
            folder = output / "parts" / split / f"{index:06d}-{condition}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "manifest.jsonl"
            write_rows(path, final)
            parts.append(
                {
                    **item,
                    "manifest": str(path.relative_to(output)),
                    "sha256": sha(path),
                    "records": len(final),
                    "condition": condition,
                }
            )
    protected_files = [output / p["manifest"] for p in parts]
    protected_files += [
        output / name
        for name in ("label_report.json", "hotword_report.json", "indexed_pronunciations.v03.jsonl")
    ]
    value = {
        "status": "new_labels_prepared",
        "parts": parts,
        "noise": dict(noise_stats),
        "noise_unit": "nearest 70% of eligible audio records PER language, fixed across epochs",
        "head_config": old["head_config"],
        "hotwords": hotwords,
        "files": {str(p.relative_to(output)): sha(p) for p in protected_files},
        "test_set_used_for_training": False,
    }
    write_json(marker, value)
    return value
