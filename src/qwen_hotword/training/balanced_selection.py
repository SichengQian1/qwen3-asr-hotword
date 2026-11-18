"""Select complete utterances by duration, independently inside each frozen split."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import finish, rows, save_rows


def select(output: Path, paths: list[Path], cfg: dict[str, Any]) -> dict[str, Any]:
    roots = {str(Path(v)): k for k, v in cfg["chinese_roots"].items()}
    pools: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    excluded: Counter[str] = Counter()
    with stage_progress("Count eligible hours by split and source", total=len(paths)) as tick:
        for index, path in enumerate(paths, 1):
            for row in rows(path):
                if row["split"] not in ("train", "validation") or row["issues"]:
                    excluded[row["split"] + "/" + row["group"]] += 1
                    continue
                group = row["group"]
                if group == "zh":
                    group = row.get("source_group") or roots[str(Path(row["source"]).parent)]
                pools[row["split"], group].append((row["id"], row["seconds"]))
            tick(index)
    chosen: dict[str, tuple[str, bool]] = {}
    stats: dict[str, Any] = {}
    for split, requested in cfg["quotas"].items():
        available = {g: sum(s for _, s in pools[split, g]) / 3600 for g in requested}
        scale = 1.0
        if split == "train" and cfg["scale_to_us"]:
            scale = min(1.0, available["us"] / requested["us"])
        if split == "validation":
            scale = min(1.0, *(available[g] / h for g, h in requested.items()))
        for group, nominal in requested.items():
            target = nominal * scale
            if target <= 0 or available[group] + 1e-8 < target:
                raise ValueError(f"{split}/{group}: need {target:.3f}h, available {available}")
            selected = []
            seconds = 0.0
            for uid, duration in sorted(pools[split, group], key=lambda x: rank(cfg["seed"], x[0])):
                if seconds >= target * 3600:
                    break
                selected.append(uid)
                seconds += duration
            n = int(len(selected) * 0.7 + 0.5) if split == "train" else 0
            noisy = set(sorted(selected, key=lambda uid: rank(cfg["seed"] + 1, uid))[:n])
            for uid in selected:
                if uid in chosen:
                    raise ValueError(f"duplicate selected id {uid}")
                chosen[uid] = (group, uid in noisy)
            stats[f"{split}/{group}"] = dict(
                requested_hours=nominal,
                target_hours=target,
                hours=seconds / 3600,
                available_hours=available[group],
                records=len(selected),
                noisy_records=n,
            )
    del pools
    folder = output / "prepared"
    buffers: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    parts: list[dict[str, Any]] = []
    files: list[Path] = []

    def flush(key: tuple[str, str, str]) -> None:
        split, source, view = key
        buffer = buffers[key]
        dest = folder / split / source / view / f"part-{len(parts):06d}.jsonl"
        save_rows(dest, buffer)
        parts.append(
            dict(
                path=str(dest),
                sha256=sha(dest),
                split=split,
                group=buffer[0]["group"],
                source_group=source,
                condition=view,
                records=len(buffer),
            )
        )
        files.append(dest)
        buffer.clear()

    seen_pcm: set[str] = set()
    with stage_progress(
        "Write balanced training and paired validation manifests", total=len(paths)
    ) as tick:
        for index, path in enumerate(paths, 1):
            for row in rows(path):
                uid = row["id"]
                if uid not in chosen:
                    continue
                group, noisy_flag = chosen.pop(uid)
                pcm = row["pcm_sha256"]
                if not pcm or pcm in seen_pcm:
                    raise ValueError(f"missing or duplicate selected PCM at {uid}")
                seen_pcm.add(pcm)
                split = row["split"]
                for view in ("mixed",) if split == "train" else ("original", "noise10db"):
                    key = (split, group, view)
                    buffers[key].append(
                        dict(
                            row,
                            base_id=uid,
                            source_group=group,
                            condition=view,
                            noisy=noisy_flag or view == "noise10db",
                            id=uid if split == "train" else f"{uid}:{view}",
                        )
                    )
                    if len(buffers[key]) == cfg["part_records"]:
                        flush(key)
            tick(index)
    if chosen:
        raise ValueError("selected rows missing during manifest export")
    for key in buffers:
        if buffers[key]:
            flush(key)
    summary = dict(
        status="balanced_labels_prepared",
        groups=stats,
        excluded=excluded,
        parts=parts,
        test_used_for_training=False,
    )
    write_json(output / "label_report.json", dict(groups=stats, excluded=excluded))
    finish(folder, {"quotas": cfg["quotas"], "seed": cfg["seed"]}, files, summary)
    return summary
