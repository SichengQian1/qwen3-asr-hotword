"""Add a Chinese corpus while preserving verified historical labels and holdouts."""

from __future__ import annotations

import os
import sqlite3
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import completed, digest, finish, read, rows, save_rows
from qwen_hotword.training.zh_en_data import chinese_rows, inspect_audio


def cpu_limit(requested: int) -> int:
    if not 1 <= requested <= 64:
        raise ValueError("CPU worker ceiling must be 1..64")
    affinity = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    )
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        limit, period = quota.read_text().split()
        if limit != "max":
            affinity = min(affinity, max(1, int(limit) // int(period)))
    return min(requested, affinity)


def legacy_parts(
    root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]
) -> list[Path]:
    source = (root / cfg["source_run"]).resolve()
    pin(source / "contract.json", cfg["source_contract_sha256"])
    old = read(source / "contract.json")
    if old["foundation"] != foundation:
        raise ValueError("legacy pronunciation foundation differs")
    for key in (
        "frontend_policy",
        "frontend_packages",
        "seed",
        "head",
        "num_classes",
        "model",
        "audio_fallbacks",
    ):
        if old["config"][key] != cfg[key]:
            raise ValueError("incompatible legacy label policy: " + key)
    for name, value in old["code_sha256"].items():
        pin(root / name, value)
    g2p_sha = sha(source / "g2p_report.json")
    paths = sorted((source / "labels").glob("*/rows.jsonl"))
    counts: Counter[str] = Counter()
    identities = {}
    with stage_progress(
        "Verify and reuse old completed label parts; no G2P rerun", total=len(paths), every=50
    ) as tick:
        for i, path in enumerate(paths, 1):
            marker = read(path.parent / "completed.json")
            if marker["identity"]["g2p_report"] != g2p_sha:
                raise ValueError("legacy G2P/label identity mismatch")
            pin(path, marker["files"]["rows.jsonl"])
            count = 0
            for row in rows(path):
                counts[row["group"]] += 1
                count += 1
            if count != marker["summary"]["rows"]:
                raise ValueError("legacy label part incomplete")
            identities[str(path)] = sha(path)
            tick(i)
    if dict(counts) != read(source / "label_report.json")["totals"]:
        raise ValueError("legacy full label population is incomplete")
    write_json(
        output / "legacy_reuse.json",
        {
            "source": str(source),
            "rows": counts,
            "label_parts": identities,
            "source_contract_sha256": cfg["source_contract_sha256"],
            "source_modified": False,
            "g2p_report_sha256": g2p_sha,
        },
    )
    return paths


def new_inventory(output: Path, cfg: dict[str, Any]) -> list[Path]:
    sources = sorted(Path(cfg["added_chinese_root"]).glob("*.tar"))
    if not sources:
        raise ValueError("new Chinese root has no supported uncompressed .tar files")
    result = []
    with (
        ThreadPoolExecutor(max_workers=cpu_limit(cfg["cpu_worker_limit"])) as pool,
        stage_progress(
            "Audit added Chinese tar audio; preserve completed parts", total=len(sources), every=1
        ) as tick,
    ):
        identities = {}
        for index, source in enumerate(sources):
            source_sha = sha(source)
            identities[str(source)] = source_sha
            batch = []
            part = 0

            def flush(
                values: list[dict[str, Any]],
                part: int,
                index: int = index,
                source: Path = source,
                source_sha: str = source_sha,
            ) -> None:
                folder = output / "new_inventory" / f"{index:05d}-{part:05d}"
                identity = {"source": str(source), "sha256": source_sha, "rows": digest(values)}
                if completed(folder, identity) is None:
                    save_rows(folder / "rows.jsonl", pool.map(inspect_audio, values))
                    finish(folder, identity, [folder / "rows.jsonl"], {"rows": len(values)})
                result.append(folder / "rows.jsonl")

            seen = 0
            for row in chinese_rows(source, index):
                seen += 1
                row["id"] = "king_" + row["id"]
                row["family"] = "king:" + row["family"]
                row["origin"] = "added_chinese"
                batch.append(row)
                if len(batch) == cfg["part_records"]:
                    flush(batch, part)
                    batch, part = [], part + 1
            if batch:
                flush(batch, part)
            if not seen:
                raise ValueError(f"no WAV/text pairs found in {source}; schema unsupported")
            tick(index + 1)
    destination = output / "added_source_identity.json"
    if destination.exists() and read(destination) != identities:
        raise ValueError("added Chinese source inventory changed; preserve this run")
    write_json(destination, identities)
    return result


def merge_splits(
    output: Path, old: list[Path], added: list[Path], seed: int
) -> tuple[list[Path], list[Path]]:
    """Old holdouts stay fixed. New duplicate-linked families inherit or are quarantined."""
    db = sqlite3.connect(output / "merged_duplicates.sqlite")
    db.executescript(
        "DROP TABLE IF EXISTS pcm; CREATE TABLE pcm "
        "(sha TEXT PRIMARY KEY, text TEXT, first_id TEXT, family TEXT, "
        "old_split TEXT, conflict INTEGER);"
    )
    parent: dict[str, str] = {}
    anchors: dict[str, set[str]] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    with stage_progress("Check cross-source PCM duplicates and preserve prior holdouts") as tick:
        number = 0
        for is_old, paths in ((True, old), (False, added)):
            for path in paths:
                for row in rows(path):
                    number += 1
                    pcm = row.get("pcm_sha256")
                    if pcm is None:
                        continue
                    family = row["family"]
                    if not is_old:
                        find(family)
                    previous = db.execute(
                        "SELECT text,first_id,family,old_split FROM pcm WHERE sha=?", (pcm,)
                    ).fetchone()
                    if previous:
                        if previous[0] != row["text"]:
                            db.execute("UPDATE pcm SET conflict=1 WHERE sha=?", (pcm,))
                        if not is_old:
                            if previous[3] is not None:
                                anchors.setdefault(family, set()).add(previous[3])
                            else:
                                union(family, previous[2])
                    else:
                        db.execute(
                            "INSERT INTO pcm VALUES (?,?,?,?,?,0)",
                            (pcm, row["text"], row["id"], family, row["split"] if is_old else None),
                        )
                tick(number)
        db.commit()
    anchored: dict[str, set[str]] = {}
    for family, splits in anchors.items():
        anchored.setdefault(find(family), set()).update(splits)
    outputs: list[list[Path]] = [[], []]
    stats: dict[str, Any] = {}
    total_hours: Counter[str] = Counter()
    for target_index, paths in enumerate((old, added)):
        for i, path in enumerate(paths):
            destination = (
                output
                / ("legacy_labels" if target_index == 0 else "new_split")
                / f"part-{i:06d}.jsonl"
            )

            def convert(
                path: Path = path, is_old: bool = target_index == 0
            ) -> Iterator[dict[str, Any]]:
                for row in rows(path):
                    row["origin"] = "legacy" if is_old else "added_chinese"
                    pcm = row.get("pcm_sha256")
                    prior = (
                        db.execute(
                            "SELECT first_id,conflict FROM pcm WHERE sha=?", (pcm,)
                        ).fetchone()
                        if pcm
                        else None
                    )
                    if not is_old:
                        family = find(row["family"])
                        fixed = anchored.get(family, set())
                        fraction = int(rank(seed, family)[:8], 16) / 2**32
                        row["split"] = (
                            next(iter(fixed))
                            if len(fixed) == 1
                            else (
                                "test"
                                if fraction < 0.02
                                else "validation"
                                if fraction < 0.04
                                else "train"
                            )
                        )
                        row["split_group"] = family
                        if len(fixed) > 1 or "quarantine" in fixed:
                            row["issues"].append("new_family_links_incompatible_legacy_splits")
                        if prior and prior[0] != row["id"]:
                            row["issues"].append("cross_source_or_new_duplicate_pcm")
                    if prior and prior[1]:
                        row["issues"].append("merged_pcm_conflicting_text")
                    key = row["group"] + "/" + row["split"]
                    entry = stats.setdefault(
                        key, {"rows": 0, "known_hours": 0.0, "eligible_before_new_g2p_hours": 0.0}
                    )
                    entry["rows"] += 1
                    hours = row.get("seconds", 0.0) / 3600
                    entry["known_hours"] += hours
                    if not row["issues"]:
                        entry["eligible_before_new_g2p_hours"] += hours
                    total_hours[row["origin"] + "/" + row["group"]] += hours
                    yield row

            save_rows(destination, convert())
            outputs[target_index].append(destination)
    db.close()
    write_json(
        output / "merged_source_report.json",
        {
            "by_split": stats,
            "known_hours": total_hours,
            "old_splits_preserved": True,
            "near_duplicates_checked": False,
            "speaker_disjoint_verified": False,
        },
    )
    return outputs[0], outputs[1]
