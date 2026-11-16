"""New-run selection prioritizes reusable rows and requires the full training hours."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, cast

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.zh_en_common import completed, digest, finish, rows, save_rows


def manifests(output: Path, sources: list[Path], cfg: dict[str, Any]) -> dict[str, Any]:
    folder = output / "prepared"
    identity = {"labels": [sha(p) for p in sources], "config": digest(cfg)}
    done = completed(folder, identity)
    if done is not None:
        return done
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "selection.sqlite")
    db.execute("DROP TABLE IF EXISTS records")
    db.execute(
        "CREATE TABLE records (id TEXT PRIMARY KEY, split TEXT, grp TEXT, "
        "ranking TEXT, seconds REAL, payload TEXT)"
    )
    import json

    rejects: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    quarantine = folder / "quarantine.jsonl"
    sealed = folder / "sealed_test.jsonl"
    with (
        quarantine.open("w", encoding="utf-8") as rejected,
        sealed.open("w", encoding="utf-8") as test,
    ):
        for source in sources:
            for row in rows(source):
                totals[row["group"]] += 1
                if row["split"] == "test":
                    test.write(json.dumps(row, ensure_ascii=False) + "\n")
                elif row["issues"]:
                    rejects[row["group"]] += 1
                    reasons.update(row["group"] + "/" + i.split(":", 1)[0] for i in row["issues"])
                    if len(examples) < 30:
                        examples.append({k: row[k] for k in ("id", "group", "text", "issues")})
                    rejected.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    db.execute(
                        "INSERT INTO records VALUES (?,?,?,?,?,?)",
                        (
                            row["id"],
                            row["split"],
                            row["group"],
                            rank(cfg["seed"], row["id"]),
                            row["seconds"],
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
    db.commit()
    report: dict[str, Any] = {
        "totals": totals,
        "quarantined_rows": rejects,
        "failure_reasons": reasons,
        "failure_examples": examples,
        "groups": {},
        "parts": [],
    }
    availability = {}
    for split, targets in cfg["minimum_hours"].items():
        for group, minimum in targets.items():
            count, seconds = db.execute(
                "SELECT count(*),coalesce(sum(seconds),0) FROM records WHERE split=? AND grp=?",
                (split, group),
            ).fetchone()
            availability[split + "/" + group] = {
                "records": count,
                "hours": seconds / 3600,
                "minimum_hours": minimum,
                "passed": seconds >= minimum * 3600,
            }
    report["availability"] = availability
    write_json(output / "label_report.json", report)
    if any(not v["passed"] for v in availability.values()):
        db.close()
        raise ValueError("insufficient complete labels; all six groups are in label_report.json")
    phone_counts: Counter[int] = Counter()
    files = [quarantine, sealed]

    def emit(items: list[dict[str, Any]], split: str, group: str) -> None:
        views = ("mixed",) if split == "train" else ("original", "noise10db")
        for view in views:
            for start in range(0, len(items), cfg["part_records"]):
                path = (
                    folder / split / group / view / f"part-{start // cfg['part_records']:06d}.jsonl"
                )
                chunk = []
                for original in items[start : start + cfg["part_records"]]:
                    row = {**original}
                    if split == "validation":
                        row.update(id=row["id"] + ":" + view, noisy=view == "noise10db")
                    row.update(condition=view, base_id=original["id"])
                    chunk.append(row)
                save_rows(path, chunk)
                files.append(path)
                report["parts"].append(
                    {
                        "path": str(path),
                        "sha256": sha(path),
                        "split": split,
                        "group": group,
                        "condition": view,
                        "records": len(chunk),
                    }
                )

    for split in ("train", "validation"):
        for group in ("zh", "us", "uk"):
            target = cfg["hours"][split][group]
            eligible = db.execute(
                "SELECT count(*),coalesce(sum(seconds),0) FROM records WHERE split=? AND grp=?",
                (split, group),
            ).fetchone()
            selected_rows = []
            seconds = 0.0
            for (payload,) in db.execute(
                "SELECT payload FROM records WHERE split=? AND grp=? "
                "ORDER BY CASE WHEN substr(id,1,5)='king_' THEN 1 ELSE 0 END,ranking,id",
                (split, group),
            ):
                if seconds >= target * 3600:
                    break
                row = json.loads(payload)
                selected_rows.append(row)
                seconds += row["seconds"]
            if not selected_rows:
                write_json(output / "label_report.json", report)
                db.close()
                raise ValueError(f"no usable {split}/{group}; inspect quarantine and grouping")
            # Report shortfalls rather than silently duplicating data or leaking holdout rows.
            report["groups"][split + "/" + group] = {
                "records": len(selected_rows),
                "hours": seconds / 3600,
                "target_hours": target,
                "available_hours": eligible[1] / 3600,
                "shortfall_hours": max(0, target - seconds / 3600),
            }
            if seconds < cfg["minimum_hours"][split][group] * 3600:
                write_json(output / "label_report.json", report)
                db.close()
                raise ValueError(
                    f"insufficient usable {split}/{group}: {seconds / 3600:.3f}h; "
                    f"minimum {cfg['minimum_hours'][split][group]}h"
                )
            noisy = set()
            if split == "train":
                n = math.floor(len(selected_rows) * 0.7 + 0.5)
                noisy = {
                    r["id"]
                    for r in sorted(selected_rows, key=lambda r: rank(cfg["seed"] + 1, r["id"]))[:n]
                }
            report["groups"][split + "/" + group]["noise_records"] = len(noisy)
            report["groups"][split + "/" + group]["noise_hours"] = (
                sum(r["seconds"] for r in selected_rows if r["id"] in noisy) / 3600
            )
            for row in selected_rows:
                if split == "train":
                    phone_counts.update(row["phoneme_token_ids"])
                row["noisy"] = row["id"] in noisy
            emit(selected_rows, split, group)
    db.close()
    report.update(
        status="new_183_class_labels_prepared",
        training_token_counts={str(k): v for k, v in phone_counts.items()},
        zero_frequency_tokens=[i for i in range(2, 183) if not phone_counts[i]],
        test_set_used_for_training=False,
        noise_percent=70,
        ratio_unit="nearest_records_per_source_group",
        labels_are_automatic=True,
        pronunciation_accuracy_verified=False,
    )
    write_json(output / "label_report.json", {k: v for k, v in report.items() if k != "parts"})
    return cast(dict[str, Any], finish(folder, identity, files, report))
