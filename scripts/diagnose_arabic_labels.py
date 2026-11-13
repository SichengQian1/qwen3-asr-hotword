#!/usr/bin/env python3
"""Read-only Arabic label statistics and listening-review selection."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.arabic_per_diagnostics import exposure
from qwen_hotword.training.stage_progress import stage_progress
from scripts.arabic_diagnostic_common import (
    completed_run,
    manifest_rows,
    protect_output,
    protected_run_directories,
    read,
    recheck,
    rows,
)
from scripts.label_arabic_native import SOURCES


def verified_labels(directory: Path) -> dict[str, Any]:
    report: dict[str, Any] = read(directory / "report.json")
    if report["status"] != "automatic_labels_completed":
        raise ValueError("label generation is incomplete")
    if report["identity"] != read(directory / "contract.json"):
        raise ValueError("label report and contract disagree")
    pin(directory / "labels.jsonl", report["labels_sha256"])
    return report


def stats(run: dict[str, Any], original: Path, labels: Path) -> dict[str, Any]:
    with stage_progress("Verify frozen label artifacts and inventory identities"):
        old, new = verified_labels(original), verified_labels(labels)
        if new["identity"]["source_report_sha256"] != sha(original / "report.json"):
            raise ValueError("repaired labels do not derive from the supplied original labels")
        if new["vocab_sha256"] != run["plan"]["vocabularies"]["ar"]["sha256"]:
            raise ValueError("label/Head vocabulary mismatch")
        inventory_counts: Counter[str] = Counter()
        for group, path in SOURCES.items():
            pin(path, new["identity"]["inventories"][group])
            inventory_counts.update(r["source"] for r in rows(path) if r["group"] == group)
    selected = {}
    distributions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for split in ("train", "validation"):
        for row in manifest_rows(run, split, ("ar",)):
            key = row["source"], row["source_row"]
            if key in selected:
                raise ValueError("duplicate or cross-split source row")
            selected[key] = row
            group_key = split, row["source_group"], row["source"]
            d = distributions.setdefault(
                group_key,
                {
                    "split": split,
                    "group": row["source_group"],
                    "source": row["source"],
                    "rows": 0,
                    "hours": 0.0,
                    "min_row": row["source_row"],
                    "max_row": row["source_row"],
                    "inventory_rows": inventory_counts[row["source"]],
                },
            )
            d["rows"] += 1
            d["hours"] += row["duration_seconds"] / 3600
            d["min_row"] = min(d["min_row"], row["source_row"])
            d["max_row"] = max(d["max_row"], row["source_row"])
    exposure_counts: dict[str, Counter[str]] = defaultdict(Counter)
    matched = set()
    with stage_progress(
        "Count actual selected Arabic label exposures",
        total=sum(v["ready_rows"] for v in new["by_group"].values()),
    ) as tick:
        for index, row in enumerate(rows(labels / "labels.jsonl"), 1):
            key = row["source"], row["row"]
            if key in selected:
                target = selected[key]
                if (
                    key in matched
                    or row["ids"] != target["phoneme_token_ids"]
                    or row["text"] != target["text"]
                    or row["group"] != target["source_group"]
                ):
                    raise ValueError(f"selected label identity mismatch: {key}")
                matched.add(key)
                exposure_counts[f"{target['split']}/{row['group']}"] += exposure(row)
            tick(index)
    if matched != set(selected):
        raise ValueError("selected manifest rows missing from labels")
    review = {}
    for group, directory, report in (("ar_egy", original, old), ("ar_msa", labels, new)):
        reasons: dict[str, Counter[str]] = defaultdict(Counter)
        count = 0
        source_count = 0
        markers = sorted((directory / "parts" / group).glob("*/completed.json"))
        with stage_progress(
            f"Inspect {group} saved review reasons", total=len(markers), every=20
        ) as tick:
            for index, marker in enumerate(markers, 1):
                piece = read(marker)
                relative = Path(piece["review"])
                if relative.is_absolute() or ".." in relative.parts or piece["group"] != group:
                    raise ValueError("invalid label part")
                pin(directory / relative, piece["review_sha256"])
                part_count = 0
                for row in rows(directory / relative):
                    reason, _, value = row["label_issue"].partition(":")
                    with suppress(ValueError, SyntaxError):
                        value = str(ast.literal_eval(value)) if value else ""
                    reasons[reason][value] += 1
                    part_count += 1
                if part_count != piece["rows"] - piece["ready_rows"]:
                    raise ValueError("review part count mismatch")
                count += part_count
                source_count += piece["rows"]
                tick(index)
        if (
            count != report["by_group"][group]["review_rows"]
            or source_count != report["by_group"][group]["source_rows"]
        ):
            raise ValueError("review parts do not cover complete source")
        review[group] = {
            "review_rows": count,
            "reason_counts": {k: sum(v.values()) for k, v in reasons.items()},
            "details": {k: v.most_common(200) for k, v in reasons.items()},
        }
    recheck(run)
    return {
        "status": "completed",
        "source_identity": run["identity"],
        "label_reports": {str(d): sha(d / "report.json") for d in (original, labels)},
        "labels_sha256": new["labels_sha256"],
        "review": review,
        "source_distribution": list(distributions.values()),
        "selected_label_exposure": dict(exposure_counts),
        "matched_rows": len(matched),
        "interpretation": (
            "D6 heuristics overlap; estimated units are not measured phoneme errors. "
            "Review values describe the first blocking issue only."
        ),
        "audio_read": False,
        "test_set_used": False,
    }


def select_review(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in samples:
        if row.get("language") == "ar":
            groups[row["source_group"]].append(row)
    chosen: list[dict[str, Any]] = []
    for _, part in sorted(groups.items()):
        worst = sorted(part, key=lambda r: (-r["phoneme_error_rate"], r["sample_id"]))[:50]
        seen = {r["sample_id"] for r in worst}
        hashed = sorted(
            (r for r in part if r["sample_id"] not in seen),
            key=lambda r: (r["audio_sha256"], r["sample_id"]),
        )[:50]
        chosen.extend({**r, "selection": "worst_per"} for r in worst)
        chosen.extend({**r, "selection": "audio_sha256_order"} for r in hashed)
    return chosen


def review_sheet(per_sample: Path, labels: Path, output: Path) -> None:
    protect_output(output, [labels])
    output.mkdir(parents=True, exist_ok=False)
    report = verified_labels(labels)
    chosen = select_review(list(rows(per_sample)))
    lookup = {(r["source"], r["source_row"]): r for r in chosen}
    with stage_progress("Join selected listening rows to original diacritization"):
        for row in rows(labels / "labels.jsonl"):
            target = lookup.get((row["source"], row["row"]))
            if target is not None:
                if target["text"] != row["text"]:
                    raise ValueError("listening reference mismatch")
                target["diacritized_words"] = row["diacritized_words"]
    fields = (
        "sample_id",
        "source_group",
        "source",
        "source_row",
        "audio_path",
        "audio_sha256",
        "duration_seconds",
        "text",
        "diacritized_words",
        "reference",
        "hypothesis",
        "phoneme_error_rate",
        "selection",
    )
    with (output / "review.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in chosen:
            if "diacritized_words" not in row:
                raise ValueError("listening sample missing from label run")
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v
                    for k, v in row.items()
                }
            )
    write_json(
        output / "report.json",
        {
            "samples": len(chosen),
            "per_sample_sha256": sha(per_sample),
            "labels_sha256": report["labels_sha256"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stats", "review-sheet"))
    parser.add_argument("--source-run", type=Path, default=ROOT / "outputs/en_ar_r15_5100_ctc_v1")
    parser.add_argument(
        "--original-labels", type=Path, default=ROOT / "outputs/arabic_native_labels_v1"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "outputs/arabic_native_labels_caphi_y_v2"
    )
    parser.add_argument("--per-sample", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protect_output(args.output_dir, [args.source_run, args.original_labels, args.labels])
    if args.action == "review-sheet":
        if args.per_sample is None:
            parser.error("review-sheet requires --per-sample")
        review_sheet(args.per_sample, args.labels, args.output_dir)
    else:
        run = completed_run(args.source_run)
        protect_output(args.output_dir, protected_run_directories(run))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = stats(run, args.original_labels, args.labels)
        if len(json.dumps(report, ensure_ascii=False, indent=2).encode()) > 199_000:
            raise ValueError("summary exceeds compact return budget")
        write_json(args.output_dir / "labels_report.json", report)
    print(
        json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
