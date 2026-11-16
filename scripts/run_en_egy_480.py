#!/usr/bin/env python3
"""Continue completed EGY labels with the existing r15 English pool and a 480h/language cap."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.en_egy_dynamic import freeze, read, save_rows, seal
from qwen_hotword.training.stage_progress import stage_progress
from scripts.arabic_diagnostic_common import completed_run
from scripts.label_arabic_native import exclusive_job
from scripts.run_en_egy_dynamic import recheck

REVIEWED_ORIGINAL = {
    "src/qwen_hotword/training/en_egy_dynamic.py": (
        "8a38ea0ad284aa93661aba49c30e3d3dd234691391a7993f01a439441cb9d88b"
    ),
}


def verify_previous(root: Path, previous: Path) -> dict[str, Any]:
    contract = read(previous / "contract.json")
    for name, checksum in contract["files"].items():
        path = Path(name)
        relative = str(path.relative_to(root)) if path.is_relative_to(root) else ""
        if relative in REVIEWED_ORIGINAL:
            if checksum != REVIEWED_ORIGINAL[relative]:
                raise ValueError("unreviewed source preparation code: " + name)
        else:
            pin(path, checksum)
    for item in contract["source_identity"].values():
        pin(Path(item["path"]), item["sha256"])
    for name, checksum in read(previous / "admission_report.json")["files"].items():
        pin(previous / name, checksum)
    labels = read(previous / "labels/report.json")
    cfg = contract["configuration"]
    if (
        labels["status"] != "automatic_labels_completed"
        or labels["MSA_used"] is not False
        or labels["vocab_sha256"] != cfg["vocab_sha256"]
        or labels["policy_sha256"] != sha(root / cfg["pronunciation_policy"])
    ):
        raise ValueError("completed matching EGY labels required")
    return dict(contract)


def prepare(root: Path, previous: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    for p in (previous, root / "outputs/en_ar_r15_5100_ctc_v1"):
        if output == p or output in p.parents or p in output.parents:
            raise ValueError("new output must be separate from existing runs")
    with stage_progress(
        "Verify completed EGY labels and reviewed source identities; no relabeling"
    ):
        original = verify_previous(root, previous)
        cfg = {
            **original["configuration"],
            "maximum_hours_per_language": 480,
            "english_pool_policy": "existing_r15_train_only",
            "output_dir": str(output),
        }
        source = completed_run(root / cfg["source_run"])
        if source["identity"] != original["source_identity"]:
            raise ValueError("r15 source run identity changed")
        protected_dirs = [
            Path(item["path"]).parent for item in source["plan"]["manifests"].values()
        ]
        for p in protected_dirs:
            if output == p or output in p.parents or p in output.parents:
                raise ValueError("output overlaps source manifests/cache")
        files = {name: sha(Path(name)) for name in original["files"]}
        for p in (
            Path(__file__),
            root / "src/qwen_hotword/training/verified_audio_hashes.py",
            previous / "contract.json",
            previous / "labels/report.json",
            previous / "admission_report.json",
            previous / "protected.jsonl",
        ):
            files[str(p)] = sha(p)
        identity = {
            "configuration": cfg,
            "files": files,
            "source_identity": source["identity"],
            "inherited_labels": str(previous / "labels"),
            "schema_version": 1,
        }
        if output.exists() and not (output / "contract.json").exists():
            raise ValueError("existing unsealed output; choose a new directory")
        output.mkdir(parents=True, exist_ok=True)
        seal(output / "contract.json", identity)
    marker = output / "import_report.json"
    if marker.exists():
        for name, checksum in read(marker)["files"].items():
            pin(output / name, checksum)
        if (output / "labels").resolve() != (previous / "labels").resolve():
            raise ValueError("inherited label directory changed")
        return cfg, source
    with stage_progress(
        "Reuse all completed Arabic labels; restrict English to old r15 training manifest"
    ):
        labels = output / "labels"
        if not labels.exists():
            labels.symlink_to(previous / "labels", target_is_directory=True)
        if labels.resolve() != (previous / "labels").resolve():
            raise ValueError("labels destination already contains different data")
        english = []
        manifest = Path(source["plan"]["manifests"]["en/train"]["path"])
        for i, row in enumerate(iter_jsonl(manifest), 1):
            group = row["source_group"]
            if group not in {"en_egy", "en_us_donor", "en_us"}:
                raise ValueError("unexpected old English source group: " + group)
            english.append(
                {
                    **row,
                    "ids": row["phoneme_token_ids"],
                    "row": row.get("source_row", i),
                    "bucket": "en_egy" if group == "en_egy" else "en_us",
                }
            )
        validation = []
        manifest = Path(source["plan"]["manifests"]["en/validation"]["path"])
        for i, row in enumerate(iter_jsonl(manifest), 1):
            validation.append({**row, "bucket": "en_validation", "row": row.get("source_row", i)})
        save_rows(output / "english_ready.jsonl", english)
        save_rows(output / "english_validation.jsonl", validation)
        save_rows(output / "protected.jsonl", iter_jsonl(previous / "protected.jsonl"))
        write_json(
            output / "english_labels_report.json",
            {
                "status": "reused_frozen_r15_English_labels",
                "ready_rows": len(english),
                "ready_hours_before_cap": sum(r["duration_seconds"] for r in english) / 3600,
                "new_G2P_run": False,
                "expanded_English_pool_used": False,
            },
        )
        # SQLite backup is consistent even with a WAL; do not copy a live .sqlite byte file.
        db_path = output / "audio_hashes.sqlite"
        if not db_path.exists() and (previous / "audio_hashes.sqlite").exists():
            pending = output / "audio_hashes.sqlite.pending"
            with (
                sqlite3.connect(
                    (previous / "audio_hashes.sqlite").as_uri() + "?mode=ro", uri=True
                ) as old_db,
                sqlite3.connect(pending) as new_db,
            ):
                old_db.backup(new_db)
            pending.replace(db_path)
        write_json(
            marker,
            {
                "status": "imported",
                "arabic_labels_recomputed": False,
                "previous": str(previous),
                "maximum_hours_per_language": 480,
                "files": {
                    name: sha(output / name)
                    for name in (
                        "english_ready.jsonl",
                        "english_validation.jsonl",
                        "protected.jsonl",
                        "english_labels_report.json",
                    )
                },
            },
        )
    return cfg, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-run", type=Path, default=ROOT / "outputs/en_egy_dynamic_ctc_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/en_egy_480_ctc_v1")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--io-workers", type=int, default=32)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.io_workers <= 64 or args.gpu < 0:
        parser.error("io-workers must be 1..64; GPU must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    previous, output = args.from_run.resolve(), args.output_dir.resolve()
    if output == previous or output in previous.parents or previous in output.parents:
        parser.error("output must be independent of source")
    try:
        with exclusive_job(output):
            with exclusive_job(previous):
                cfg, source = prepare(ROOT, previous, output)
            prepared = freeze(ROOT, cfg, output, args.io_workers)
            recheck(output)
            print(__import__("json").dumps(prepared, ensure_ascii=False, indent=2), flush=True)
            if not args.prepare_only:
                from qwen_hotword.training.en_egy_dynamic_train import run

                run(ROOT, cfg, output, source, args.gpu)
                recheck(output)
            with zipfile.ZipFile(
                output / "return_summary.zip", "w", zipfile.ZIP_DEFLATED
            ) as archive:
                for name in (
                    "import_report.json",
                    "contract.json",
                    "selection_contract.json",
                    "audio_duration_report.json",
                    "capacity_report.json",
                    "prepare_report.json",
                    "english_labels_report.json",
                    "feature_reuse.json",
                    "head/report.json",
                    "report.json",
                ):
                    if (output / name).exists():
                        archive.write(output / name, name)
                summary = read(output / "labels/report.json")
                summary.pop("pieces")
                write_json(output / "arabic_labels_summary.json", summary)
                archive.write(output / "arabic_labels_summary.json", "arabic_labels_summary.json")
            print(f"Return summary: {output / 'return_summary.zip'}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(
            __import__("json").dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
