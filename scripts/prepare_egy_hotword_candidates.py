#!/usr/bin/env python3
"""Review candidate terms first; never run Top5/Top7 or open held-out manifests."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation.egy_hotword_review import (
    load_candidates,
    scan_training,
    verify_run,
    write_review,
)
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.stage_progress import stage_progress


def package(output: Path) -> None:
    archive = output / "return_review.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (
            "report.json",
            "candidates.json",
            "candidates.tsv",
            "nested_pairs.json",
            "候选词表.md",
        ):
            bundle.write(output / name, name)
    payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    size = 30 * 1024
    parts = (len(payload) + size - 1) // size
    for index in range(parts):
        header = {
            "encoding": "base64",
            "archive": archive.name,
            "archive_sha256": sha(archive),
            "part": index + 1,
            "parts": parts,
        }
        (output / f"return_part_{index + 1:03d}.txt").write_text(
            json.dumps(header) + "\n" + payload[index * size : (index + 1) * size] + "\n",
            encoding="ascii",
        )
    write_json(
        output / "return_files.json",
        {
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha(archive),
            "text_parts": parts,
        },
    )


def build(root: Path, config_path: Path, output: Path, *, preview: bool) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate_path = root / config["candidate_file"]
    if output.exists():
        raise ValueError("output exists; choose a new directory; existing reviews are preserved")
    candidates = load_candidates(candidate_path)
    identity = {
        str(p.resolve()): sha(p)
        for p in (
            config_path,
            candidate_path,
            Path(__file__),
            root / "src/qwen_hotword/evaluation/egy_hotword_review.py",
        )
    }
    stats: dict[str, Any] = {}
    if not preview:
        with stage_progress("Verify new EGY x3 epoch29 and frozen TRAIN manifest; no audio reads"):
            source = verify_run(root, config)
        identity.update(source["files"])
        with (
            stage_progress(
                "Count candidate evidence in TRAIN text only", total=source["expected_rows"]
            ) as tick,
            Path(source["train_manifest"]).open(encoding="utf-8") as stream,
        ):
            candidates, stats = scan_training(
                (json.loads(line) for line in stream), candidates, tick
            )
        if stats["rows"] != source["expected_rows"]:
            raise ValueError("training row count mismatch")
    report = {
        "status": "candidate_review_ready" if not preview else "editorial_preview_only",
        "corpus_scan_completed": not preview,
        "training_counts": stats,
        "input_identity": identity,
        "selected_model": config["expected"],
        "future_evaluation": config["future_evaluation"],
        "selection_basis": "curated entities and technical terms; no frequency filtering",
        "matching_policy": "NFC_strip_marks_tatweel; exact tokens; punctuation blocks phrases; "
        "attached prefixes are review-only; no alef/ya/ta merging",
        "test_coverage": "not_measured",
        "phoneme_coverage": "not_measured",
    }
    with stage_progress("Recheck inputs and write candidate review; NO retrieval"):
        for path, digest in identity.items():
            pin(Path(path), digest)
        write_review(output, candidates, report)
        package(output)
    return {
        "status": report["status"],
        "output_dir": str(output),
        "candidates": len(candidates),
        "training_counts": stats,
        "return_file": str(output / "return_review.zip"),
        "retrieval_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true", help="editorial draft only, no corpus access"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/egy_hotword_review.workzone.json"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        output = args.output_dir.resolve()
    else:
        base = ROOT / "outputs"
        base.mkdir(exist_ok=True)
        # Reserve a unique parent, leaving the review itself new for no-overwrite semantics.
        output = Path(tempfile.mkdtemp(prefix="egy_hotword_review_", dir=base)) / "review"
    try:
        result = build(ROOT, args.config.resolve(), output, preview=args.preview)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
