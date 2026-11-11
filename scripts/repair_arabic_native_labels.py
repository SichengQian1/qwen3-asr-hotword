#!/usr/bin/env python3
"""Retry only saved CAPHI-y OOV rejects, preserving the original Arabic label run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
from qwen_hotword.phonemes.arabic_native import CalimaFrontend
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.experiment_a import ctc_minimum_input_length
from qwen_hotword.training.stage_progress import stage_progress
from scripts.label_arabic_native import CONFIG, VOCAB, exclusive_job, verify_assets

OLD_BACKEND_SHA = "2d0f0349436ceb4a2618bb61d105e1225a719cbc4db62481c4ba76ae516cfc0f"
RETRY_ISSUES = {"phone_OOV:'ar_ipa:y'", "phone_OOV:'ar_ipa:yː'"}
REVIEWED_SERIAL_SHA = "7bfa6321e9036929e4a335a652f85d2d7c5c617c55b53df4da1afd31081a7d31"
_WORKER_SOURCE = Path()
_WORKER_OUTPUT = Path()
_WORKER_CONFIG: dict[str, Any] = {}
_WORKER_TOKENS: list[str] = []
_WORKER_FRONTEND: CalimaFrontend | None = None


def _silent_tick(_value: int) -> None:
    pass


def recovered_record(
    row: dict[str, Any], frontend: CalimaFrontend, vocab_sha256: str
) -> dict[str, Any]:
    generated = frontend.label(row["text"])
    minimum = ctc_minimum_input_length(generated["ids"])
    frames = qwen3_asr_audio_output_length(round(row["duration_seconds"] * 16000) // 160)
    if minimum > 2 * frames:
        raise ValueError("estimated_CTC_length_infeasible")
    return {
        **{
            key: row[key]
            for key in ("group", "source", "row", "text", "audio_path", "duration_seconds")
        },
        **generated,
        "ctc_minimum_input_length": minimum,
        "vocab_sha256": vocab_sha256,
    }


def repair_part(
    source: Path,
    output: Path,
    marker: Path,
    config: dict[str, Any],
    tokens: list[str],
    frontend: CalimaFrontend | None,
    *,
    progress: bool = True,
) -> tuple[dict[str, Any], CalimaFrontend | None]:
    old = json.loads(marker.read_text(encoding="utf-8"))
    for key in ("labels", "review"):
        relative = Path(old[key])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid source part path")
        pin(source / relative, old[key + "_sha256"])
    folder = output / marker.parent.relative_to(source)
    completed = folder / "completed.json"
    if completed.exists():
        piece = json.loads(completed.read_text(encoding="utf-8"))
        if piece["source_piece_sha256"] != sha(marker):
            raise ValueError("source part changed since repair")
        for key in ("labels", "review"):
            pin(output / piece[key], piece[key + "_sha256"])
        return piece, frontend
    folder.mkdir(parents=True, exist_ok=True)
    ready, hours = old["ready_rows"], old["ready_hours"]
    reasons: Counter[str] = Counter()
    retried: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    recovered = 0
    recovered_hours = 0.0
    total = old["rows"] - ready
    with (
        (folder / "labels.pending").open("wb") as labels,
        (folder / "review.pending").open("w", encoding="utf-8") as review,
        (
            stage_progress(f"Reuse/retry {old['group']} part {marker.parent.name}", total=total)
            if progress
            else nullcontext(_silent_tick)
        ) as tick,
    ):
        with (source / old["labels"]).open("rb") as existing:
            shutil.copyfileobj(existing, labels, length=1024 * 1024)
        processed = 0
        for processed, row in enumerate(iter_jsonl(source / old["review"]), 1):
            original_issue = row["label_issue"]
            if old["group"] == "ar_msa" and original_issue in RETRY_ISSUES:
                retried[original_issue] += 1
                if frontend is None:
                    with stage_progress("Load pinned MSA CALIMA for CAPHI-y rejects only"):
                        frontend = CalimaFrontend(ROOT / config["assets"], "ar_msa", tokens)
                try:
                    record = recovered_record(row, frontend, old["vocab_sha256"])
                except ValueError as error:
                    row["label_issue"] = str(error)
                    failures[str(error).split(":", 1)[0]] += 1
                else:
                    labels.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
                    recovered += 1
                    recovered_hours += row["duration_seconds"] / 3600
                    tick(processed)
                    continue
            reasons[row["label_issue"].split(":", 1)[0]] += 1
            review.write(json.dumps(row, ensure_ascii=False) + "\n")
            tick(processed)
        if processed != total:
            raise ValueError("source review count differs from completed part")
    for key in ("labels", "review"):
        (folder / f"{key}.pending").replace(folder / f"{key}.jsonl")
    piece = {
        **old,
        "ready_rows": ready + recovered,
        "ready_hours": hours + recovered_hours,
        "reasons": dict(reasons),
        "source_piece_sha256": sha(marker),
        "retried_issues": dict(retried),
        "retry_failures": dict(failures),
        "recovered_rows": recovered,
        "recovered_hours": recovered_hours,
        **{key + "_sha256": sha(folder / f"{key}.jsonl") for key in ("labels", "review")},
    }
    write_json(completed, piece)
    return piece, frontend


def _initialize_worker(
    source: Path, output: Path, config: dict[str, Any], tokens: list[str]
) -> None:
    global _WORKER_SOURCE, _WORKER_OUTPUT, _WORKER_CONFIG, _WORKER_TOKENS, _WORKER_FRONTEND
    _WORKER_SOURCE, _WORKER_OUTPUT = source, output
    _WORKER_CONFIG, _WORKER_TOKENS = config, tokens
    _WORKER_FRONTEND = None


def _repair_worker(marker: Path) -> dict[str, Any]:
    global _WORKER_FRONTEND
    piece, _WORKER_FRONTEND = repair_part(
        _WORKER_SOURCE,
        _WORKER_OUTPUT,
        marker,
        _WORKER_CONFIG,
        _WORKER_TOKENS,
        _WORKER_FRONTEND,
        progress=False,
    )
    return piece


def repair_pieces(
    config: dict[str, Any], source: Path, output: Path, markers: list[Path], *, workers: int
) -> list[dict[str, Any]]:
    tokens = json.loads(VOCAB.read_text())["tokens"]
    if workers == 1:
        frontend = None
        pieces = []
        for index, marker in enumerate(markers, 1):
            piece, frontend = repair_part(source, output, marker, config, tokens, frontend)
            pieces.append(piece)
            print(f"Serial repair: {index:,}/{len(markers):,} parts completed", flush=True)
        return pieces
    completed: dict[Path, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(source, output, config, tokens),
    )
    try:
        with stage_progress(
            f"Parallel Arabic repair ({workers} CPU processes)", total=len(markers), every=1
        ) as tick:
            pending = {pool.submit(_repair_worker, marker): marker for marker in markers}
            for future in as_completed(pending):
                marker = pending[future]
                piece = future.result()
                completed[marker] = piece
                counts[piece["group"]] += piece["rows"]
                tick(len(completed))
                print(
                    f"{piece['group']}: checked {counts[piece['group']]:,} source rows; "
                    f"part {marker.parent.name} recovered {piece['recovered_rows']:,} rows",
                    flush=True,
                )
    finally:
        # On failure/interruption, cancel queued work; completed part markers remain reusable.
        pool.shutdown(wait=True, cancel_futures=True)
    # Completion order must never change label order, IDs, counts or manifest SHA256.
    return [completed[marker] for marker in markers]


def repair(
    config: dict[str, Any], source: Path, output: Path, *, resume: bool, workers: int = 1
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and repair output must be separate directories")
    report_path = source / "report.json"
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    original_identity = previous["identity"]
    if (
        previous["status"] != "automatic_labels_completed"
        or previous["vocab_sha256"] != sha(VOCAB)
        or original_identity["backend_sha256"] != OLD_BACKEND_SHA
        or original_identity["config_sha256"] != sha(CONFIG)
        or original_identity["script_sha256"] != sha(ROOT / "scripts/label_arabic_native.py")
    ):
        raise ValueError("source is not the reviewed original CAPHI-y label run")
    with stage_progress("Verify original report, contract and complete label SHA256"):
        if json.loads((source / "contract.json").read_text()) != original_identity:
            raise ValueError("source report and contract disagree")
        pin(source / "labels.jsonl", previous["labels_sha256"])
    identity = {
        **original_identity,
        "script_sha256": sha(Path(__file__)),
        "backend_sha256": sha(ROOT / "src/qwen_hotword/phonemes/arabic_native.py"),
        "source_report_sha256": sha(report_path),
        "source_contract_sha256": sha(source / "contract.json"),
        "source_labels_sha256": previous["labels_sha256"],
        "repair_policy": "retry_only_saved_MSA_CAPHI_y_OOV_with_y_to_IPA_j",
    }
    output.mkdir(parents=True, exist_ok=resume)
    contract = output / "contract.json"
    if resume:
        recorded = json.loads(contract.read_text())
        if recorded != identity:
            comparable = {**recorded, "script_sha256": identity["script_sha256"]}
            if recorded.get("script_sha256") != REVIEWED_SERIAL_SHA or comparable != identity:
                raise ValueError("repair inputs or code changed; use a new output directory")
            print(
                "Reviewed serial-to-parallel upgrade: "
                "original contract and completed parts retained",
                flush=True,
            )
            identity = recorded
    else:
        write_json(contract, identity)
    markers = [
        marker
        for group in previous["by_group"]
        for marker in sorted((source / "parts" / group).glob("*/completed.json"))
    ]
    pieces = repair_pieces(config, source, output, markers, workers=workers)
    totals = {}
    for group, original in previous["by_group"].items():
        group_pieces = [piece for piece in pieces if piece["group"] == group]
        rows = sum(piece["rows"] for piece in group_pieces)
        if rows != original["source_rows"]:
            raise ValueError("source completed part count differs from original report")
        ready = sum(piece["ready_rows"] for piece in group_pieces)
        totals[group] = {
            "source_rows": rows,
            "ready_rows": ready,
            "ready_hours": sum(piece["ready_hours"] for piece in group_pieces),
            "review_rows": rows - ready,
            "reasons": dict(sum((Counter(piece["reasons"]) for piece in group_pieces), Counter())),
        }
    manifest = output / "labels.jsonl"
    with (
        stage_progress("Assemble repaired Arabic labels"),
        (output / "labels.pending").open("wb") as dst,
    ):
        for piece in pieces:
            with (output / piece["labels"]).open("rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    if manifest.exists():
        pin(manifest, sha(output / "labels.pending"))
        (output / "labels.pending").unlink()
    else:
        (output / "labels.pending").replace(manifest)
    pin(report_path, identity["source_report_sha256"])
    pin(source / "contract.json", identity["source_contract_sha256"])
    report = {
        **previous,
        "output_dir": str(output),
        "by_group": totals,
        "labels_sha256": sha(manifest),
        "identity": identity,
        "execution": {"workers": workers, "script_sha256": sha(Path(__file__))},
        "repair": {
            "source_dir": str(source),
            "recovered_rows": sum(piece["recovered_rows"] for piece in pieces),
            "recovered_hours": sum(piece["recovered_hours"] for piece in pieces),
            "retried_issues": dict(sum((Counter(p["retried_issues"]) for p in pieces), Counter())),
            "retry_failures": dict(sum((Counter(p["retry_failures"]) for p in pieces), Counter())),
            "original_ready_rows_reused": sum(
                v["ready_rows"] for v in previous["by_group"].values()
            ),
        },
    }
    write_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes, recommended: 4")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    try:
        if not args.worker:
            python = ROOT / config["runtime"] / "bin/python"
            return subprocess.run(
                [str(python), "-B", __file__, *sys.argv[1:], "--worker"],
                env={**os.environ, "CAMELTOOLS_DATA": str(ROOT / config["assets"])},
            ).returncode
        if importlib.metadata.version("camel-tools") != config["camel_tools_version"]:
            raise ValueError("CAMeL version changed")
        verify_assets(config)
        with exclusive_job(args.output_dir.resolve()):
            report = repair(
                config,
                args.source_dir.resolve(),
                args.output_dir.resolve(),
                resume=args.resume,
                workers=args.workers,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
