#!/usr/bin/env python3
"""Prepare isolated CALIMA assets and generate versioned automatic Arabic IPA labels."""

from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import ssl
import subprocess
import sys
import urllib.request
import uuid
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.modeling.ctc_tap import qwen3_asr_audio_output_length
from qwen_hotword.phonemes.arabic_native import CalimaFrontend
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.experiment_a import ctc_minimum_input_length
from qwen_hotword.training.stage_progress import stage_progress

CONFIG = ROOT / "configs/arabic_native_frontend.workzone.json"
VOCAB = ROOT / "configs/phonemes/en_ar_native_ipa_vocab.v1.1.json"
SOURCES = {
    "ar_egy": ROOT / "outputs/en_ar_training_source_audit_5bda8943/source_inventory.jsonl",
    "ar_msa": ROOT / "outputs/en_ar_training_source_audit_7987bde0/source_inventory.jsonl",
}
DEPENDENCIES = [
    "docopt==0.6.2",
    "editdistance==0.8.1",
    "emoji==2.16.0",
    "pyrsistent==0.20.0",
    "muddler==0.1.3",
    "cachetools==6.2.1",
    "future==1.0.0",
    "tabulate==0.9.0",
]


@contextmanager
def exclusive_job(output: Path) -> Iterator[None]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.name + ".lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns {output}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield


def verify_assets(config: dict[str, Any]) -> None:
    with stage_progress("Verify pinned CALIMA model files"):
        for spec in config["packages"].values():
            for entry in spec["files"]:
                pin(ROOT / config["assets"] / spec["destination"] / entry["path"], entry["sha256"])


def setup(config: dict[str, Any]) -> None:
    runtime = ROOT / config["runtime"]
    runtime.parent.mkdir(parents=True, exist_ok=True)
    python = runtime / "bin/python"
    installation = config["installation"]
    context = ssl.create_default_context()
    pip_install = [str(python), "-m", "pip", "install"]
    if not installation["verify_ssl"]:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        for host in installation["pip_trusted_hosts"]:
            pip_install.extend(["--trusted-host", host])
        print(
            "Work-zone installation: SSL certificate verification disabled; "
            "pinned model SHA256 checks retained",
            flush=True,
        )
    with stage_progress("Prepare isolated Arabic morphology Python"):
        if not python.is_file():
            if runtime.exists():
                raise ValueError(f"incomplete runtime directory exists: {runtime}")
            subprocess.run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(runtime)], check=True
            )
        subprocess.run([*pip_install, "--no-deps", "camel-tools==1.6.0"], check=True)
        subprocess.run([*pip_install, *DEPENDENCIES], check=True)
    assets = ROOT / config["assets"]
    assets.mkdir(parents=True, exist_ok=True)
    # Direct DB paths are used, so an empty local catalogue prevents implicit network access.
    catalogue = assets / "catalogue.json"
    if not catalogue.exists():
        write_json(catalogue, {"version": "1.6", "packages": {}, "components": {}})
    for name, spec in config["packages"].items():
        destination = assets / spec["destination"]
        if all((destination / entry["path"]).is_file() for entry in spec["files"]):
            for entry in spec["files"]:
                pin(destination / entry["path"], entry["sha256"])
            continue
        destination.mkdir(parents=True, exist_ok=True)
        archive = assets / (name + ".zip")
        with stage_progress(f"Acquire pinned model {name}"):
            if not archive.exists():
                pending = archive.with_suffix(".pending")
                with (
                    urllib.request.urlopen(spec["url"], timeout=90, context=context) as response,
                    pending.open("wb") as target,
                ):
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
                pin(pending, spec["archive_sha256"])
                pending.replace(archive)
            pin(archive, spec["archive_sha256"])
            with zipfile.ZipFile(archive) as zipped:
                for entry in spec["files"]:
                    target = destination / entry["path"]
                    if target.exists():
                        pin(target, entry["sha256"])
                        continue
                    pending = target.with_suffix(".pending")
                    pending.write_bytes(zipped.read(entry["path"]))
                    pin(pending, entry["sha256"])
                    pending.replace(target)
    verify_assets(config)
    subprocess.run(
        [
            str(python),
            "-c",
            "from camel_tools.disambig.mle import MLEDisambiguator; print('CALIMA import passed')",
        ],
        env={**os.environ, "CAMELTOOLS_DATA": str(assets)},
        check=True,
    )
    write_json(
        runtime.parent / "setup_completed.json",
        {
            "config_sha256": sha(CONFIG),
            "backend_version": "1.6.0",
            "installation": installation,
            "existing_conda_environments_modified": False,
        },
    )


def label(config: dict[str, Any], output: Path, resume: bool) -> dict[str, Any]:
    version = importlib.metadata.version("camel-tools")
    if version != config["camel_tools_version"]:
        raise ValueError(f"expected CAMeL 1.6.0, got {version}")
    verify_assets(config)
    output.mkdir(parents=True, exist_ok=resume)
    with stage_progress("Record Arabic input and code identities"):
        identity = {
            "config_sha256": sha(CONFIG),
            "vocab_sha256": sha(VOCAB),
            "script_sha256": sha(Path(__file__)),
            "backend_sha256": sha(ROOT / "src/qwen_hotword/phonemes/arabic_native.py"),
            "inventories": {group: sha(path) for group, path in SOURCES.items()},
            "camel_tools_version": version,
        }
    contract = output / "contract.json"
    if resume:
        if json.loads(contract.read_text()) != identity:
            raise ValueError("resume inputs, vocabulary or code changed; use a new output")
    else:
        write_json(contract, identity)
    tokens = json.loads(VOCAB.read_text(encoding="utf-8"))["tokens"]
    pieces: list[dict[str, Any]] = []
    totals: dict[str, dict[str, Any]] = {}
    for group, source in SOURCES.items():
        frontend = None
        batch: list[dict[str, Any]] = []
        part_number = 0
        seen = 0
        for row in iter_jsonl(source):
            if row["group"] != group:
                continue
            batch.append(row)
            if len(batch) < 2000:
                continue
            piece, frontend = label_part(
                config, output, group, batch, part_number, tokens, frontend, identity
            )
            pieces.append(piece)
            seen += len(batch)
            print(f"{group}: completed {seen:,} source rows", flush=True)
            batch, part_number = [], part_number + 1
        if batch:
            piece, frontend = label_part(
                config, output, group, batch, part_number, tokens, frontend, identity
            )
            pieces.append(piece)
        selected = [piece for piece in pieces if piece["group"] == group]
        totals[group] = {
            "source_rows": sum(piece["rows"] for piece in selected),
            "ready_rows": sum(piece["ready_rows"] for piece in selected),
            "ready_hours": sum(piece["ready_hours"] for piece in selected),
            "review_rows": sum(piece["rows"] - piece["ready_rows"] for piece in selected),
            "reasons": dict(sum((Counter(piece["reasons"]) for piece in selected), Counter())),
        }
        del frontend
    manifest = output / "labels.jsonl"
    pending = output / "labels.pending"
    with stage_progress("Assemble complete Arabic label manifest"), pending.open("wb") as target:
        for piece in pieces:
            with (output / piece["labels"]).open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    if manifest.exists():
        if sha(manifest) != sha(pending):
            raise ValueError("completed label manifest differs")
        pending.unlink()
    else:
        pending.replace(manifest)
    report = {
        "status": "automatic_labels_completed",
        "output_dir": str(output),
        "vocab": str(VOCAB),
        "vocab_sha256": sha(VOCAB),
        "num_classes": len(tokens),
        "english_prefix_unchanged": True,
        "by_group": totals,
        "labels_sha256": sha(manifest),
        "identity": identity,
        "label_policy": "CALIMA_MLE_Top1_EGY_diac_rules_MSA_CAPHI",
        "disambiguation_context": "word_based_MLE",
        "human_pronunciation_review_completed": False,
        "audio_read": False,
        "qwen_loaded": False,
        "training_started": False,
    }
    write_json(output / "report.json", report)
    return report


def label_part(
    config: dict[str, Any],
    output: Path,
    group: str,
    batch: list[dict[str, Any]],
    number: int,
    tokens: list[str],
    frontend: CalimaFrontend | None,
    identity: dict[str, Any],
) -> tuple[dict[str, Any], CalimaFrontend | None]:
    folder = output / "parts" / group / f"{number:05d}"
    marker = folder / "completed.json"
    if marker.is_file():
        piece = json.loads(marker.read_text())
        for key in ("labels", "review"):
            pin(output / piece[key], piece[key + "_sha256"])
        if piece["vocab_sha256"] != identity["vocab_sha256"]:
            raise ValueError("part vocabulary mismatch")
        return piece, frontend
    folder.mkdir(parents=True, exist_ok=True)
    if frontend is None:
        with stage_progress(f"Load {group} CALIMA model"):
            frontend = CalimaFrontend(ROOT / config["assets"], group, tokens)
    reasons: Counter[str] = Counter()
    ready, seconds = 0, 0.0
    with (
        (folder / "labels.pending").open("w", encoding="utf-8") as labels,
        (folder / "review.pending").open("w", encoding="utf-8") as review,
        stage_progress(f"Label {group} part {number}", total=len(batch), every=250) as tick,
    ):
        for index, row in enumerate(batch, 1):
            try:
                if set(row["issues"]) - {"assistant_punctuation_or_spacing_only"}:
                    raise ValueError("source_structure_or_transcript_conflict")
                if not row["duration_seconds"]:
                    raise ValueError("audio_header_unavailable")
                generated = frontend.label(row["text"])
                minimum = ctc_minimum_input_length(generated["ids"])
                frames = qwen3_asr_audio_output_length(
                    round(row["duration_seconds"] * 16000) // 160
                )
                if minimum > 2 * frames:
                    raise ValueError("estimated_CTC_length_infeasible")
                record = {
                    "group": group,
                    "source": row["source"],
                    "row": row["row"],
                    "text": row["text"],
                    "audio_path": row["audio_path"],
                    "duration_seconds": row["duration_seconds"],
                    **generated,
                    "ctc_minimum_input_length": minimum,
                    "vocab_sha256": identity["vocab_sha256"],
                }
                labels.write(json.dumps(record, ensure_ascii=False) + "\n")
                ready += 1
                seconds += row["duration_seconds"]
            except ValueError as error:
                reason = str(error)
                reasons[reason.split(":", 1)[0]] += 1
                review.write(json.dumps({**row, "label_issue": reason}, ensure_ascii=False) + "\n")
            tick(index)
    for name in ("labels", "review"):
        (folder / f"{name}.pending").replace(folder / f"{name}.jsonl")
    piece = {
        "group": group,
        "rows": len(batch),
        "ready_rows": ready,
        "ready_hours": seconds / 3600,
        "reasons": dict(reasons),
        "vocab_sha256": identity["vocab_sha256"],
        **{
            name: str((folder / f"{name}.jsonl").relative_to(output))
            for name in ("labels", "review")
        },
        **{name + "_sha256": sha(folder / f"{name}.jsonl") for name in ("labels", "review")},
    }
    write_json(marker, piece)
    return piece, frontend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "label", "_label"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = (
        args.output_dir or ROOT / "outputs" / f"arabic_native_labels_{uuid.uuid4().hex[:8]}"
    ).resolve()
    try:
        if args.action == "setup":
            with exclusive_job(ROOT / config["runtime"]):
                setup(config)
            return 0
        if args.action == "label":
            python = ROOT / config["runtime"] / "bin/python"
            if not python.is_file():
                raise ValueError("run setup first")
            command = [str(python), "-B", __file__, "_label", "--output-dir", str(output)]
            if args.resume:
                command.append("--resume")
            return subprocess.run(
                command, env={**os.environ, "CAMELTOOLS_DATA": str(ROOT / config["assets"])}
            ).returncode
        with exclusive_job(output):
            report = label(config, output, args.resume)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(
            json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
