"""H200-side pilot orchestration. Never run this against local full-model assets."""

from __future__ import annotations

import array
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.training.g2p_prep import digit_fragments, extract_word_tokens
from qwen_hotword.training.pt_mfa_pilot import (
    inspect_alignment,
    select_pilot,
    sha256_file,
    write_json_new,
)


def run_command(args: list[str], log: Path, env: dict[str, str]) -> None:
    if not log.name.endswith("_ffmpeg.log"):
        print(f"Running: {' '.join(args[:4])} ... (log: {log})", file=sys.stderr, flush=True)
    with log.open("xb") as handle:
        process = subprocess.run(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=3600,
            check=False,
        )
    if process.returncode:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 3000))
            tail = handle.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Exit {process.returncode}; {log.name}: {tail}")


def wav_metrics(path: Path) -> dict[str, float]:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != 16000
        ):
            raise ValueError("Expected mono 16kHz PCM16 conversion")
        samples = array.array("h", handle.readframes(handle.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError("Empty converted audio")
    return {
        "duration_seconds": len(samples) / 16000,
        "rms": (sum(float(s) ** 2 for s in samples) / len(samples)) ** 0.5 / 32768,
        "clipped_sample_fraction": sum(abs(s) >= 32767 for s in samples) / len(samples),
    }


def alignment_command(
    mfa: str,
    root: Path,
    dictionary: Path,
    acoustic: Path,
    jobs: int,
) -> list[str]:
    return [
        mfa,
        "align",
        str(root / "corpus"),
        str(dictionary),
        str(acoustic),
        str(root / "aligned"),
        "--output_format",
        "json",
        "--single_speaker",
        "--num_jobs",
        str(jobs),
        "--temporary_directory",
        str(root / "mfa_tmp"),
        "--no_clean",
        "--no_final_clean",
        "--no_overwrite",
        "--no_textgrid_cleanup",
    ]


def run_pilot(
    config: dict[str, Any], repo_root: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Fresh output only; any failure is preserved with a compact JSON report."""
    manifest = repo_root / config["validation_manifest"]
    selected, strata = select_pilot(
        manifest,
        config["validation_sha256"],
        config["sources"],
        config["samples_per_source"],
        config["seed"],
    )
    if not isinstance(config["jobs"], int) or not 1 <= config["jobs"] <= 16:
        raise ValueError("jobs must be 1..16")
    if output_dir is not None:
        root = output_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        parent = (repo_root / config["output_parent"]).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="pt_mfa_pilot_v1_", dir=parent))
    print(f"Pilot output: {root}", file=sys.stderr, flush=True)
    logs, corpus = root / "logs", root / "corpus"
    logs.mkdir()
    corpus.mkdir()
    env = dict(os.environ)
    # Isolate all MFA-managed global state, downloads and scratch from existing outputs/caches.
    env["MFA_ROOT_DIR"] = str(root / "mfa_root")
    env["TMPDIR"] = str(root / "scratch")
    (root / "scratch").mkdir()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    report: dict[str, Any] = {
        "status": "failed",
        "output_dir": str(root),
        "config": config,
        "validation_sha256": sha256_file(manifest),
        "sampling_policy": "equal_source_release_density_tertiles_hash_v1",
        "selected_count": len(selected),
        "strata": strata,
        "thresholds_seconds": {"minimum_phone": 0.0101, "long_phone": 0.300, "edge": 0.020},
        "minimum_phone_fraction_flag": 0.25,
        "very_low_rms_flag": 0.001,
        "clipped_sample_fraction_flag": 0.01,
        "manifest_duration_difference_flag_seconds": 0.1,
        "full_qwen_loaded": False,
        "training_started": False,
        "test_set_used": False,
        "labels_changed": False,
        "mfa_alignment_invoked": False,
        "limitations": [
            "Flags are review signals, not confirmed label errors; aligned does not mean clean.",
            "Equal-source/stratum oversampling: raw flag rates are not pool error prevalence.",
            "MFA acoustic training includes CV/MLS; read-speech bias may affect Noah results.",
            "Brazilian dictionary may mismatch non-Brazilian or unknown-accent samples.",
            "No phoneme PER, actual-phone gold labels, or deletion attribution is produced.",
            "No speaker adaptation; speaker identities are not inferred from filenames.",
            "Audio is resampled/downmixed to mono PCM16 without cropping; originals remain intact.",
            "Dictionary OOV uses the existing G2P word tokenizer; it is not MFA's own OOV count.",
        ],
    }
    results: list[dict[str, Any]] = []
    try:
        write_json_new(root / "config.json", config)
        write_json_new(root / "selection.json", selected)
        ids_path = root / "selected_ids.txt"
        with ids_path.open("x", encoding="utf-8") as handle:
            handle.write("".join(row["id"] + "\n" for row in selected))
        report["selected_ids_sha256"] = sha256_file(ids_path)
        report["selection_sha256"] = sha256_file(root / "selection.json")
        report["code_sha256"] = {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("pt_mfa_runner.py", "pt_mfa_pilot.py", "g2p_prep.py")
        }
        identity = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        report["git_commit"] = identity.stdout.strip() if identity.returncode == 0 else None
        report["mfa_version"] = importlib.metadata.version("montreal-forced-aligner")
        if report["mfa_version"] != "3.4.0":
            raise ValueError("This pilot is prepared for the confirmed MFA 3.4.0 environment")
        mfa, ffmpeg = shutil.which("mfa"), shutil.which("ffmpeg")
        if not mfa or not ffmpeg:
            raise ValueError("Run in the aligner environment; mfa and ffmpeg are required")
        run_command([mfa, "--help"], logs / "mfa_preflight.log", env)
        for kind, name in (
            ("acoustic", config["acoustic_model"]),
            ("dictionary", config["dictionary"]),
        ):
            run_command(
                [mfa, "model", "download", kind, name, "--version", config["model_version"]],
                logs / f"download_{kind}.log",
                env,
            )
        assets = root / "mfa_root/pretrained_models"
        acoustic = assets / "acoustic" / f"{config['acoustic_model']}.zip"
        dictionary = assets / "dictionary" / f"{config['dictionary']}.dict"
        report["assets"] = {
            "acoustic": {"path": str(acoustic), "sha256": sha256_file(acoustic)},
            "dictionary": {"path": str(dictionary), "sha256": sha256_file(dictionary)},
        }
        vocabulary: set[str] = set()
        with dictionary.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    vocabulary.add(line.split()[0].casefold())

        for index, row in enumerate(selected):
            if index % 20 == 0:
                print(f"Preparing audio {index}/{len(selected)}", file=sys.stderr, flush=True)
            pilot_id = row["pilot_id"]
            result: dict[str, Any] = {**row, "status": "preparation_failed", "flags": []}
            try:
                result["source_audio_sha256"] = sha256_file(Path(row["audio_path"]))
                wav = corpus / f"{pilot_id}.wav"
                run_command(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-n",
                        "-v",
                        "error",
                        "-i",
                        row["audio_path"],
                        "-map",
                        "0:a:0",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        str(wav),
                    ],
                    logs / f"{pilot_id}_ffmpeg.log",
                    env,
                )
                result["audio"] = wav_metrics(wav)
                result["converted_audio_sha256"] = sha256_file(wav)
                with (corpus / f"{pilot_id}.lab").open("x", encoding="utf-8") as handle:
                    # No word, punctuation or number removal; whitespace only.
                    handle.write(" ".join(row["text"].split()) + "\n")
                result["oov_words"] = sorted(set(extract_word_tokens(row["text"])) - vocabulary)
                result["digit_fragments"] = digit_fragments(row["text"])
                result["status"] = "prepared"
            except (
                OSError,
                ValueError,
                RuntimeError,
                wave.Error,
                subprocess.TimeoutExpired,
            ) as error:
                result["error"] = str(error)[-2000:]
                result["flags"] = ["preparation_failed"]
            results.append(result)
        write_json_new(root / "preparation.json", results)
        if not any(r["status"] == "prepared" for r in results):
            raise ValueError("No audio prepared; see preparation.json")
        command = alignment_command(mfa, root, dictionary, acoustic, config["jobs"])
        report["alignment_command"] = command
        report["mfa_alignment_invoked"] = True
        # MFA rejects dictionary/acoustic phoneset incompatibility; do not map/merge phones.
        run_command(command, logs / "alignment.log", env)
        for result in results:
            if result["status"] != "prepared":
                continue
            paths = list((root / "aligned").rglob(f"{result['pilot_id']}.json"))
            if len(paths) > 1:
                raise ValueError(f"Duplicate alignment export for {result['pilot_id']}")
            alignment = paths[0] if paths else root / "aligned" / f"{result['pilot_id']}.json"
            try:
                result.update(inspect_alignment(alignment, result["audio"]["duration_seconds"]))
            except (ValueError, KeyError, TypeError) as error:
                result.update(
                    status="invalid_alignment", flags=["invalid_alignment"], error=str(error)
                )
            if result["oov_words"]:
                result["flags"].append("dictionary_oov")
            if result["digit_fragments"]:
                result["flags"].append("unresolved_digits")
            if result["audio"]["rms"] < 0.001:
                result["flags"].append("very_low_audio_energy")
            if result["audio"]["clipped_sample_fraction"] > 0.01:
                result["flags"].append("possible_clipping")
            if abs(result["audio"]["duration_seconds"] - result["duration_seconds"]) > 0.1:
                result["flags"].append("manifest_duration_mismatch")
        report["status"] = "completed"
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        report["error"] = str(error)[-3000:]
    report["counts_by_status"] = dict(Counter(r["status"] for r in results))
    report["by_release_density"] = {
        stratum: {
            "status_counts": dict(Counter(r["status"] for r in results if r["stratum"] == stratum)),
            "flag_counts": dict(
                Counter(flag for r in results if r["stratum"] == stratum for flag in r["flags"])
            ),
        }
        for stratum in sorted({r["stratum"] for r in selected})
    }
    report["by_source"] = {
        source: {
            "status_counts": dict(
                Counter(r["status"] for r in results if r["source_corpus"] == source)
            ),
            "flag_counts": dict(
                Counter(
                    flag for r in results if r["source_corpus"] == source for flag in r["flags"]
                )
            ),
        }
        for source in config["sources"]
    }
    report["review_examples"] = [
        {
            **{
                key: r[key]
                for key in (
                    "id",
                    "source_corpus",
                    "release_source",
                    "ref_per_frame",
                    "status",
                    "flags",
                )
            },
            "minimum_duration_phone_fraction": r.get("minimum_duration_phone_fraction"),
            "oov_word_count": len(r.get("oov_words", [])),
        }
        for source in config["sources"]
        for r in sorted(
            (r for r in results if r["source_corpus"] == source and r["flags"]),
            key=lambda r: (-len(r["flags"]), r["id"]),
        )[:2]
    ]
    write_json_new(root / "sample_diagnostics.json", results)
    write_json_new(root / "report.json", report)
    with (root / "sha256.txt").open("x", encoding="utf-8") as handle:
        for name in (
            "report.json",
            "selection.json",
            "selected_ids.txt",
            "sample_diagnostics.json",
        ):
            path = root / name
            if path.is_file():
                handle.write(f"{sha256_file(path)}  {name}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(root),
                "counts_by_status": report["counts_by_status"],
                "return_files": [str(root / "report.json"), str(root / "sha256.txt")],
                "error": report.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report
