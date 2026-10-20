"""Direct full EN/ES/PT causal retrieval, with compact text-returnable metrics."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.multilingual_stream_assets import prepare_multilingual_stream
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.stream_retrieval_metrics import score_stream_contexts
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.reviewed_retrieval import NAMES
from qwen_hotword.inference.stream_context_contract import EFFECTS, MODES, validate_contexts
from qwen_hotword.inference.stream_context_run import _check_inputs, _run_prefixes
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def _small_bundle(output: Path) -> None:
    names = ("terminal_summary.json", "report.json", "checkpoint_metadata.json")
    content = {name: (output / name).read_bytes() for name in names}
    content["sha256.txt"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in content.items()
    ).encode()
    final = output / "summary_bundle.zip"
    if final.exists():
        with zipfile.ZipFile(final) as archive:
            if set(archive.namelist()) != set(content) or any(
                archive.read(name) != value for name, value in content.items()
            ):
                raise ValueError("existing small summary bundle differs")
        return
    with tempfile.TemporaryDirectory(dir=output) as directory:
        temporary = Path(directory) / "bundle.zip"
        with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED) as archive:
            for name, value in content.items():
                archive.writestr(name, value)
        os.link(temporary, final)


def run_multilingual_stream(
    root: Path,
    config: Path,
    output: Path,
    *,
    effect: str,
    resume: bool = False,
    audit_only: bool = False,
    detector_factory: Callable[..., Any] | None = None,
    retrieve_function: external.RetrieveFunction | None = None,
) -> dict[str, Any]:
    if effect not in EFFECTS:
        raise ValueError("explicit same-step or next-step required")
    if output.exists() and not resume and not audit_only:
        raise FileExistsError(f"output exists; --resume only for identical inputs: {output}")
    with stage_progress("Verify all three languages, frozen targets and noisy inputs"):
        plan, runtime = prepare_multilingual_stream(root, config)
    plan.update(stage="full", effect=effect)
    for filename in plan["inputs"]:
        path = Path(filename)
        if path.is_relative_to(output) or (
            path.parent != root and output.is_relative_to(path.parent)
        ):
            raise ValueError("output overlaps a protected input directory")
    audited = {
        "status": "audit_pass",
        "sample_count": plan["selected_samples"],
        "by_language": plan["dataset_counts"],
        "prefix_count": plan["prefix_count"],
        "checkpoint_sha256": plan["checkpoint"]["sha256"],
        "model_loaded": False,
    }
    if audit_only:
        return audited
    print(f"Full stream inventory: {audited}", flush=True)
    lock = output.parent / f".{output.name}.orchestration"
    with exclusive_feature_cache_run(lock):
        external._prepare_output(output, plan, resume=resume)
        if (output / "sha256.txt").exists():
            with stage_progress("Verify completed full run; no model reload"):
                seal(output)
            completed: dict[str, Any] = _json(output / "terminal_summary.json")
            return completed
        external._write_or_verify_json(output / "checkpoint_metadata.json", plan["checkpoint"])
        contexts: dict[str, Any] = {mode: {} for mode in MODES}
        durations: dict[str, Any] = {}
        audit, timings = [], []
        for lang in ("en", "es", "pt"):
            samples = [s for s in plan["samples"] if s["language"] == lang]

            def factory(_language: str = lang, **kwargs: Any) -> Any:
                # One detector per language; retain the original language-specific prompt.
                return (detector_factory or external._load_detector)(
                    **{**kwargs, "language": NAMES[_language]}
                )

            with stage_progress(f"Stream {NAMES[lang]} ({len(samples)} audio files)"):
                data, lengths, checks, steps = _run_prefixes(
                    output,
                    {
                        **plan,
                        "samples": samples,
                        "prefix_count": sum(s["step_count"] for s in samples),
                    },
                    runtime,
                    factory,
                    retrieve_function,
                )
            for mode in MODES:
                contexts[mode].update(data[mode])
            durations.update(lengths)
            audit.extend(checks)
            timings.extend(steps)
        with stage_progress("Validate delivery and score full-reference retrieval metrics"):
            validation = validate_contexts(contexts, durations)
            delivery = output / "delivery"
            delivery.mkdir(exist_ok=True)
            external._write_or_verify_json(delivery / "stream_contexts.json", contexts)
            external._write_or_verify_json(delivery / "durations.json", durations)
            manifest: dict[str, Any] = {}
            for sample in plan["samples"]:
                manifest.setdefault(sample["key"], {})[sample["sample_id"]] = sample
            external._write_or_verify_json(delivery / "audio_manifest.json", manifest)
            external._write_or_verify_jsonl(output / "causality_audit.jsonl", audit)
            validator = (
                Path(__file__).resolve().parents[3]
                / "tests/fixtures/stream_interface_v1_validator.py.txt"
            )
            checked = subprocess.run(
                [
                    sys.executable,
                    str(validator),
                    "--ctx",
                    str(delivery / "stream_contexts.json"),
                    "--durations",
                    str(delivery / "durations.json"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if checked.returncode:
                raise ValueError(f"downstream format validation failed: {checked.stderr}")
            metrics, details = score_stream_contexts(
                contexts, plan["samples"], runtime["evaluation_groups"], runtime["bundle"].entries
            )
            external._write_or_verify_jsonl(output / "sample_metrics.jsonl", details)
        with stage_progress("Recheck input identities and seal outputs"):
            _check_inputs(plan)
            times = [s["prefix_preparation_seconds"] + s["retrieval_wall_seconds"] for s in timings]
            summary = {
                "status": "completed",
                "effect": effect,
                "sample_count": len(plan["samples"]),
                "sample_counts_by_language": plan["dataset_counts"],
                "prefix_count": len(times),
                "checkpoint_sha256": plan["checkpoint"]["sha256"],
                "checkpoint_epoch": plan["checkpoint"]["epoch_metrics"]["epoch"],
                "table_sha256": plan["source_configuration"]["table_sha256"],
                "vocab_sha256": plan["source_configuration"]["vocab_sha256"],
                "run_config_sha256": _sha(output / "run_config.json"),
                "by_language": metrics["by_language"],
                "prefix_compute_seconds": {
                    "sum": sum(times),
                    "mean": sum(times) / len(times),
                    "max": max(times),
                    "over_2s_count": sum(t > 2 for t in times),
                },
                "schema_validation": validation,
                "downstream_format_validator_passed": True,
                "qwen_decoder_used": False,
                "word_timestamps_available": False,
                "output_dir": str(output),
            }
            report = {
                **summary,
                "by_dataset": metrics["by_dataset"],
                "code_sha256": plan["code_sha256"],
                "audio_pairing": plan["audio_pairing"],
                "candidate_pool_surfaces": len(runtime["bundle"].entries),
                "candidate_pronunciations": len(runtime["variants"]),
                "causality_checked": True,
                "snr_verified": False,
                "full_run_authorization": plan["configuration"]["full_run_authorization"],
                "downstream_validator_stdout": checked.stdout,
                "limitations": [
                    "last_context is the last delivered TopK; next-step omits the final retrieval.",
                    "ever_delivered unions contexts per audio, not fixed-TopK or on-time recall.",
                    "Outside-reference counts use full references; prefix correctness is unknown.",
                    "No word timestamps, ASR WER, decoder adoption or end-to-end latency measured.",
                    "SNR unverified; reference-conditioned table construction remains unchanged.",
                    "Old-head clean results verify identity, not paired streaming performance.",
                ],
            }
            external._write_or_verify_json(output / "report.json", report)
            external._write_or_verify_json(output / "terminal_summary.json", summary)
            _small_bundle(output)
            seal(output)
    return summary
