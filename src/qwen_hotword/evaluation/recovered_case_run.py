"""Run missing baseline/prompted ASR calls, stop at real gains, package original WAVs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.prompt_order import digest
from qwen_hotword.evaluation.recovered_audio_export import sha, text_recoveries, write_package
from qwen_hotword.evaluation.recovered_case_assets import prepare_cases, read_json
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.prompt_order_runtime import inspect_runtime
from qwen_hotword.inference.recovered_case_runtime import LANGUAGES, RecoveredCaseRuntime
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress

PROMPTS = {
    "en": "The following terms are spelling references. Use them only when supported by the audio. "
    "Do not add a term that was not spoken.\n{words}",
    "es": "Los siguientes términos son referencias de escritura. Úsalos solo si el audio los "
    "respalda. No añadas términos que no se hayan pronunciado.\n{words}",
    "pt": "Os termos abaixo são referências de escrita. Use-os apenas quando houver apoio no "
    "áudio. Não acrescente termos que não foram falados.\n{words}",
}


def prompt(language: str, words: list[str]) -> str:
    if (
        not words
        or len(words) > 7
        or any(
            not word.strip() or any(s in word for s in ("\n", "\r", "<|", "|>")) for word in words
        )
    ):
        raise ValueError("one to seven original, safe Top7 terms required")
    return PROMPTS[language].format(words="\n".join(words))


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        if read_json(path) != value:
            raise ValueError(f"existing artifact differs; refusing overwrite: {path}")
        return
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_audio(row: dict[str, Any]) -> None:
    import soundfile as sf

    path = Path(row["audio_path"])
    if path.suffix.lower() != ".wav" or sha(path) != row["audio_sha256"]:
        raise ValueError(f"original WAV identity differs: {path}")
    info = sf.info(str(path))
    if info.format not in {"WAV", "WAVEX", "RF64"} or (
        info.frames,
        info.samplerate,
        info.channels,
    ) != (row["frames"], row["sample_rate"], row["channels"]):
        raise ValueError(f"original WAV header differs: {path}")


def validate_result(result: dict[str, Any], request: dict[str, Any]) -> None:
    if (
        not isinstance(result.get("prediction"), str)
        or result.get("actual_context") != request["context"]
        or result.get("prompt_tokens_verified") is not True
        or not isinstance(result.get("generation_limit_reached"), bool)
        or not re.fullmatch("[0-9a-f]{64}", result.get("normalized_audio_sha256", ""))
    ):
        raise ValueError("ASR cell lacks audited context/text/audio/truncation evidence")


def _verify_completed(output: Path, completed: dict[str, Any], plan_sha: str) -> dict[str, Any]:
    if completed["plan_sha256"] != plan_sha:
        raise ValueError("completed marker belongs to another plan")
    package = output / "package"
    names = ["sha256.txt"]
    for line in (package / "sha256.txt").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        names.append(name)
        path = (package / name).resolve()
        if not path.is_relative_to(package.resolve()) or sha(path) != expected:
            raise ValueError(f"completed package changed: {name}")
    if sha(package / "return_bundle.zip") != completed["zip_sha256"]:
        raise ValueError("completed audio ZIP changed")
    with zipfile.ZipFile(package / "return_bundle.zip") as archive:
        if sorted(archive.namelist()) != sorted(names) or any(
            archive.read(name) != (package / name).read_bytes() for name in names
        ):
            raise ValueError("audio ZIP contents differ from the checked package files")
    summary: dict[str, Any] = completed["summary"]
    return summary


def run_case_hunt(
    root: Path,
    source: Path,
    table_path: Path,
    output: Path,
    *,
    count: int = 50,
    seed: int = 20261028,
    runtime_factory: Callable[..., Any] | None = None,
    runtime_probe: Callable[[], dict[str, Any]] = inspect_runtime,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("positive case count required")
    with stage_progress("Verify latest saved Top7, references and model metadata (no model load)"):
        assets, table = prepare_cases(root, source, table_path, seed)
    protected = [source, Path(assets["model"]), table_path.parent]
    if any(
        output == path or output.is_relative_to(path) or path.is_relative_to(output)
        for path in protected
    ):
        raise ValueError("output overlaps protected inputs")
    code_root = Path(__file__).resolve().parents[3]
    code_names = (
        "scripts/find_recovered_hotword_cases.py",
        "src/qwen_hotword/evaluation/recovered_case_assets.py",
        "src/qwen_hotword/evaluation/recovered_case_run.py",
        "src/qwen_hotword/evaluation/recovered_audio_export.py",
        "src/qwen_hotword/evaluation/prompt_order.py",
        "src/qwen_hotword/inference/recovered_case_runtime.py",
        "src/qwen_hotword/inference/prompt_order_runtime.py",
        "src/qwen_hotword/inference/hotword_prompt.py",
        "src/qwen_hotword/inference/stream_context_contract.py",
        "src/qwen_hotword/training/spanish_capacity.py",
        "src/qwen_hotword/training/feature_cache.py",
        "src/qwen_hotword/training/stage_progress.py",
    )
    plan = {
        **assets,
        "schema_version": 1,
        "requested_cases": count,
        "seed": seed,
        "max_new_tokens": 1024,
        "prompts": PROMPTS,
        "runtime": runtime_probe(),
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_sha256": {name: sha(code_root / name) for name in code_names},
    }
    plan_sha = digest(plan)
    lock = output.parent / f".{output.name}.case-hunt-lock"
    with exclusive_feature_cache_run(lock):
        if output.exists() and not (output / "run_config.json").exists() and any(output.iterdir()):
            raise ValueError("nonempty output has no matching case-hunt plan; use a NEW directory")
        atomic_json(output / "run_config.json", plan)
        if (output / "completed.json").exists():
            return _verify_completed(output, read_json(output / "completed.json"), plan_sha)
        selected: list[dict[str, Any]] = []
        outcomes: Counter[str] = Counter()
        engine: Any = None
        new_calls = 0
        reused_calls = 0
        checked = 0

        def progress(status: str, error: str | None = None) -> None:
            atomic_json(
                output / "progress.json",
                {
                    "status": status,
                    "plan_sha256": plan_sha,
                    "checked_audio": checked,
                    "pool_audio": len(plan["pool"]),
                    "recovered_cases": len(selected),
                    "requested_cases": count,
                    "new_ASR_calls_this_invocation": new_calls,
                    "reused_ASR_calls_this_invocation": reused_calls,
                    "outcomes": dict(outcomes),
                    "error": error,
                },
                replace=True,
            )

        def call(row: dict[str, Any], context: str, mode: str) -> dict[str, Any]:
            nonlocal engine, new_calls, reused_calls
            request = {
                "audio_path": row["audio_path"],
                "audio_sha256": row["audio_sha256"],
                "language": LANGUAGES[row["language"]],
                "context": context,
                "seed": seed,
            }
            path = output / "cells" / row["uid"] / f"{mode}.json"
            if path.exists():
                cell = read_json(path)
                if (
                    cell["sha256"] != digest({k: v for k, v in cell.items() if k != "sha256"})
                    or cell["plan_sha256"] != plan_sha
                    or cell["request"] != request
                ):
                    raise ValueError(f"saved ASR cell changed or belongs to another plan: {path}")
                result: dict[str, Any] = cell["result"]
                validate_result(result, request)
                reused_calls += 1
                return result
            if engine is None:
                with stage_progress("Load local Qwen3-ASR-1.7B for REAL final-ASR inference"):
                    engine = (runtime_factory or RecoveredCaseRuntime)(Path(plan["model"]), plan)
                atomic_json(output / "runtime.json", engine.metadata)
            with stage_progress(f"ASR {mode} {row['language']}/{row['sample_id']}"):
                started = time.perf_counter()
                result = engine.transcribe(request)
                validate_result(result, request)
                result["wall_seconds"] = time.perf_counter() - started
            cell = {"plan_sha256": plan_sha, "request": request, "result": result}
            atomic_json(path, {**cell, "sha256": digest(cell)})
            new_calls += 1
            return result

        progress("running")
        print(
            f"Case hunt: pool={len(plan['pool'])}, by_language={plan['pool_by_language']}, "
            f"stop_after={count} actual recovery cases",
            flush=True,
        )
        try:
            with stage_progress(
                "Check case-hunting audio pool", total=len(plan["pool"]), every=1
            ) as tick:
                for row in plan["pool"]:
                    verify_audio(row)
                    before = call(row, "", "baseline")
                    if (
                        before["generation_limit_reached"]
                        or before["returned_language"] != LANGUAGES[row["language"]]
                    ):
                        outcomes["baseline_truncated_or_wrong_language"] += 1
                    elif all(
                        strict_phrase_match(before["prediction"], w)
                        for w in row["potential_targets"]
                    ):
                        outcomes["baseline_already_has_all_retrieved_targets"] += 1
                    else:
                        after = call(
                            row, prompt(row["language"], row["injected_hotwords"]), "retrieved"
                        )
                        if after["normalized_audio_sha256"] != before["normalized_audio_sha256"]:
                            raise ValueError(
                                "baseline and prompted calls used different normalized audio"
                            )
                        if (
                            after["generation_limit_reached"]
                            or after["returned_language"] != LANGUAGES[row["language"]]
                        ):
                            outcomes["prompted_truncated_or_wrong_language"] += 1
                        else:
                            gains = text_recoveries(
                                row["reference_text"],
                                before["prediction"],
                                after["prediction"],
                                row["injected_hotwords"],
                                table,
                            )
                            if gains:
                                selected.append(
                                    {
                                        "case_id": row["uid"],
                                        "sample_id": row["sample_id"],
                                        "language": row["language"],
                                        "source_audio": row["audio_path"],
                                        "audio_sha256": row["audio_sha256"],
                                        "reference_text": row["reference_text"],
                                        "without_hotwords": before["prediction"],
                                        "with_hotwords": after["prediction"],
                                        "recovered_hotwords": gains,
                                        "injected_hotwords": row["injected_hotwords"],
                                        "retrieved_candidates": row["retrieved_candidates"],
                                        "source_run": str(output),
                                        "comparison": "full_audio_baseline_vs_actual_last_Top7",
                                        "retrieval_source_run": str(source),
                                        "historical_audio_sha_recorded": True,
                                    }
                                )
                                outcomes["recovered"] += 1
                                print(
                                    f"Recovered cases {len(selected)}/{count}: "
                                    f"{row['language']} {gains}",
                                    flush=True,
                                )
                            else:
                                outcomes["prompt_did_not_recover_a_missing_target"] += 1
                    checked += 1
                    progress("running")
                    tick(checked)
                    if len(selected) == count:
                        break
            status = "completed" if len(selected) == count else "insufficient_verified_cases"
            report = {
                "status": status,
                "requested_cases": count,
                "exported_cases": len(selected),
                "shortfall": count - len(selected),
                "checked_audio": checked,
                "eligible_pool_audio": len(plan["pool"]),
                "outcomes": dict(outcomes),
                "by_language": dict(Counter(r["language"] for r in selected)),
                "plan_sha256": plan_sha,
                "table_sha256": plan["table_sha256"],
                "retrieval_head_sha256": plan["retrieval_head_sha256"],
                "retrieval_source_run": str(source),
                "qwen_decoder_used": checked > 0,
                "scope": plan["scope"],
                "selection": plan["selection"],
                "reference_audio_audit": plan["reference_audio_audit"],
                "pool_exclusions": plan["pool_exclusions"],
                "blocker_summary": "复用最新联合4k的实际末次Top7，新跑整段音频两版转写；"
                "这是成功案例筛选，不用于估计总体提升。"
                + ("候选池已遍历完，真实恢复例子不足请求数量。" if len(selected) < count else ""),
            }
            with stage_progress("Copy unchanged original WAVs and package audio plus Markdown"):
                package = output / "package"
                if package.exists():
                    # A package rename can finish before the completion marker; verify exact report
                    # and audio list before recovering that narrowly defined commit boundary.
                    if read_json(package / "report.json") != report:
                        raise ValueError(
                            "existing package differs from reconstructed saved ASR results"
                        )
                    recorded_cases = [
                        json.loads(line)
                        for line in (package / "cases.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                    ]
                    if [
                        {k: v for k, v in case.items() if k != "exported_audio"}
                        for case in recorded_cases
                    ] != selected:
                        raise ValueError("existing package audio cases differ from saved ASR")
                else:
                    with tempfile.TemporaryDirectory(prefix=".package-", dir=output) as tmp:
                        staged = Path(tmp) / "package"
                        write_package(staged, selected, report)
                        os.rename(staged, package)
                summary = {
                    **report,
                    "return_bundle": str(package / "return_bundle.zip"),
                    "audio_directory": str(package / "audio"),
                }
                completed = {
                    "plan_sha256": plan_sha,
                    "summary": summary,
                    "zip_sha256": sha(package / "return_bundle.zip"),
                }
                _verify_completed(output, completed, plan_sha)
                atomic_json(output / "completed.json", completed)
            progress(status)
            return summary
        except BaseException as error:
            progress("interrupted_or_failed", f"{type(error).__name__}: {error}")
            raise
