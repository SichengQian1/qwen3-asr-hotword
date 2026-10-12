"""Append-only request cells, strict resume identity, and paired ASR artifacts."""

from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.prompt_order import digest, freeze, make_plan, preparation_report
from qwen_hotword.evaluation.prompt_order_metrics import score, summarize
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.prompt_order_runtime import QwenPromptRuntime, inspect_runtime
from qwen_hotword.training.spanish_capacity import _sha, _verified


def prepare(
    root: Path, config: Path, *, probe: Callable[[], dict[str, Any]] = inspect_runtime
) -> dict[str, Any]:
    output = root / _json(config)["output"]
    if output.exists():
        raise FileExistsError(f"already frozen: {output}; use audit, not prepare")
    return freeze(make_plan(root, config, probe()), output)


def audit(
    root: Path, config: Path, *, probe: Callable[[], dict[str, Any]] = inspect_runtime
) -> tuple[dict[str, Any], Path]:
    output = root / _json(config)["output"]
    folder = output / "freeze"
    if folder.is_dir() and (folder / "sha256.txt").is_file():
        seal(folder)
    plan = _json(_verified(folder, "plan.json", {}))
    current = make_plan(root, config, probe())
    if plan != current:
        raise ValueError(
            "frozen sample/audio/candidate/prompt/model/config/code/runtime identity changed; "
            "use a new experiment directory"
        )
    if not plan["ready"]:
        raise ValueError(
            "insufficient frozen samples; see freeze/preparation_report.json; no inference"
        )
    return plan, output


def requests(plan: dict[str, Any]) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    result = []
    for sample in plan["samples"]:
        order = sample["execution_order"] + (["B_repeat"] if sample["repeat_b"] else [])
        for group in order:
            condition = sample["conditions"]["B" if group == "B_repeat" else group]
            # Repeat requests have exactly the same input identity as B, including the random seed.
            request = {
                "uid": sample["uid"],
                "audio_path": sample["audio_path"],
                "audio_sha256": sample["audio_sha256"],
                "reference_sha256": sample["reference_sha256"],
                "seed": sample["request_seed"],
                "language": "English",
                "return_time_stamps": False,
                **condition,
            }
            result.append((sample, group, request))
    return result


def cell_path(output: Path, sample: dict[str, Any], group: str) -> Path:
    return output / "results" / digest(sample["uid"]) / f"{group}.json"


def read_cell(
    path: Path, request: dict[str, Any], group: str, plan_sha: str, runtime_sha: str
) -> dict[str, Any]:
    cell = _json(path)
    row = cell["result"]
    if (
        cell["result_sha256"] != digest(row)
        or row["request"] != request
        or row["group"] != group
        or row["plan_sha256"] != plan_sha
        or row["runtime_sha256"] != runtime_sha
        or row["actual_context"] != request["context"]
        or row["prompt_tokens_verified"] is not True
    ):
        raise ValueError(f"completed request identity/integrity mismatch: {path}")
    return dict(row)


def _attempt(output: Path, value: dict[str, Any]) -> None:
    # A start without a finish is reported as an uncertain interrupted attempt, never a result.
    with (output / "attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"existing final artifact differs: {path}")
    else:
        external._atomic_write_text(path, text)


def finish(
    output: Path, plan: dict[str, Any], runtime: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics, paired = summarize(plan, rows)
    attempts = [json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()]
    starts = sum(a["event"] == "start" for a in attempts)
    ends = sum(a["event"] == "saved" for a in attempts)
    report = {
        "status": "completed",
        "output_dir": str(output),
        "scope": "offline_final_asr_exploratory",
        "sample_count": len(plan["samples"]),
        "main_completed_calls": plan["main_calls"],
        "extra_completed_repeat_calls": plan["extra_repeat_calls"],
        "request_attempts_started": starts,
        "starts_without_saved_event": starts - ends,
        "attempt_note": "An interrupted attempt may or may not have reached GPU generation.",
        "plan_sha256": digest(plan),
        "runtime_sha256": digest(runtime),
        "preparation": preparation_report(plan, output),
        "runtime": runtime,
        "metric_unit": "unique term/audio pairs; micro aggregation; [0,1] rates except WER",
        "retrieval_rerun": False,
        "training_started": False,
        **metrics,
        "return_files": [
            str(output / n) for n in ("report.json", "representative_examples.md", "sha256.txt")
        ],
    }
    external._write_or_verify_jsonl(output / "predictions.jsonl", rows)
    external._write_or_verify_jsonl(output / "paired_results.jsonl", paired)
    examples = metrics["representative_selection"]
    selected = {x["uid"] for x in examples if x["uid"] is not None}
    chunks = [
        "# Prompt order / priority: exploratory paired examples\n",
        "Selection: one seeded eligible case per comparison and WER-outcome bucket. "
        "All buckets (including missing cases) follow. WER improvement need not imply "
        "target improvement; reference-relative wrong terms are not audited hallucinations.\n",
        "```json\n" + json.dumps(examples, ensure_ascii=False, indent=2) + "\n```\n",
    ]
    for row in paired:
        if row["uid"] in selected:
            chunks.append(
                f"## {row['uid']} ({row['stratum']})\n\n```json\n"
                + json.dumps(row, ensure_ascii=False, indent=2)
                + "\n```\n"
            )
    _write_text_once(output / "representative_examples.md", "\n".join(chunks))
    external._write_or_verify_json(output / "report.json", report)
    seal(output)
    return report


def run(
    root: Path,
    config: Path,
    *,
    resume: bool = False,
    probe: Callable[[], dict[str, Any]] = inspect_runtime,
    factory: Callable[[Path, dict[str, Any]], Any] = QwenPromptRuntime,
) -> dict[str, Any]:
    plan, output = audit(root, config, probe=probe)
    # flock is process-scoped; a crash does not leave an unrecoverable stale lock.
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(root, output, plan, resume, factory)


def _run_locked(
    root: Path,
    output: Path,
    plan: dict[str, Any],
    resume: bool,
    factory: Callable[[Path, dict[str, Any]], Any],
) -> dict[str, Any]:
    runtime_path = output / "runtime.json"
    existing = runtime_path.exists()
    if existing and not resume:
        raise FileExistsError(
            "inference already initialized; use --resume only for identical inputs"
        )
    schedule = requests(plan)
    expected_paths = {cell_path(output, s, g) for s, g, _ in schedule}
    if set((output / "results").rglob("*.json")) - expected_paths:
        raise ValueError("unexpected completed request cells")
    if not existing and ((output / "results").exists() or (output / "attempts.jsonl").exists()):
        raise ValueError("orphan results without runtime identity")
    runtime = _json(runtime_path) if existing else None
    plan_sha = digest(plan)
    saved: dict[tuple[str, str], dict[str, Any]] = {}
    if runtime is not None:
        for sample, group, request in schedule:
            path = cell_path(output, sample, group)
            if path.exists():
                saved[sample["uid"], group] = read_cell(
                    path, request, group, plan_sha, digest(runtime)
                )
    if len(saved) == len(schedule):
        assert runtime is not None
        if (output / "sha256.txt").exists():
            seal(output)
            return dict(_json(output / "report.json"))
        return finish(output, plan, runtime, [saved[s["uid"], g] for s, g, _ in schedule])
    if (output / "sha256.txt").exists() or (output / "report.json").exists():
        raise ValueError("finalized experiment has missing results; refusing partial overwrite")
    print(
        f"Loading full ASR model on H200; {len(saved)}/{len(schedule)} requests reusable",
        flush=True,
    )
    engine = factory(root / plan["configuration"]["model"], plan["configuration"])
    current_runtime = {"plan_sha256": plan_sha, "engine": engine.metadata}
    if runtime is not None and current_runtime != runtime:
        raise ValueError(
            "effective decoding or hardware/runtime identity changed; new directory required"
        )
    runtime = current_runtime
    external._write_or_verify_json(runtime_path, runtime)
    rows = []
    for sample, group, request in schedule:
        if _sha(Path(request["audio_path"])) != request["audio_sha256"]:
            raise ValueError("audio changed after preflight")
        stored = saved.get((sample["uid"], group))
        if stored is not None:
            rows.append(stored)
            continue
        path = cell_path(output, sample, group)
        _attempt(
            output,
            {
                "event": "start",
                "uid": sample["uid"],
                "group": group,
                "request_sha256": digest(request),
            },
        )
        start = time.monotonic()
        prediction = engine.transcribe(request)
        if (
            prediction["actual_context"] != request["context"]
            or prediction["prompt_tokens_verified"] is not True
        ):
            raise ValueError("runtime did not verify the actual complete prompt")
        row = {
            **prediction,
            "uid": sample["uid"],
            "dataset": sample["dataset"],
            "stratum": sample["stratum"],
            "group": group,
            "candidates": request["candidates"],
            "request": request,
            "elapsed_seconds": time.monotonic() - start,
            "plan_sha256": plan_sha,
            "runtime_sha256": digest(runtime),
        }
        row["metrics"] = score(
            sample, request, row["prediction"], plan["targets_by_dataset"][sample["dataset"]]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        external._write_or_verify_json(path, {"result": row, "result_sha256": digest(row)})
        _attempt(
            output,
            {"event": "saved", "uid": sample["uid"], "group": group, "result_sha256": digest(row)},
        )
        rows.append(row)
        print(f"Completed {len(rows)}/{len(schedule)}: {sample['uid']} {group}", flush=True)
    return finish(output, plan, runtime, rows)
