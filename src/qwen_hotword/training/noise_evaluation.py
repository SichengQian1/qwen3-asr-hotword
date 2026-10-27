"""Paired full-audio Top5/7 retrieval for baseline and both noise Heads; no decoder."""

from __future__ import annotations

import gc
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.multilingual_stream_assets import prepare_multilingual_stream
from qwen_hotword.evaluation.stream_retrieval_metrics import reference_matcher
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.reviewed_retrieval import NAMES
from qwen_hotword.training.english_newval import pin
from qwen_hotword.training.noise_cache import publish
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def row_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps({k: v for k, v in row.items() if k != "row_sha256"}, sort_keys=True).encode()
    ).hexdigest()


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"samples": len(rows)}
    for k in (5, 7):
        entries = [r[f"top{k}"] for r in rows]
        expected = sum(r["expected"] for r in entries)
        hits = sum(r["hits"] for r in entries)
        negatives = [r for r in entries if r["no_reference_hotwords"]]
        result[f"top{k}"] = {
            "target_hits": hits,
            "target_pairs": expected,
            "recall": hits / expected if expected else None,
            "selected": sum(r["selected"] for r in entries),
            "outside_reference": sum(r["outside_reference"] for r in entries),
            "no_reference_audio_count": len(negatives),
            "negative_audio_false_positive_rate": (
                sum(r["selected"] > 0 for r in negatives) / len(negatives) if negatives else None
            ),
            "no_target_audio_count": sum(r["expected"] == 0 for r in entries),
        }
    timing = sorted(r["pure_ms"] for r in rows)
    result["pure_ms"] = {
        "mean": sum(timing) / len(timing),
        "p50": timing[len(timing) // 2],
        "p95": timing[min(len(timing) - 1, int(len(timing) * 0.95))],
    }
    return result


def paired_delta(base: list[dict[str, Any]], new: list[dict[str, Any]], k: int) -> dict[str, Any]:
    old = {(r["group"], r["sample_id"]): r[f"top{k}"] for r in base}
    current = {(r["group"], r["sample_id"]): r[f"top{k}"] for r in new}
    if old.keys() != current.keys():
        raise ValueError("paired evaluation sample sets differ")
    values = []
    for key in sorted(old):
        a, b = old[key], current[key]
        if a["expected"] != b["expected"]:
            raise ValueError("paired target denominators differ")
        values.append((a["hits"], b["hits"], a["expected"]))
    denominator = sum(v[2] for v in values)
    differences = []
    generator = random.Random(20261027)
    for _ in range(500):
        sample = generator.choices(values, k=len(values))
        total = sum(v[2] for v in sample)
        if total:
            differences.append(100 * sum(v[1] - v[0] for v in sample) / total)
    differences.sort()
    return {
        "delta_recall_pp": (
            100 * sum(v[1] - v[0] for v in values) / denominator if denominator else None
        ),
        "paired_audio_bootstrap_95pct_pp": (
            [differences[int(len(differences) * 0.025)], differences[int(len(differences) * 0.975)]]
            if differences
            else None
        ),
        "bootstrap_unit": "audio; speaker/session independence unverified",
    }


def evaluate(root: Path, output: Path, plan: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    import torch

    destination = output / "evaluation"
    if destination.exists() and not resume:
        raise FileExistsError("evaluation output exists; use --resume")
    checkpoints = {"baseline": Path(plan["initial_head"])}
    for ratio in (30, 50):
        folder = output / f"head_{ratio}"
        marker = json.loads((folder / "completed.json").read_text())
        pin(folder / "report.json", {}, marker["report_sha256"])
        path = folder / "ctc_head_best.pt"
        pin(path, {}, marker["checkpoint_sha256"])
        checkpoints[str(ratio)] = path
    with stage_progress("Freeze paired 6800-audio original/noisy evaluation inventory"):
        inventory, runtime = prepare_multilingual_stream(
            root, root / plan["configuration"]["evaluation_config"]
        )
    # The legacy entry point verifies the baseline Head/table and the unchanged retrieval code.
    source = root / inventory["source_configuration"]["baseline_run"]
    original = json.loads((source / "run_config.json").read_text())
    original_records = {
        (group, r["sample_id"]): r
        for group, data in original["groups"].items()
        for r in data["records"]
    }
    contract: dict[str, Any] = {
        "plan_sha256": _sha(output / "plan.json"),
        "checkpoints": {name: _sha(path) for name, path in checkpoints.items()},
        "inventory": inventory,
        "scope": "full_audio_top5_top7_no_decoder",
        "evaluation_code_sha256": _sha(Path(__file__)),
    }
    destination.mkdir(exist_ok=True)
    publish(destination / "run_config.json", contract)
    match = reference_matcher(runtime["bundle"].entries)
    all_rows = []
    for tag, checkpoint in checkpoints.items():
        detector = None
        try:
            samples = inventory["samples"]
            with stage_progress(
                f"Evaluate {tag}: original and noise", total=len(samples) * 2, every=100
            ) as tick:
                done = 0
                for sample in samples:
                    group, sample_id, lang = (
                        sample["group"],
                        sample["sample_id"],
                        sample["language"],
                    )
                    reference = original_records[group, sample_id]["reference_text"]
                    present = set(match(reference))
                    targets = set(runtime["evaluation_groups"][group]["target_ids"])
                    for condition in ("original", "noise10db"):
                        audio = (
                            Path(original_records[group, sample_id]["audio_path"])
                            if condition == "original"
                            else Path(sample["audio_path"])
                        )
                        expected_sha = (
                            sample["clean_sha256"]
                            if condition == "original"
                            else sample["audio_sha256"]
                        )
                        key = hashlib.sha256(
                            f"{tag}:{group}:{sample_id}:{condition}".encode()
                        ).hexdigest()
                        shard = destination / "samples" / f"{key}.json"
                        identity = {
                            "tag": tag,
                            "condition": condition,
                            "language": lang,
                            "group": group,
                            "sample_id": sample_id,
                            "audio_sha256": expected_sha,
                            "checkpoint_sha256": contract["checkpoints"][tag],
                        }
                        if shard.exists():
                            row = json.loads(shard.read_text())
                            if any(row.get(k) != v for k, v in identity.items()) or row_digest(
                                row
                            ) != row.get("row_sha256"):
                                raise ValueError(
                                    "evaluation sample shard identity/corruption mismatch"
                                )
                        else:
                            pin(audio, {}, expected_sha)
                            if detector is None:
                                detector = external._load_detector(
                                    model=runtime["model"],
                                    checkpoint=checkpoint,
                                    vocab=runtime["vocab"],
                                    hotwords=runtime["variants"],
                                    device="cuda:0",
                                    dtype="bfloat16",
                                    language=NAMES[lang],
                                )
                            detector.language = NAMES[lang]
                            waveform, _ = external._load_audio(audio)
                            retrieval = external._retrieve_waveform(
                                detector, waveform, runtime["vocab"], saved_raw_rank_depth=64
                            )
                            retrieval = collapse_variants(
                                retrieval,
                                runtime["variants"],
                                runtime["by_variant"],
                                inventory["gate"][group],
                            )
                            row = {
                                **identity,
                                "pure_ms": retrieval["timing"]["pure_retrieval_seconds"] * 1000,
                            }
                            for k in (5, 7):
                                selected = _select(
                                    retrieval["raw_ranked_matches"],
                                    gate=inventory["gate"][group],
                                    top_k=k,
                                )
                                ids = {entry["hotword_id"] for entry in selected}
                                row[f"top{k}"] = {
                                    "expected": len(present & targets),
                                    "hits": len(ids & present & targets),
                                    "selected": len(ids),
                                    "outside_reference": len(ids - present),
                                    "no_reference_hotwords": not present,
                                    "selected_ids": sorted(ids),
                                }
                            row["row_sha256"] = row_digest(row)
                            shard.parent.mkdir(exist_ok=True)
                            publish(shard, row)
                        all_rows.append(row)
                        done += 1
                        tick(done)
        finally:
            detector = None
            gc.collect()
            torch.cuda.empty_cache()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[f"{row['tag']}/{row['condition']}/{row['language']}"].append(row)
    summary = {key: aggregate(rows) for key, rows in grouped.items()}
    comparisons = {}
    for tag in ("30", "50"):
        for condition in ("original", "noise10db"):
            for lang in ("en", "es", "pt"):
                comparisons[f"{tag}/{condition}/{lang}"] = {
                    f"top{k}": paired_delta(
                        grouped[f"baseline/{condition}/{lang}"],
                        grouped[f"{tag}/{condition}/{lang}"],
                        k,
                    )
                    for k in (5, 7)
                }
    for filename, identity in inventory["inputs"].items():
        pin(Path(filename), {}, identity["sha256"])
    result = {
        "status": "completed",
        "by_language": summary,
        "paired_vs_baseline": comparisons,
        "rows": len(all_rows),
        "checkpoints": contract["checkpoints"],
        "table_sha256": inventory["source_configuration"]["table_sha256"],
        "qwen_decoder_used": False,
        "business_snr_verified": False,
        "business_noise_assumed_external_not_verified": True,
        "limitations": [
            "Full-audio retrieval, not the earlier cumulative-prefix latency measurement.",
            "Reference-conditioned joint table retained; "
            "no claim of unseen business generalization.",
            "Outside-reference candidates are text-based mismatches, not audited hallucinations.",
            "No zero-percent continued-training control; "
            "gains cannot be solely attributed to noise.",
            "CTC automatic-label PER and retrieval recall do not establish final ASR WER.",
        ],
    }
    publish(destination / "report.json", result)
    return result
