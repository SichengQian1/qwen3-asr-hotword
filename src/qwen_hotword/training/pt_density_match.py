"""Prediction-independent Portuguese subset matched to Spanish sample density."""

from __future__ import annotations

import hashlib
import json
import math
from array import array
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.training.pt_mfa_pilot import sha256_file, write_json_new


def read_validation(path: Path, expected_sha: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"Manifest SHA mismatch: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, paths = set(), set()
    for row in rows:
        if row.get("split") != "validation" or row.get("experiment") != "full-ctc-v1":
            raise ValueError("Only full-ctc-v1 validation is accepted")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Missing sample ID")
        audio = str(Path(row["audio_path"]).resolve())
        if row["id"] in ids or audio in paths:
            raise ValueError("Duplicate validation ID/audio")
        ids.add(row["id"])
        paths.add(audio)
        tokens = row.get("phoneme_token_ids")
        if not isinstance(tokens, list) or not tokens or len(tokens) != row.get("label_length"):
            raise ValueError("Reference length disagrees with saved tokens")
        if any(type(t) is not int or t <= 0 for t in tokens):
            raise ValueError("Invalid reference token ID")
        frames = row.get("effective_ctc_input_length")
        if type(frames) is not int or frames <= 0:
            raise ValueError("Invalid effective CTC frames")
        if row.get("ctc_time_upsampling_factor") != 2:
            raise ValueError("Temporal 2x required")
        if frames != 2 * row.get("estimated_ctc_input_length", -1):
            raise ValueError("Effective frames must equal 2x encoder frames")
        duration = row.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid duration")
        for key in ("source_corpus", "release_source", "language"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Missing {key}")
    if not rows:
        raise ValueError("Empty validation")
    return rows


def density(row: dict[str, Any]) -> float:
    return float(row["label_length"] / row["effective_ctc_input_length"])


def optimal_pairs(
    pool: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Exact 1D ordered injective minimum-L1 assignment; ties fixed by ID hash.

    O(N*M) CPU and O(N*M) bytes for traceback. No replacement, cutoff, bin merging,
    dropping target tails, or model-error selection. References are all retained.
    """

    def key(row: dict[str, Any]) -> tuple[float, str, str]:
        tie = hashlib.sha256(f"{seed}\0{row['id']}".encode()).hexdigest()
        return density(row), tie, row["id"]

    candidates = sorted(pool, key=key)
    targets = sorted(reference, key=key)
    n, m = len(targets), len(candidates)
    if not n or m < n:
        raise ValueError("Need at least as many unique PT candidates as ES references")
    previous = array("d", [0.0]) * (m + 1)
    trace = []
    values = [density(r) for r in candidates]
    for i, target in enumerate(targets, 1):
        current = array("d", [math.inf]) * (m + 1)
        choices = bytearray(m + 1)
        value = density(target)
        for j in range(i, m - n + i + 1):
            match = previous[j - 1] + abs(value - values[j - 1])
            skip = current[j - 1]
            if match < skip:
                current[j] = match
                choices[j] = 1
            else:
                current[j] = skip
        trace.append(choices)
        previous = current
    i, j = n, m
    pairs = []
    while i:
        if trace[i - 1][j]:
            pairs.append((targets[i - 1], candidates[j - 1]))
            i -= 1
        j -= 1
    return list(reversed(pairs))


def quantiles(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        str(p): values[min(len(values) - 1, math.ceil(p * len(values)) - 1)]
        for p in (0.05, 0.25, 0.5, 0.75, 0.95)
    }


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(density(r) for r in rows)
    return {
        "samples": len(rows),
        "hours": sum(r["duration_seconds"] for r in rows) / 3600,
        "density_mean_per_sample": sum(values) / len(values),
        "density_aggregate_tokens_per_frame": (
            sum(r["label_length"] for r in rows)
            / sum(r["effective_ctc_input_length"] for r in rows)
        ),
        "density_quantiles": quantiles(values),
        "density_histogram_0_025": dict(Counter(str(math.floor(v / 0.025)) for v in values)),
        "duration_quantiles": quantiles([r["duration_seconds"] for r in rows]),
        "reference_length_quantiles": quantiles([float(r["label_length"]) for r in rows]),
        "by_source": {
            s: {
                "samples": sum(r["source_corpus"] == s for r in rows),
                "hours": sum(r["duration_seconds"] for r in rows if r["source_corpus"] == s) / 3600,
            }
            for s in sorted({r["source_corpus"] for r in rows})
        },
        "release_counts": dict(Counter(r["release_source"] for r in rows)),
    }


def match_quality(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    es = sorted(density(a) for a, _ in pairs)
    pt = sorted(density(b) for _, b in pairs)
    gaps = [abs(a - b) for a, b in zip(es, pt, strict=True)]
    ks = max(abs(bisect_right(es, x) - bisect_right(pt, x)) / len(es) for x in es + pt)
    return {
        "ks": ks,
        "wasserstein_1": sum(gaps) / len(gaps),
        "p95_pair_gap": quantiles(gaps)["0.95"],
        "max_pair_gap": max(gaps),
    }


def write_checksums(root: Path, names: list[str]) -> None:
    with (root / "sha256.txt").open("x") as handle:
        for name in names:
            handle.write(f"{sha256_file(root / name)}  {name}\n")


def build_subset(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    root = repo / config["output_dir"]
    if root.exists():
        raise FileExistsError(f"Output exists; preserve it and choose a new output_dir: {root}")
    pool = read_validation(repo / config["pt_manifest"], config["pt_sha256"])
    full = read_validation(repo / config["reference_manifest"], config["reference_sha256"])
    es = [r for r in full if r.get("balanced_language_bucket") == "es"]
    legacy = [r for r in full if r.get("balanced_language_bucket") == "pt"]
    if not es or not legacy or any(not r["language"].lower().startswith("pt") for r in pool):
        raise ValueError("Expected Spanish reference and Portuguese-only pool")
    pairs = optimal_pairs(pool, es, config["seed"])
    selected = sorted((p for _, p in pairs), key=lambda r: r["id"])
    quality = match_quality(pairs)
    passed = (
        quality["ks"] <= config["max_ks"]
        and quality["p95_pair_gap"] <= config["max_p95_pair_gap"]
        and quality["max_pair_gap"] <= config["max_pair_gap"]
    )
    root.mkdir(parents=True, exist_ok=False)
    # Preserve complete source records and original labels byte-equivalent as JSON values.
    with (root / "full_ctc_validation.jsonl").open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "selected_ids.txt").open("x") as handle:
        handle.write("".join(r["id"] + "\n" for r in selected))
    write_json_new(
        root / "pairs.json",
        [
            {
                "es_id": e["id"],
                "pt_id": p["id"],
                "es_density": density(e),
                "pt_density": density(p),
                "gap": abs(density(e) - density(p)),
            }
            for e, p in pairs
        ],
    )
    report = {
        "status": "matched" if passed else "insufficient_density_match",
        "config": config,
        "density_definition": "label_length / effective_ctc_input_length",
        "weighting": "one_sample_one_vote; total hours and token-weighting not matched",
        "selection_policy": "sorted_1d_injective_minimum_l1_v1",
        "prediction_used": False,
        "labels_changed": False,
        "test_used": False,
        "quality": quality,
        "spanish_reference": distribution(es),
        "portuguese_pool": distribution(pool),
        "portuguese_selected": distribution(selected),
        "portuguese_legacy": distribution(legacy),
        "selected_legacy_overlap": len({r["id"] for r in selected} & {r["id"] for r in legacy}),
        "selected_manifest_sha256": sha256_file(root / "full_ctc_validation.jsonl"),
        "selected_ids_sha256": sha256_file(root / "selected_ids.txt"),
        "code_sha256": sha256_file(Path(__file__)),
        "limitations": [
            "Density matching is not label accuracy certification or a causal attribution.",
            "Source, duration, accent and difficulty may shift; inspect group summaries.",
            "PER is token-weighted; matched density is sample-weighted.",
            "Historical 9.662% refers to legacy PT, not the whole 16h pool.",
        ],
    }
    write_json_new(root / "selection_report.json", report)
    write_checksums(
        root,
        ["full_ctc_validation.jsonl", "selected_ids.txt", "pairs.json", "selection_report.json"],
    )
    return report
