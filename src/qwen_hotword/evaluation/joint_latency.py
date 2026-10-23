"""Read saved joint stream timings without opening audio, models, or old outputs for writing."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.inference.stream_context_contract import boundaries, digest
from qwen_hotword.training.edit_distance import sequence_edit_distance_backend
from qwen_hotword.training.stage_progress import stage_progress

TABLE_SHA256 = "6bfc29496cd014608ab5890cc7f4dc7fd65fcd2c194e0f2aca021d1264729a1b"
STAGE_FIELDS = {
    "decode_ms": "ctc_decode_seconds",
    "anchor_ms": "anchor_query_seconds",
    "rerank_ms": "hotword_matching_seconds",
    "sort_ms": "hotword_sorting_seconds",
    "gate_ms": "hotword_selection_seconds",
    "merge_ms": "joint_word_merge_seconds",
}
STAGES = tuple(STAGE_FIELDS)


def distribution(values: list[float]) -> dict[str, Any]:
    """Nearest-rank tails; retain mean and count, never subtract marginal percentiles."""
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
        "p99": ordered[math.ceil(len(values) * 0.99) - 1],
        "max": ordered[-1],
    }


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a finite nonnegative number: {value!r}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"expected a finite nonnegative number: {value!r}")
    return result


def _correlation(rows: list[dict[str, Any]], x: str, y: str) -> float | None:
    xs, ys = [r[x] for r in rows], [r[y] for r in rows]
    if len(rows) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    cross = sum((a - xm) * (b - ym) for a, b in zip(xs, ys, strict=True))
    scale = math.sqrt(sum((a - xm) ** 2 for a in xs) * sum((b - ym) ** 2 for b in ys))
    return float(cross / scale)


def observation(row: dict[str, Any], index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    retrieval = row["retrieval"]
    raw = retrieval["raw_variant_ranked_matches"]
    if len(raw) != retrieval["variant_shortlist_candidates"] or len(raw) > 64:
        raise ValueError("complete variant shortlist (not just Top5/Top7) required")
    ids = [m["hotword_id"] for m in raw]
    if len(ids) != len(set(ids)) or any(i not in index for i in ids):
        raise ValueError("duplicate or unknown variant in saved shortlist")
    lengths = [len(index[i]["token_ids"]) for i in ids]
    timing = retrieval["timing"]
    phases = {name: _number(timing[field]) * 1000 for name, field in STAGE_FIELDS.items()}
    total = _number(timing["pure_retrieval_seconds"]) * 1000
    residual = total - sum(phases.values())
    if residual < -0.001:
        raise ValueError("recorded component times exceed pure retrieval; incompatible scope")
    decoded = retrieval["decoded_token_ids"]
    if not isinstance(decoded, list):
        raise ValueError("saved decoded token IDs required")
    return {
        "group": row["group"],
        "sample_id": row["sample_id"],
        "step": row["step"],
        "audio_seconds": _number(row["audio_end_seconds"]),
        "decoded_phones": len(decoded),
        "postings": _number(retrieval["postings_visited"]),
        "variant_candidates": len(raw),
        "surface_candidates": _number(retrieval["shortlist_candidates"]),
        "candidate_phone_mean": statistics.mean(lengths) if lengths else 0.0,
        "candidate_phone_max": max(lengths, default=0),
        "candidate_phone_sum": sum(lengths),
        "candidate_phone_square_sum": sum(n * n for n in lengths),
        "candidate_languages": dict(Counter(index[i]["language"] for i in ids)),
        **phases,
        "unattributed_ms": residual,
        "pure_ms": total,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = (*STAGES, "unattributed_ms", "pure_ms")
    work = (
        "audio_seconds",
        "decoded_phones",
        "postings",
        "variant_candidates",
        "surface_candidates",
        "candidate_phone_mean",
        "candidate_phone_max",
        "candidate_phone_sum",
        "candidate_phone_square_sum",
    )
    total = sum(r["pure_ms"] for r in rows)
    return {
        "queries": len(rows),
        "utterances": len({(r["group"], r["sample_id"]) for r in rows}),
        "timing_ms": {k: distribution([r[k] for r in rows]) for k in stages},
        "workload": {k: distribution([r[k] for r in rows]) for k in work},
        "time_share_of_sum": {
            k: sum(r[k] for r in rows) / total if total else None
            for k in (*STAGES, "unattributed_ms")
        },
        "over_50ms": sum(r["pure_ms"] > 50 for r in rows),
        "correlations_descriptive_only": {
            f"{x}__{y}": _correlation(rows, x, y)
            for x, y in (
                ("audio_seconds", "pure_ms"),
                ("decoded_phones", "anchor_ms"),
                ("postings", "anchor_ms"),
                ("variant_candidates", "rerank_ms"),
                ("candidate_phone_sum", "rerank_ms"),
                ("candidate_phone_square_sum", "rerank_ms"),
            )
        },
        "candidate_language_occurrences": dict(
            sum((Counter(r["candidate_languages"]) for r in rows), Counter())
        ),
    }


class SealedReader:
    """Verify only the small evidence actually consumed, not referenced models/audio."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.manifest_bytes = (root / "sha256.txt").read_bytes()
        self.checksums: dict[str, str] = {}
        for line in self.manifest_bytes.decode("utf-8").splitlines():
            sha, name = line.split(maxsplit=1)
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("invalid SHA256 manifest")
            if name in self.checksums or not (self.root / name).resolve().is_relative_to(self.root):
                raise ValueError("duplicate or escaping evidence filename")
            self.checksums[name] = sha
        self.consumed = hashlib.sha256()
        self.files_read = 0

    def read(self, name: str) -> Any:
        if name not in self.checksums:
            raise ValueError(f"evidence not covered by source SHA256 manifest: {name}")
        raw = (self.root / name).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != self.checksums[name]:
            raise ValueError(f"source evidence checksum mismatch: {name}")
        self.consumed.update(f"{sha}  {name}\n".encode())
        self.files_read += 1
        return (
            json.loads(raw)
            if not name.endswith(".jsonl")
            else [json.loads(line) for line in raw.splitlines() if line.strip()]
        )

    def identity(self) -> dict[str, Any]:
        if (self.root / "sha256.txt").read_bytes() != self.manifest_bytes:
            raise ValueError("source manifest changed during inspection")
        return {
            "root": str(self.root),
            "files_read": self.files_read,
            "source_manifest_sha256": hashlib.sha256(self.manifest_bytes).hexdigest(),
            "consumed_file_manifest_sha256": self.consumed.hexdigest(),
        }


def analyze_run(run: Path, assets: Path, *, table_sha256: str = TABLE_SHA256) -> dict[str, Any]:
    source, asset = SealedReader(run), SealedReader(assets)
    table = asset.read("reviewed_keyword_table.json")
    if asset.checksums["reviewed_keyword_table.json"] != table_sha256:
        raise ValueError("not the fixed reviewed joint 4000 table")
    allowed = {
        (e["normalized"], tuple(v["token_ids"])) for e in table["entries"] for v in e["variants"]
    }
    indexed = asset.read("indexed_pronunciations.jsonl")
    index = {e["hotword_id"]: e for e in indexed}
    if (
        len(index) != len(indexed)
        or {(e["normalized"], tuple(e["token_ids"])) for e in indexed} != allowed
        or any(not e["token_ids"] for e in indexed)
    ):
        raise ValueError("pronunciation index differs from frozen table")
    plan, status = source.read("run_config.json"), source.read("report.json")
    if status.get("status") != "completed":
        raise ValueError("a completed, sealed prefix run is required")
    if plan["source_configuration"]["table_sha256"] != table_sha256:
        raise ValueError("source run declares a different table")
    declared_index = {
        value["sha256"]
        for name, value in plan["inputs"].items()
        if Path(name).name == "indexed_pronunciations.jsonl"
    }
    if declared_index != {asset.checksums["indexed_pronunciations.jsonl"]}:
        raise ValueError("source run declares a different pronunciation index")
    samples = {(s["group"], s["sample_id"]): s for s in plan["samples"]}
    if len(samples) != len(plan["samples"]) or len(samples) != plan["selected_samples"]:
        raise ValueError("duplicate sample or incorrect selected sample count in source plan")
    expected: dict[tuple[str, str, int], tuple[dict[str, Any], int]] = {}
    for (group, identifier), sample in samples.items():
        ends = boundaries(sample["frames"], sample["sample_rate"])
        if len(ends) != sample["step_count"] or sample["language"] not in {"en", "es", "pt"}:
            raise ValueError("source plan has invalid language or prefix count")
        for step, end in enumerate(ends):
            expected[group, identifier, step] = sample, end
    files = sorted((run / "prefix_shards").glob("*/*/*.json"))
    if len(files) != len(expected) or len(files) != plan["prefix_count"]:
        raise ValueError("incomplete/extra prefix shards; no partial-run language comparison")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    with stage_progress(
        "Read saved prefix timings (no audio/model)", total=len(files), every=1000
    ) as tick:
        for completed, filename in enumerate(files, 1):
            name = filename.relative_to(run).as_posix()
            row = source.read(name)
            key = row["group"], row["sample_id"], row["step"]
            if row.get("sha256") != digest(row) or key not in expected or key in seen:
                raise ValueError(f"corrupt, unknown or duplicate prefix: {name}")
            sample, end = expected[key]
            if any(
                row[k] != v
                for k, v in {
                    "audio_sha256": sample["audio_sha256"],
                    "source_end_frame": end,
                    "source_sample_rate": sample["sample_rate"],
                    "key": sample["key"],
                    "audio_end_seconds": end / sample["sample_rate"],
                }.items()
            ):
                raise ValueError(f"prefix identity/cutoff differs from run plan: {name}")
            item = observation(row, index)
            item["evidence_file"] = name
            grouped[sample["language"]].append(item)
            seen.add(key)
            tick(completed)
    if seen != set(expected) or set(grouped) != {"en", "es", "pt"}:
        raise ValueError("complete three-language prefix coverage required")
    by_language = {lang: summarize(rows) for lang, rows in sorted(grouped.items())}
    last_prefix = {}
    for lang, rows in sorted(grouped.items()):
        last: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            sample_key = row["group"], row["sample_id"]
            if sample_key not in last or row["step"] > last[sample_key]["step"]:
                last[sample_key] = row
        last_prefix[lang] = summarize(list(last.values()))
    buckets: dict[str, Any] = {}
    for lang, rows in grouped.items():
        durations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            upper = 2 * math.ceil(row["audio_seconds"] / 2)
            durations[f"({upper - 2},{upper}]s" if upper <= 12 else "(12,inf)s"].append(row)
        buckets[lang] = {
            bucket: {
                "queries": len(values),
                "pure_ms": distribution([v["pure_ms"] for v in values]),
                "stage_mean_ms": {k: statistics.mean(v[k] for v in values) for k in STAGES},
                "mean_decoded_phones": statistics.mean(v["decoded_phones"] for v in values),
                "mean_postings": statistics.mean(v["postings"] for v in values),
                "mean_candidate_phone_sum": statistics.mean(
                    v["candidate_phone_sum"] for v in values
                ),
            }
            for bucket, values in sorted(durations.items())
        }
    group_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in grouped.values():
        for row in rows:
            group_rows[row["group"]].append(row)
    return {
        "status": "completed",
        "scope": "saved_joint_prefix_latency_no_inference",
        "model_loaded": False,
        "audio_opened": False,
        "retrieval_replayed": False,
        "table_sha256": table_sha256,
        "surfaces": len(table["entries"]),
        "indexed_pronunciations": len(index),
        "checkpoint_declared": plan["checkpoint"],
        "source_code_sha256_declared": plan.get("code_sha256"),
        "source_git_commit_declared": plan.get("git_commit"),
        "source_backend_and_gc_policy": "not inferred from timings or current environment",
        "inspection_environment_only": {
            "python": platform.python_version(),
            "edit_distance_backend": sequence_edit_distance_backend(),
        },
        "evidence": {"run": source.identity(), "assets": asset.identity()},
        "percentile_method": "median P50; nearest-rank P95/P99; units milliseconds",
        "interpretation_limits": [
            "Repeated prefixes are correlated; query count is not independent utterance count.",
            "Language is confounded with source, phone sequence, candidate mix and runtime load.",
            "Bucket and correlation comparisons are descriptive, not causal proof.",
            "Historical timings are reused; no local timing is presented as H200 measurement.",
        ],
        "by_language": by_language,
        "last_prefix_by_language": last_prefix,
        "by_prefix_duration": buckets,
        "by_dataset": {
            name: {
                "queries": len(rows),
                "pure_ms": distribution([r["pure_ms"] for r in rows]),
                "stage_mean_ms": {k: statistics.mean(r[k] for r in rows) for k in STAGES},
            }
            for name, rows in sorted(group_rows.items())
        },
        "slowest_5_per_language": {
            lang: sorted(rows, key=lambda r: -r["pure_ms"])[:5]
            for lang, rows in sorted(grouped.items())
        },
    }


def write_report(
    report: dict[str, Any], output: Path, protected: tuple[Path, ...]
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    if any(
        output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected
    ):
        raise ValueError("diagnostic output must be separate from source evidence")
    summary = {
        "status": report["status"],
        "scope": report["scope"],
        "output": str(output),
        "table_sha256": report["table_sha256"],
        "by_language": {
            lang: {
                "queries": data["queries"],
                "utterances": data["utterances"],
                "pure_ms": data["timing_ms"]["pure_ms"],
                "stage_mean_ms": {k: data["timing_ms"][k]["mean"] for k in STAGES},
                "over_50ms": data["over_50ms"],
            }
            for lang, data in report["by_language"].items()
        },
        "last_prefix_by_language": {
            lang: {
                "queries": data["queries"],
                "pure_ms": data["timing_ms"]["pure_ms"],
                "stage_mean_ms": {k: data["timing_ms"][k]["mean"] for k in STAGES},
            }
            for lang, data in report["last_prefix_by_language"].items()
        },
    }
    contents = {
        f"{name}.json": (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode()
        for name, value in (("report", report), ("summary", summary))
    }
    contents["sha256.txt"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in sorted(contents.items())
    ).encode()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in contents.items():
        (output / name).write_bytes(value)
    with zipfile.ZipFile(output / "return_bundle.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, value in contents.items():
            archive.writestr(name, value)
    return summary
