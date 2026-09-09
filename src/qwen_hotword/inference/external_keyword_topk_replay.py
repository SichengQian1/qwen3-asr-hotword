from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_hotword.inference.external_keyword_retrieval import (
    KeywordBundle,
    load_keyword_bias,
)
from qwen_hotword.phonemes.coverage import load_phoneme_vocab

SOURCE_OUTPUTS = {
    "mls_portuguese": "mls_retrieval_output.json",
    "delivery_20260706_ptbr": "delivery_retrieval_output.json",
}
FINAL_FILES = (
    "run_config.json",
    "top7_replay_details.jsonl",
    "topk_comparison.json",
    "topk_comparison.md",
    "mls_retrieval_output.json",
    "delivery_retrieval_output.json",
    "README.md",
)


def replay_external_keyword_top7(
    *,
    source_run: str | Path,
    vocab_path: str | Path,
    keyword_bias_path: str | Path,
    output_dir: str | Path,
    keyword_set: str = "hard_k266",
) -> dict[str, object]:
    """Exactly replay the fixed Top-7 gate from a completed external Top-5 run."""
    source = Path(source_run).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    keyword_file = Path(keyword_bias_path).expanduser()
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    for path in (
        source / "run_config.json",
        source / "retrieval_details.jsonl",
        source / "evaluation_summary.json",
        source / "sha256.txt",
        vocab_file,
        keyword_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required input does not exist: {path}")
    _verify_hash_manifest(source)

    source_config = _read_json(source / "run_config.json")
    source_summary = _read_json(source / "evaluation_summary.json")
    gate = _validated_source_gate(source_config)
    _verify_identity(_mapping(source_config, "vocab"), vocab_file, "vocab")
    _verify_identity(_mapping(source_config, "keyword_bias"), keyword_file, "keyword bias")
    if source_config.get("keyword_set") != keyword_set:
        raise ValueError("keyword set differs from the completed source run")

    vocab = load_phoneme_vocab(vocab_file)
    bundle = load_keyword_bias(keyword_file, vocab=vocab, keyword_set=keyword_set)
    rows = _read_jsonl(source / "retrieval_details.jsonl")
    expected_count = _required_int(_mapping(source_summary, "overall"), "sample_count")
    if len(rows) != expected_count:
        raise ValueError(
            f"retrieval detail count differs from source summary: {len(rows)} != {expected_count}"
        )
    replay_rows = replay_rows_top7(rows, gate=gate, bundle=bundle)

    source_names = {str(row["source"]) for row in replay_rows}
    if source_names != set(SOURCE_OUTPUTS):
        raise ValueError(
            f"source names do not match the required MLS/Delivery split: {sorted(source_names)}"
        )

    profiles: dict[str, object] = {}
    for label, top_k in (("top5", 5), ("top7", 7)):
        profiles[label] = {
            "gate": {**gate, "top_k": top_k},
            "overall": _metric_summary(replay_rows, top_k=top_k),
            "by_source": {
                source_name: _metric_summary(
                    [row for row in replay_rows if row["source"] == source_name],
                    top_k=top_k,
                )
                for source_name in sorted(source_names)
            },
        }
    _verify_source_top5_metrics(_mapping(profiles, "top5"), source_summary)
    comparison = {
        "schema_version": 1,
        "status": "pass",
        "evaluation_scope": source_summary.get("evaluation_scope"),
        "replay_mode": "exact_from_saved_anchor_raw_rank",
        "only_changed_parameter": "top_k_5_to_7",
        "sample_count": len(replay_rows),
        "source_run": {
            "path": str(source),
            "git_commit": source_config.get("git_commit"),
            "retrieval_details_sha256": _sha256(source / "retrieval_details.jsonl"),
            "sha256_manifest_verified": True,
        },
        "identity_checks": {
            "source_top5_reproduced_for_every_sample": True,
            "source_top5_summary_reproduced": True,
            "top7_exact_for_every_sample": True,
            "vocab_bound_to_source_run": True,
            "keyword_bias_bound_to_source_run": True,
            "transcripts_used_for_candidate_generation": False,
            "model_rerun": False,
        },
        "profiles": profiles,
        "delta_top7_minus_top5": _comparison_delta(profiles),
        "observed_source_latency": _latency_summary(source_summary),
    }
    run_config = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "source_run": comparison["source_run"],
        "vocab": _identity(vocab_file),
        "keyword_bias": _identity(keyword_file),
        "keyword_set": keyword_set,
        "baseline_top_k": 5,
        "candidate_top_k": 7,
        "shared_gate": gate,
        "output_source_mapping": SOURCE_OUTPUTS,
    }

    destination.mkdir(parents=True)
    _write_json(destination / "run_config.json", run_config)
    _write_jsonl(destination / "top7_replay_details.jsonl", replay_rows)
    _write_json(destination / "topk_comparison.json", comparison)
    _atomic_write_text(destination / "topk_comparison.md", _markdown_table(comparison))
    for source_name, filename in SOURCE_OUTPUTS.items():
        selected_output = {
            str(row["sample_id"]): _delivery_items(row, bundle=bundle, top_k=7)
            for row in replay_rows
            if row["source"] == source_name
        }
        _write_json(destination / filename, selected_output)
    _atomic_write_text(destination / "README.md", _readme(source, comparison))
    _write_hash_manifest(destination)
    return comparison


def replay_rows_top7(
    rows: Sequence[Mapping[str, Any]],
    *,
    gate: Mapping[str, object],
    bundle: KeywordBundle,
) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("source retrieval details are empty")
    known_ids = {entry.hotword_id for entry in bundle.entries}
    seen_samples: set[tuple[str, str]] = set()
    replayed: list[dict[str, object]] = []
    for line_number, row in enumerate(rows, start=1):
        sample_id = _required_string(row, "sample_id")
        source = _required_string(row, "source")
        identity = (source, sample_id)
        if identity in seen_samples:
            raise ValueError(f"duplicate source/sample identity at row {line_number}: {identity}")
        seen_samples.add(identity)
        raw = _list_of_mappings(row, "raw_ranked_matches")
        _validate_ranked_matches(raw, known_ids=known_ids, line_number=line_number)
        shortlist_count = _required_int(row, "shortlist_candidates")
        if shortlist_count < len(raw):
            raise ValueError(f"row {line_number} saved rank exceeds shortlist candidate count")
        if len(raw) < 7 and len(raw) != shortlist_count:
            raise ValueError(f"row {line_number} does not save an exact raw Top-7")

        selected5 = _select(raw, gate=gate, top_k=5)
        stored5 = _string_list(row, "selected_hotword_ids")
        replayed5 = [str(match["hotword_id"]) for match in selected5]
        if replayed5 != stored5:
            raise ValueError(
                f"row {line_number} cannot reproduce stored Top-5 selection: "
                f"{replayed5} != {stored5}"
            )
        selected7 = _select(raw, gate=gate, top_k=7)
        if len(selected7) < 7 and len(raw) != shortlist_count:
            tail_score = _required_number(raw[-1], "score") if raw else -math.inf
            if tail_score >= _required_number(gate, "threshold"):
                raise ValueError(
                    f"row {line_number} saved rank is insufficient for an exact Top-7 gate"
                )

        expected = set(_string_list(row, "expected_hotword_ids"))
        raw_ids = [str(match["hotword_id"]) for match in raw]
        selected7_ids = [str(match["hotword_id"]) for match in selected7]
        replayed.append(
            {
                "sample_id": sample_id,
                "source": source,
                "expected_hotword_ids": sorted(expected),
                "raw_top5_hotword_ids": raw_ids[:5],
                "raw_top7_hotword_ids": raw_ids[:7],
                "top5_selected_hotword_ids": replayed5,
                "top7_selected_matches": selected7,
                "top7_selected_hotword_ids": selected7_ids,
                "top7_hit_expected_hotword_ids": sorted(expected.intersection(selected7_ids)),
                "top7_missed_expected_hotword_ids": sorted(expected.difference(selected7_ids)),
                "top7_wrong_selected_hotword_ids": sorted(set(selected7_ids).difference(expected)),
            }
        )
    return replayed


def _select(
    raw: Sequence[Mapping[str, object]], *, gate: Mapping[str, object], top_k: int
) -> list[Mapping[str, object]]:
    threshold = _required_number(gate, "threshold")
    maximum_edit_ratio = _required_number(gate, "maximum_edit_ratio")
    minimum_posterior = _required_number(gate, "minimum_posterior_confidence")
    minimum_margin = _required_number(gate, "minimum_top1_margin")
    qualified = [
        match
        for match in raw
        if _required_number(match, "score") >= threshold
        and _required_number(match, "edit_ratio") <= maximum_edit_ratio
        and _required_number(match, "posterior_confidence") >= minimum_posterior
    ]
    if not qualified:
        return []
    if (
        len(qualified) > 1
        and _required_number(qualified[0], "score") - _required_number(qualified[1], "score")
        < minimum_margin
    ):
        return []
    return qualified[:top_k]


def _metric_summary(rows: Sequence[Mapping[str, object]], *, top_k: int) -> dict[str, object]:
    expected = sum(len(_string_list(row, "expected_hotword_ids")) for row in rows)
    raw_field = f"raw_top{top_k}_hotword_ids"
    selected_field = f"top{top_k}_selected_hotword_ids"
    raw_hits = 0
    selected_hits = 0
    selected_count = 0
    negative_count = 0
    negative_false_positives = 0
    for row in rows:
        expected_ids = set(_string_list(row, "expected_hotword_ids"))
        raw_ids = set(_string_list(row, raw_field))
        selected_ids = _string_list(row, selected_field)
        raw_hits += len(expected_ids.intersection(raw_ids))
        selected_hits += len(expected_ids.intersection(selected_ids))
        selected_count += len(selected_ids)
        if not expected_ids:
            negative_count += 1
            negative_false_positives += bool(selected_ids)
    return {
        "sample_count": len(rows),
        "expected_hotwords": expected,
        "raw_k": top_k,
        "raw_retrieved_expected_hotwords": raw_hits,
        "raw_recall": _ratio(raw_hits, expected),
        f"raw_retrieved_expected_hotwords_at_{top_k}": raw_hits,
        f"raw_recall_at_{top_k}": _ratio(raw_hits, expected),
        "selected_hotwords": selected_count,
        "selected_true_positive_hotwords": selected_hits,
        "final_retrieval_recall": _ratio(selected_hits, expected),
        "final_retrieval_precision": _ratio(selected_hits, selected_count),
        "negative_sample_count": negative_count,
        "negative_sample_false_positives": negative_false_positives,
        "negative_sample_false_positive_rate": _ratio(negative_false_positives, negative_count),
        "mean_selected_hotwords_per_sample": (selected_count / len(rows) if rows else None),
    }


def _delivery_items(
    row: Mapping[str, object], *, bundle: KeywordBundle, top_k: int
) -> list[dict[str, str]]:
    entry_by_id = {entry.hotword_id: entry for entry in bundle.entries}
    items = []
    for hotword_id in _string_list(row, f"top{top_k}_selected_hotword_ids"):
        entry = entry_by_id[hotword_id]
        items.append({"word": entry.surface, "phoneme": bundle.phonemes[entry.surface]})
    return items


def _validated_source_gate(config: Mapping[str, Any]) -> dict[str, object]:
    gate = dict(_mapping(config, "gate"))
    required = {
        "threshold": 0.75,
        "top_k": 5,
        "minimum_phonemes": 1,
        "maximum_edit_ratio": 0.35,
        "posterior_weight": 0.25,
        "minimum_posterior_confidence": 0.5,
        "minimum_top1_margin": 0.0,
    }
    if gate != required:
        raise ValueError(f"source run is not the fixed D5 gate: {gate}")
    retrieval = _mapping(config, "retrieval")
    if retrieval.get("backend") != "anchor_guided":
        raise ValueError("source run did not use Anchor-guided retrieval")
    if _required_int(retrieval, "saved_raw_rank_depth") < 7:
        raise ValueError("source run saved fewer than seven raw ranked matches")
    gate.pop("top_k")
    return gate


def _comparison_delta(profiles: Mapping[str, object]) -> dict[str, object]:
    top5 = _mapping(_mapping(profiles, "top5"), "overall")
    top7 = _mapping(_mapping(profiles, "top7"), "overall")
    keys = (
        "raw_retrieved_expected_hotwords",
        "raw_recall",
        "selected_hotwords",
        "selected_true_positive_hotwords",
        "final_retrieval_recall",
        "final_retrieval_precision",
        "negative_sample_false_positives",
        "negative_sample_false_positive_rate",
        "mean_selected_hotwords_per_sample",
    )
    overall = {key: _optional_delta(top7, top5, key) for key in keys}
    by_source: dict[str, object] = {}
    top5_sources = _mapping(_mapping(profiles, "top5"), "by_source")
    top7_sources = _mapping(_mapping(profiles, "top7"), "by_source")
    for source_name in sorted(top5_sources):
        old = _mapping(top5_sources, source_name)
        new = _mapping(top7_sources, source_name)
        by_source[source_name] = {key: _optional_delta(new, old, key) for key in keys}
    return {"overall": overall, "by_source": by_source}


def _verify_source_top5_metrics(
    replayed: Mapping[str, Any], source_summary: Mapping[str, Any]
) -> None:
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = [
        (
            _mapping(replayed, "overall"),
            _mapping(source_summary, "overall"),
            "overall",
        )
    ]
    replayed_sources = _mapping(replayed, "by_source")
    source_summaries = _mapping(source_summary, "by_source")
    if set(replayed_sources) != set(source_summaries):
        raise ValueError("source names differ from the completed Top-5 summary")
    pairs.extend(
        (
            _mapping(replayed_sources, source_name),
            _mapping(source_summaries, source_name),
            source_name,
        )
        for source_name in replayed_sources
    )
    keys = (
        "sample_count",
        "expected_hotwords",
        "raw_retrieved_expected_hotwords_at_5",
        "raw_recall_at_5",
        "selected_hotwords",
        "selected_true_positive_hotwords",
        "final_retrieval_recall",
        "final_retrieval_precision",
        "negative_sample_count",
        "negative_sample_false_positives",
        "negative_sample_false_positive_rate",
        "mean_selected_hotwords_per_sample",
    )
    for current, recorded, label in pairs:
        for key in keys:
            current_value = current.get(key)
            recorded_value = recorded.get(key)
            if current_value is None or recorded_value is None:
                equal = current_value is recorded_value
            elif isinstance(current_value, (int, float)) and isinstance(
                recorded_value, (int, float)
            ):
                equal = math.isclose(
                    float(current_value), float(recorded_value), rel_tol=0.0, abs_tol=1e-12
                )
            else:
                equal = current_value == recorded_value
            if not equal:
                raise ValueError(
                    f"replayed Top-5 metric differs from source summary for {label}.{key}: "
                    f"{current_value} != {recorded_value}"
                )


def _latency_summary(source_summary: Mapping[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    metrics: dict[str, Mapping[str, Any]] = {"overall": _mapping(source_summary, "overall")}
    metrics.update(
        {
            str(name): _mapping(_mapping(source_summary, "by_source"), str(name))
            for name in _mapping(source_summary, "by_source")
        }
    )
    for name, metric in metrics.items():
        latency = _mapping(metric, "latency_seconds")
        result[name] = {
            "pure_retrieval_seconds": _mapping(latency, "pure_retrieval_seconds"),
            "audio_to_result_seconds": _mapping(latency, "audio_to_result_seconds"),
            "audio_to_result_real_time_factor": metric.get("audio_to_result_real_time_factor"),
            "measurement_note": (
                "Observed in the source Top-5 model run; the CPU Top-7 replay does not "
                "remeasure Encoder or Anchor latency."
            ),
        }
    return result


def _markdown_table(comparison: Mapping[str, Any]) -> str:
    profiles = _mapping(comparison, "profiles")
    lines = [
        "# External Portuguese keyword retrieval: Top-5 vs Top-7\n\n",
        "Shared gate: threshold 0.75, posterior minimum 0.5, maximum edit ratio 0.35. "
        "Only Top-K changes. Top-7 is exactly replayed from the saved Anchor raw ranks; "
        "the model is not rerun.\n\n",
        "| Source | Profile | Raw recall | Final recall | Final precision | Selected | "
        "Negative sample FPR |\n",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    display_names = {
        "overall": "Overall",
        "mls_portuguese": "MLS",
        "delivery_20260706_ptbr": "Delivery",
    }
    for source_name in ("overall", "mls_portuguese", "delivery_20260706_ptbr"):
        for profile_name, top_k in (("top5", 5), ("top7", 7)):
            profile = _mapping(profiles, profile_name)
            metrics = (
                _mapping(profile, "overall")
                if source_name == "overall"
                else _mapping(_mapping(profile, "by_source"), source_name)
            )
            raw_hits = _required_int(metrics, f"raw_retrieved_expected_hotwords_at_{top_k}")
            expected = _required_int(metrics, "expected_hotwords")
            selected_hits = _required_int(metrics, "selected_true_positive_hotwords")
            selected = _required_int(metrics, "selected_hotwords")
            negative_fp = _required_int(metrics, "negative_sample_false_positives")
            negative_count = _required_int(metrics, "negative_sample_count")
            lines.append(
                f"| {display_names[source_name]} | Top-{top_k} | "
                f"{raw_hits}/{expected} ({_percent(metrics[f'raw_recall_at_{top_k}'])}) | "
                f"{selected_hits}/{expected} ({_percent(metrics['final_retrieval_recall'])}) | "
                f"{selected_hits}/{selected} ({_percent(metrics['final_retrieval_precision'])}) | "
                f"{selected} | {negative_fp}/{negative_count} "
                f"({_percent(metrics['negative_sample_false_positive_rate'])}) |\n"
            )
    lines.extend(
        [
            "\n## Latency scope\n\n",
            "Latency is reused from the completed source model run. It is not a Top-7 "
            "latency measurement; this replay only changes selection over saved ranks.\n",
        ]
    )
    return "".join(lines)


def _readme(source: Path, comparison: Mapping[str, object]) -> str:
    return (
        "# External Portuguese Top-7 replay\n\n"
        "Exact CPU replay of threshold 0.75 / posterior 0.5 / Top-7 from the completed "
        "Anchor raw ranks. The source Top-5 selection is reproduced for every sample "
        "before any output is written. No model, audio encoder, CTC Head, or Anchor query "
        "is rerun.\n\n"
        f"Source run: `{source}`\n\n"
        f"Samples: `{comparison['sample_count']}`\n\n"
        "Downstream files are split into `mls_retrieval_output.json` and "
        "`delivery_retrieval_output.json`.\n"
    )


def _validate_ranked_matches(
    matches: Sequence[Mapping[str, object]], *, known_ids: set[str], line_number: int
) -> None:
    previous_score = math.inf
    seen: set[str] = set()
    for match in matches:
        hotword_id = _required_string(match, "hotword_id")
        if hotword_id not in known_ids:
            raise ValueError(f"row {line_number} contains unknown hotword ID: {hotword_id}")
        if hotword_id in seen:
            raise ValueError(f"row {line_number} contains duplicate ranked hotword ID")
        seen.add(hotword_id)
        score = _required_number(match, "score")
        if score > previous_score:
            raise ValueError(f"row {line_number} raw matches are not score-sorted")
        previous_score = score
        _required_number(match, "edit_ratio")
        _required_number(match, "posterior_confidence")


def _verify_hash_manifest(root: Path) -> None:
    manifest = root / "sha256.txt"
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        pieces = raw_line.split(maxsplit=1)
        if len(pieces) != 2:
            raise ValueError(f"invalid SHA256 manifest row {line_number}: {manifest}")
        expected, raw_name = pieces
        name = raw_name.lstrip("* ")
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch for {name}")


def _verify_identity(identity: Mapping[str, Any], path: Path, label: str) -> None:
    if identity.get("sha256") != _sha256(path) or identity.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"{label} identity differs from the completed source run")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ValueError(f"field {key} is not an object")
    return nested


def _list_of_mappings(value: Mapping[str, Any], key: str) -> list[Mapping[str, object]]:
    nested = value.get(key)
    if not isinstance(nested, list) or any(not isinstance(item, dict) for item in nested):
        raise ValueError(f"field {key} is not a list of objects")
    return nested


def _string_list(value: Mapping[str, object], key: str) -> list[str]:
    nested = value.get(key)
    if not isinstance(nested, list) or any(not isinstance(item, str) for item in nested):
        raise ValueError(f"field {key} is not a list of strings")
    return nested


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"field {key} is not a non-empty string")
    return item


def _required_int(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"field {key} is not an integer")
    return item


def _required_number(value: Mapping[str, object], key: str) -> float:
    item = value.get(key)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
        raise ValueError(f"field {key} is not a finite number")
    return float(item)


def _optional_delta(
    candidate: Mapping[str, object], baseline: Mapping[str, object], key: str
) -> float | None:
    new = candidate.get(key)
    old = baseline.get(key)
    if new is None and old is None:
        return None
    return _required_number(candidate, key) - _required_number(baseline, key)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percent(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_hash_manifest(destination: Path) -> None:
    lines = [f"{_sha256(destination / name)}  {name}\n" for name in FINAL_FILES]
    _atomic_write_text(destination / "sha256.txt", "".join(lines))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"
