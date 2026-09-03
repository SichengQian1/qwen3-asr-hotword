from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LANGUAGES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}
GROUPS = ("C", "D", "E")
EXPECTED_CTC_CHECKPOINT_SHA256 = "bd9df8072b7efe7fafa599e958bbd7ca8405b289d0a353913d865340764d01a0"
QUALITY_FIELDS = (
    "expected_hotwords",
    "correct_prompt_injected_hotwords",
    "prompt_hotword_recall",
    "correct_prompt_adopted_hotwords",
    "correct_prompt_adoption_rate",
    "wrong_prompt_injected_hotwords",
    "wrong_prompt_written_hotwords",
    "wrong_prompt_landing_rate",
    "final_hotword_recall",
    "final_hotword_precision",
    "sample_hotword_hit_rate",
    "negative_hotword_hallucination_rate",
    "wer",
    "cer",
    "mean_inference_seconds",
)
D5_GATE = {
    "threshold": 0.86,
    "top_k": 5,
    "maximum_edit_ratio": 0.35,
    "posterior_weight": 0.25,
    "minimum_posterior_confidence": 0.0,
    "minimum_top1_margin": 0.0,
}


def summarize_multilingual_streaming_e2e(
    language_run_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
) -> dict[str, object]:
    if set(language_run_dirs) != set(LANGUAGES):
        raise ValueError("language runs must contain exactly en, es, and pt")
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(f"multilingual summary output already exists: {destination}")

    checkpoint_sha: str | None = None
    per_language: dict[str, object] = {}
    rows_by_language: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for code in LANGUAGES:
        root = Path(language_run_dirs[code]).expanduser()
        _verify_sha256_manifest(root / "sha256.txt")
        config = _read_mapping(root / "run_config.json")
        summary = _read_mapping(root / "summary.json")
        if summary.get("status") != "pass" or summary.get("test_set_used") is not False:
            raise ValueError(f"{code} streaming run is not a passed non-test run")
        if config.get("language") != LANGUAGES[code]:
            raise ValueError(f"{code} run language does not match {LANGUAGES[code]}")
        if tuple(config.get("groups", ())) != GROUPS:
            raise ValueError(f"{code} run must contain only C,D,E")
        _validate_d5_gate(config, code)
        current_checkpoint = _checkpoint_sha(config)
        if current_checkpoint != EXPECTED_CTC_CHECKPOINT_SHA256:
            raise ValueError(f"{code} run does not use the sealed multilingual checkpoint")
        if checkpoint_sha is None:
            checkpoint_sha = current_checkpoint
        elif current_checkpoint != checkpoint_sha:
            raise ValueError("language runs use different CTC checkpoints")

        raw_groups = summary.get("groups")
        if not isinstance(raw_groups, Mapping) or set(raw_groups) != set(GROUPS):
            raise ValueError(f"{code} summary groups are not exactly C,D,E")
        rows = _read_jsonl(root / "sample_results.jsonl")
        grouped_rows = {
            group: [row for row in rows if row.get("experiment_group") == group] for group in GROUPS
        }
        selections = {_selection(grouped_rows[group]) for group in GROUPS}
        if len(selections) != 1 or not next(iter(selections)):
            raise ValueError(f"{code} C,D,E sample selections differ or are empty")
        if len(next(iter(selections))) != 100:
            raise ValueError(f"{code} run does not contain exactly 100 samples per group")
        if len(rows) != 300:
            raise ValueError(f"{code} run contains rows outside the 100-sample C,D,E scope")
        rows_by_language[code] = grouped_rows
        quality = {group: _selected_quality(_mapping(raw_groups, group)) for group in GROUPS}
        per_language[code] = {
            "language": LANGUAGES[code],
            "run_dir": str(root),
            "run_config_sha256": _sha256(root / "run_config.json"),
            "summary_sha256": _sha256(root / "summary.json"),
            "sample_count": len(next(iter(selections))),
            "groups": quality,
            "d_minus_c": _metric_delta(quality["D"], quality["C"]),
            "e_minus_d": _metric_delta(quality["E"], quality["D"]),
        }

    assert checkpoint_sha is not None
    aggregate = {
        group: _aggregate_group(
            [_mapping(_mapping(per_language, code), "groups", group) for code in LANGUAGES],
            group=group,
        )
        for group in GROUPS
    }
    sample_ids = {
        code: {str(row.get("sample_id")) for row in rows_by_language[code]["C"]}
        for code in LANGUAGES
    }
    overlaps = {
        f"{left}-{right}": len(sample_ids[left] & sample_ids[right])
        for index, left in enumerate(LANGUAGES)
        for right in tuple(LANGUAGES)[index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"cross-language sample ID overlap detected: {overlaps}")

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "test_set_used": False,
        "evaluation": "qwen3_asr_multilingual_4k_streaming_e2e_formal100",
        "ctc_checkpoint_sha256": checkpoint_sha,
        "gate": D5_GATE,
        "languages": list(LANGUAGES),
        "sample_count_per_language": {
            code: _mapping(per_language, code)["sample_count"] for code in LANGUAGES
        },
        "total_sample_count": sum(
            int(_mapping(per_language, code)["sample_count"]) for code in LANGUAGES
        ),
        "identity_checks": {
            "streaming_run_sha256_manifests_verified": True,
            "same_ctc_checkpoint": True,
            "same_d5_gate": True,
            "c_d_e_selection_equal_within_each_language": True,
            "cross_language_sample_id_overlaps": overlaps,
        },
        "per_language": per_language,
        "aggregate": {
            "groups": aggregate,
            "d_minus_c": _metric_delta(aggregate["D"], aggregate["C"]),
            "e_minus_d": _metric_delta(aggregate["E"], aggregate["D"]),
            "wer_cer_scope": "unweighted macro average across en/es/pt",
            "count_metric_scope": "micro aggregate across all expected hotwords",
        },
    }
    destination.mkdir(parents=True)
    _write_json(destination / "multilingual_e2e_summary.json", report)
    (destination / "README.md").write_text(
        "# Multilingual streaming end-to-end summary\n\n"
        "This report combines sealed English, Spanish, and Portuguese formal100 C/D/E "
        "runs at the shared 4k D5 operating point. Count metrics are micro-aggregated; "
        "WER and CER are language-macro diagnostics. No test split is used.\n",
        encoding="utf-8",
    )
    _write_hashes(destination)
    return report


def _aggregate_group(
    qualities: Sequence[Mapping[str, Any]],
    *,
    group: str,
) -> dict[str, object]:
    expected = sum(_integer(value, "expected_hotwords") for value in qualities)
    matched = sum(
        _derived_count(value, "final_hotword_recall", "expected_hotwords") for value in qualities
    )
    result: dict[str, object] = {
        "expected_hotwords": expected,
        "final_correct_hotwords": matched,
        "final_hotword_recall": _ratio(matched, expected),
        "sample_hotword_hit_rate_macro": _mean_field(qualities, "sample_hotword_hit_rate"),
        "negative_hotword_hallucination_rate_macro": _mean_field(
            qualities, "negative_hotword_hallucination_rate"
        ),
        "wer_macro": _mean_field(qualities, "wer"),
        "cer_macro": _mean_field(qualities, "cer"),
        "mean_inference_seconds_macro": _mean_field(qualities, "mean_inference_seconds"),
    }
    if group == "C":
        result["final_hotword_precision"] = None
        return result
    wrong_written = sum(_integer(value, "wrong_prompt_written_hotwords") for value in qualities)
    result.update(
        {
            "wrong_prompt_written_hotwords": wrong_written,
            "final_hotword_precision": _ratio(matched, matched + wrong_written),
        }
    )
    if group == "D":
        injected = sum(_integer(value, "correct_prompt_injected_hotwords") for value in qualities)
        adopted = sum(_integer(value, "correct_prompt_adopted_hotwords") for value in qualities)
        wrong_injected = sum(
            _integer(value, "wrong_prompt_injected_hotwords") for value in qualities
        )
        result.update(
            {
                "correct_prompt_injected_hotwords": injected,
                "prompt_hotword_recall": _ratio(injected, expected),
                "correct_prompt_adopted_hotwords": adopted,
                "correct_prompt_adoption_rate": _ratio(adopted, injected),
                "wrong_prompt_injected_hotwords": wrong_injected,
                "wrong_prompt_landing_rate": _ratio(wrong_written, wrong_injected),
            }
        )
    return result


def _validate_d5_gate(config: Mapping[str, Any], code: str) -> None:
    actual = {key: config.get(key) for key in D5_GATE}
    if actual != D5_GATE:
        raise ValueError(f"{code} run does not use the sealed D5 gate")


def _checkpoint_sha(config: Mapping[str, Any]) -> str:
    inputs = config.get("inputs")
    checkpoint = inputs.get("checkpoint") if isinstance(inputs, Mapping) else None
    sha = checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError("run config has no valid checkpoint SHA256")
    return sha


def _selection(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(row.get("case_id")),
                str(row.get("sample_id")),
                str(row.get("reference_text")),
            )
            for row in rows
        )
    )


def _selected_quality(value: Mapping[str, Any]) -> dict[str, object]:
    return {name: value.get(name) for name in QUALITY_FIELDS}


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float | None]:
    names = (
        "prompt_hotword_recall",
        "final_hotword_recall",
        "final_hotword_precision",
        "sample_hotword_hit_rate",
        "sample_hotword_hit_rate_macro",
        "negative_hotword_hallucination_rate",
        "negative_hotword_hallucination_rate_macro",
        "wer",
        "wer_macro",
        "cer",
        "cer_macro",
    )
    return {name: _numeric_delta(left.get(name), right.get(name)) for name in names}


def _numeric_delta(left: object, right: object) -> float | None:
    if (
        isinstance(left, int | float)
        and not isinstance(left, bool)
        and isinstance(right, int | float)
        and not isinstance(right, bool)
    ):
        return float(left) - float(right)
    return None


def _integer(value: Mapping[str, Any], name: str) -> int:
    raw = value.get(name)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"quality field {name} is not an integer")
    return raw


def _derived_count(value: Mapping[str, Any], rate_name: str, total_name: str) -> int:
    rate = value.get(rate_name)
    total = _integer(value, total_name)
    if not isinstance(rate, int | float) or isinstance(rate, bool):
        raise ValueError(f"quality field {rate_name} is not numeric")
    count = round(float(rate) * total)
    if abs(float(rate) - _ratio(count, total)) > 1e-9:
        raise ValueError(f"quality field {rate_name} cannot be mapped to an exact count")
    return count


def _mean_field(values: Sequence[Mapping[str, Any]], name: str) -> float:
    numbers: list[float] = []
    for value in values:
        raw = value.get(name)
        if not isinstance(raw, int | float) or isinstance(raw, bool):
            raise ValueError(f"quality field {name} is not numeric for every language")
        numbers.append(float(raw))
    return sum(numbers) / len(numbers)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mapping(value: Mapping[str, Any], name: str, child: str | None = None) -> Mapping[str, Any]:
    selected = value.get(name)
    if child is not None:
        selected = selected.get(child) if isinstance(selected, Mapping) else None
    if not isinstance(selected, Mapping):
        suffix = f".{child}" if child is not None else ""
        raise ValueError(f"missing mapping {name}{suffix}")
    return selected


def _verify_sha256_manifest(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"SHA256 manifest does not exist: {path}")
    entries = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        digest, separator, raw_name = line.partition("  ")
        if not separator or len(digest) != 64 or not raw_name:
            raise ValueError(f"invalid SHA256 manifest row: {path}:{line_number}")
        local = path.parent / raw_name
        fallback = Path(raw_name).expanduser()
        target = local if local.is_file() else fallback
        if not target.is_file():
            raise FileNotFoundError(f"SHA256 target does not exist for {path}: {raw_name}")
        if _sha256(target) != digest:
            raise ValueError(f"SHA256 mismatch for {target} listed by {path}")
        entries += 1
    if not entries:
        raise ValueError(f"SHA256 manifest is empty: {path}")


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_hashes(destination: Path) -> None:
    names = ("multilingual_e2e_summary.json", "README.md")
    (destination / "sha256.txt").write_text(
        "".join(f"{_sha256(destination / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
