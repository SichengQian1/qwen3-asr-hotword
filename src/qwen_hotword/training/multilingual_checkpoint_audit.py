from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EXPECTED_GROUPS = ("en", "es", "pt")
EXPECTED_SELECTION = (
    "validation_macro_phoneme_error_rate_then_worst_group_"
    "phoneme_error_rate_then_validation_loss"
)
OUTPUT_FILES = ("checkpoint_audit.json", "README.md", "sha256.txt")


def audit_multilingual_ctc_checkpoint(
    training_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_groups: Sequence[str] = EXPECTED_GROUPS,
) -> dict[str, object]:
    """Freeze and summarize a completed multilingual Macro-PER CTC run."""

    source = Path(training_dir).expanduser()
    destination = Path(output_dir).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"training directory does not exist: {source}")
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {destination}"
        )

    report_path = source / "report.json"
    metrics_path = source / "metrics.jsonl"
    checkpoint_path = source / "ctc_head_best.pt"
    latest_checkpoint_path = source / "ctc_head_latest.pt"
    training_state_path = source / "training_state_latest.pt"
    for path in (
        report_path,
        metrics_path,
        checkpoint_path,
        latest_checkpoint_path,
        training_state_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required training artifact does not exist: {path}")

    report = _read_mapping(report_path)
    metrics = _read_jsonl(metrics_path)
    normalized_groups = tuple(sorted(set(expected_groups)))
    if not normalized_groups:
        raise ValueError("expected validation groups must not be empty")
    _validate_report(report, metrics, expected_groups=normalized_groups)

    best_epoch = _required_int(report, "best_epoch")
    best_metric = next(row for row in metrics if _required_int(row, "epoch") == best_epoch)
    best_by_group = _group_per(report, "best_validation_by_group", normalized_groups)
    final_by_group = _group_per(report, "final_validation_by_group", normalized_groups)
    initial_by_group = _group_per(report, "initial_validation_by_group", normalized_groups)
    checkpoint_identity = _file_identity(checkpoint_path)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "purpose": "freeze_multilingual_macro_ctc_checkpoint_for_end_to_end_regression",
        "test_set_used": False,
        "training_dir": str(source),
        "training_report": _file_identity(report_path),
        "metrics": _file_identity(metrics_path),
        "best_checkpoint": checkpoint_identity,
        "latest_checkpoint": _file_identity(latest_checkpoint_path),
        "training_state": _file_identity(training_state_path),
        "selection_contract": {
            "checkpoint_selection_metric": report["checkpoint_selection_metric"],
            "selection_metric": report["selection_metric"],
            "validation_groups": list(normalized_groups),
            "cache_sha256_verified": report["cache_sha256_verified"],
        },
        "training_progress": {
            "epochs_requested": report["epochs_requested"],
            "epochs_completed": report["epochs_completed"],
            "resumed_from_epoch": report["resumed_from_epoch"],
            "early_stopped": report["early_stopped"],
            "best_epoch": best_epoch,
            "final_learning_rate": report["final_learning_rate"],
        },
        "head_contract": report["head_config"],
        "best_validation": {
            "mixed_per": report["best_validation_phoneme_error_rate"],
            "macro_per": report["best_validation_macro_phoneme_error_rate"],
            "worst_group": report["best_validation_worst_group"],
            "worst_group_per": report[
                "best_validation_worst_group_phoneme_error_rate"
            ],
            "by_group_per": best_by_group,
        },
        "initial_validation": {
            "mixed_per": report["initial_validation_phoneme_error_rate"],
            "macro_per": report["initial_validation_macro_phoneme_error_rate"],
            "by_group_per": initial_by_group,
        },
        "final_validation": {
            "mixed_per": report["final_validation_phoneme_error_rate"],
            "macro_per": report["final_validation_macro_phoneme_error_rate"],
            "worst_group": report["final_validation_worst_group"],
            "worst_group_per": report[
                "final_validation_worst_group_phoneme_error_rate"
            ],
            "by_group_per": final_by_group,
        },
        "best_epoch_metrics_sha256": _mapping_sha256(best_metric),
    }

    destination.mkdir(parents=True)
    _write_json(destination / "checkpoint_audit.json", result)
    (destination / "README.md").write_text(
        "# Multilingual CTC checkpoint audit\n\n"
        "This compact, read-only audit freezes the selected en/es/pt Macro-PER "
        "checkpoint identity before end-to-end regression. It does not load Qwen, "
        "read the sealed test set, or modify the source training directory.\n",
        encoding="utf-8",
    )
    _write_hashes(destination)
    return result


def _validate_report(
    report: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    expected_groups: tuple[str, ...],
) -> None:
    if report.get("status") != "completed":
        raise ValueError("training report is not completed")
    if report.get("test_set_used") is not False:
        raise ValueError("training report must explicitly record test_set_used=false")
    if report.get("cache_sha256_verified") is not True:
        raise ValueError("training report did not verify feature-cache SHA256 identities")
    if report.get("checkpoint_selection_metric") != "validation_macro_per":
        raise ValueError("checkpoint was not selected by validation_macro_per")
    if report.get("selection_metric") != EXPECTED_SELECTION:
        raise ValueError("checkpoint selection tie-break contract is unexpected")
    groups = report.get("validation_groups")
    if not isinstance(groups, list) or tuple(sorted(str(value) for value in groups)) != (
        expected_groups
    ):
        raise ValueError("training report validation groups differ from expected groups")
    head = report.get("head_config")
    expected_head = {
        "head_type": "temporal_upsample",
        "input_dimension": 1024,
        "num_classes": 90,
        "hidden_dimension": 512,
        "kernel_size": 5,
        "time_upsampling_factor": 2,
    }
    if not isinstance(head, Mapping) or any(
        head.get(key) != value for key, value in expected_head.items()
    ):
        raise ValueError("training report Head contract is not temporal-2x h512-k5 / 90 classes")

    epochs_completed = _required_int(report, "epochs_completed")
    if epochs_completed <= 0 or len(metrics) != epochs_completed:
        raise ValueError("metrics row count does not match epochs_completed")
    epochs = [_required_int(row, "epoch") for row in metrics]
    if epochs != list(range(1, epochs_completed + 1)):
        raise ValueError("metrics epochs are not contiguous from 1 through epochs_completed")
    best_epoch = _required_int(report, "best_epoch")
    if best_epoch not in epochs:
        raise ValueError("best_epoch is absent from metrics.jsonl")

    best_groups = _group_per(report, "best_validation_by_group", expected_groups)
    macro = sum(best_groups.values()) / len(best_groups)
    reported_macro = _required_float(report, "best_validation_macro_phoneme_error_rate")
    if abs(macro - reported_macro) > 1e-12:
        raise ValueError("best validation Macro PER does not equal the group mean")
    worst_group = max(best_groups, key=lambda group: (best_groups[group], group))
    if report.get("best_validation_worst_group") != worst_group:
        raise ValueError("best validation worst group is inconsistent")
    if abs(
        best_groups[worst_group]
        - _required_float(report, "best_validation_worst_group_phoneme_error_rate")
    ) > 1e-12:
        raise ValueError("best validation worst-group PER is inconsistent")

    best_metric = next(row for row in metrics if _required_int(row, "epoch") == best_epoch)
    validation = best_metric.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("best metrics row has no validation mapping")
    if abs(
        _required_float(report, "best_validation_phoneme_error_rate")
        - _required_float(validation, "phoneme_error_rate")
    ) > 1e-12:
        raise ValueError("best report mixed PER differs from its metrics row")
    if abs(
        reported_macro
        - _required_float(best_metric, "validation_macro_phoneme_error_rate")
    ) > 1e-12:
        raise ValueError("best report Macro PER differs from its metrics row")


def _group_per(
    report: Mapping[str, Any],
    key: str,
    expected_groups: Sequence[str],
) -> dict[str, float]:
    raw = report.get(key)
    if not isinstance(raw, Mapping) or set(raw) != set(expected_groups):
        raise ValueError(f"{key} does not contain the expected groups")
    result: dict[str, float] = {}
    for group in expected_groups:
        metrics = raw.get(group)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{key}.{group} is not a metric mapping")
        result[group] = _required_float(metrics, "phoneme_error_rate")
    return result


def _required_int(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"{key} is not an integer")
    return result


def _required_float(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if not isinstance(result, int | float) or isinstance(result, bool):
        raise ValueError(f"{key} is not numeric")
    return float(result)


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON input is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    lines = []
    for name in OUTPUT_FILES:
        if name == "sha256.txt":
            continue
        path = destination / name
        lines.append(f"{_sha256(path)}  {name}\n")
    (destination / "sha256.txt").write_text("".join(lines), encoding="utf-8")
