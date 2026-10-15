"""Explicitly reviewed preparation-code upgrades; never relax dataset identity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_capacity import _sha, _verified

# These are whole-file SHA256 values, not Git commit IDs. Only this exact file and
# reviewed source/target pairs may differ from an existing preparation contract.
FAST_MODULE = Path(__file__).with_name("english_refresh_fast.py")
REVIEWED_TARGET = "ed283d77e9c7443c99bd4e682af51283300f93c217d7691226656667c14821dc"
REVIEWED_SOURCES = {
    "5d69e4a99650ec42469c87c274f1975c3a1e18c8d2fd3b6bc0590a5620cb83cf": "2615e3d",
    "59ac7fc038578cfe22481ae337eacb6d865a77a40b72b073cd6d763e0fe5f6b1": "6b85bfc",
}


def verify_preparation_inputs(output: Path) -> list[dict[str, str]]:
    checked: dict[str, Any] = {}
    contract_path = _verified(output, "preparation_contract.json", checked)
    saved = json.loads(contract_path.read_text())
    changes: list[dict[str, str]] = []
    for filename, expected in saved["input_sha256"].items():
        path = Path(filename)
        actual = _sha(path)
        if actual == expected:
            continue
        if path != FAST_MODULE or expected not in REVIEWED_SOURCES or actual != REVIEWED_TARGET:
            raise ValueError(f"input changed: {filename}")
        changes.append(
            {
                "path": filename,
                "recorded_sha256": expected,
                "current_sha256": actual,
                "recorded_version": REVIEWED_SOURCES[expected],
                "reason": "Reviewed runtime-only update; preparation rules unchanged",
            }
        )
    if changes:
        # Preserve the original contract/seal/configuration. Publish an additive,
        # immutable audit only after every unchanged input has passed its checksum.
        audit = {
            "status": "reviewed_code_upgrade",
            "schema_version": 1,
            "preparation_contract_sha256": _sha(contract_path),
            "checked_inputs": len(saved["input_sha256"]),
            "code_changes": changes,
            "dataset_checks_relaxed": False,
        }
        content = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        digest = hashlib.sha256(content).hexdigest()
        audit_path = output / f"code_compatibility_{digest[:16]}.json"
        try:
            with audit_path.open("xb") as handle:
                handle.write(content)
        except FileExistsError:
            if audit_path.read_bytes() != content:
                raise ValueError(f"existing compatibility audit differs: {audit_path}") from None
        for change in changes:
            print(
                "Reviewed preparation-code upgrade: "
                + change["path"]
                + " "
                + change["recorded_sha256"][:12]
                + " -> "
                + change["current_sha256"][:12]
                + "; data/label checks retained; audit="
                + str(audit_path),
                flush=True,
            )
    return changes
