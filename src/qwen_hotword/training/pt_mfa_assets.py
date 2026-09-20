"""Separate host-side download and offline verification of fixed MFA pilot assets."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from qwen_hotword.training.pt_mfa_pilot import sha256_file, write_json_new

VERSION = "v2.0.0a"
DEFAULT_ASSETS = "models/mfa/pt_pilot_v2_0_0a"
BASE_URL = "https://github.com/MontrealCorpusTools/mfa-models/releases/download"
ASSETS = {
    "acoustic": {
        "name": "portuguese_mfa.zip",
        "size_bytes": 91604181,
        "url": f"{BASE_URL}/acoustic-portuguese_mfa-{VERSION}/portuguese_mfa.zip",
    },
    "dictionary": {
        "name": "portuguese_brazil_mfa.dict",
        "size_bytes": 1583153,
        "url": f"{BASE_URL}/dictionary-portuguese_brazil_mfa-{VERSION}/portuguese_brazil_mfa.dict",
    },
}


def verify_assets(root: Path) -> dict[str, Any]:
    """Check pinned metadata and download-time digests without network or writes."""
    receipt = root / "assets.json"
    if not receipt.is_file():
        raise ValueError(
            f"Missing {receipt}; outside the container run: "
            "python3 -B scripts/download_pt_mfa_assets.py"
        )
    data = json.loads(receipt.read_text(encoding="utf-8"))
    if data.get("model_version") != VERSION:
        raise ValueError("MFA asset version mismatch")
    result = {}
    for kind, spec in ASSETS.items():
        recorded = data.get("assets", {}).get(kind, {})
        path = root / str(spec["name"])
        if any(recorded.get(key) != value for key, value in spec.items()):
            raise ValueError(f"MFA {kind} receipt metadata mismatch")
        if path.stat().st_size != spec["size_bytes"]:
            raise ValueError(f"MFA {kind} size mismatch")
        digest = sha256_file(path)
        if digest != recorded.get("sha256"):
            raise ValueError(f"MFA {kind} SHA256 mismatch; do not overwrite this asset directory")
        result[kind] = {"path": str(path.resolve()), "sha256": digest, **spec}
    return result


def download_assets(destination: Path) -> dict[str, Any]:
    """Download with host curl trust, publish only complete verified assets; never overwrite."""
    destination = destination.resolve()
    if destination.exists():
        return {"status": "verified_existing", "assets": verify_assets(destination)}
    curl = shutil.which("curl")
    if not curl:
        raise ValueError("curl is required on the download host")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + "_download_", dir=destination.parent))
    print(f"Download staging (retained on failure): {staging}", flush=True)
    receipt: dict[str, Any] = {"model_version": VERSION, "assets": {}}
    for kind, spec in ASSETS.items():
        path = staging / str(spec["name"])
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--show-error",
                "--retry",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                "600",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--output",
                str(path),
                str(spec["url"]),
            ],
            check=True,
            timeout=1900,
        )
        if path.stat().st_size != spec["size_bytes"]:
            raise ValueError(f"Unexpected {kind} size; preserved in {staging}")
        if kind == "acoustic":
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Acoustic ZIP CRC check failed")
        elif not path.read_text(encoding="utf-8-sig").strip():
            raise ValueError("Empty MFA dictionary")
        receipt["assets"][kind] = {**spec, "sha256": sha256_file(path)}
    write_json_new(staging / "assets.json", receipt)
    verify_assets(staging)
    # Reserve the final directory exclusively; partial publication is never reused silently.
    destination.mkdir(exist_ok=False)
    for spec in ASSETS.values():
        (staging / str(spec["name"])).rename(destination / str(spec["name"]))
    (staging / "assets.json").rename(destination / "assets.json")
    staging.rmdir()  # Our now-empty staging directory only; no existing outputs are touched.
    return {"status": "downloaded", "assets": verify_assets(destination)}
