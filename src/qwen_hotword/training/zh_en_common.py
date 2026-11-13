"""Immutable, resumable artifacts for the Chinese/US/UK training run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, cast

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save_rows(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    pending.replace(path)


def completed(folder: Path, identity: Any) -> dict[str, Any] | None:
    marker = folder / "completed.json"
    if not marker.exists():
        return None
    value = read(marker)
    if value["identity"] != identity:
        raise ValueError(f"changed stage identity: {folder}; use a new run directory")
    for relative, expected in value["files"].items():
        pin(folder / relative, expected)
    return cast(dict[str, Any], value["summary"])


def finish(folder: Path, identity: Any, files: list[Path], summary: Any) -> Any:
    write_json(
        folder / "completed.json",
        {
            "identity": identity,
            "files": {str(p.relative_to(folder)): sha(p) for p in files},
            "summary": summary,
        },
    )
    return summary


def contract(root: Path, output: Path, cfg: dict[str, Any], foundation: dict[str, Any]) -> None:
    if cfg["noise_percent"] != 70 or cfg["snr_db"] != 10:
        raise ValueError("this run is fixed to 70 percent at 10 dB")
    if cfg["encoder_frozen"] is not True or cfg["head_initialization"] != "random":
        raise ValueError("original frozen Encoder and random Head required")
    if cfg["head"]["type"] != "temporal_upsample":
        raise ValueError("this run uses the established temporal Head")
    if cfg["num_classes"] != 183 or cfg["head"]["time_upsampling_factor"] != 2:
        raise ValueError("approved 183-class / 2x initial Head required")
    protected = [
        Path(cfg["model"]),
        Path(cfg["noise_root"]),
        Path(foundation["chinese_root"]),
        *(Path(p) for p in foundation["english_sources"].values()),
        root / "models",
    ]
    for source in protected:
        source = source.resolve()
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError(f"output overlaps protected source {source}")
    paths = [
        *sorted((root / "src/qwen_hotword/training").glob("zh_en_*.py")),
        root / "scripts/run_zh_en_training.py",
        root / cfg["foundation_config"],
        root / foundation["vocab"],
        root / "scripts/prepare_mfa_v03.py",
    ]
    for name in (
        "training/feature_cache.py",
        "training/ctc_overfit.py",
        "training/sharded_ctc.py",
        "training/ctc_diagnostics.py",
        "training/noise_audio.py",
        "training/g2p_prep.py",
        "training/en_ar_source_audit.py",
        "training/noah_source_inspection.py",
        "training/experiment_a.py",
        "modeling/audio_encoder.py",
        "modeling/ctc_head.py",
        "modeling/qwen_backbone.py",
        "phonemes/coverage.py",
        "phonemes/mfa_v03_assets.py",
        "phonemes/mfa_v03_runtime.py",
    ):
        paths.append(root / "src/qwen_hotword" / name)
    value = {
        "config": cfg,
        "foundation": foundation,
        "code_sha256": {str(p.relative_to(root)): sha(p) for p in paths},
    }
    path = output / "contract.json"
    if path.exists():
        if read(path) != value:
            raise ValueError("code/config differs from this run; preserve it and use a new output")
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("refusing unowned nonempty output")
        write_json(path, value)
        write_json(
            output / "identity.json",
            {
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "contract_sha256": sha(path),
                "training_started": False,
            },
        )
