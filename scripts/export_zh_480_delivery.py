"""Copy the six original Chinese groups from the completed bilingual delivery."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read

COUNTS = {
    "zh-aishell-test@@zh": 7176,
    "zh-aishell-kw_subset@@zh": 2135,
    "zh-smarthome-val_v1@@zh": 258,
    "zh-smarthome-val_v2@@zh": 200,
    "zh-smarthome-val_n1@@zh": 60,
    "zh-smarthome-val_n2@@zh": 150,
}
PLAN = "37b0d91c93abc9b9920336a45fa06f27deaf6455cdc498aa85102381987d52a0"


def chinese(groups: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    selected = {k: records for k, records in groups.items() if k.endswith("@@zh")}
    if {k: len(records) for k, records in selected.items()} != counts:
        raise ValueError("Chinese dataset keys or record counts differ from the source run")
    return selected


def export(source: Path, output: Path, counts: dict[str, int]) -> Path:
    with zipfile.ZipFile(source) as original:

        def load(name: str) -> Any:
            with original.open(f"delivery/{name}.json") as stream:
                return json.load(stream)

        with stage_progress("Read Chinese durations, audio manifest and original Top5/Top7"):
            durations = chinese(load("durations"), counts)
            manifest = chinese(load("audio_manifest"), counts)
            all_modes = load("stream_contexts")
            if all_modes.keys() != MODES.keys():
                raise ValueError("original delivery modes differ")
            contexts = {m: chinese(all_modes[m], counts) for m in MODES}
            del all_modes
    with stage_progress("Validate six Chinese datasets", total=len(counts), every=1) as tick:
        for i, (key, records) in enumerate(manifest.items(), 1):
            if records.keys() != durations[key].keys():
                raise ValueError(f"manifest and duration IDs differ: {key}")
            for uid, s in records.items():
                if (s["key"], s["sample_id"], s["language"], s["duration_seconds"]) != (
                    key,
                    uid,
                    "zh",
                    durations[key][uid],
                ):
                    raise ValueError(f"manifest identity differs: {key}/{uid}")
            validate_contexts({m: {key: contexts[m][key]} for m in MODES}, {key: durations[key]})
            print(f"{key}: {len(records):,}", flush=True)
            tick(i)
    with stage_progress("Write and verify Chinese delivery"):
        folder = Path(tempfile.mkdtemp(prefix="zh_480_delivery_", dir=output))
        pending = folder / "delivery.pending"
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in [
                ("durations", durations),
                ("audio_manifest", manifest),
                ("stream_contexts", contexts),
            ]:
                archive.writestr(
                    f"delivery/{name}.json",
                    json.dumps(
                        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ).encode(),
                )
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise ValueError("Chinese delivery archive CRC failed")
        result = folder / "delivery.zip"
        pending.replace(result)
    return result


def main() -> None:
    source = ROOT / "outputs/zh_en183_480_joint2351_eleven_v1"
    report = read(source / "report.json")
    if (report["status"], report["plan_sha256"], digest(read(source / "run_config.json"))) != (
        "completed",
        PLAN,
        PLAN,
    ):
        raise ValueError("expected completed 480h epoch-25 source run")
    result = export(source / "delivery.zip", source.parent, COUNTS)
    print("请交给下游（中文六组）：", result, flush=True)


if __name__ == "__main__":
    main()
