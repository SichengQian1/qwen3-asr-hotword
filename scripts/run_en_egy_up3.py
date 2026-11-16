#!/usr/bin/env python3
"""Run x3 from frozen new EGY labels, independently of x2 cache/training completion."""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_egy_dynamic import read, seal
from qwen_hotword.training.stage_progress import stage_progress
from scripts.arabic_diagnostic_common import completed_run
from scripts.label_arabic_native import exclusive_job


def copy_sealed(source: Path, target: Path, digest: str) -> None:
    pin(source, digest)
    if target.exists():
        pin(target, digest)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".pending")
    shutil.copyfile(source, pending)
    pin(pending, digest)
    pending.replace(target)


def prepare(root: Path, source: Path, output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("x3 output must be independent of the x2 source")
    with stage_progress("Read frozen NEW EGY labels/selection; x2 final report NOT required"):
        prepared = read(source / "prepare_report.json")
        original = read(source / "contract.json")
        if prepared["status"] != "prepared" or prepared["arabic_msa_used"] is not False:
            raise ValueError("requires frozen new pure-EGY data")
        cfg = {
            **original["configuration"],
            "head_time_upsampling_factor": 3,
            "output_dir": str(output),
        }
        if cfg.get("maximum_hours_per_language") != 480:
            raise ValueError("requires the new capped 480h/language selection")
        base = completed_run(root / cfg["source_run"])
        if base["identity"] != original["source_identity"]:
            raise ValueError("frozen r15 Encoder source identity changed")
        vocab = root / cfg["vocab"]
        pin(vocab, cfg["vocab_sha256"])
        if prepared["vocab_sha256"] != cfg["vocab_sha256"]:
            raise ValueError("frozen labels vocabulary mismatch")
        files = {
            str(source / name): sha(source / name)
            for name in ("contract.json", "prepare_report.json")
        }
        files[str(vocab)] = sha(vocab)
        for split in ("train", "validation"):
            item = prepared["manifests"][split]
            path = source / f"{split}.jsonl"
            if path.resolve() != Path(item["path"]).resolve():
                raise ValueError("manifest does not belong to selected source")
            pin(path, item["sha256"])
            files[str(path)] = item["sha256"]
        code = [Path(__file__), root / "scripts/arabic_diagnostic_common.py"]
        code.extend(
            root / "src/qwen_hotword/training" / name
            for name in (
                "en_egy_dynamic_train.py",
                "arabic_refresh_train.py",
                "feature_cache.py",
                "sharded_ctc.py",
                "ctc_overfit.py",
            )
        )
        contract = {
            "configuration": cfg,
            "source_files": files,
            "source_identity": base["identity"],
            "selection_source": str(source),
            "code": {str(p): sha(p) for p in code},
            "fresh_head": True,
            "wait_for_x2_completion": False,
            "test_set_used": False,
        }
        if output.exists() and not (output / "standalone_contract.json").exists():
            raise ValueError("existing unsealed output; preserve it and choose a new directory")
        output.mkdir(parents=True, exist_ok=True)
        seal(output / "standalone_contract.json", contract)
        selected = copy.deepcopy(prepared)
        # Keep identical bytes/IDs and the x2 feasibility floor, changing only Head expansion.
        selected["manifests"] = {}
        for split in ("train", "validation"):
            item = prepared["manifests"][split]
            destination = output / f"{split}.jsonl"
            copy_sealed(Path(item["path"]), destination, item["sha256"])
            selected["manifests"][split] = {**item, "path": str(destination)}
        selected["selection_source"] = str(source)
        selected["test_manifest_not_read"] = True
        seal(output / "prepare_report.json", selected)
    return cfg, base


def snapshot_shards(source: Path, output: Path, model: dict[str, Any], vocab: Path) -> None:
    """Snapshot only atomically published shard pairs; never write through to the x2 run."""
    marker = output / "cache_snapshot.json"
    if marker.exists():
        return
    result = {}
    for split in ("train", "validation"):
        folder = source / "cache" / split
        if not (folder / "cache_config.json").exists():
            result[split] = {"reused_shards": 0, "reason": "no_source_cache_yet"}
            continue
        config = read(folder / "cache_config.json")
        expected = {
            "path": model["view"],
            "dtype": "bfloat16",
            "config_sha256": model["files"]["config.json"]["sha256"],
            "weight_index_sha256": model["index_sha256"],
        }
        if config["model"] != expected or config["vocab"]["sha256"] != sha(vocab):
            raise ValueError("source partial cache Encoder/vocabulary identity mismatch")
        pin(source / f"{split}.jsonl", config["source_manifest"]["sha256"])
        metas = sorted((folder / "shards").glob("shard-*.json"))
        copied = linked = 0
        with stage_progress(
            f"Snapshot completed {split} shards; ignore unfinished pairs", total=len(metas)
        ) as tick:
            for i, metadata_path in enumerate(metas, 1):
                old = metadata_path.with_suffix(".pt")
                if not old.is_file():
                    tick(i)
                    continue
                metadata = read(metadata_path)
                if metadata["split"] != split or metadata["feature_bytes"] != old.stat().st_size:
                    raise ValueError("published source shard identity mismatch: " + str(old))
                digest = metadata["feature_sha256"]
                target = output / "cache" / split / "shards" / old.name
                target.parent.mkdir(parents=True, exist_ok=True)
                pin(old, digest)
                if target.exists():
                    pin(target, digest)
                else:
                    try:
                        os.link(old, target)
                        linked += 1
                    except OSError as error:
                        if error.errno != errno.EXDEV:
                            raise
                        if shutil.disk_usage(output).free < old.stat().st_size + 2 * 1024**3:
                            raise ValueError(
                                "insufficient space for cross-filesystem shard copy"
                            ) from error
                        copy_sealed(old, target, digest)
                        copied += 1
                seal(target.with_suffix(".json"), {**metadata, "feature_path": str(target)})
                tick(i)
        result[split] = {
            "published_pairs": len(metas),
            "new_hardlinks": linked,
            "new_copies": copied,
        }
    write_json(marker, {"source": str(source), "splits": result, "source_unchanged": True})


def run(root: Path, source: Path, output: Path, gpu: int) -> dict[str, Any]:
    from qwen_hotword.training.en_egy_dynamic_train import run as train

    source, output = source.resolve(), output.resolve()
    # Lock only our independent output, never block on the running x2 orchestrator.
    with exclusive_job(output):
        cfg, base = prepare(root, source, output)
        snapshot_shards(source, output, base["plan"]["model"], root / cfg["vocab"])
        result = train(root, cfg, output, base, gpu)
        contract = read(output / "standalone_contract.json")
        for name, digest in {**contract["source_files"], **contract["code"]}.items():
            pin(Path(name), digest)
        for item in contract["source_identity"].values():
            pin(Path(item["path"]), item["sha256"])
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=ROOT / "outputs/en_egy_480_ctc_v1")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/en_egy_480_up3_standalone_v1"
    )
    parser.add_argument("--gpu", type=int, default=4)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        print(
            json.dumps(
                run(ROOT, args.source_run, args.output_dir, args.gpu), ensure_ascii=False, indent=2
            )
        )
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
