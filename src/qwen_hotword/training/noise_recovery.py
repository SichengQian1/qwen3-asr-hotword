"""Audit immutable noise caches, then resume with explicit boundary-frame exceptions.

Only the two documented, additive source upgrades are compatible with the old
plan. No plan, labels, waveform, feature tensor or training fingerprint is edited.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_hotword.training.english_newval import pin, save
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.noise_audio import read_audio
from qwen_hotword.training.noise_cache import compose_views
from qwen_hotword.training.noise_run import bundle, train
from qwen_hotword.training.noise_status import _object, _tensor_index
from qwen_hotword.training.spanish_480_plan import _rows
from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress

RECOVERY = "recovery_boundary_v1"
HISTORICAL_CODE = {
    "sharded_ctc.py": "00003bb68f646627eeb7c963874ba550620642c54d69001a11057bc74dfa2dee",
    "noise_run.py": "45a2ada53ea1891caf2c705bd6fef2daad1355da39356bd82495d0e6cd234e55",
}


def encoder_frames(mel_frames: int) -> int:
    """Qwen3-ASR 0.0.6's 100-mel-frame chunk / three stride-2 convolutions."""
    return (mel_frames // 100) * 13 + ((mel_frames % 100) + 7) // 8


def frame_bounds(samples: int) -> tuple[int, int]:
    if type(samples) is not int or samples <= 0:
        raise ValueError("invalid recorded 16 kHz sample count")
    return encoder_frames(samples // 160), encoder_frames((samples + 159) // 160)


def processor_probe() -> dict[str, Any]:
    """Exercise real preprocessing, without a tokenizer, model weights or CUDA."""
    import numpy as np
    from transformers import WhisperFeatureExtractor

    versions = {
        "transformers": importlib.metadata.version("transformers"),
        "qwen_asr": importlib.metadata.version("qwen-asr"),
    }
    if versions != {"transformers": "4.57.6", "qwen_asr": "0.0.6"}:
        raise ValueError(f"boundary recovery requires the recorded source runtime: {versions}")
    # Upstream's constructor is untyped and its padding annotation excludes bool.
    extractor_factory: Any = WhisperFeatureExtractor
    extractor = extractor_factory(feature_size=128)
    wave = np.zeros(108944, dtype=np.float32)
    values = []
    for audio in ([wave], [wave, np.zeros(160000, dtype=np.float32)]):
        features = extractor(
            audio,
            sampling_rate=16000,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="np",
        )
        values.append(int(features["attention_mask"][0].sum()))
    if values != [680, 681]:
        raise ValueError(f"installed preprocessing does not match boundary policy: {values}")
    return {
        **versions,
        "samples": len(wave),
        "mel_frames_alone_with_longer": values,
        "qwen_frames_alone_with_longer": [encoder_frames(v) for v in values],
        "gpu_used": False,
        "model_loaded": False,
    }


def verify_source(root: Path, run: Path, config: dict[str, Any]) -> dict[str, Any]:
    plan = _object(run / "plan.json")
    pin(run / "plan.json", {}, _object(run / "prepared.json")["plan_sha256"])
    if config != plan["configuration"]:
        raise ValueError("recovery configuration differs from the sealed experiment")
    upgraded = {
        (root / "src/qwen_hotword/training" / name).resolve(): digest
        for name, digest in HISTORICAL_CODE.items()
    }
    compatibility = []
    with stage_progress("Verify recovery source plan, inputs and explicit code compatibility"):
        for name, expected in plan["input_sha256"].items():
            path = Path(name)
            actual = _sha(path)
            if actual != expected:
                if upgraded.get(path.resolve()) != expected:
                    raise ValueError(f"unapproved input/code drift during recovery: {path}")
                compatibility.append({"path": str(path), "historical": expected, "current": actual})
        pin(run / "noise_inventory.json", {}, plan["noise_inventory_sha256"])
        for part in plan["parts"]:
            pin(run / part["manifest"], {}, part["sha256"])
    marker = _object(run / "cache_completed.json")
    if marker["status"] != "noise_cache_completed" or marker["parts"] != len(plan["parts"]):
        raise ValueError("recovery requires a completed shared noise cache")
    # Authenticate metadata/audits now; compose_views rehashes all feature tensors
    # before a Head can train. The read-only audit does not page feature values in.
    with stage_progress("Verify saved mixing and feature metadata identities"):
        for name, digest in marker["sha256"].items():
            if Path(name).suffix != ".pt":
                pin(Path(name), {}, digest)
    sources = [
        Path(__file__),
        root / "scripts/recover_noise_finetune.py",
        root / "src/qwen_hotword/training/noise_status.py",
        *(root / "src/qwen_hotword/training" / name for name in HISTORICAL_CODE),
    ]
    return {
        "plan": plan,
        "identity": {
            "plan_sha256": _sha(run / "plan.json"),
            "cache_marker_sha256": _sha(run / "cache_completed.json"),
            "recovery_source_sha256": {str(p): _sha(p) for p in sources},
            "additive_code_compatibility": compatibility,
            "runtime_versions": {
                package: importlib.metadata.version(package)
                for package in ("transformers", "qwen-asr", "librosa", "soundfile", "torch")
            },
        },
    }


def verify_audio(row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    import librosa

    path = Path(row["audio_path"])
    pin(path, {}, audit["original_audio_sha256"])
    noisy_loader_length = len(read_audio(path))
    original_loader_length = len(librosa.load(str(path), sr=16000, mono=True)[0])
    if (
        noisy_loader_length != original_loader_length
        or noisy_loader_length != audit["source_samples"]
    ):
        raise ValueError(
            f"waveform length mismatch: {row['id']}; original_loader={original_loader_length}, "
            f"noise_loader={noisy_loader_length}, recorded={audit['source_samples']}"
        )
    return {
        "audio_path": str(path),
        "audio_sha256": audit["original_audio_sha256"],
        "samples_16k_both_loaders": noisy_loader_length,
    }


def audit_all(run: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Check every selected 50%/nested-30% training row and all noisy validation rows."""
    counts: Counter[str] = Counter()
    pairs = []
    batch_size = plan["encoder_batch_size"]
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("invalid sealed Encoder batch size")
    with stage_progress(
        "Audit ALL noise/original labels, CTC lengths and boundary frames",
        total=len(plan["parts"]),
        every=25,
    ) as tick:
        for position, part in enumerate(plan["parts"], 1):
            split = part["split"]
            manifest = run / part["manifest"]
            rows = list(_rows(manifest))
            noise_folder = manifest.parent / "features/shards"
            original_folder = Path(plan[f"{split}_cache"]) / "shards"
            index = part["base_shard"]
            original_path = original_folder / f"shard-{index:06d}.pt"
            noise_path = noise_folder / "shard-000000.pt"
            original_meta = _object(original_path.with_suffix(".json"))
            noise_meta = _object(noise_path.with_suffix(".json"))
            for path, meta in ((original_path, original_meta), (noise_path, noise_meta)):
                if path.stat().st_size != meta["feature_bytes"]:
                    raise ValueError(f"cached tensor size differs from metadata: {path}")
            if (
                len(rows) != part["records"]
                or [r["id"] for r in rows] != noise_meta["sample_ids"]
                or original_meta["ctc_time_upsampling_factor"] != 2
                or noise_meta["ctc_time_upsampling_factor"] != 2
            ):
                raise ValueError(f"part order/count/CTC-time contract mismatch: {manifest}")
            original = _tensor_index(original_path, original_meta)
            noisy = _tensor_index(noise_path, noise_meta)
            audit_rows = json.loads((manifest.parent / "mix_audit.json").read_text())
            audits = {a["id"]: a for a in audit_rows}
            if len(audits) != len(audit_rows) or audits.keys() != noisy.keys():
                raise ValueError(f"mix audit IDs differ from noise cache: {manifest}")
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                maximum_samples = max(audits[r["id"]]["source_samples"] for r in batch)
                for row in batch:
                    sample_id = row["id"]
                    base_id = (
                        sample_id.removeprefix("noise10db::")
                        if split == "validation"
                        else sample_id
                    )
                    a, b = original[base_id], noisy[sample_id]
                    audit = audits[sample_id]
                    phones = tuple(row["phoneme_token_ids"])
                    if (
                        a[1] != b[1]
                        or phones != b[1]
                        or not phones
                        or any(type(t) is not int or not 0 < t < 90 for t in phones)
                    ):
                        raise ValueError(f"label drift/invalid labels: {sample_id}")
                    minimum = len(phones) + sum(
                        x == y for x, y in zip(phones, phones[1:], strict=False)
                    )
                    if min(a[0][0], b[0][0]) * 2 < minimum:
                        raise ValueError(f"CTC-infeasible cached sample: {sample_id}")
                    samples = audit["source_samples"]
                    bounds = frame_bounds(samples)
                    # In HF Whisper, STFT drops its final frame and attention masks
                    # are trimmed to floor(batch_max / hop), while a shorter row's
                    # mask counts ceil(row_samples / hop), bounded by that maximum.
                    mel = min((samples + 159) // 160, maximum_samples // 160)
                    if a[0][0] not in bounds or b[0][0] != encoder_frames(mel):
                        raise ValueError(
                            f"frame counts not explained by batch boundary: {sample_id}; "
                            f"original={a[0][0]}, noise={b[0][0]}, bounds={bounds}, "
                            f"predicted_noise={encoder_frames(mel)}, samples={samples}, "
                            f"batch_max_samples={maximum_samples}"
                        )
                    if not math.isfinite(audit["snr_db"]) or abs(audit["snr_db"] - 10.0) > 1e-5:
                        raise ValueError(f"recorded mixing SNR differs from 10dB: {sample_id}")
                    counts[split + "_records"] += 1
                    if split == "train" and row["selected_30"]:
                        counts["train_selected_30"] += 1
                    if a[0] != b[0]:
                        if abs(a[0][0] - b[0][0]) != 1:
                            raise ValueError(f"frame difference is not exactly one: {sample_id}")
                        evidence = verify_audio(row, audit)
                        pairs.append(
                            {
                                "split": split,
                                "base_shard": index,
                                "sample_id": base_id,
                                "language": row["language"],
                                "selected_30": row["selected_30"],
                                "original_frames": a[0][0],
                                "noise_frames": b[0][0],
                                "floor_ceil_encoder_frames": list(bounds),
                                "predicted_noise_mel_frames": mel,
                                **evidence,
                            }
                        )
                        counts[split + "_boundary_pairs"] += 1
            tick(position)
    return {
        "status": "all_boundary_checks_passed",
        "counts": dict(counts),
        "parts_checked": len(plan["parts"]),
        "frame_pairs": pairs,
        "scope": "all 50% training pool (including 30% subset) plus all noisy validation",
        "hidden_values_read": False,
        "full_tensor_sha_verified": False,
        "audio_rechecked": "both waveform loader lengths and source bytes, for changed-frame rows",
        "limitation": (
            "historical clean audio bytes/batch membership were not recorded; "
            "clean frames are checked against the same-waveform floor/ceil envelope, "
            "noise frames against the actual sealed noise-batch membership"
        ),
        "test_set_used": False,
    }


def audited_report(run: Path, source: dict[str, Any]) -> dict[str, Any]:
    folder = run / RECOVERY
    contract_path = folder / "contract.json"
    report_path = folder / "audit.json"
    if contract_path.exists():
        contract = _object(contract_path)
        if contract["identity"] != source["identity"]:
            raise ValueError("saved recovery identity changed; preserve the existing recovery")
        pin(report_path, {}, contract["audit_sha256"])
        report = _object(report_path)
        with stage_progress("Recheck source audio identities for saved boundary exceptions"):
            for pair in report["frame_pairs"]:
                pin(Path(pair["audio_path"]), {}, pair["audio_sha256"])
        return report
    if folder.exists():
        raise ValueError("partial recovery audit exists; preserve it and inspect before retrying")
    probe = processor_probe()
    report = {**audit_all(run, source["plan"]), "processor_probe": probe}
    folder.mkdir()
    save(report_path, report)
    save(contract_path, {"identity": source["identity"], "audit_sha256": _sha(report_path)})
    return report


def recovered_views(run: Path, plan: dict[str, Any], ratio: int, report: dict[str, Any]) -> Any:
    train_cache, validation, groups = compose_views(run, plan, ratio)
    allowed: dict[int, list[tuple[str, int, int]]] = {}
    for pair in report["frame_pairs"]:
        if pair["split"] == "train" and (ratio == 50 or pair["selected_30"]):
            allowed.setdefault(pair["base_shard"], []).append(
                (pair["sample_id"], pair["original_frames"], pair["noise_frames"])
            )
    shards = tuple(
        replace(s, replacement_frame_pairs=tuple(allowed.get(i, [])))
        for i, s in enumerate(train_cache.shards)
    )
    # The data fingerprint is deliberately unchanged: same actual tensors, IDs,
    # labels, order and sample budget. Only strict loading of audited pairs changes.
    return replace(train_cache, shards=shards), validation, groups


def recovery_bundle(run: Path) -> Path:
    original = bundle(run)
    with zipfile.ZipFile(original) as archive:
        payload = {n: archive.read(n) for n in archive.namelist() if n != "sha256.txt"}
    for name in ("audit.json", "contract.json"):
        payload[f"{RECOVERY}/{name}"] = (run / RECOVERY / name).read_bytes()
    checks = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(payload.items())
    )
    temporary = run / ".recovery-return.pending.zip"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
        archive.writestr("sha256.txt", checks)
    temporary.replace(original)
    return original


def recover(
    root: Path, run: Path, config: dict[str, Any], stage: str, *, ratio: int | None = None
) -> dict[str, Any]:
    # Same lock as the original orchestrator; never race an original training job.
    with exclusive_feature_cache_run(run.parent / f".{run.name}.noise-orchestration"):
        source = verify_source(root, run, config)
        plan = source["plan"]
        report = audited_report(run, source)
        result: dict[str, Any] = {
            "status": report["status"],
            "all_parts_checked": report["parts_checked"],
            "counts": report["counts"],
            "features_recomputed": False,
        }
        print(json.dumps(result, ensure_ascii=False), flush=True)
        # Always package the audit first, so even a subsequent interrupted Head
        # run has a compact recovery record. Old plans/caches are never rewritten.
        result["return_bundle"] = str(recovery_bundle(run))
        if stage in {"train", "all"}:
            for percent in (ratio,) if ratio else (30, 50):
                with stage_progress(f"Resume audited noise Head {percent}%"):
                    if (run / f"head_{percent}/completed.json").exists():
                        views = None
                    else:
                        views = recovered_views(run, plan, percent, report)
                    trained = train(run, plan, percent, resume=True, audited_views=views)
                result[str(percent)] = {
                    "best_epoch": trained["best_epoch"],
                    "epochs_completed": trained["epochs_completed"],
                    "macro_per": trained["best_validation_macro_phoneme_error_rate"],
                }
                recovery_bundle(run)
            result["status"] = "training_completed"
        if stage in {"evaluate", "all"}:
            from qwen_hotword.training.noise_evaluation import evaluate

            with stage_progress("Paired baseline / 30% / 50% full-audio retrieval evaluation"):
                result["evaluation"] = evaluate(root, run, plan, resume=True)
            result["status"] = "completed"
        result["return_bundle"] = str(recovery_bundle(run))
        return result
