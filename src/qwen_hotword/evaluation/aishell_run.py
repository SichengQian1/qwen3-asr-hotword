"""Independent fixed-policy AISHELL retrieval evaluation, no training or ASR."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import traceback
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.aishell_reviewed import (
    assemble_entries,
    check_result_counts,
    select_dataset,
    selections,
    summarize,
)
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, run_worker, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import completed, digest, finish, read, rows, save_rows
from qwen_hotword.training.zh_en_foundation import verify_assets
from qwen_hotword.training.zh_en_labels import frontend, install_frontend, make_worker


def check_checkpoint_metadata(
    payload: dict[str, Any], marker: dict[str, Any], report: dict[str, Any], vocab: PhonemeVocab
) -> None:
    if (
        marker.get("status") != "fresh_head_training_completed"
        or marker.get("num_classes") != 183
        or marker.get("encoder_frozen") is not True
        or marker.get("head_initialization") != "random"
        or marker.get("test_used") is not False
        or report.get("status") != "completed"
        or report.get("test_set_used") is not False
        or report.get("num_classes") != 183
    ):
        raise ValueError("requires completed random-initialized frozen-Encoder Chinese/US/UK Head")
    if tuple(payload.get("vocab_tokens", ())) != vocab.tokens or len(vocab.tokens) != 183:
        raise ValueError("checkpoint vocabulary/token-ID order differs from approved 183 classes")
    head = payload.get("head_config", {})
    if (
        payload.get("num_classes") != 183
        or payload.get("blank_id") != 0
        or head.get("head_type") != "temporal_upsample"
        or head.get("input_dimension") != 1024
        or head.get("num_classes") != 183
        or head.get("time_upsampling_factor") != 2
        or head != report.get("head_config")
    ):
        raise ValueError("checkpoint is not the completed 183-class Temporal 2x Head")
    if payload["epoch_metrics"]["epoch"] != marker["best_epoch"] or (
        marker["best_epoch"] != report["best_epoch"]
    ):
        raise ValueError("checkpoint epoch differs from completed training")


def identity(
    root: Path, cfg: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], PhonemeVocab]:
    train = (root / cfg["training_run"]).resolve()
    if not (train / "training_completed.json").is_file():
        raise ValueError(f"Chinese training not verified complete: {train}/training_completed.json")
    train_cfg = read(train / "config.json")
    expected_cfg = read(root / cfg["training_config"])
    # GPU allocation may differ; training/data/frontend policies may not.
    if {k: v for k, v in train_cfg.items() if k != "gpu_ids"} != {
        k: v for k, v in expected_cfg.items() if k != "gpu_ids"
    }:
        raise ValueError("training run configuration differs from this Chinese/US/UK run")
    foundation = read(root / train_cfg["foundation_config"])
    contract = read(train / "contract.json")
    if contract["foundation"] != foundation or contract["config"] != train_cfg:
        raise ValueError("training contract/config/foundation differ")
    for name in (
        "src/qwen_hotword/training/zh_en_labels.py",
        "src/qwen_hotword/phonemes/coverage.py",
        "src/qwen_hotword/phonemes/mfa_v03_runtime.py",
        "src/qwen_hotword/modeling/audio_encoder.py",
        "src/qwen_hotword/modeling/ctc_head.py",
    ):
        pin(root / name, contract["code_sha256"][name])
    cache_contract = read(train / "cache_contract.json")
    for name in ("prepared.json", "model_identity.json", "contract.json"):
        pin(train / name, cache_contract[name])
    if cfg["num_classes"] != 183 or foundation["vocab_sha256"] != cfg["vocab_sha256"]:
        raise ValueError("evaluation must use approved 183-class vocabulary")
    verify_assets(root, foundation)
    vocab = load_phoneme_vocab(root / foundation["vocab"])
    marker = read(train / "training_completed.json")
    checkpoint = train / "head/ctc_head_best.pt"
    pin(checkpoint, marker["identity"]["checkpoint_sha256"])
    if cfg.get("required_checkpoint_sha256"):
        pin(checkpoint, cfg["required_checkpoint_sha256"])
    pin(train / "head/report.json", marker["identity"]["report_sha256"])
    pin(train / "validation_per.json", marker["validation_per_sha256"])
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report = read(train / "head/report.json")
    check_checkpoint_metadata(payload, marker, report, vocab)
    del payload
    model = Path(train_cfg["model"])
    if model.name != "Qwen3-ASR-1.7B":
        raise ValueError("requires original Qwen3-ASR-1.7B")
    model_identity = read(train / "model_identity.json")
    required = {"config.json", "preprocessor_config.json", "model.safetensors.index.json"}
    required.update(read(model / "model.safetensors.index.json")["weight_map"].values())
    if not required <= model_identity.keys():
        raise ValueError("incomplete frozen Encoder identity")
    with stage_progress(
        "Verify original Qwen weight/processor hashes", total=len(model_identity)
    ) as tick:
        for i, (name, checksum) in enumerate(model_identity.items(), 1):
            pin(model / name, checksum)
            tick(i)
    files = [
        train / "training_completed.json",
        train / "head/report.json",
        checkpoint,
        train / "contract.json",
        train / "config.json",
        train / "model_identity.json",
        train / "prepared.json",
        root / cfg["review_policy"],
        root / foundation["vocab"],
        root / train_cfg["foundation_config"],
    ]
    code = [
        "evaluation/aishell_reviewed.py",
        "evaluation/aishell_run.py",
        "inference/external_keyword_retrieval.py",
        "inference/streaming_backends.py",
        "hotwords/anchor_index.py",
        "hotwords/scoring.py",
        "hotwords/registry.py",
        "modeling/audio_encoder.py",
        "modeling/ctc_head.py",
        "modeling/qwen_backbone.py",
        "training/zh_en_labels.py",
        "training/zh_en_data.py",
        "training/ctc_overfit.py",
        "phonemes/coverage.py",
        "phonemes/mfa_v03_runtime.py",
        "phonemes/mfa_v03_assets.py",
    ]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "qwen-asr", "transformers", "numpy", "soundfile", "librosa")
    }
    value = {
        "configuration": cfg,
        "policy_sha256": digest(policy),
        "versions": versions,
        "inputs": {str(p): sha(p) for p in files},
        "code_sha256": {n: sha(root / "src/qwen_hotword" / n) for n in code},
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_epoch": marker["best_epoch"],
        "vocab_sha256": cfg["vocab_sha256"],
        "model_identity": model_identity,
        "checkpoint_path": str(checkpoint),
        "model_path": str(model),
    }
    return value, train_cfg, foundation, vocab


def audio_inventory(
    dataset: Path, selected: list[dict[str, Any]], folder: Path, source_identity: str
) -> list[dict[str, Any]]:
    from qwen_hotword.training.zh_en_data import waveform

    manifest = folder / "audio_manifest.jsonl"
    done = completed(folder, source_identity)
    if done is not None:
        inventory = list(rows(manifest))
        with stage_progress(
            "Verify saved AISHELL WAV/reference identities", total=len(inventory)
        ) as tick:
            for i, row in enumerate(inventory, 1):
                pin(Path(row["audio_path"]), row["audio_sha256"])
                pin(Path(row["reference_path"]), row["reference_sha256"])
                tick(i)
        return inventory
    inventory = []
    with stage_progress(
        "Read selected WAVs and verify reference/PCM", total=len(selected), every=50
    ) as tick:
        for i, row in enumerate(selected, 1):
            path = dataset / "wav/test" / (row["sample_id"] + ".wav")
            ref = path.with_suffix(".txt")
            if "".join(ref.read_text(encoding="utf-8-sig").split()) != "".join(
                row["reference_text"].split()
            ):
                raise ValueError(f"WAV sidecar reference differs: {ref}")
            data = path.read_bytes()
            audio = waveform(data)
            inventory.append(
                {
                    **row,
                    "audio_path": str(path),
                    "reference_path": str(ref),
                    "audio_sha256": hashlib.sha256(data).hexdigest(),
                    "reference_sha256": sha(ref),
                    "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                    "duration_seconds": len(audio) / 16000,
                }
            )
            tick(i)
    save_rows(manifest, inventory)
    finish(folder, source_identity, [manifest], {"sample_count": len(inventory)})
    return inventory


def overlap_audit(train: Path, inventory: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Compare canonical float32/16k/mono hashes to selected train/validation metadata only."""
    parts = read(train / "prepared.json")["parts"]
    pcm = {r["pcm_sha256"] for r in inventory}
    byte_hashes = {r["audio_sha256"] for r in inventory}
    overlaps, checked, missing = [], 0, 0
    with stage_progress(
        "Check selected training/validation PCM overlap", total=len(parts), every=20
    ) as tick:
        for i, part in enumerate(parts, 1):
            path = Path(part["path"])
            pin(path, part["sha256"])
            for row in rows(path):
                checked += 1
                missing += not bool(row.get("pcm_sha256"))
                if row.get("pcm_sha256") in pcm or row.get("audio_sha256") in byte_hashes:
                    overlaps.append({"split": part["split"], "id": row["id"], "part": str(path)})
            tick(i)
    result = {
        "selected_training_validation_rows_checked": checked,
        "missing_pcm_hashes": missing,
        "overlap_count": len(overlaps),
        "overlap_examples": overlaps[:20],
        "evaluation_repeated_pcm_count": len(inventory) - len(pcm),
        "scope": "exact normalized PCM and file bytes; not speaker/near-duplicate isolation",
    }
    write_json(output / "overlap_audit.json", result)
    if not checked or missing or overlaps:
        raise ValueError("training/validation overlap audit failed; see overlap_audit.json")
    return result


def pronunciations(
    root: Path,
    output: Path,
    surfaces: list[str],
    train_cfg: dict[str, Any],
    foundation: dict[str, Any],
    vocab: PhonemeVocab,
    source_identity: str,
) -> tuple[HotwordEntry, ...]:
    folder = output / "phonemes"
    done = completed(folder, source_identity)
    index = folder / "indexed_pronunciations.jsonl"
    if done is not None:
        return tuple(
            HotwordEntry(
                **{**r, **{k: tuple(r[k]) for k in ("words", "phoneme_tokens", "token_ids")}}
            )
            for r in rows(index)
        )
    folder.mkdir(parents=True, exist_ok=True)
    install_frontend(folder, train_cfg["frontend_packages"])
    worker, assets, prefix = make_worker(root, folder, foundation, train_cfg)
    prepared: list[dict[str, Any]] = []
    unique: set[tuple[str, str]] = set()
    with stage_progress(
        "Chinese hotwords through pinned training frontend", total=len(surfaces)
    ) as tick:
        for i, surface in enumerate(surfaces, 1):
            errors = []
            try:
                units = frontend(surface, "zh")
            except ValueError as error:
                units, errors = [], [str(error)]
            unique.update((u["language"], u["word"]) for u in units)
            prepared.append({"surface": surface, "units": units, "errors": errors})
            tick(i)
    save_rows(folder / "frontend.jsonl", prepared)
    predictions: list[dict[str, Any]] = []
    # Each small batch is atomic and resumable; all batches use the same pinned Top1 runtime.
    requests = [{"language": lang, "word": word} for lang, word in sorted(unique)]
    batches = list(range(0, len(requests), 256))
    with stage_progress(
        "MFA Top1 hotword pronunciation batches", total=len(batches), every=1
    ) as tick:
        for number, start in enumerate(batches):
            part = folder / f"batch-{number:04d}"
            request, prediction = part / "request.jsonl", part / "predictions.jsonl"
            batch_id = {"run": source_identity, "requests": requests[start : start + 256]}
            if completed(part, batch_id) is None:
                save_rows(request, requests[start : start + 256])
                run_worker(root, worker, assets, prefix, request, prediction)
                finish(
                    part,
                    batch_id,
                    [request, prediction],
                    {"requests": len(requests[start : start + 256])},
                )
            predictions.extend(rows(prediction))
            tick(number + 1)
    entries, failures = assemble_entries(prepared, predictions, vocab)
    save_rows(folder / "failures.jsonl", failures)
    counts = Counter(e.token_ids for e in entries)
    report = {
        "requested_surfaces": len(surfaces),
        "converted_surfaces": len(entries),
        "failed_surfaces": len(failures),
        "failure_examples": failures[:20],
        "coverage": len(entries) / len(surfaces),
        "num_classes": len(vocab.tokens),
        "homophone_groups": sum(n > 1 for n in counts.values()),
        "homophones_kept_as_distinct_surfaces": True,
        "mfa_models": {
            k: {"name": v["name"], "g2p_version": v["g2p_version"]}
            for k, v in foundation["models"].items()
        },
    }
    write_json(output / "g2p_report.json", report)
    if failures or len(entries) != len(surfaces):
        raise ValueError(
            f"hotword G2P incomplete; refusing to shrink the {len(surfaces)}-word table"
        )
    save_rows(index, (e.to_dict() for e in entries))
    finish(
        folder,
        source_identity,
        [index, folder / "frontend.jsonl", folder / "failures.jsonl"],
        report,
    )
    return tuple(entries)


def check_gpu(gpu: int, minimum_free: float) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = result.stdout.strip().split(",")
    if len(values) != 3 or int(values[0]) != gpu or float(values[1]) < minimum_free * 1024:
        raise ValueError(
            f"GPU {gpu} needs {minimum_free} GiB free; observed {result.stdout.strip()}"
        )
    print(
        f"Selected physical GPU {gpu}: {result.stdout.strip()} (MiB); no other jobs stopped",
        flush=True,
    )


def infer(
    output: Path,
    plan: dict[str, Any],
    inventory: list[dict[str, Any]],
    entries: tuple[HotwordEntry, ...],
    vocab: PhonemeVocab,
    gpu: int,
) -> list[dict[str, Any]]:
    cfg = plan["configuration"]
    plan_id = digest(plan)
    shards = output / "sample_shards"
    shards.mkdir(exist_ok=True)
    detector, details = None, []
    with stage_progress(
        "AISHELL full-audio CTC + Anchor + rerank", total=len(inventory), every=10
    ) as tick:
        for i, record in enumerate(inventory, 1):
            path = shards / (record["sample_id"] + ".json")
            if path.exists():
                saved = read(path)
                checksum = saved.pop("row_sha256")
                if (
                    checksum != digest(saved)
                    or saved["plan_sha256"] != plan_id
                    or saved["record"] != record
                ):
                    raise ValueError(f"changed/corrupt saved sample: {path}")
                detail = saved["detail"]
            else:
                if detector is None:
                    check_gpu(gpu, cfg["minimum_free_gpu_gib"])
                    from qwen_hotword.hotwords.scoring import HotwordScoringConfig
                    from qwen_hotword.inference.streaming_backends import (
                        load_cumulative_ctc_detector,
                    )

                    with stage_progress("Load original Qwen and new Chinese/US/UK 183-class Head"):
                        anchor = cfg["anchor"]
                        detector = load_cumulative_ctc_detector(
                            model_path=plan["model_path"],
                            checkpoint_path=plan["checkpoint_path"],
                            vocab=vocab,
                            hotwords=entries,
                            language="Chinese",
                            device="cuda:0",
                            dtype=cfg["dtype"],
                            scoring_config=HotwordScoringConfig(**cfg["scoring"]),
                            retrieval_mode="operating",
                            retrieval_backend="anchor_guided",
                            anchor_shortlist_size=anchor["shortlist_size"],
                            anchor_start_radius=anchor["start_radius"],
                            anchor_ngram_sizes=tuple(anchor["ngram_sizes"]),
                            anchors_per_entry=anchor["anchors_per_entry"],
                            anchor_offset_tolerance=anchor["offset_tolerance"],
                        )
                pin(Path(record["audio_path"]), record["audio_sha256"])
                audio, timing = external._load_audio(Path(record["audio_path"]))
                result = dict(
                    external._retrieve_waveform(detector, audio, vocab, saved_raw_rank_depth=64)
                )
                selected = selections(result, cfg["scoring"])
                if any(w not in {e.surface for e in entries} for v in selected.values() for w in v):
                    raise ValueError("retrieved word outside fixed table")
                detail = {
                    "sample_id": record["sample_id"],
                    "reference_text": record["reference_text"],
                    "targets": record["targets"],
                    "selections": selected,
                    "retrieval": result,
                    "audio": timing,
                }
                saved = {"plan_sha256": plan_id, "record": record, "detail": detail}
                write_json(path, {**saved, "row_sha256": digest(saved)})
            details.append(detail)
            tick(i)
    return details


def return_bundle(output: Path) -> Path:
    destination = output / "return_bundle.zip"
    with zipfile.ZipFile(destination.with_suffix(".pending.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for name in (
            "report.json",
            "g2p_report.json",
            "overlap_audit.json",
            "selection_summary.json",
            "asset_summary.json",
            "failure_examples.json",
            "error_traceback.txt",
        ):
            if (output / name).is_file():
                if (
                    name == "error_traceback.txt"
                    and read(output / "report.json")["status"] == "completed"
                ):
                    continue
                z.write(output / name, name)
    destination.with_suffix(".pending.zip").replace(destination)
    return destination


def run(root: Path, config: Path, output: Path, gpu: int) -> dict[str, Any]:
    cfg = read(config)
    protected = [
        (root / cfg["training_run"]).resolve(),
        Path(cfg["dataset_root"]).resolve(),
        Path(read(root / cfg["training_config"])["model"]).resolve(),
        root / "models",
        root / "configs",
        root / "src",
    ]
    if output == root or any(
        output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected
    ):
        raise ValueError("evaluation output overlaps protected source")
    with exclusive_feature_cache_run(output.parent / ("." + output.name + ".lock")):
        policy = read(root / cfg["review_policy"])
        # Reject unrelated/pre-existing directories before writing any artifact.
        contract_path = output / "evaluation_config.json"
        contract = {"configuration": cfg, "policy": policy}
        if contract_path.exists():
            if read(contract_path) != contract:
                raise ValueError("changed evaluation configuration; use a new output directory")
        elif output.exists() and any(output.iterdir()):
            raise ValueError("nonempty output without this evaluation contract")
        output.mkdir(parents=True, exist_ok=True)
        write_json(contract_path, contract)
        stage = "verify_model"
        try:
            with stage_progress("Verify completed Chinese/US/UK model and pinned assets"):
                plan, train_cfg, foundation, vocab = identity(root, cfg, policy)
            external._write_or_verify_json(output / "source_identity.json", plan)
            source_id = digest(plan)
            write_json(
                output / "asset_summary.json",
                {
                    k: plan[k]
                    for k in (
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_epoch",
                        "vocab_sha256",
                        "model_path",
                        "versions",
                    )
                },
            )
            stage = "prepare"
            surfaces, selected = select_dataset(Path(cfg["dataset_root"]), policy)
            write_json(output / "selection_summary.json", policy["expected_counts"])
            inventory = audio_inventory(
                Path(cfg["dataset_root"]), selected, output / "audio", source_id
            )
            overlap = overlap_audit(root / cfg["training_run"], inventory, output)
            stage = "g2p"
            entries = pronunciations(
                root, output, surfaces, train_cfg, foundation, vocab, source_id
            )
            if len(entries) != policy["expected_counts"]["candidates"]:
                raise ValueError("active index differs from the fixed candidate count")
            plan = {
                **plan,
                "audio_manifest_sha256": sha(output / "audio/audio_manifest.jsonl"),
                "index_sha256": sha(output / "phonemes/indexed_pronunciations.jsonl"),
            }
            external._write_or_verify_json(output / "run_config.json", plan)
            stage = "retrieval"
            details = infer(output, plan, inventory, entries, vocab, gpu)
            result = summarize(details)
            check_result_counts(result, policy["expected_counts"])
            save_rows(output / "retrieval_details.jsonl", details)
            failure_examples = [
                r for r in details if not set(r["targets"]) <= set(r["selections"]["gated_top7"])
            ]
            write_json(
                output / "failure_examples.json",
                [
                    {k: r[k] for k in ("sample_id", "reference_text", "targets", "selections")}
                    for r in failure_examples[:20]
                ],
            )
            with stage_progress("Recheck immutable evaluation inputs"):
                for name, expected in plan["inputs"].items():
                    pin(Path(name), expected)
                for row in inventory:
                    pin(Path(row["audio_path"]), row["audio_sha256"])
                    pin(Path(row["reference_path"]), row["reference_sha256"])
            result.update(
                {
                    "status": "completed",
                    "scope": "AISHELL_"
                    + policy.get("selection_mode", "reviewed")
                    + "_offline_full_audio_retrieval",
                    "selection_mode": policy.get("selection_mode", "reviewed"),
                    "active_surfaces": len(entries),
                    "num_classes": 183,
                    "checkpoint_sha256": plan["checkpoint_sha256"],
                    "checkpoint_epoch": plan["checkpoint_epoch"],
                    "vocab_sha256": cfg["vocab_sha256"],
                    "index_sha256": plan["index_sha256"],
                    "plan_sha256": digest(plan),
                    "audio_condition": "original_no_added_noise",
                    "qwen_decoder_used": False,
                    "training_started": False,
                    "retrieval_seconds": external._distribution(
                        [r["retrieval"]["timing"]["pure_retrieval_seconds"] for r in details]
                    ),
                    "overlap_audit": overlap,
                    "limitations": policy["limitations"]
                    + [
                        "Positive subset only; no population false-positive rate "
                        "or ASR WER measured.",
                        "Exact PCM exclusion does not prove speaker/near-duplicate "
                        "or Qwen pretraining disjointness.",
                        "Cached timings describe original per-sample inference, "
                        "not current resume wall time.",
                        "Thresholds inherited without tuning on this test; "
                        "raw TopK is after Anchor shortlist.",
                    ],
                }
            )
        except (Exception, KeyboardInterrupt) as error:
            (output / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            result = {
                "status": "failed",
                "stage": stage,
                "error": f"{type(error).__name__}: {error}",
            }
        result["output"] = str(output)
        result["return_bundle"] = str(output / "return_bundle.zip")
        write_json(output / "report.json", result)
        return_bundle(output)
        return result
