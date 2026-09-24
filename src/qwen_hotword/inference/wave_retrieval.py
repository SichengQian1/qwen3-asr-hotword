"""Eight wave evaluations with fixed tables, exact Top5/7 replay and 16 deliveries."""

from __future__ import annotations

import gc
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.wave_keywords import WAVES
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import (
    _delivery_items,
    _metric_summary,
    _validated_source_gate,
    _verify_source_top5_metrics,
    replay_rows_top7,
)
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified

LANGUAGES = ("es", "pt")
NAMES = {"es": "Spanish", "pt": "Portuguese"}
TAGS = {"es": "es", "pt": "pt-BR"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_run(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate all eight groups and bind audio bytes before loading a model."""
    cfg = _read(config_path)
    identities: dict[str, Any] = {}

    def identify(path: Path, expected: str | None = None) -> None:
        actual = _sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        identities[str(path)] = {"sha256": actual, "size_bytes": path.stat().st_size}

    identify(config_path)
    tables = root / cfg["tables"]
    table_report = _read(_verified(tables, "report.json", identities))
    if table_report.get("status") != "completed" or not table_report.get("tables_created"):
        raise ValueError("keyword table release is not completed")
    inventory = _read(_verified(root / cfg["inventory"], "report.json", identities))
    vocab_path, checkpoint, model = (root / cfg[k] for k in ("vocab", "checkpoint", "model"))
    identify(vocab_path, cfg["vocab_sha256"])
    provenance = cfg.get("training_provenance")
    if provenance is not None:
        # First bind the user-selected completed training run; its new Head hash is
        # captured on H200 and persisted in run_config, then required unchanged on resume.
        plan_path = root / provenance["run_plan"]
        report_path = root / provenance["report"]
        identify(plan_path, provenance["run_plan_sha256"])
        identify(report_path)
        trained = _read(report_path)
        training_plan = _read(plan_path)
        if (
            trained.get("status") != "completed"
            or trained.get("test_set_used") is not False
            or trained.get("cache_sha256_verified") is not True
            or trained.get("best_epoch") != provenance["best_epoch"]
            or Path(trained["best_checkpoint_path"]).resolve() != checkpoint.resolve()
            or trained["train_sample_count"] != training_plan["train"]["records"]
            or trained["validation_sample_count"] != training_plan["validation"]["records"]
            or training_plan["config"]["vocab_sha256"] != cfg["vocab_sha256"]
        ):
            raise ValueError("completed training provenance does not match selected best Head")
        identify(checkpoint, cfg.get("checkpoint_sha256"))
        cfg["checkpoint_sha256"] = identities[str(checkpoint)]["sha256"]
    identify(checkpoint, cfg["checkpoint_sha256"])
    identify(model / "config.json")
    # Bind model metadata/index, not all multi-GB weights. Actual load remains H200-only.
    index = model / "model.safetensors.index.json"
    if index.is_file():
        identify(index)
    vocab = load_phoneme_vocab(vocab_path)
    groups: dict[str, Any] = {}
    bundles: dict[str, external.KeywordBundle] = {}
    records_by_group: dict[str, tuple[external.ExternalAudioRecord, ...]] = {}
    for lang in LANGUAGES:
        keyword_path = _verified(tables, f"{lang}/keyword_bias_phoneme.json", identities)
        target_path = _verified(tables, f"{lang}/targets_by_wave.json", identities)
        bundle = external.load_keyword_bias(
            keyword_path, vocab=vocab, keyword_set="all_keywords", language=lang
        )
        if len(bundle.entries) != cfg["expected_keywords"]:
            raise ValueError(f"{lang}: wrong keyword count")
        targets = _read(target_path)
        if not isinstance(targets, dict) or set(targets) != set(WAVES):
            raise ValueError(f"{lang}: missing wave target view")
        for values in targets.values():
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(v, str) for v in values)
            ):
                raise ValueError(f"{lang}: invalid wave targets")
            if len(values) != len(set(values)):
                raise ValueError(f"{lang}: duplicate wave target")
        by_normalized = {e.normalized: e.hotword_id for e in bundle.entries}
        union = {v for values in targets.values() for v in values}
        if len(union) != cfg["expected_primary"][lang] or not union <= by_normalized.keys():
            raise ValueError(f"{lang}: mandatory targets missing or changed")
        bundles[lang] = bundle
        for wave in WAVES:
            name = f"{wave}_{lang}"
            base = root / wave / lang
            transcripts = base / "transcripts.txt"
            identify(transcripts, inventory["datasets"][f"{wave}/{lang}"]["transcripts_sha256"])
            source = external.SourceSpec(name, base / "wav", transcripts)
            records, audit = external.build_external_dataset((source,), language=TAGS[lang])
            if len(records) != cfg["expected_samples_per_group"]:
                raise ValueError(f"{name}: wrong audio/transcript count: {len(records)}")
            for record in records:
                identify(record.audio_path)
            group_config = external._run_config(
                model=model,
                checkpoint=checkpoint,
                vocab_path=vocab_path,
                keyword_path=keyword_path,
                sources=(source,),
                keyword_set="all_keywords",
                device="cuda:0",
                dtype="bfloat16",
                language=lang,
                saved_raw_rank_depth=64,
            )
            groups[name] = {
                "language": lang,
                "wave": wave,
                "audit": audit,
                "config": group_config,
                "records": [r.to_dict() for r in records],
                "target_ids": sorted(by_normalized[t] for t in targets[wave]),
            }
            records_by_group[name] = records
    if "comparison_run_config" in cfg:
        previous_path = root / cfg["comparison_run_config"]
        identify(previous_path)
        previous = _read(previous_path)
        for filename, identity in previous["inputs"].items():
            identify(Path(filename), identity["sha256"])
        old_cfg = previous["configuration"]
        for key in (
            "tables",
            "inventory",
            "model",
            "vocab",
            "vocab_sha256",
            "expected_keywords",
            "expected_primary",
            "expected_samples_per_group",
        ):
            if cfg[key] != old_cfg[key]:
                raise ValueError(f"comparison input configuration changed: {key}")
        for name, group in groups.items():
            old_group = previous["groups"][name]
            if (
                {
                    k: v
                    for k, v in group["config"].items()
                    if k not in {"git_commit", "ctc_checkpoint"}
                }
                != {
                    k: v
                    for k, v in old_group["config"].items()
                    if k not in {"git_commit", "ctc_checkpoint"}
                }
                or group["records"] != old_group["records"]
                or group["target_ids"] != old_group["target_ids"]
            ):
                raise ValueError(f"comparison retrieval/data contract changed: {name}")
    plan = {
        "schema_version": 1,
        "git_commit": external._git_commit(),
        "configuration": cfg,
        "inputs": identities,
        "groups": groups,
        "audio_hashes_verified": True,
        "model_weight_shards_hashed": False,
        "qwen_decoder_used": False,
        "saved_raw_rank_depth": 64,
    }
    return plan, {
        "vocab": vocab,
        "bundles": bundles,
        "records": records_by_group,
        "model": model,
        "checkpoint": checkpoint,
    }


def _row_digest(row: Mapping[str, Any]) -> str:
    content = {k: v for k, v in row.items() if k != "row_sha256"}
    return hashlib.sha256(
        json.dumps(content, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _view_metrics(
    rows: Sequence[Mapping[str, Any]], targets: set[str], *, top_k: int
) -> dict[str, Any]:
    """Target recall excludes fillers; false positives use all transcript-present words."""
    projected = [
        {**r, "expected_hotword_ids": sorted(set(r["expected_hotword_ids"]) & targets)}
        for r in rows
    ]
    primary = _metric_summary(projected, top_k=top_k)
    # A present filler is not a false positive merely because it is outside this wave's targets.
    selected_field = f"top{top_k}_selected_hotword_ids"
    target_predictions = sum(len(set(r[selected_field]) & targets) for r in rows)
    target_hits = sum(
        len(set(r["expected_hotword_ids"]) & targets & set(r[selected_field])) for r in rows
    )
    total_selected = sum(len(r[selected_field]) for r in rows)
    false_selected = sum(len(set(r[selected_field]) - set(r["expected_hotword_ids"])) for r in rows)
    return {
        "sample_count": len(rows),
        "declared_wave_target_count": len(targets),
        "expected_target_sample_pairs": primary["expected_hotwords"],
        "raw_target_hits": primary["raw_retrieved_expected_hotwords"],
        "raw_target_recall": primary["raw_recall"],
        "gated_target_hits": target_hits,
        "gated_target_recall": primary["final_retrieval_recall"],
        "selected_target_sample_pairs": target_predictions,
        "target_precision": target_hits / target_predictions if target_predictions else None,
        "selected_all_words": total_selected,
        "selected_words_absent_from_transcript": false_selected,
        "false_selection_fraction": false_selected / total_selected if total_selected else None,
        "full_4000_view": _metric_summary(rows, top_k=top_k),
    }


def export_group(
    output: Path,
    name: str,
    group: Mapping[str, Any],
    bundle: external.KeywordBundle,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate D5 reproduction, replay D7 exactly, then serialize the unchanged schema."""
    d5, _, summary, _ = external.summarize_retrieval(rows, bundle=bundle)
    gate = _validated_source_gate(group["config"])
    replayed = replay_rows_top7(rows, gate=gate, bundle=bundle)
    _verify_source_top5_metrics(
        {
            "overall": _metric_summary(replayed, top_k=5),
            "by_source": {name: _metric_summary(replayed, top_k=5)},
        },
        summary,
    )
    primary = set(group["target_ids"])
    metrics = {}
    for k in (5, 7):
        delivery = {
            str(r["sample_id"]): _delivery_items(r, bundle=bundle, top_k=k) for r in replayed
        }
        if len(delivery) != len(rows) or (k == 5 and delivery != d5):
            raise ValueError(f"{name}: downstream ID/Top5 reproduction mismatch")
        external._write_or_verify_json(output / "delivery" / f"{name}_top{k}.json", delivery)
        metrics[f"top{k}"] = _view_metrics(replayed, primary, top_k=k)
    directory = output / name
    external._write_or_verify_jsonl(directory / "retrieval_details.jsonl", rows)
    external._write_or_verify_json(directory / "evaluation_summary.json", summary)
    external._write_or_verify_json(directory / "topk_metrics.json", metrics)
    return metrics


def run_waves(
    root: Path,
    config_path: Path,
    output: Path,
    *,
    resume: bool = False,
    audit_only: bool = False,
    detector_factory: Callable[..., Any] | None = None,
    retrieve_function: Callable[[Any, Any, PhonemeVocab], Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    if output.exists() and not resume and not audit_only:
        raise FileExistsError(f"output exists; use --resume for this identical run: {output}")
    print(
        "Verifying tables, selected Head, transcripts and all 800 audio identities...", flush=True
    )
    plan, runtime = prepare_run(root, config_path)
    if audit_only:
        return {
            "status": "audit_pass",
            "groups": len(plan["groups"]),
            "sample_count": sum(len(v) for v in runtime["records"].values()),
            "keyword_counts": {k: len(v.entries) for k, v in runtime["bundles"].items()},
            "model_loaded": False,
            "files_written": False,
        }
    if output.exists():
        if not (output / "run_config.json").is_file() or _read(output / "run_config.json") != plan:
            raise ValueError(
                "resume input/config/code identity differs; use a new output directory"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        external._write_json(output / "run_config.json", plan)
    (output / "delivery").mkdir(exist_ok=True)
    vocab = runtime["vocab"]
    metrics = {}
    for lang in LANGUAGES:
        bundle = runtime["bundles"][lang]
        detector = None
        try:
            for wave in WAVES:
                name = f"{wave}_{lang}"
                group = plan["groups"][name]
                directory = output / name
                shards = directory / "sample_shards"
                shards.mkdir(parents=True, exist_ok=True)
                rows = []
                for n, record in enumerate(runtime["records"][name], 1):
                    shard = external._shard_path(shards, record)
                    audio_sha = plan["inputs"][str(record.audio_path)]["sha256"]
                    if shard.is_file():
                        row = _read(shard)
                        external._validate_shard(row, record)
                        if (
                            row.get("row_sha256") != _row_digest(row)
                            or row.get("audio_sha256") != audio_sha
                            or any(row.get(k) != v for k, v in record.to_dict().items())
                        ):
                            raise ValueError(
                                f"resume sample changed or corrupted: {name}/{record.sample_id}"
                            )
                    else:
                        # The config binds bytes at preflight; check again just before decoding.
                        if _sha(record.audio_path) != audio_sha:
                            raise ValueError(f"audio changed after preflight: {record.audio_path}")
                        if detector is None:
                            detector = (detector_factory or external._load_detector)(
                                model=runtime["model"],
                                checkpoint=runtime["checkpoint"],
                                vocab=vocab,
                                hotwords=bundle.entries,
                                device="cuda:0",
                                dtype="bfloat16",
                                language=NAMES[lang],
                            )
                        waveform, timing = external._load_audio(record.audio_path)
                        if retrieve_function is None:
                            result = external._retrieve_waveform(
                                detector, waveform, vocab, saved_raw_rank_depth=64
                            )
                        else:
                            result = retrieve_function(detector, waveform, vocab)
                        row = external._build_result_row(
                            record, bundle=bundle, retrieval=result, audio_timing=timing
                        )
                        row["audio_sha256"] = audio_sha
                        row["row_sha256"] = _row_digest(row)
                        external._write_json(shard, row)
                    rows.append(row)
                    if n == 1 or n % 20 == 0 or n == len(runtime["records"][name]):
                        print(f"{name}: {n}/{len(runtime['records'][name])}", flush=True)
                metrics[name] = export_group(output, name, group, bundle, rows)
        finally:
            detector = None
            gc.collect()
            # No torch import for audit-only/local mocked runs.
            if detector_factory is None:
                import torch

                torch.cuda.empty_cache()
    expected = {f"{w}_{lang}_top{k}.json" for w in WAVES for lang in LANGUAGES for k in (5, 7)}
    delivered = {p.name for p in (output / "delivery").glob("*.json")}
    if delivered != expected:
        raise ValueError("expected exactly 16 downstream files")
    report = {
        "status": "completed",
        "output_dir": str(output),
        "groups": metrics,
        "delivery_files": sorted(expected),
        "delivery_file_count": 16,
        "sample_count": sum(len(v) for v in runtime["records"].values()),
        "checkpoint_sha256": plan["configuration"]["checkpoint_sha256"],
        "top5_reproduced": True,
        "top7_exact_replay": True,
        "qwen_decoder_used": False,
        "parameters_tuned": False,
        "metric_unit": "distinct keyword per audio, not repeated token occurrences",
        "limitations": [
            "Transcript phrase matching is not a manual pronunciation audit.",
            "ASR WER and CTC PER are not evaluated by this retrieval run.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    external._write_or_verify_json(output / "report.json", report)
    # The final checksum list is deterministic; resume verifies rather than replacing it.
    lines = "".join(
        f"{_sha(p)}  {p.relative_to(output).as_posix()}\n"
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "sha256.txt"
    )
    checks = output / "sha256.txt"
    if checks.exists() and checks.read_text() != lines:
        raise ValueError("completed output checksum manifest changed")
    if not checks.exists():
        external._atomic_write_text(checks, lines)
    return report
