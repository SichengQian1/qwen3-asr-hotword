"""Run reviewed targets through the existing retrieval and downstream Top5/Top7 path."""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.reviewed_targets import import_review
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import LANGUAGES, _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import (
    _validated_source_gate,
    replay_rows_top7,
)
from qwen_hotword.inference.reviewed_joint import (
    collapse_variants,
    export_joint_group,
    joint_bundle,
)
from qwen_hotword.inference.wave_retrieval import _row_digest, _view_metrics
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified

NAMES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}


def prepare_reviewed_run(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _json(config)
    if cfg.get("candidate_scope") != "joint_multilingual":
        raise ValueError("this run requires the full joint multilingual reviewed table")
    identities: dict[str, Any] = {}
    _checked(config, identities)
    vocab_path, model, head, table = (
        (root / cfg[key]).resolve() for key in ("vocab", "model", "checkpoint", "reviewed_table")
    )
    _checked(vocab_path, identities, cfg["vocab_sha256"])
    _checked(head, identities, cfg["checkpoint_sha256"])
    _checked(model / "config.json", identities)
    if (model / "model.safetensors.index.json").exists():
        _checked(model / "model.safetensors.index.json", identities)
    vocab = load_phoneme_vocab(vocab_path)
    bundles, targets, review = import_review(table, vocab, identities)
    joint, variants, by_variant = joint_bundle(bundles)
    review.update(
        active_unique_surfaces_per_audio=len(joint.entries),
        active_pronunciation_variants_per_audio=len(variants),
        variant_policy="identical surface+phones deduplicated; best-ranked variant per surface",
        shortlist_unit="pronunciation variant; 64 maximum before surface collapse",
    )
    groups, records_by_group, previous = {}, {}, {}
    for lang in LANGUAGES:
        baseline = (root / cfg["baselines"][lang]).resolve()
        old = _json(_verified(baseline, "run_config.json", identities))
        report = _json(_verified(baseline, "report.json", identities))
        if (
            report.get("status") != "completed"
            or report.get("checkpoint_sha256") != cfg["checkpoint_sha256"]
            or report.get("top5_reproduced") is not True
            or report.get("top7_exact_replay") is not True
            or "audio_root" in old["configuration"]
        ):
            raise ValueError(f"completed clean baseline with the same Head required: {lang}")
        for name, info in cfg["datasets"].items():
            if info["language"] != lang:
                continue
            if not name.replace("_", "").isalnum():
                raise ValueError("unsafe dataset name")
            old_group = old["groups"][name]
            old_config = old_group["config"]
            old_source = old_config["sources"][0]
            if len(old_config["sources"]) != 1 or old_source["name"] != name:
                raise ValueError(f"unexpected baseline source: {name}")
            source = external.SourceSpec(
                name, Path(old_source["audio_dir"]), Path(old_source["transcripts"]["path"])
            )
            _checked(source.transcript_path, identities, old_source["transcripts"]["sha256"])
            records, audit = external.build_external_dataset(
                (source,), language={"en": "en", "es": "es", "pt": "pt-BR"}[lang]
            )
            if (
                len(records) != info["expected_samples"]
                or [r.to_dict() for r in records] != old_group["records"]
            ):
                raise ValueError(f"audio/transcript sample identity differs from baseline: {name}")
            for record in records:
                _checked(
                    record.audio_path,
                    identities,
                    old["inputs"][str(record.audio_path)]["sha256"],
                )
            group_config = external._run_config(
                model=model,
                checkpoint=head,
                vocab_path=vocab_path,
                keyword_path=table,
                sources=(source,),
                keyword_set="all_keywords",
                device="cuda:0",
                dtype="bfloat16",
                language=lang,
                saved_raw_rank_depth=64,
            )
            # Only table identity and the orchestration Git revision may differ.
            ignored = {"git_commit", "keyword_bias", "table_filler_exclusion_used_transcripts"}
            if {k: v for k, v in group_config.items() if k not in ignored} != {
                k: v for k, v in old_config.items() if k not in ignored
            }:
                raise ValueError(f"retrieval settings/model differ from clean baseline: {name}")
            old_keyword = Path(old_config["keyword_bias"]["path"])
            _checked(old_keyword, identities, old_config["keyword_bias"]["sha256"])
            old_bundle = external.load_keyword_bias(
                old_keyword, vocab=vocab, keyword_set="all_keywords", language=lang
            )
            old_entries = {e.normalized: e for e in old_bundle.entries}
            for entry in bundles[lang].entries:
                original = old_entries.get(entry.normalized)
                if original is None or (entry.surface, entry.token_ids) != (
                    original.surface,
                    original.token_ids,
                ):
                    raise ValueError(
                        f"reviewed word/label is not in the clean baseline: {entry.surface}"
                    )
            target_ids = sorted(targets[lang].get(info["source_group"], []))
            wanted = {e.normalized for e in bundles[lang].entries if e.hotword_id in target_ids}
            old_target_ids = {old_entries[key].hotword_id for key in wanted}
            if not old_target_ids <= set(old_group["target_ids"]):
                raise ValueError(f"reviewed target provenance differs from baseline: {name}")
            details = _verified(baseline, f"{name}/retrieval_details.jsonl", identities)
            groups[name] = {
                "language": lang,
                "config": group_config,
                "audit": audit,
                "records": [r.to_dict() for r in records],
                "target_ids": sorted({by_variant[i].hotword_id for i in target_ids}),
                "original_declared_targets": len(old_group["target_ids"]),
                "retained_declared_targets": len(target_ids),
                "baseline_details": str(details),
                "baseline_target_ids": sorted(old_target_ids),
            }
            records_by_group[name] = records
            previous[name] = {"bundle": old_bundle, "config": old_config}
    if set(groups) != set(cfg["datasets"]) or not groups:
        raise ValueError("datasets must belong to en/es/pt")
    plan = {
        "schema_version": 1,
        "git_commit": external._git_commit(),
        "configuration": cfg,
        "inputs": identities,
        "review": review,
        "groups": groups,
        "audio_hashes_verified": True,
        "model_weight_shards_hashed": False,
        "code_sha256": {
            rel: _sha(Path(__file__).resolve().parents[1] / rel)
            for rel in (
                "inference/reviewed_retrieval.py",
                "inference/reviewed_joint.py",
                "evaluation/reviewed_targets.py",
                "evaluation/target_review.py",
                "inference/external_keyword_retrieval.py",
                "inference/external_keyword_topk_replay.py",
                "inference/wave_retrieval.py",
                "inference/streaming_backends.py",
                "inference/hotword_prompt.py",
                "hotwords/anchor_index.py",
                "hotwords/scoring.py",
                "training/edit_distance.py",
                "phonemes/coverage.py",
            )
        },
    }
    return plan, {
        "bundle": joint,
        "variants": variants,
        "by_variant": by_variant,
        "vocab": vocab,
        "model": model,
        "checkpoint": head,
        "records": records_by_group,
        "previous": previous,
        "table": table,
    }


def _old_metrics(group: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    rows = [json.loads(s) for s in Path(group["baseline_details"]).read_text().splitlines() if s]
    if len(rows) != len(group["records"]) or any(
        any(row.get(k) != v for k, v in record.items())
        for row, record in zip(rows, group["records"], strict=True)
    ):
        raise ValueError("baseline detail records differ from frozen audio/reference")
    replay = replay_rows_top7(
        rows, gate=_validated_source_gate(previous["config"]), bundle=previous["bundle"]
    )
    return {
        f"top{k}": _view_metrics(replay, set(group["baseline_target_ids"]), top_k=k) for k in (5, 7)
    }


def _aggregate(groups: dict[str, Any], names: list[str], *, baseline: bool) -> dict[str, Any]:
    result = {}
    for key in ("top5", "top7"):
        views = [
            groups[n]["baseline_same_retained_targets" if baseline else "current"][key]
            for n in names
        ]
        counts = {
            field: sum(v[field] for v in views)
            for field in (
                "sample_count",
                "expected_target_sample_pairs",
                "gated_target_hits",
                "raw_target_hits",
                "selected_all_words",
                "selected_words_absent_from_transcript",
            )
        }
        denom = counts["expected_target_sample_pairs"]
        counts["gated_target_recall"] = counts["gated_target_hits"] / denom if denom else None
        counts["raw_target_recall"] = counts["raw_target_hits"] / denom if denom else None
        result[key] = counts
    return result


def run_reviewed_retrieval(
    root: Path,
    config: Path,
    output: Path,
    *,
    resume: bool = False,
    audit_only: bool = False,
    detector_factory: Callable[..., Any] | None = None,
    retrieve_function: external.RetrieveFunction | None = None,
) -> dict[str, Any]:
    if output.exists() and not resume and not audit_only:
        raise FileExistsError(f"output exists; --resume only for this identical run: {output}")
    cfg = _json(config)
    protected = [root / cfg["reviewed_table"]] + [root / p for p in cfg["baselines"].values()]
    if any(
        output.resolve().is_relative_to((p.parent if p.is_file() else p).resolve())
        for p in protected
    ):
        raise ValueError("output must be separate from the edited table and old retrieval outputs")
    print("Checking reviewed targets, baseline settings and all audio SHA256...", flush=True)
    plan, runtime = prepare_reviewed_run(root, config)
    audit = {
        "status": "audit_pass",
        "candidate_scope": cfg["candidate_scope"],
        **plan["review"],
        "samples": {n: len(r) for n, r in runtime["records"].items()},
        "delivery_file_count": len(plan["groups"]) * 2,
        "model_loaded": False,
        "files_written": False,
    }
    if audit_only:
        return audit
    if output.exists():
        if not (output / "run_config.json").is_file() or _json(output / "run_config.json") != plan:
            raise ValueError(
                "resume input/table/audio/config/code identity differs; use a new output"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
        external._write_json(output / "run_config.json", plan)
    external._write_or_verify_json(output / "reviewed_keyword_table.json", _json(runtime["table"]))
    external._write_or_verify_jsonl(
        output / "indexed_pronunciations.jsonl", [e.to_dict() for e in runtime["variants"]]
    )
    metrics = {}
    for lang in LANGUAGES:
        bundle, detector = runtime["bundle"], None
        language_output = output / "languages" / lang
        (language_output / "delivery").mkdir(parents=True, exist_ok=True)
        try:
            for name, records in runtime["records"].items():
                group = plan["groups"][name]
                if group["language"] != lang:
                    continue
                shards = language_output / name / "sample_shards"
                shards.mkdir(parents=True, exist_ok=True)
                rows = []
                for n, record in enumerate(records, 1):
                    shard = external._shard_path(shards, record)
                    digest = plan["inputs"][str(record.audio_path)]["sha256"]
                    if shard.exists():
                        row = _json(shard)
                        external._validate_shard(row, record)
                        if (
                            row.get("row_sha256") != _row_digest(row)
                            or row.get("audio_sha256") != digest
                            or any(row.get(k) != v for k, v in record.to_dict().items())
                        ):
                            raise ValueError(f"corrupted sample shard: {name}/{record.sample_id}")
                    else:
                        if _sha(record.audio_path) != digest:
                            raise ValueError(f"audio changed after preflight: {record.audio_path}")
                        if detector is None:
                            detector = (detector_factory or external._load_detector)(
                                model=runtime["model"],
                                checkpoint=runtime["checkpoint"],
                                vocab=runtime["vocab"],
                                hotwords=runtime["variants"],
                                device="cuda:0",
                                dtype="bfloat16",
                                language=NAMES[lang],
                            )
                        waveform, timing = external._load_audio(record.audio_path)
                        retrieval = (
                            retrieve_function(detector, waveform, runtime["vocab"])
                            if retrieve_function
                            else external._retrieve_waveform(
                                detector, waveform, runtime["vocab"], saved_raw_rank_depth=64
                            )
                        )
                        retrieval = collapse_variants(
                            retrieval,
                            runtime["variants"],
                            runtime["by_variant"],
                            _validated_source_gate(group["config"]),
                        )
                        row = external._build_result_row(
                            record, bundle=bundle, retrieval=retrieval, audio_timing=timing
                        )
                        row["raw_variant_ranked_matches"] = retrieval["raw_variant_ranked_matches"]
                        row["variant_shortlist_candidates"] = retrieval[
                            "variant_shortlist_candidates"
                        ]
                        row["audio_sha256"] = digest
                        row["row_sha256"] = _row_digest(row)
                        external._write_json(shard, row)
                    rows.append(row)
                    if n == 1 or n % 50 == 0 or n == len(records):
                        print(f"{name}: {n}/{len(records)}", flush=True)
                current = export_joint_group(
                    language_output,
                    name,
                    group,
                    bundle,
                    rows,
                )
                previous = _old_metrics(group, runtime["previous"][name])
                for key in ("top5", "top7"):
                    if (
                        current[key]["expected_target_sample_pairs"]
                        != previous[key]["expected_target_sample_pairs"]
                    ):
                        raise ValueError(f"retained-target denominator mismatch: {name}/{key}")
                metrics[name] = {"current": current, "baseline_same_retained_targets": previous}
        finally:
            detector = None
            gc.collect()
            if detector_factory is None:
                import torch

                torch.cuda.empty_cache()
    expected = sorted(
        f"languages/{g['language']}/delivery/{name}_top{k}.json"
        for name, g in plan["groups"].items()
        for k in (5, 7)
    )
    if (
        sorted(p.relative_to(output).as_posix() for p in output.glob("languages/*/delivery/*.json"))
        != expected
    ):
        raise ValueError("unexpected downstream file set")
    # No input changes during a long run may silently enter a completed report.
    for filename, identity in plan["inputs"].items():
        if _sha(Path(filename)) != identity["sha256"]:
            raise ValueError(f"input changed during inference: {filename}")
    report = {
        "status": "completed",
        "output_dir": str(output),
        "candidate_scope": cfg["candidate_scope"],
        "sample_count": sum(len(r) for r in runtime["records"].values()),
        "review": plan["review"],
        "groups": metrics,
        "by_language": {
            lang: {
                label: _aggregate(
                    metrics,
                    [n for n, g in plan["groups"].items() if g["language"] == lang],
                    baseline=old,
                )
                for label, old in (("current", False), ("baseline_same_retained_targets", True))
            }
            for lang in LANGUAGES
        },
        "delivery_files": expected,
        "delivery_file_count": len(expected),
        "checkpoint_sha256": cfg["checkpoint_sha256"],
        "top5_reproduced": True,
        "top7_exact_replay": True,
        "parameters_tuned": False,
        "qwen_decoder_used": False,
        "metric_unit": "distinct keyword per audio, not repeated token occurrences",
        "limitations": [
            "Every audio searches the full reviewed multilingual table, including other languages.",
            "Baseline recall uses identical retained targets, not the old full denominator.",
            "Table size, composition and cross-language competition change together.",
            "The fixed Top64 shortlist counts pronunciation variants before surface deduplication.",
            "Transcript matching is not audio adjudication; no final ASR WER or CTC PER measured.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    external._write_or_verify_json(output / "report.json", report)
    seal(output)
    return report
