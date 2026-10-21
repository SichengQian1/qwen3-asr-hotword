"""Reference-count selection, independent of retrieval outcomes; single-file delivery."""

from __future__ import annotations

import gc
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.english_best_noise import load_completed_best
from qwen_hotword.inference.english_head_preview import _index
from qwen_hotword.inference.external_keyword_topk_replay import (
    _validated_source_gate,
    replay_rows_top7,
)
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.wave_retrieval import _row_digest
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress

GROUPS = {1: "one_target", 2: "two_targets", 3: "three_targets"}


def choose_samples(
    records: Sequence[external.ExternalAudioRecord],
    targets: Sequence[HotwordEntry],
    *,
    count: int,
    seed: str,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if count <= 0 or len({e.hotword_id for e in targets}) != len(targets):
        raise ValueError("invalid sample count or duplicate target IDs")
    candidates = []
    distribution: Counter[str] = Counter()
    for completed, record in enumerate(records, 1):
        expected = [e for e in targets if strict_phrase_match(record.reference_text, e.surface)]
        distribution[str(len(expected))] += 1
        if len(expected) in GROUPS:
            candidates.append(
                {
                    **record.to_dict(),
                    "case_id": "english_"
                    + hashlib.sha256(f"{record.source}\0{record.sample_id}".encode()).hexdigest(),
                    "case_type": "positive",
                    "expected_hotword_ids": [e.hotword_id for e in expected],
                    "expected_surfaces": [e.surface for e in expected],
                    "containment_expected_ids": [e.hotword_id for e in expected],
                    "redundant_family_ids": [],
                    "primary_group": GROUPS[len(expected)],
                    "selection_reason": (
                        f"deterministic_seed={seed}; source={record.source}; "
                        f"true_target_count={len(expected)}; uniform_without_replacement; "
                        "eligible_target_count=1..3; retrieval_outcomes_not_used"
                    ),
                }
            )
        if progress:
            progress(completed)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} eligible 1–3-target cases; need {count}")
    candidates.sort(
        key=lambda row: hashlib.sha256(f"{seed}\0{row['case_id']}".encode()).hexdigest()
    )
    return candidates[:count], dict(sorted(distribution.items(), key=lambda p: int(p[0])))


def prepare_selection(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _json(config)
    inputs: dict[str, Any] = {}
    _checked(config, inputs)
    source = (root / cfg["baseline_run"]).resolve()
    _checked(source / "sha256.txt", inputs)
    old = _json(_verified(source, "run_config.json", inputs))
    report = _json(_verified(source, "report.json", inputs))
    if (
        report.get("status") != "completed"
        or report.get("candidate_scope") != "joint_multilingual"
        or report.get("checkpoint_sha256") != cfg["baseline_checkpoint_sha256"]
        or not report.get("top7_exact_replay")
        or report["review"]["expanded_table_sha256"] != cfg["table_sha256"]
        or report["review"]["active_unique_surfaces_per_audio"] != cfg["expected_surfaces"]
    ):
        raise ValueError("sealed clean 4k baseline identity differs")
    previous_cfg = old["configuration"]
    table = _checked(root / previous_cfg["expanded_table"], inputs, cfg["table_sha256"])
    if _json(table) != _json(_verified(source, "reviewed_keyword_table.json", inputs)):
        raise ValueError("baseline table snapshot differs")
    vocab_path = _checked(root / previous_cfg["vocab"], inputs, cfg["vocab_sha256"])
    vocab = load_phoneme_vocab(vocab_path)
    bundle, variants, by_variant = _index(
        _verified(source, "indexed_pronunciations.jsonl", inputs), vocab
    )
    if len(bundle.entries) != cfg["expected_surfaces"]:
        raise ValueError("full multilingual surface count differs")
    head, checkpoint = load_completed_best(root, cfg, vocab, inputs)
    if checkpoint["sha256"] != cfg["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the fixed epoch27 SHA256")
    model = (root / previous_cfg["model"]).resolve()
    for name in ("config.json", "model.safetensors.index.json"):
        path = model / name
        if name == "config.json" or path.exists():
            _checked(path, inputs, old["inputs"][str(path)]["sha256"])
    package = Path(__file__).resolve().parents[1]
    for relative, digest in old["code_sha256"].items():
        _checked(package / relative, inputs, digest)
    for path in (
        Path(__file__),
        Path(__file__).resolve().parents[3] / "scripts/build_english_sample_selection.py",
        package / "inference/english_best_noise.py",
        package / "inference/english_head_preview.py",
    ):
        _checked(path, inputs)
    if set(cfg["datasets"]) != {"en_chile", "en_colombia", "en_mexico"}:
        raise ValueError("exactly the three original English datasets are required")
    by_id = {e.hotword_id: e for e in bundle.entries}
    samples, groups = [], {}
    total_expected = 0
    for name, expected_count in cfg["datasets"].items():
        previous = old["groups"][name]
        old_config = previous["config"]
        gate = _validated_source_gate(old_config)
        if previous["language"] != "en" or len(old_config["sources"]) != 1:
            raise ValueError("original group must be a single English source")
        original = old_config["sources"][0]
        spec = external.SourceSpec(
            name, Path(original["audio_dir"]), Path(original["transcripts"]["path"])
        )
        _checked(spec.transcript_path, inputs, original["transcripts"]["sha256"])
        with stage_progress(f"Scan clean {name} paths and reference metadata"):
            records, _ = external.build_external_dataset((spec,), language="en")
        if len(records) != expected_count or [r.to_dict() for r in records] != previous["records"]:
            raise ValueError(f"original clean sample/reference identity differs: {name}")
        ids = previous["target_ids"]
        if len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError(f"invalid frozen target membership: {name}")
        with stage_progress(
            f"Count frozen targets and select {name}", total=len(records), every=250
        ) as tick:
            selected, distribution = choose_samples(
                records,
                [by_id[i] for i in ids],
                count=cfg["samples_per_dataset"],
                seed=cfg["seed"],
                progress=tick,
            )
        total_expected += sum(int(k) * v for k, v in distribution.items())
        current = external._run_config(
            model=model,
            checkpoint=head,
            vocab_path=vocab_path,
            keyword_path=table,
            sources=(spec,),
            keyword_set="all_keywords",
            device="cuda:0",
            dtype="bfloat16",
            language="en",
            saved_raw_rank_depth=64,
        )
        ignored = {"git_commit", "ctc_checkpoint"}
        if {k: v for k, v in current.items() if k not in ignored} != {
            k: v for k, v in old_config.items() if k not in ignored
        }:
            raise ValueError(f"retrieval settings differ beyond the Head: {name}")
        groups[name] = {
            "config": current,
            "target_ids": ids,
            "source_samples": len(records),
            "source_target_count_distribution": distribution,
            "eligible_samples": sum(v for k, v in distribution.items() if int(k) in GROUPS),
            "selected_samples": len(selected),
            "selected_target_count_distribution": dict(
                Counter(str(len(s["expected_hotword_ids"])) for s in selected)
            ),
            "gate": gate,
        }
        print(
            f"{name}: source={len(records)}, eligible={groups[name]['eligible_samples']}, "
            f"selected={len(selected)}",
            flush=True,
        )
        samples.extend(selected)
    if total_expected != cfg["expected_target_sample_pairs"]:
        raise ValueError("frozen full-English target denominator differs")
    if len({s["case_id"] for s in samples}) != len(samples) or len(
        {s["audio_path"] for s in samples}
    ) != len(samples):
        raise ValueError("duplicate selected case or audio path")
    with stage_progress(
        "Verify ONLY selected clean WAV bytes", total=len(samples), every=25
    ) as tick:
        for n, sample in enumerate(samples, 1):
            path = Path(sample["audio_path"])
            _checked(path, inputs, old["inputs"][str(path)]["sha256"])
            tick(n)
    return {
        "schema_version": 1,
        "git_commit": external._git_commit(),
        "configuration": cfg,
        "inputs": inputs,
        "checkpoint": checkpoint,
        "groups": groups,
        "samples": samples,
        "audio_condition": "clean",
        "retrieval_mode": "complete_audio_offline_top7",
    }, {
        "model": model,
        "checkpoint": head,
        "vocab": vocab,
        "bundle": bundle,
        "variants": variants,
        "by_variant": by_variant,
    }


def _reuse_rows(
    source: Path, plan: dict[str, Any], inputs: dict[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    _checked(source / "sha256.txt", inputs)
    old = _json(_verified(source, "run_config.json", inputs))
    report = _json(_verified(source, "report.json", inputs))
    if (
        report.get("status") != "completed"
        or report.get("checkpoint", {}).get("sha256") != plan["checkpoint"]["sha256"]
        or old.get("checkpoint", {}).get("sha256") != plan["checkpoint"]["sha256"]
        or report.get("table_sha256") != plan["configuration"]["table_sha256"]
        or "audio_pairing" in old
    ):
        raise ValueError("reuse requires a sealed epoch27 clean offline run with the same table")
    package = Path(__file__).resolve().parents[1]
    for relative, digest in old["code_sha256"].items():
        _checked(package / relative, inputs, digest)
    result = {}
    wanted = {(s["source"], s["sample_id"]): s for s in plan["samples"]}
    for name, group in plan["groups"].items():
        ignored = {"git_commit"}
        if {k: v for k, v in old["groups"][name]["config"].items() if k not in ignored} != {
            k: v for k, v in group["config"].items() if k not in ignored
        }:
            raise ValueError("reuse config/audio/Head differs")
        path = _verified(source, f"languages/en/{name}/retrieval_details.jsonl", inputs)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["source"], row["sample_id"])
            if key in wanted:
                if key in result:
                    raise ValueError("duplicate reused sample")
                sample = wanted[key]
                if any(
                    row.get(k) != sample[k] for k in ("audio_path", "reference_text", "language")
                ):
                    raise ValueError("reused sample reference/path differs")
                if row.get("audio_sha256") != inputs[sample["audio_path"]]["sha256"]:
                    raise ValueError("reused sample audio SHA256 differs")
                result[key] = row
    if set(result) != set(wanted):
        raise ValueError("reuse run lacks selected samples")
    return result


def run_selection(
    root: Path,
    config: Path,
    output: Path,
    *,
    resume: bool = False,
    prepare_only: bool = False,
    reuse_run: Path | None = None,
    detector_factory: Callable[..., Any] | None = None,
    retrieve_function: external.RetrieveFunction | None = None,
) -> dict[str, Any]:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; use --resume for identical inputs: {output}")
    with stage_progress("Verify frozen targets, select 100 per source, bind final Head"):
        plan, runtime = prepare_selection(root, config)
    reused = _reuse_rows(reuse_run, plan, plan["inputs"]) if reuse_run else {}
    for filename in plan["inputs"]:
        path = Path(filename).resolve()
        if path.is_relative_to(output) or (
            path.parent != root and output.is_relative_to(path.parent)
        ):
            raise ValueError("output overlaps a protected input directory")
    plan["reuse_run"] = str(reuse_run) if reuse_run else None
    external._prepare_output(output, plan, resume=resume)
    with exclusive_feature_cache_run(output):
        summary = {
            "status": "selection_prepared",
            "output_dir": str(output),
            "sample_count": len(plan["samples"]),
            "checkpoint": plan["checkpoint"],
            "by_source": {
                name: {k: v for k, v in g.items() if k not in {"config", "gate", "target_ids"}}
                for name, g in plan["groups"].items()
            },
        }
        if prepare_only:
            return summary
        contract = _sha(output / "run_config.json")
        shards = output / "sample_shards"
        shards.mkdir(exist_ok=True)
        detector = None
        delivered = []
        try:
            with stage_progress(
                "Epoch27 clean whole-audio Top7", total=len(plan["samples"]), every=10
            ) as tick:
                for n, sample in enumerate(plan["samples"], 1):
                    record = external.ExternalAudioRecord(
                        **{
                            **{
                                k: sample[k]
                                for k in ("sample_id", "source", "reference_text", "language")
                            },
                            "audio_path": Path(sample["audio_path"]),
                        }
                    )
                    shard = external._shard_path(shards, record)
                    digest = plan["inputs"][sample["audio_path"]]["sha256"]
                    if shard.exists():
                        envelope = _json(shard)
                        if envelope.get("contract_sha256") != contract:
                            raise ValueError("sample shard belongs to another run")
                        row = envelope["row"]
                    elif (record.source, record.sample_id) in reused:
                        row = reused[(record.source, record.sample_id)]
                    else:
                        if _sha(record.audio_path) != digest:
                            raise ValueError("selected audio changed after preparation")
                        if detector is None:
                            with stage_progress("Load frozen Qwen encoder and final epoch27 Head"):
                                detector = (detector_factory or external._load_detector)(
                                    model=runtime["model"],
                                    checkpoint=runtime["checkpoint"],
                                    vocab=runtime["vocab"],
                                    hotwords=runtime["variants"],
                                    device="cuda:0",
                                    dtype="bfloat16",
                                    language="English",
                                )
                        waveform, timing = external._load_audio(record.audio_path)
                        raw = (
                            retrieve_function(detector, waveform, runtime["vocab"])
                            if retrieve_function
                            else external._retrieve_waveform(
                                detector, waveform, runtime["vocab"], saved_raw_rank_depth=64
                            )
                        )
                        retrieval = collapse_variants(
                            raw,
                            runtime["variants"],
                            runtime["by_variant"],
                            plan["groups"][record.source]["gate"],
                        )
                        row = external._build_result_row(
                            record,
                            bundle=runtime["bundle"],
                            retrieval=retrieval,
                            audio_timing=timing,
                        )
                        row["audio_sha256"] = digest
                        row["row_sha256"] = _row_digest(row)
                    if (
                        any(row.get(k) != v for k, v in record.to_dict().items())
                        or row.get("audio_sha256") != digest
                        or row.get("row_sha256") != _row_digest(row)
                    ):
                        raise ValueError("sample shard/reuse identity or checksum differs")
                    replayed = replay_rows_top7(
                        [row], gate=plan["groups"][record.source]["gate"], bundle=runtime["bundle"]
                    )[0]
                    matches = external._list_of_mappings(replayed, "top7_selected_matches")
                    sample = {
                        **sample,
                        "selected_matches": [
                            {
                                k: match[k]
                                for k in (
                                    "hotword_id",
                                    "surface",
                                    "score",
                                    "edit_ratio",
                                    "posterior_confidence",
                                )
                            }
                            for match in matches
                        ],
                    }
                    external._write_or_verify_json(shard, {"contract_sha256": contract, "row": row})
                    delivered.append(sample)
                    tick(n)
        finally:
            detector = None
            gc.collect()
            if detector_factory is None and not reuse_run:
                import torch

                torch.cuda.empty_cache()
        with stage_progress(
            "Recheck inputs and publish single downstream JSON", total=len(plan["inputs"]), every=25
        ) as tick:
            for n, (filename, identity) in enumerate(plan["inputs"].items(), 1):
                if _sha(Path(filename)) != identity["sha256"]:
                    raise ValueError(f"input changed during execution: {filename}")
                tick(n)
            selection = {
                "schema_version": 1,
                "evaluation_scope": "external_english_clean_300_target_count_1_to_3",
                "test_set_used": True,
                "seed": plan["configuration"]["seed"],
                "selection_profile": "three_sources_uniform_from_1_to_3_targets",
                "retrieval_mode": "operating",
                "retrieval_config": {"top_k": 7, **next(iter(plan["groups"].values()))["gate"]},
                "checkpoint": plan["checkpoint"],
                "table_sha256": plan["configuration"]["table_sha256"],
                "vocab_sha256": plan["configuration"]["vocab_sha256"],
                "audio_condition": "clean",
                "sample_count": len(delivered),
                "positive_count": len(delivered),
                "negative_count": 0,
                "actual_group_counts": dict(Counter(s["primary_group"] for s in delivered)),
                "sample_counts_by_source": dict(Counter(s["source"] for s in delivered)),
                "target_policy": (
                    "all_frozen_source_targets; distinct_surface_presence; "
                    "no_longest_match_collapse"
                ),
                "group_policy": (
                    "target_count_only; no_synthetic_nested_families; redundant_family_ids_empty"
                ),
                "qwen_decoder_used": False,
                "samples": delivered,
            }
            path = output / "sample_selection.json"
            external._write_or_verify_json(path, selection)
            summary.update(status="completed", delivery_file=str(path), delivery_sha256=_sha(path))
            external._write_or_verify_json(output / "report.json", summary)
        return summary
