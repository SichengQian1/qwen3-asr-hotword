"""Fixed English 3000 table, three datasets, unchanged wave Top5/Top7 delivery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.english_keywords import (
    DATASETS,
    checked,
    read_targets,
    transcript_words,
)
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.wave_retrieval import _row_digest, export_group
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified


def prepare_english_run(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _json(config_path)
    identities: dict[str, Any] = {}
    checked(config_path, identities)
    tables = root / cfg["tables"]
    report = _json(_verified(tables, "report.json", identities))
    if (
        report.get("status") != "completed"
        or not report.get("tables_created")
        or report.get("selected_filler_transcript_overlap") != 0
        or report.get("table_filler_exclusion_used_transcripts") is not True
    ):
        raise ValueError("completed transcript-excluded English table required")
    # Bind every build input, including the train dictionary and all three transcripts.
    for filename, identity in report["inputs"].items():
        checked(Path(filename), identities, identity["sha256"])
    vocab_path, model, head = (root / cfg[k] for k in ("vocab", "model", "checkpoint"))
    checked(vocab_path, identities, cfg["vocab_sha256"])
    checked(head, identities, cfg["checkpoint_sha256"])
    checked(model / "config.json", identities)
    if (model / "model.safetensors.index.json").is_file():
        checked(model / "model.safetensors.index.json", identities)
    vocab = load_phoneme_vocab(vocab_path)
    keyword_path = _verified(tables, "keyword_bias_phoneme.json", identities)
    bundle = external.load_keyword_bias(
        keyword_path, vocab=vocab, keyword_set="all_keywords", language="en"
    )
    union, targets, transcripts, _ = read_targets(root, cfg, vocab, identities)
    frozen_targets = _json(_verified(tables, "targets_by_dataset.json", identities))
    frozen_union = _json(_verified(tables, "targets.json", identities))
    by_key = {e.normalized: e for e in bundle.entries}
    if (
        len(bundle.entries) != cfg["target_size"]
        or len(bundle.entries) != report["total"]
        or targets != frozen_targets
        or sorted(union) != frozen_union
        or len(union) != report["mandatory_union"]
        or not set(union) <= by_key.keys()
        or any(
            by_key[k].token_ids != e.token_ids
            or by_key[k].surface != e.surface
            or by_key[k].pronunciation != e.pronunciation
            for k, e in union.items()
        )
    ):
        raise ValueError("frozen table/target identity or labels changed")
    forbidden = transcript_words(transcripts)
    for key, entry in by_key.items():
        if key not in union and (len(entry.words) != 1 or key in forbidden):
            raise ValueError(f"filler appears in transcripts or is not a single word: {key}")
    groups = {}
    records_by_group = {}
    for name in DATASETS:
        info = cfg["sources"][name]
        source = external.SourceSpec(name, root / info["audio"], root / info["transcripts"])
        # Separate groups deliberately permit the same stem in different datasets.
        records, audit = external.build_external_dataset((source,), language="en")
        if len(records) != report["transcript_counts"][name]:
            raise ValueError(f"{name}: sample count changed since table build")
        if "expected_samples" in info and len(records) != info["expected_samples"]:
            raise ValueError(f"{name}: unexpected sample count {len(records)}")
        for record in records:
            checked(record.audio_path, identities)
        pairs = [
            sum(strict_phrase_match(r.reference_text, union[t].surface) for t in targets[name])
            for r in records
        ]
        group_config = external._run_config(
            model=model,
            checkpoint=head,
            vocab_path=vocab_path,
            keyword_path=keyword_path,
            sources=(source,),
            keyword_set="all_keywords",
            device="cuda:0",
            dtype="bfloat16",
            language="English",
            saved_raw_rank_depth=64,
        )
        group_config["table_filler_exclusion_used_transcripts"] = True
        groups[name] = {
            "config": group_config,
            "audit": audit,
            "records": [r.to_dict() for r in records],
            "target_ids": sorted(by_key[t].hotword_id for t in targets[name]),
            "target_presence": {
                "expected_target_sample_pairs": sum(pairs),
                "samples_with_target": sum(n > 0 for n in pairs),
                "samples_without_target": sum(n == 0 for n in pairs),
            },
        }
        records_by_group[name] = records
    plan = {
        "schema_version": 1,
        "git_commit": external._git_commit(),
        "retrieval_source_sha256": {
            relative: _sha(Path(__file__).resolve().parents[1] / relative)
            for relative in (
                "inference/english_retrieval.py",
                "inference/external_keyword_retrieval.py",
                "inference/wave_retrieval.py",
                "inference/external_keyword_topk_replay.py",
                "inference/hotword_prompt.py",
                "hotwords/anchor_index.py",
                "hotwords/scoring.py",
                "training/edit_distance.py",
                "phonemes/coverage.py",
                "evaluation/english_keywords.py",
            )
        },
        "configuration": cfg,
        "inputs": identities,
        "groups": groups,
        "audio_hashes_verified": True,
        "model_weight_shards_hashed": False,
        "table_filler_exclusion_used_transcripts": True,
    }
    return plan, {
        "vocab": vocab,
        "bundle": bundle,
        "records": records_by_group,
        "model": model,
        "checkpoint": head,
        "table_report": report,
    }


def run_english_retrieval(
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
        raise FileExistsError(f"output exists; --resume only for identical inputs: {output}")
    print("Checking English table, Head, transcripts and audio SHA256...", flush=True)
    plan, runtime = prepare_english_run(root, config_path)
    if audit_only:
        return {
            "status": "audit_pass",
            "model_loaded": False,
            "files_written": False,
            "samples": {name: len(rows) for name, rows in runtime["records"].items()},
            "keywords": len(runtime["bundle"].entries),
            "mandatory_union": runtime["table_report"]["mandatory_union"],
            "selected_filler_transcript_overlap": 0,
            "target_presence": {name: g["target_presence"] for name, g in plan["groups"].items()},
            "checkpoint_sha256": plan["configuration"]["checkpoint_sha256"],
        }
    if output.exists():
        if not (output / "run_config.json").is_file() or _json(output / "run_config.json") != plan:
            raise ValueError("resume input/config/code identity differs; use a new output")
    else:
        output.mkdir(parents=True, exist_ok=False)
        external._write_json(output / "run_config.json", plan)
    (output / "delivery").mkdir(exist_ok=True)
    bundle, vocab = runtime["bundle"], runtime["vocab"]
    detector = None
    metrics = {}
    for name, records in runtime["records"].items():
        shards = output / name / "sample_shards"
        shards.mkdir(parents=True, exist_ok=True)
        rows = []
        for n, record in enumerate(records, 1):
            shard = external._shard_path(shards, record)
            audio_sha = plan["inputs"][str(record.audio_path)]["sha256"]
            if shard.exists():
                row = _json(shard)
                external._validate_shard(row, record)
                if (
                    row.get("row_sha256") != _row_digest(row)
                    or row.get("audio_sha256") != audio_sha
                    or any(row.get(k) != v for k, v in record.to_dict().items())
                ):
                    raise ValueError(f"corrupted/changed sample: {name}/{record.sample_id}")
            else:
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
                        language="English",
                    )
                waveform, timing = external._load_audio(record.audio_path)
                result = (
                    retrieve_function(detector, waveform, vocab)
                    if retrieve_function
                    else external._retrieve_waveform(
                        detector, waveform, vocab, saved_raw_rank_depth=64
                    )
                )
                row = external._build_result_row(
                    record, bundle=bundle, retrieval=result, audio_timing=timing
                )
                row["audio_sha256"] = audio_sha
                row["row_sha256"] = _row_digest(row)
                external._write_json(shard, row)
            rows.append(row)
            if n == 1 or n % 50 == 0 or n == len(records):
                print(f"{name}: {n}/{len(records)}", flush=True)
        metrics[name] = export_group(
            output,
            name,
            plan["groups"][name],
            bundle,
            rows,
            full_table_metric_key="full_3000_view",
            table_filler_exclusion_used_transcripts=True,
        )
    expected = sorted(f"{name}_top{k}.json" for name in DATASETS for k in (5, 7))
    if sorted(p.name for p in (output / "delivery").glob("*.json")) != expected:
        raise ValueError("expected exactly six downstream JSON files")
    report = {
        "status": "completed",
        "output_dir": str(output),
        "groups": metrics,
        "sample_count": sum(len(v) for v in runtime["records"].values()),
        "keyword_count": len(bundle.entries),
        "delivery_files": expected,
        "delivery_file_count": 6,
        "checkpoint_sha256": plan["configuration"]["checkpoint_sha256"],
        "table_sha256": plan["inputs"][
            str(root / plan["configuration"]["tables"] / "keyword_bias_phoneme.json")
        ]["sha256"],
        "top5_reproduced": True,
        "top7_exact_replay": True,
        "parameters_tuned": False,
        "qwen_decoder_used": False,
        "table_filler_exclusion_used_transcripts": True,
        "target_scope": "per-dataset declared targets; full_3000_view covers the target union",
        "metric_unit": "distinct keyword per audio, not repeated token occurrences",
        "limitations": [
            "Transcript-conditioned filler exclusion; not a blind vocabulary experiment.",
            "Absence is based on supplied transcripts, not a manual audio audit.",
            "Natural phonetic similarities are retained; no WER or CTC PER is measured.",
            "Directory names are dataset names, not verified speaker accent metadata.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    external._write_or_verify_json(output / "report.json", report)
    seal(output)
    return report
