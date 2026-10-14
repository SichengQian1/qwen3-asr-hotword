"""JSON-listed noisy English audio, unchanged joint Anchor scoring and Top5/Top7."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.evaluation.ziyan_assets import (
    NAME,
    bundle_from_table,
    presence,
    read_dataset,
    read_targets,
)
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import (
    _validated_source_gate,
    replay_rows_top7,
)
from qwen_hotword.inference.reviewed_joint import collapse_variants, export_joint_group
from qwen_hotword.inference.wave_retrieval import _row_digest, _view_metrics
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified


def prepare(root: Path, config: Path, assets: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import soundfile as sf

    identities: dict[str, Any] = {}
    cfg = _json(_checked(config, identities))
    _checked(assets / "sha256.txt", identities)
    report = _json(_verified(assets, "report.json", identities))
    if report.get("status") != "completed" or report.get("table_ready") is not True:
        raise ValueError("completed MFA replacement table required")
    if _json(_verified(assets, "config.json", identities)) != cfg:
        raise ValueError("assets config differs; prepare a new assets directory")
    for path, identity in report["inputs"].items():
        _checked(Path(path), identities, identity["sha256"])
    dataset = _checked(Path(cfg["dataset"]), identities, cfg["dataset_sha256"])
    records = read_dataset(dataset, cfg["expected_samples"])
    targets_path = _verified(assets, "targets.json", identities)
    targets = read_targets(
        _checked(root / cfg["targets"], identities, cfg["targets_sha256"]), records
    )
    if _json(targets_path) != targets or targets != report["targets"]:
        raise ValueError("frozen target identities differ")
    manifest = _verified(assets, "dataset_manifest.jsonl", identities)
    if [json.loads(s) for s in manifest.read_text().splitlines()] != [r.to_dict() for r in records]:
        raise ValueError("frozen audio/reference records differ")
    _verified(assets, "english_raw.dict", identities)
    table = _verified(assets, "keyword_table.json", identities)
    vocab_path = _checked(root / cfg["vocab"], identities, cfg["vocab_sha256"])
    vocab = load_phoneme_vocab(vocab_path)
    bundle, variants, mapping = bundle_from_table(_json(table), table, vocab)
    index = {e.normalized: e.hotword_id for e in bundle.entries}
    if (
        not set(targets) <= index.keys()
        or len(bundle.entries) != report["selection"]["unique_surfaces"]
    ):
        raise ValueError("runtime table count/target coverage differs")
    model = Path(cfg["model"])
    head = _checked(root / cfg["checkpoint"], identities, cfg["checkpoint_sha256"])
    _checked(model / "config.json", identities)
    if (model / "model.safetensors.index.json").is_file():
        _checked(model / "model.safetensors.index.json", identities)
    metadata, hashes = [], []
    for record in records:
        _checked(record.audio_path, identities)
        info = sf.info(str(record.audio_path))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError(f"invalid audio header: {record.audio_path}")
        metadata.append(
            {
                "sample_id": record.sample_id,
                "duration_seconds": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
            }
        )
        hashes.append(identities[str(record.audio_path)]["sha256"])
    group_config = external._run_config(
        model=model,
        checkpoint=head,
        vocab_path=vocab_path,
        keyword_path=table,
        sources=(),
        keyword_set="all_keywords",
        device="cuda:0",
        dtype="bfloat16",
        language="English",
        saved_raw_rank_depth=64,
    )
    group = {
        "language": "en",
        "config": group_config,
        "target_ids": sorted(index[t] for t in targets),
    }
    plan = {
        "configuration": cfg,
        "inputs": identities,
        "group": group,
        "target_presence": presence(records, targets),
        "records": [r.to_dict() for r in records],
        "audio_metadata": metadata,
        "duplicate_audio_file_hashes": len(hashes) - len(set(hashes)),
        "model_weight_shards_hashed": False,
        "code_sha256": {
            rel: _sha(Path(__file__).parents[1] / rel)
            for rel in (
                "evaluation/ziyan_assets.py",
                "inference/ziyan_retrieval.py",
                "inference/reviewed_joint.py",
                "inference/external_keyword_retrieval.py",
                "inference/external_keyword_topk_replay.py",
                "inference/wave_retrieval.py",
                "inference/streaming_backends.py",
                "inference/hotword_prompt.py",
                "hotwords/anchor_index.py",
                "hotwords/exact_automaton.py",
                "hotwords/scoring.py",
                "phonemes/coverage.py",
                "hotwords/registry.py",
                "evaluation/target_review.py",
                "evaluation/wave_keywords.py",
            )
        },
    }
    plan["code_sha256"]["scripts/run_ziyan_english.py"] = _sha(
        Path(__file__).parents[3] / "scripts/run_ziyan_english.py"
    )
    return plan, {
        "records": records,
        "bundle": bundle,
        "variants": variants,
        "mapping": mapping,
        "model": model,
        "head": head,
        "vocab": vocab,
        "table": table,
        "asset_report": report,
    }


def run(
    root: Path,
    config: Path,
    assets: Path,
    output: Path,
    *,
    resume: bool = False,
    audit_only: bool = False,
    detector_factory: Any = None,
    retrieve_function: Any = None,
) -> dict[str, Any]:
    if output.exists() and not resume and not audit_only:
        raise FileExistsError("output exists; --resume only for this identical experiment")
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("retrieval output must be separate from prepared assets")
    plan, runtime = prepare(root, config, assets)
    if any(output.is_relative_to(Path(p).parent) for p in plan["inputs"]):
        raise ValueError("retrieval output must be outside all input asset directories")
    if audit_only:
        return {
            "status": "audit_pass",
            "files_written": False,
            "model_loaded": False,
            "sample_count": len(runtime["records"]),
            "keyword_count": len(runtime["bundle"].entries),
            "target_presence": plan["target_presence"],
            "audio_hours": sum(m["duration_seconds"] for m in plan["audio_metadata"]) / 3600,
            "duplicate_audio_file_hashes": plan["duplicate_audio_file_hashes"],
        }
    if output.exists():
        if not (output / "run_config.json").is_file() or _json(output / "run_config.json") != plan:
            raise ValueError("resume audio/targets/table/model/config/code differs; use new output")
    else:
        output.mkdir(parents=True, exist_ok=False)
        external._write_json(output / "run_config.json", plan)
    (output / "delivery").mkdir(exist_ok=True)
    external._write_or_verify_json(output / "keyword_table.json", _json(runtime["table"]))
    shards = output / NAME / "sample_shards"
    shards.mkdir(parents=True, exist_ok=True)
    rows, detector = [], None
    gate = _validated_source_gate(plan["group"]["config"])
    try:
        for n, record in enumerate(runtime["records"], 1):
            shard = external._shard_path(shards, record)
            digest = plan["inputs"][str(record.audio_path)]["sha256"]
            if shard.exists():
                row = _json(shard)
                if (
                    row.get("row_sha256") != _row_digest(row)
                    or row.get("audio_sha256") != digest
                    or any(row.get(k) != v for k, v in record.to_dict().items())
                ):
                    raise ValueError(f"corrupt/changed saved sample: {record.sample_id}")
            else:
                if _sha(record.audio_path) != digest:
                    raise ValueError("audio changed after preflight")
                if detector is None:
                    detector = (detector_factory or external._load_detector)(
                        model=runtime["model"],
                        checkpoint=runtime["head"],
                        vocab=runtime["vocab"],
                        hotwords=runtime["variants"],
                        device="cuda:0",
                        dtype="bfloat16",
                        language="English",
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
                    retrieval, runtime["variants"], runtime["mapping"], gate
                )
                row = external._build_result_row(
                    record, bundle=runtime["bundle"], retrieval=retrieval, audio_timing=timing
                )
                row["raw_variant_ranked_matches"] = retrieval["raw_variant_ranked_matches"]
                row["variant_shortlist_candidates"] = retrieval["variant_shortlist_candidates"]
                row["audio_sha256"] = digest
                row["row_sha256"] = _row_digest(row)
                external._write_json(shard, row)
            rows.append(row)
            if n == 1 or n % 50 == 0 or n == len(runtime["records"]):
                print(f"{NAME}: {n}/{len(runtime['records'])}", flush=True)
    finally:
        detector = None
        gc.collect()
    metrics = export_joint_group(output, NAME, plan["group"], runtime["bundle"], rows)
    replay = replay_rows_top7(rows, gate=gate, bundle=runtime["bundle"])
    targets = set(plan["group"]["target_ids"])
    by_scenario = {
        scenario: {
            f"top{k}": _view_metrics(
                [r for r in replay if str(r["sample_id"]).split("_")[0] == scenario],
                targets,
                top_k=k,
            )
            for k in (5, 7)
        }
        for scenario in sorted({str(r["sample_id"]).split("_")[0] for r in replay})
    }
    for path, identity in plan["inputs"].items():
        if _sha(Path(path)) != identity["sha256"]:
            raise ValueError(f"input changed during retrieval: {path}")
    report = {
        "status": "completed",
        "sample_count": len(rows),
        "candidate_scope": "joint_multilingual",
        "keyword_count": len(runtime["bundle"].entries),
        "metrics": metrics,
        "by_scenario": by_scenario,
        "target_presence": plan["target_presence"],
        "table_replacement": runtime["asset_report"]["selection"],
        "retained_filler_reference_overlap": runtime["asset_report"][
            "retained_filler_reference_overlap"
        ],
        "checkpoint_sha256": plan["configuration"]["checkpoint_sha256"],
        "table_sha256": plan["inputs"][str(runtime["table"])]["sha256"],
        "delivery_file_count": 2,
        "top5_reproduced": True,
        "top7_exact_replay": True,
        "snr_verified": False,
        "qwen_decoder_used": False,
        "parameters_tuned": False,
        "metric_unit": "distinct new target per audio, not repeated word occurrences",
        "limitations": [
            "Transcript-selected targets; exploratory, not a blind vocabulary benchmark.",
            "Old fillers retained, including overlaps with new transcripts; counted separately.",
            "New noisy data, targets and table size differ from previous English tests.",
            "10dB is provider naming; no paired clean audio or measured SNR in this run.",
            "MFA labels are not acoustic correctness certification; no final ASR WER or CTC PER.",
            "Theoretical TopK capacity bound ignores ranking, gates and acoustic errors.",
        ],
    }
    external._write_or_verify_json(output / "report.json", report)
    seal(output)
    return report
