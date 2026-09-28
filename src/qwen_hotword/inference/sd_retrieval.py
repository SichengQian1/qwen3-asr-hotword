"""Fixed SD 4000-word table on MLS and Delivery, using the existing Anchor path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import strict_phrase_match
from qwen_hotword.inference.wave_retrieval import _row_digest, export_group
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified


def prepare_sd_run(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _json(config_path)
    identities: dict[str, Any] = {}

    def checked(path: Path, expected: str | None = None) -> None:
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        identities[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}

    checked(config_path)
    tables = root / cfg["tables"]
    report = _json(_verified(tables, "report.json", identities))
    if report.get("status") != "completed" or not report.get("tables_created"):
        raise ValueError("keyword table is not completed")
    vocab_path, model, head = (root / cfg[k] for k in ("vocab", "model", "checkpoint"))
    checked(vocab_path, cfg["vocab_sha256"])
    checked(head, cfg["checkpoint_sha256"])
    checked(model / "config.json")
    if (model / "model.safetensors.index.json").is_file():
        checked(model / "model.safetensors.index.json")
    vocab = load_phoneme_vocab(vocab_path)
    keyword_path = _verified(tables, "keyword_bias_phoneme.json", identities)
    bundle = external.load_keyword_bias(keyword_path, vocab=vocab, keyword_set="all_keywords")
    targets = _json(_verified(tables, "targets.json", identities))
    by_key = {e.normalized: e.hotword_id for e in bundle.entries}
    if (
        len(bundle.entries) != cfg["expected_keywords"]
        or not isinstance(targets, list)
        or any(not isinstance(t, str) for t in targets)
        or len(targets) != len(set(targets))
        or len(targets) != cfg["expected_primary"]
        or not set(targets) <= by_key.keys()
    ):
        raise ValueError("table/target counts or identities mismatch")
    # Bind target labels back to the source whose SHA was recorded at table build time.
    primary = root / cfg["primary"]
    checked(primary, report["inputs"][str(primary)]["sha256"])
    original = external.load_keyword_bias(primary, vocab=vocab, keyword_set="all_keywords")
    labels = {e.normalized: e.token_ids for e in bundle.entries if e.normalized in targets}
    if labels != {e.normalized: e.token_ids for e in original.entries}:
        raise ValueError("SD target labels changed")
    groups = {}
    records_by_group = {}
    if set(cfg["sources"]) != {"mls_portuguese", "delivery_20260706_ptbr"}:
        raise ValueError("expected MLS and Delivery sources")
    for name, info in cfg["sources"].items():
        source = external.SourceSpec(name, root / info["audio"], root / info["transcripts"])
        checked(source.transcript_path)
        records, audit = external.build_external_dataset((source,))
        if len(records) != info["expected_samples"]:
            raise ValueError(f"{name}: expected {info['expected_samples']}, got {len(records)}")
        for record in records:
            checked(record.audio_path)
        target_pairs = [
            sum(strict_phrase_match(r.reference_text, e.surface) for e in original.entries)
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
            saved_raw_rank_depth=64,
        )
        groups[name] = {
            "config": group_config,
            "audit": audit,
            "records": [r.to_dict() for r in records],
            "target_ids": sorted(by_key[t] for t in targets),
            "target_presence": {
                "expected_target_sample_pairs": sum(target_pairs),
                "samples_with_target": sum(n > 0 for n in target_pairs),
                "samples_without_target": sum(n == 0 for n in target_pairs),
            },
        }
        records_by_group[name] = records
    return {
        "git_commit": external._git_commit(),
        "configuration": cfg,
        "inputs": identities,
        "groups": groups,
        "audio_hashes_verified": True,
        "model_weight_shards_hashed": False,
    }, {
        "vocab": vocab,
        "bundle": bundle,
        "records": records_by_group,
        "model": model,
        "checkpoint": head,
    }


def run_sd_retrieval(
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
    print("Checking table, Head, MLS/SD transcripts and audio SHA256...", flush=True)
    plan, runtime = prepare_sd_run(root, config_path)
    if audit_only:
        return {
            "status": "audit_pass",
            "model_loaded": False,
            "files_written": False,
            "samples": {k: len(v) for k, v in runtime["records"].items()},
            "keywords": len(runtime["bundle"].entries),
            "targets": plan["configuration"]["expected_primary"],
            "target_presence": {k: v["target_presence"] for k, v in plan["groups"].items()},
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
        metrics[name] = export_group(output, name, plan["groups"][name], bundle, rows)
    expected = sorted(f"{name}_top{k}.json" for name in metrics for k in (5, 7))
    if sorted(p.name for p in (output / "delivery").glob("*.json")) != expected:
        raise ValueError("expected exactly four downstream JSON files")
    report = {
        "status": "completed",
        "output_dir": str(output),
        "groups": metrics,
        "sample_count": sum(len(v) for v in runtime["records"].values()),
        "delivery_files": expected,
        "delivery_file_count": 4,
        "checkpoint_sha256": plan["configuration"]["checkpoint_sha256"],
        "top5_reproduced": True,
        "top7_exact_replay": True,
        "parameters_tuned": False,
        "qwen_decoder_used": False,
        "target_scope": "SD all_keywords only on both datasets",
        "metric_unit": "distinct target per audio, not repeated token occurrences",
        "limitations": [
            "MLS target coverage uses SD362, not the historical MLS266 denominator.",
            "Legacy comparisons also used a different Head; not a table-only ablation.",
            "Fillers can naturally be near-homophones; no claim of acoustic dissimilarity.",
            "No final ASR WER or CTC PER measured.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    external._write_or_verify_json(output / "report.json", report)
    seal(output)
    return report
