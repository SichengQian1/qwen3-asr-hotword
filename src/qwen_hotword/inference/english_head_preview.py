"""Head-only comparison against sealed English results; keep the full joint index."""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _validated_source_gate
from qwen_hotword.inference.reviewed_joint import (
    collapse_variants,
    export_joint_group,
    joint_bundle,
)
from qwen_hotword.inference.reviewed_retrieval import _aggregate, _old_metrics
from qwen_hotword.inference.wave_retrieval import _row_digest
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress


def _snapshot(
    root: Path, cfg: dict[str, Any], vocab: PhonemeVocab, inputs: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    import torch

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
        ctc_head_config,
    )

    head = (root / cfg["checkpoint"]).resolve()
    checksum = _checked((root / cfg["checkpoint_checksum"]).resolve(), inputs)
    lines = [line.split(maxsplit=1) for line in checksum.read_text().splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0]) != 2:
        raise ValueError("snapshot checksum must identify exactly one checkpoint")
    digest, filename = lines[0]
    if (root / filename.removeprefix("*")).resolve() != head:
        raise ValueError("snapshot checksum path differs from the requested checkpoint")
    _checked(head, inputs, digest)
    payload = torch.load(head, map_location="cpu", weights_only=True)
    if tuple(payload.get("vocab_tokens", ())) != tuple(vocab.tokens):
        raise ValueError("snapshot vocabulary differs")
    metrics = payload["epoch_metrics"]
    if metrics["epoch"] != cfg["expected_epoch"]:
        raise ValueError("snapshot epoch differs from the requested preview")
    model = build_ctc_head_from_checkpoint(payload)
    if not isinstance(model, TemporalUpsampleCtcHead) or model.time_upsampling_factor != 2:
        raise ValueError("snapshot must be a Temporal 2x Head")
    model.load_state_dict(payload["state_dict"], strict=True)
    return head, {"sha256": digest, "epoch_metrics": metrics, "head_config": ctc_head_config(model)}


def _index(
    path: Path, vocab: PhonemeVocab
) -> tuple[external.KeywordBundle, tuple[HotwordEntry, ...], dict[str, HotwordEntry]]:
    # Use the sealed, already deduplicated index, including ES/PT pronunciations.
    raw = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    entries = tuple(
        HotwordEntry(**{**r, **{k: tuple(r[k]) for k in ("words", "phoneme_tokens", "token_ids")}})
        for r in raw
    )
    if len({e.hotword_id for e in entries}) != len(entries) or any(
        not e.token_ids
        or any(i <= 0 or i >= len(vocab.tokens) for i in e.token_ids)
        or tuple(vocab.tokens[i] for i in e.token_ids) != e.phoneme_tokens
        for e in entries
    ):
        raise ValueError("invalid sealed pronunciation index")
    bundles = {
        lang: external.KeywordBundle(tuple(e for e in entries if e.language == label), {}, {})
        for lang, label in (("en", "en"), ("es", "es"), ("pt", "pt-BR"))
    }
    bundle, variants, by_variant = joint_bundle(bundles)
    if [e.to_dict() for e in variants] != raw:
        raise ValueError("sealed index ordering/languages differ")
    return bundle, variants, by_variant


def prepare_preview(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
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
        or not report.get("top5_reproduced")
        or not report.get("top7_exact_replay")
        or report["review"]["expanded_table_sha256"] != cfg["table_sha256"]
        or report["review"]["active_unique_surfaces_per_audio"] != cfg["expected_surfaces"]
    ):
        raise ValueError("completed joint 4k baseline identity differs")
    cfg_old = old["configuration"]
    table = _checked((root / cfg_old["expanded_table"]).resolve(), inputs, cfg["table_sha256"])
    frozen = _verified(source, "reviewed_keyword_table.json", inputs)
    if _json(table) != _json(frozen):
        raise ValueError("baseline table snapshot differs")
    vocab_path = _checked((root / cfg_old["vocab"]).resolve(), inputs, cfg["vocab_sha256"])
    vocab = load_phoneme_vocab(vocab_path)
    head, checkpoint = _snapshot(root, cfg, vocab, inputs)
    if checkpoint["sha256"] == cfg["baseline_checkpoint_sha256"]:
        raise ValueError("preview checkpoint is the old baseline Head")
    model = (root / cfg_old["model"]).resolve()
    for name in ("config.json", "model.safetensors.index.json"):
        path = model / name
        if name == "config.json" or path.exists():
            _checked(path, inputs, old["inputs"][str(path)]["sha256"])
    package = Path(__file__).resolve().parents[1]
    for relative, digest in old["code_sha256"].items():
        if _sha(package / relative) != digest:
            raise ValueError(f"baseline retrieval code changed: {relative}")
    bundle, variants, by_variant = _index(
        _verified(source, "indexed_pronunciations.jsonl", inputs), vocab
    )
    if (
        len(bundle.entries) != cfg["expected_surfaces"]
        or len(variants) != report["review"]["active_pronunciation_variants_per_audio"]
    ):
        raise ValueError("baseline index size differs")
    groups, records_by_group, baseline = {}, {}, {}
    if set(cfg["datasets"]) != {"en_chile", "en_colombia", "en_mexico"}:
        raise ValueError("preview requires exactly the three clean English datasets")
    for name, count in cfg["datasets"].items():
        previous = old["groups"][name]
        old_config = previous["config"]
        if previous["language"] != "en" or len(old_config["sources"]) != 1:
            raise ValueError(f"baseline group is not one English source: {name}")
        original = old_config["sources"][0]
        spec = external.SourceSpec(
            name, Path(original["audio_dir"]), Path(original["transcripts"]["path"])
        )
        _checked(spec.transcript_path, inputs, original["transcripts"]["sha256"])
        records, audit = external.build_external_dataset((spec,), language="en")
        if len(records) != count or [r.to_dict() for r in records] != previous["records"]:
            raise ValueError(f"baseline sample/reference identity differs: {name}")
        with stage_progress(f"Verify baseline audio SHA256 {name}", total=count, every=100) as tick:
            for n, record in enumerate(records, 1):
                _checked(record.audio_path, inputs, old["inputs"][str(record.audio_path)]["sha256"])
                tick(n)
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
        } or old_config["ctc_checkpoint"]["sha256"] != cfg["baseline_checkpoint_sha256"]:
            raise ValueError(f"settings differ beyond Head/revision: {name}")
        details = _verified(source, f"languages/en/{name}/retrieval_details.jsonl", inputs)
        groups[name] = {
            **previous,
            "config": current,
            "audit": audit,
            "baseline_details": str(details),
            "baseline_target_ids": previous["target_ids"],
        }
        metrics = _old_metrics(groups[name], {"config": old_config, "bundle": bundle})
        fields = (
            "sample_count",
            "expected_target_sample_pairs",
            "gated_target_hits",
            "raw_target_hits",
            "selected_all_words",
            "selected_words_absent_from_transcript",
            "gated_target_recall",
            "raw_target_recall",
        )
        if any(
            metrics[k][field] != report["groups"][name]["current"][k][field]
            for k in ("top5", "top7")
            for field in fields
        ):
            raise ValueError(f"baseline detail/report metrics differ: {name}")
        baseline[name] = {"baseline_same_retained_targets": metrics}
        records_by_group[name] = records
    aggregate = _aggregate(baseline, list(groups), baseline=True)
    expected = cfg["expected_baseline"]
    for k in ("top5", "top7"):
        if any(
            aggregate[k][f] != expected[f] for f in ("sample_count", "expected_target_sample_pairs")
        ):
            raise ValueError("baseline sample count/target denominator differs")
        if aggregate[k]["gated_target_hits"] != expected[f"{k}_hits"]:
            raise ValueError("baseline hit count differs")
        if any(
            aggregate[k][f] != report["by_language"]["en"]["current"][k][f] for f in aggregate[k]
        ):
            raise ValueError("baseline English aggregate differs")
    plan = {
        "schema_version": 1,
        "git_commit": external._git_commit(),
        "configuration": cfg,
        "inputs": inputs,
        "groups": groups,
        "review": report["review"],
        "checkpoint": checkpoint,
        "baseline_same_retained_targets": aggregate,
        "audio_hashes_verified": True,
        "model_weight_shards_hashed": False,
        "code_sha256": {
            **old["code_sha256"],
            "inference/english_head_preview.py": _sha(Path(__file__)),
        },
    }
    return plan, {
        "bundle": bundle,
        "variants": variants,
        "by_variant": by_variant,
        "model": model,
        "checkpoint": head,
        "vocab": vocab,
        "table": table,
        "records": records_by_group,
        "baseline": baseline,
    }


def _return_bundle(output: Path) -> Path:
    # Small evidence only: no weights, audio, tensors or 6k per-sample rows.
    files = {
        name: (output / name).read_bytes()
        for name in ("report.json", "run_config.json", "checkpoint_metadata.json")
    }
    files["sha256.txt"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())
    ).encode()
    path = output / "return_bundle.zip"
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != set(files) or any(
                archive.read(name) != data for name, data in files.items()
            ):
                raise ValueError("existing return bundle differs")
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        # Publish a complete archive; interruption cannot leave a partial final ZIP.
        with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(buffer.getvalue())
                handle.flush()
                os.fsync(handle.fileno())
                os.link(temporary, path)
            finally:
                temporary.unlink()
    return path


def run_preview(
    root: Path,
    config: Path,
    output: Path,
    *,
    resume: bool = False,
    audit_only: bool = False,
    detector_factory: Callable[..., Any] | None = None,
    retrieve_function: external.RetrieveFunction | None = None,
) -> dict[str, Any]:
    cfg = _json(config)
    protected = [(root / cfg[k]).resolve() for k in ("baseline_run", "checkpoint")]
    if any(
        output == p
        or output.is_relative_to(p if p.is_dir() else p.parent)
        or p.is_relative_to(output)
        for p in protected
    ):
        raise ValueError("output must be separate from the baseline and Head snapshot")
    if output.exists() and not resume and not audit_only:
        raise FileExistsError(f"output exists; use --resume for identical inputs: {output}")
    with stage_progress("Verify frozen baseline, index, Head snapshot and English samples"):
        plan, runtime = prepare_preview(root, config)
    old_top7 = plan["baseline_same_retained_targets"]["top7"]
    print(
        f"Frozen Head epoch={plan['checkpoint']['epoch_metrics']['epoch']} "
        f"SHA256={plan['checkpoint']['sha256']}; baseline Top7="
        f"{old_top7['gated_target_hits']}/{old_top7['expected_target_sample_pairs']}; "
        f"full joint index={len(runtime['bundle'].entries)} surfaces",
        flush=True,
    )
    if audit_only:
        return {
            "status": "audit_pass",
            "checkpoint": plan["checkpoint"],
            "samples": {n: len(r) for n, r in runtime["records"].items()},
            "baseline": plan["baseline_same_retained_targets"],
            "model_loaded": False,
        }
    # Also protect every input directory discovered from the source plan.
    if any(
        Path(p).is_relative_to(output)
        or (Path(p).parent != root and output.is_relative_to(Path(p).parent))
        for p in plan["inputs"]
    ):
        raise ValueError("output overlaps a protected input directory")
    external._prepare_output(output, plan, resume=resume)
    external._write_or_verify_json(output / "checkpoint_metadata.json", plan["checkpoint"])
    external._write_or_verify_json(output / "reviewed_keyword_table.json", _json(runtime["table"]))
    external._write_or_verify_jsonl(
        output / "indexed_pronunciations.jsonl", [e.to_dict() for e in runtime["variants"]]
    )
    metrics, detector = {}, None
    language_output = output / "languages/en"
    (language_output / "delivery").mkdir(parents=True, exist_ok=True)
    try:
        for name, records in runtime["records"].items():
            group = plan["groups"][name]
            shards = language_output / name / "sample_shards"
            shards.mkdir(parents=True, exist_ok=True)
            rows = []
            with stage_progress(
                f"English CTC/retrieval {name}", total=len(records), every=50
            ) as tick:
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
                            with stage_progress("Load frozen Qwen and snapshot Head"):
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
                            record,
                            bundle=runtime["bundle"],
                            retrieval=retrieval,
                            audio_timing=timing,
                        )
                        for key in ("raw_variant_ranked_matches", "variant_shortlist_candidates"):
                            row[key] = retrieval[key]
                        row["audio_sha256"] = digest
                        row["row_sha256"] = _row_digest(row)
                        external._write_json(shard, row)
                    rows.append(row)
                    tick(n)
            with stage_progress(f"Export and verify Top5/Top7 {name}"):
                current = export_joint_group(language_output, name, group, runtime["bundle"], rows)
            baseline = runtime["baseline"][name]["baseline_same_retained_targets"]
            if any(
                current[k]["expected_target_sample_pairs"]
                != baseline[k]["expected_target_sample_pairs"]
                for k in ("top5", "top7")
            ):
                raise ValueError(f"target denominator changed: {name}")
            metrics[name] = {"current": current, "baseline_same_retained_targets": baseline}
    finally:
        detector = None
        gc.collect()
        if detector_factory is None:
            import torch

            torch.cuda.empty_cache()
    with stage_progress("Recheck immutable evaluation inputs"):
        for filename, identity in plan["inputs"].items():
            if _sha(Path(filename)) != identity["sha256"]:
                raise ValueError(f"input changed during inference: {filename}")
    current = _aggregate(metrics, list(metrics), baseline=False)
    baseline = plan["baseline_same_retained_targets"]
    report = {
        "status": "completed",
        "checkpoint": plan["checkpoint"],
        "candidate_scope": "joint_multilingual",
        "sample_count": cfg["expected_baseline"]["sample_count"],
        "groups": metrics,
        "current": current,
        "baseline_same_retained_targets": baseline,
        "recall_delta_percentage_points": {
            k: 100 * (current[k]["gated_target_recall"] - baseline[k]["gated_target_recall"])
            for k in ("top5", "top7")
        },
        "table_sha256": cfg["table_sha256"],
        "active_unique_surfaces_per_audio": cfg["expected_surfaces"],
        "active_pronunciation_variants_per_audio": len(runtime["variants"]),
        "delivery_files": [
            f"languages/en/delivery/{n}_top{k}.json" for n in metrics for k in (5, 7)
        ],
        "top5_reproduced": True,
        "top7_exact_replay": True,
        "parameters_tuned": False,
        "qwen_decoder_used": False,
        "limitations": [
            "English audio only; every audio searches the original full EN/ES/PT joint table.",
            "Only the frozen Head changes; this is an interim preview while training continues.",
            "Retrieval recall is not CTC PER or final ASR WER; no LLM transcription is run.",
            "The original fillers were conditioned on reference absence; this is not a blind test.",
            "No new byte-level or speaker-level training/test isolation claim is made.",
        ],
        "return_file": str(output / "return_bundle.zip"),
    }
    with stage_progress("Publish comparison, return bundle and output checksums"):
        external._write_or_verify_json(output / "report.json", report)
        _return_bundle(output)
        seal(output)
    return report
