"""Read frozen retrieval assets; inspect headers and hash only selected stream audio."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.english_best_noise import load_completed_best
from qwen_hotword.inference.english_head_preview import _index
from qwen_hotword.inference.external_keyword_topk_replay import _validated_source_gate
from qwen_hotword.inference.stream_context_contract import boundaries
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress

KEYS = {"en_chile": "encl@@en", "en_colombia": "encol@@en", "en_mexico": "enmx@@en"}


def select_samples(rows: list[dict[str, Any]], limit: int, seed: str) -> list[dict[str, Any]]:
    if limit == 0 or len(rows) <= limit:
        return rows
    if limit < len(KEYS):
        raise ValueError("smoke must cover all three datasets")
    ranked = sorted(
        rows,
        key=lambda r: hashlib.sha256(f"{seed}:{r['group']}:{r['sample_id']}".encode()).hexdigest(),
    )
    selected: list[dict[str, Any]] = [
        next(r for r in ranked if r["group"] == group) for group in KEYS
    ]
    # Cover available duration buckets without changing, padding or fabricating audio.
    for bucket in ("under_2s", "exact_2s_multiple", "tail"):
        for group in KEYS:
            match = next(
                (
                    r
                    for r in ranked
                    if r["group"] == group and r["duration_bucket"] == bucket and r not in selected
                ),
                None,
            )
            if match is not None and len(selected) < limit:
                selected.append(match)
    while len(selected) < limit:
        counts = Counter(r["group"] for r in selected)
        for group in sorted(KEYS, key=lambda g: (counts[g], g)):
            match = next((r for r in ranked if r["group"] == group and r not in selected), None)
            if match is not None:
                selected.append(match)
                break
    return sorted(selected, key=lambda r: (r["group"], r["sample_id"]))


def prepare_stream(root: Path, config: Path, limit: int) -> tuple[dict[str, Any], dict[str, Any]]:
    import soundfile as sf

    cfg = _json(config)
    if (
        cfg["dataset_keys"] != KEYS
        or cfg["chunk_seconds"] != 2
        or cfg["retention_policy"] != "current_prefix_only"
        or cfg["candidate_form"] != "objects_truncated_per_mode"
    ):
        raise ValueError("unsupported stream contract configuration")
    inputs: dict[str, Any] = {}
    _checked(config, inputs)
    retrieval_path = _checked((root / cfg["retrieval_config"]).resolve(), inputs)
    source_cfg = _json(retrieval_path)
    source = (root / source_cfg["baseline_run"]).resolve()
    _checked(source / "sha256.txt", inputs)
    old = _json(_verified(source, "run_config.json", inputs))
    report = _json(_verified(source, "report.json", inputs))
    if (
        report["status"] != "completed"
        or report["candidate_scope"] != "joint_multilingual"
        or report["checkpoint_sha256"] != source_cfg["baseline_checkpoint_sha256"]
        or report["review"]["expanded_table_sha256"] != source_cfg["table_sha256"]
        or report["review"]["active_unique_surfaces_per_audio"] != source_cfg["expected_surfaces"]
    ):
        raise ValueError("sealed joint baseline identity differs")
    for k in ("top5", "top7"):
        metrics = report["by_language"]["en"]["current"][k]
        expected = source_cfg["expected_baseline"]
        if (
            any(metrics[f] != expected[f] for f in ("sample_count", "expected_target_sample_pairs"))
            or metrics["gated_target_hits"] != expected[f"{k}_hits"]
        ):
            raise ValueError("baseline English counts differ")
    vocab_path = _checked(
        (root / old["configuration"]["vocab"]).resolve(), inputs, source_cfg["vocab_sha256"]
    )
    vocab = load_phoneme_vocab(vocab_path)
    head, checkpoint = load_completed_best(root, source_cfg, vocab, inputs)
    table = _checked(
        (root / old["configuration"]["expanded_table"]).resolve(),
        inputs,
        source_cfg["table_sha256"],
    )
    if _json(table) != _json(_verified(source, "reviewed_keyword_table.json", inputs)):
        raise ValueError("frozen joint table differs")
    bundle, variants, by_variant = _index(
        _verified(source, "indexed_pronunciations.jsonl", inputs), vocab
    )
    if (
        len(bundle.entries) != source_cfg["expected_surfaces"]
        or len(variants) != report["review"]["active_pronunciation_variants_per_audio"]
    ):
        raise ValueError("full pronunciation index size differs")
    model = (root / old["configuration"]["model"]).resolve()
    for name in ("config.json", "model.safetensors.index.json"):
        path = model / name
        if name == "config.json" or path.exists():
            _checked(path, inputs, old["inputs"][str(path)]["sha256"])
    package = Path(__file__).resolve().parents[1]
    for relative, sha in old["code_sha256"].items():
        if _sha(package / relative) != sha:
            raise ValueError(f"retrieval implementation changed: {relative}")
    rows, gates = [], {}
    if set(source_cfg["datasets"]) != set(KEYS):
        raise ValueError("three English groups required")
    for group, expected_count in source_cfg["datasets"].items():
        previous = old["groups"][group]
        original = previous["config"]["sources"][0]
        text = _checked(
            Path(original["transcripts"]["path"]), inputs, original["transcripts"]["sha256"]
        )
        audio = (root / source_cfg["audio_root"] / group / "en/wav").resolve()
        records, _ = external.build_external_dataset(
            (external.SourceSpec(group, audio, text),), language="en"
        )
        if len(records) != expected_count or [
            {k: v for k, v in r.to_dict().items() if k != "audio_path"} for r in records
        ] != [{k: v for k, v in r.items() if k != "audio_path"} for r in previous["records"]]:
            raise ValueError(f"sample/reference coverage differs: {group}")
        gates[group] = _validated_source_gate(previous["config"])
        current: dict[str, Any] = external._run_config(
            model=model,
            checkpoint=head,
            vocab_path=vocab_path,
            keyword_path=table,
            sources=(external.SourceSpec(group, audio, text),),
            keyword_set="all_keywords",
            device="cuda:0",
            dtype="bfloat16",
            language="en",
            saved_raw_rank_depth=64,
        )
        current["sources"] = [{**s, "audio_dir": original["audio_dir"]} for s in current["sources"]]
        ignored = {"git_commit", "ctc_checkpoint"}
        if {k: v for k, v in current.items() if k not in ignored} != {
            k: v for k, v in previous["config"].items() if k not in ignored
        }:
            raise ValueError(f"retrieval settings differ: {group}")
        with stage_progress(f"Read audio headers {group}", total=len(records), every=100) as tick:
            for n, (record, clean) in enumerate(zip(records, previous["records"], strict=True), 1):
                info = sf.info(str(record.audio_path))
                ends = boundaries(info.frames, info.samplerate)
                bucket = (
                    "under_2s"
                    if info.frames < 2 * info.samplerate
                    else "exact_2s_multiple"
                    if info.frames % (2 * info.samplerate) == 0
                    else "tail"
                )
                rows.append(
                    {
                        "group": group,
                        "key": KEYS[group],
                        "sample_id": record.sample_id,
                        "audio_path": str(record.audio_path),
                        "frames": info.frames,
                        "sample_rate": info.samplerate,
                        "channels": info.channels,
                        "duration_seconds": info.frames / info.samplerate,
                        "step_count": len(ends),
                        "duration_bucket": bucket,
                        "clean_sha256": old["inputs"][clean["audio_path"]]["sha256"],
                    }
                )
                tick(n)
    chosen = select_samples(rows, limit, cfg["selection_seed"])
    with stage_progress("Hash selected audio only", total=len(chosen), every=20) as tick:
        for n, row in enumerate(chosen, 1):
            path = _checked(Path(row["audio_path"]), inputs)
            row["audio_sha256"] = inputs[str(path)]["sha256"]
            tick(n)
    for group in KEYS:
        selected = [r for r in chosen if r["group"] == group]
        if selected and all(r["audio_sha256"] == r["clean_sha256"] for r in selected):
            raise ValueError(f"all selected noisy audio equals clean audio: {group}")
    code = {**old["code_sha256"]}
    for relative in (
        "evaluation/stream_context_assets.py",
        "inference/stream_context_contract.py",
        "inference/stream_context_run.py",
        "inference/english_best_noise.py",
        "inference/english_head_preview.py",
        "modeling/ctc_head.py",
        "modeling/audio_encoder.py",
        "modeling/qwen_backbone.py",
        "training/feature_cache.py",
        "training/stage_progress.py",
    ):
        code[relative] = _sha(package / relative)
    for relative, sha in code.items():
        _checked(package / relative, inputs, sha)
    for relative in (
        "scripts/run_stream_contexts.py",
        "tests/fixtures/stream_interface_v1_validator.py.txt",
    ):
        _checked(package.parents[1] / relative, inputs)
    plan = {
        "schema_version": 1,
        "git_commit": external._git_commit(),
        "configuration": cfg,
        "inputs": inputs,
        "checkpoint": checkpoint,
        "source_configuration": source_cfg,
        "samples": chosen,
        "gate": gates,
        "code_sha256": code,
        "inventory_samples": len(rows),
        "selected_samples": len(chosen),
        "inventory_duration_buckets": dict(Counter(r["duration_bucket"] for r in rows)),
        "selected_duration_buckets": dict(Counter(r["duration_bucket"] for r in chosen)),
        "prefix_count": sum(r["step_count"] for r in chosen),
        "resample_policy": "slice/read source frames first, then resample current prefix only",
        "retention_policy": cfg["retention_policy"],
        "snr_verified": False,
    }
    return plan, {
        "model": model,
        "checkpoint": head,
        "vocab": vocab,
        "bundle": bundle,
        "variants": variants,
        "by_variant": by_variant,
        "table": table,
    }
