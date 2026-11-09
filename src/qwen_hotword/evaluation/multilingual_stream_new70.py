"""Bind the v0.3 retrieval assets to the returned same-step audio inventory."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.stream_retrieval_metrics import reference_matcher
from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.english_head_preview import _index
from qwen_hotword.inference.external_keyword_retrieval import _git_commit
from qwen_hotword.inference.stream_context_contract import boundaries
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_review import rows
from qwen_hotword.training.spanish_capacity import _verified
from qwen_hotword.training.stage_progress import stage_progress


def prepare_new70_stream(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = _json(config)
    inputs: dict[str, Any] = {}
    _checked(config, inputs)
    source = (root / cfg["source_stream"]).resolve()
    source_config = _checked(source / "run_config.json", inputs, cfg["source_run_config_sha256"])
    old = _json(source_config)
    report = _json(_verified(source, "report.json", inputs))
    samples = old["samples"]
    if (
        old["effect"] != "same-step"
        or report["status"] != "completed"
        or report["effect"] != "same-step"
        or report["run_config_sha256"] != cfg["source_run_config_sha256"]
        or dict(Counter(s["language"] for s in samples)) != cfg["dataset_counts"]
        or len({(s["key"], s["sample_id"]) for s in samples}) != len(samples)
        or sum(s["step_count"] for s in samples) != cfg["prefix_count"]
        or report["sample_count"] != len(samples)
        or report["prefix_count"] != cfg["prefix_count"]
    ):
        raise ValueError("completed same-step source inventory differs")

    # Resolve canonical surfaces and targets against the original sealed 4k plan.
    baseline = (root / old["source_configuration"]["baseline_run"]).resolve()
    baseline_path = baseline / "run_config.json"
    original = _json(_checked(baseline_path, inputs, old["inputs"][str(baseline_path)]["sha256"]))
    old_vocab_path = (root / original["configuration"]["vocab"]).resolve()
    old_vocab = load_phoneme_vocab(
        _checked(old_vocab_path, inputs, old["source_configuration"]["vocab_sha256"])
    )
    original_index = baseline / "indexed_pronunciations.jsonl"
    bundle, _, _ = _index(
        _checked(original_index, inputs, old["inputs"][str(original_index)]["sha256"]), old_vocab
    )
    canonical = {entry.normalized: entry for entry in bundle.entries}
    if len(canonical) != cfg["declared_surfaces"]:
        raise ValueError("original declared surface count differs")

    run = (root / cfg["new70_source"]).resolve()
    checkpoint = _checked(run / "head/ctc_head_best.pt", inputs, cfg["checkpoint_sha256"])
    marker = _json(_checked(run / "training_completed.json", inputs))
    if (
        marker["status"] != "fresh_head_training_completed"
        or marker["checkpoint_sha256"] != cfg["checkpoint_sha256"]
        or marker["num_classes"] != 100
        or marker["head_initialization"] != "random"
    ):
        raise ValueError("new70 training identity differs")
    head_report = _json(_checked(run / "head/report.json", inputs, marker["report_sha256"]))
    if head_report["best_epoch"] != marker["best_epoch"]:
        raise ValueError("new70 checkpoint epoch differs")
    vocab = load_phoneme_vocab(_checked(root / cfg["vocab"], inputs, cfg["vocab_sha256"]))
    if len(vocab.tokens) != 100 or vocab.tokenization_policy != "mfa_units_v0.3":
        raise ValueError("requires the v0.3 100-class vocabulary")
    index = _checked(run / "indexed_pronunciations.v03.jsonl", inputs, cfg["index_sha256"])
    variants = tuple(
        HotwordEntry(
            **{**row, **{k: tuple(row[k]) for k in ("words", "phoneme_tokens", "token_ids")}}
        )
        for row in rows(index)
    )
    if (
        len(variants) != cfg["pronunciations"]
        or len({e.hotword_id for e in variants}) != len(variants)
        or len({e.normalized for e in variants}) != cfg["active_surfaces"]
        or any(
            e.language not in {"en", "es", "pt-BR"}
            or e.normalized not in canonical
            or not e.token_ids
            or any(type(i) is not int or not 2 <= i < len(vocab.tokens) for i in e.token_ids)
            or tuple(vocab.tokens[i] for i in e.token_ids) != e.phoneme_tokens
            for e in variants
        )
    ):
        raise ValueError("new70 pronunciation index differs or is incompatible with v0.3")
    by_variant = {e.hotword_id: canonical[e.normalized] for e in variants}

    # References and targets are read only for post-retrieval scoring.
    grouped = defaultdict(set)
    keys = defaultdict(set)
    with stage_progress(
        "Verify saved noisy audio and prefix boundaries", total=len(samples), every=100
    ) as tick:
        for number, sample in enumerate(samples, 1):
            group = sample["group"]
            if (
                sample["language"] != original["groups"][group]["language"]
                or sample["step_count"] != len(boundaries(sample["frames"], sample["sample_rate"]))
                or sample["duration_seconds"] != sample["frames"] / sample["sample_rate"]
            ):
                raise ValueError(
                    f"sample language or boundary differs: {group}/{sample['sample_id']}"
                )
            _checked(Path(sample["audio_path"]), inputs, sample["audio_sha256"])
            grouped[group].add(sample["sample_id"])
            keys[group].add(sample["key"])
            tick(number)
    if (
        set(grouped) != set(original["groups"])
        or set(grouped) != set(old["gate"])
        or any(len(v) != 1 for v in keys.values())
        or len({key for v in keys.values() for key in v}) != len(keys)
    ):
        raise ValueError("source dataset routing or gates differ")
    match = reference_matcher(bundle.entries)
    canonical_ids = {e.hotword_id for e in bundle.entries}
    evaluation = {}
    for group, data in original["groups"].items():
        refs = {r["sample_id"]: r["reference_text"] for r in data["records"]}
        targets = set(data["target_ids"])
        if set(refs) != grouped[group] or not targets <= canonical_ids:
            raise ValueError(f"reference or target coverage differs: {group}")
        evaluation[group] = {
            "language": data["language"],
            "target_ids": sorted(targets),
            "references": refs,
            "expected_target_sample_pairs": sum(
                len(match(text) & targets) for text in refs.values()
            ),
        }

    model = (root / original["configuration"]["model"]).resolve()
    if model.name != "Qwen3-ASR-1.7B":
        raise ValueError("requires the original Qwen3-ASR-1.7B snapshot")
    for name in ("config.json", "model.safetensors.index.json"):
        path = model / name
        _checked(path, inputs, old["inputs"][str(path)]["sha256"])
    _checked(model / "preprocessor_config.json", inputs)
    package = Path(__file__).resolve().parents[1]
    code = {}
    for name in sorted(
        set(old["code_sha256"])
        | {
            "evaluation/multilingual_stream_new70.py",
            "inference/multilingual_stream_run.py",
            "phonemes/coverage.py",
            "phonemes/mfa_v03_review.py",
        }
    ):
        path = _checked(package / name, inputs)
        code[name] = inputs[str(path)]["sha256"]
    for name in (
        "scripts/run_multilingual_stream.py",
        "tests/fixtures/stream_interface_v1_validator.py.txt",
    ):
        _checked(package.parents[1] / name, inputs)

    plan = {
        **old,
        "git_commit": _git_commit(),
        "configuration": cfg,
        "inputs": inputs,
        "code_sha256": code,
        "checkpoint": {
            "sha256": cfg["checkpoint_sha256"],
            "epoch_metrics": {"epoch": marker["best_epoch"]},
        },
        "source_configuration": {
            "baseline_run": str(baseline),
            "table_sha256": old["source_configuration"]["table_sha256"],
            "vocab_sha256": cfg["vocab_sha256"],
            "index_sha256": cfg["index_sha256"],
        },
        "delivery_asset_identity": {
            "asset_version": "v0.3_new70",
            "index_sha256": cfg["index_sha256"],
            "source_run_config_sha256": cfg["source_run_config_sha256"],
            "declared_surfaces": cfg["declared_surfaces"],
            "active_surfaces": cfg["active_surfaces"],
            "pronunciations": cfg["pronunciations"],
        },
    }
    runtime = {
        "model": model,
        "checkpoint": checkpoint,
        "vocab": vocab,
        "bundle": bundle,
        "variants": variants,
        "by_variant": by_variant,
        "evaluation_groups": evaluation,
    }
    return plan, runtime
