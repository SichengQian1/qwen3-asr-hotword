"""Immutable inputs and independent outputs for the fresh v0.3/70% experiment."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.g2p_prep import extract_word_tokens
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.noise_prepare import language
from qwen_hotword.training.sharded_ctc import DiskFeatureCache, load_disk_feature_cache
from qwen_hotword.training.stage_progress import stage_progress


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"object required: {path}")
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def words(text: str) -> tuple[list[str], list[str]]:
    """Never silently verbalize/drop numbers, symbols or non-whitespace controls."""
    issues = []
    if any(c.isdigit() for c in text):
        issues.append("number_requires_verbalization")
    if any(unicodedata.category(c).startswith("S") or c in "&@#%*" for c in text):
        issues.append("symbol_requires_verbalization")
    if any(unicodedata.category(c).startswith("C") and not c.isspace() for c in text):
        issues.append("unsupported_control_character")
    result = extract_word_tokens(text)
    if not result:
        issues.append("no_words")
    return result, issues


def select_noise(items: list[dict[str, Any]], old_ids: set[str], seed: int) -> set[str]:
    """Exactly nearest 70% of eligible RECORDS per language, not 70% extra copies.

    Retain usable old random noise selections first to reuse the SAME mixtures.
    The remainder is deterministic and independent of model predictions.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for row in items:
        groups[language(row)].append(row["id"])
    selected = set()
    for group in groups.values():
        if len(set(group)) != len(group):
            raise ValueError("duplicate IDs in noise selection")
        group.sort(key=lambda i: (i not in old_ids, rank(seed, i)))
        selected.update(group[: math.floor(0.7 * len(group) + 0.5)])
    return selected


def contract(root: Path, cfg: dict[str, Any], output: Path) -> None:
    if (
        cfg["noise_percent"] != 70
        or cfg["snr_db"] != 10
        or cfg["head_initialization"] != "random"
        or cfg["encoder_frozen"] is not True
        or cfg["human_review_required"] is not False
    ):
        raise ValueError("this delivery is fresh 100-class Head / frozen Encoder / 70% at 10dB")
    protected = [
        root / cfg["source_run"],
        root / cfg["noise_source"],
        root / cfg["foundation_output"],
        root / cfg["assets_dir"],
        Path(cfg["noise_root"]),
        Path(cfg["model"]),
    ]
    inherited_plan = root / cfg["noise_source"] / "plan.json"
    if inherited_plan.is_file():
        source_info = read(inherited_plan)
        protected.extend(Path(source_info[f"{split}_cache"]) for split in ("train", "validation"))
    for p in protected:
        p = p.resolve()
        if output == p or output.is_relative_to(p) or p.is_relative_to(output):
            raise ValueError("new output overlaps an immutable source")
    files = [
        root / cfg["config_path"],
        root / cfg["foundation_config"],
        root / cfg["vocab"],
        root / "scripts/run_mfa_v03_training.py",
        root / "scripts/prepare_mfa_v03.py",
        *sorted((root / "src/qwen_hotword/training").glob("mfa_round_*.py")),
        root / "src/qwen_hotword/evaluation/mfa_round.py",
        *sorted((root / "src/qwen_hotword/phonemes").glob("mfa_v03_*.py")),
    ]
    for relative in (
        "phonemes/coverage.py",
        "training/sharded_ctc.py",
        "training/ctc_overfit.py",
        "training/feature_cache.py",
        "training/noise_audio.py",
        "training/noise_cache.py",
        "modeling/audio_encoder.py",
        "modeling/ctc_head.py",
        "modeling/qwen_backbone.py",
        "inference/external_keyword_retrieval.py",
        "inference/reviewed_joint.py",
        "inference/external_keyword_topk_replay.py",
        "inference/hotword_prompt.py",
        "hotwords/anchor_index.py",
        "hotwords/anchor_rerank.py",
        "evaluation/stream_retrieval_metrics.py",
        "evaluation/recovered_case_run.py",
        "training/noise_evaluation.py",
        "inference/recovered_case_runtime.py",
        "inference/prompt_order_runtime.py",
    ):
        files.append(root / "src/qwen_hotword" / relative)
    value = {"configuration": cfg, "source_sha256": {str(p): sha(p) for p in files}}
    path = output / "contract.json"
    if path.exists():
        if read(path) != value:
            raise ValueError("code/config changed; preserve this experiment and use a NEW output")
        return
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("refusing an unowned nonempty output directory")
    write_json(path, value)
    write_json(
        output / "identity.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "contract_sha256": sha(path),
            "head_initialization": "random",
            "encoder_frozen": True,
            "automatic_labels_not_human_verified": True,
            "human_gate_omitted_by_user": True,
        },
    )


def sources(
    root: Path, cfg: dict[str, Any], *, verify_features: bool = False
) -> tuple[dict[str, Any], dict[str, DiskFeatureCache]]:
    source, previous = root / cfg["source_run"], root / cfg["noise_source"]
    pin(source / "plan.json", cfg["source_plan_sha256"])
    pin(source / "head/report.json", cfg["source_report_sha256"])
    old = read(previous / "plan.json")
    pin(previous / "plan.json", read(previous / "prepared.json")["plan_sha256"])
    if (
        old["configuration"]["source_plan_sha256"] != cfg["source_plan_sha256"]
        or old["configuration"]["seed"] != cfg["seed"]
        or Path(old["configuration"]["noise_root"]).resolve() != Path(cfg["noise_root"]).resolve()
        or old.get("test_set_used") is not False
    ):
        raise ValueError("old mixture provenance differs")
    pin(Path(old["vocab"]), cfg["old_vocab_sha256"])
    pin(root / cfg["vocab"], cfg["vocab_sha256"])
    pin(previous / "noise_inventory.json", old["noise_inventory_sha256"])
    completion = read(previous / "cache_completed.json")
    if completion.get("status") != "noise_cache_completed" or completion.get("parts") != len(
        old["parts"]
    ):
        raise ValueError("expected the completed historical noise feature pool")
    baseline = read(source / "plan.json")
    if (
        Path(old["model"]["path"]).resolve() != Path(cfg["model"]).resolve()
        or old["head_config"] != read(source / "head/report.json")["head_config"]
    ):
        raise ValueError("baseline Encoder path or Head structure differs")
    if Path(old["train_manifest"]).resolve() != Path(baseline["train_manifest"]).resolve():
        raise ValueError("actual baseline training manifest differs")
    if Path(old["train_cache"]).resolve() != Path(baseline["train_cache"]).resolve():
        raise ValueError("actual baseline training cache differs")
    caches = {}
    for split in ("train", "validation"):
        path = Path(old[f"{split}_manifest"])
        expected = old["input_sha256"].get(str(path))
        if expected is None:
            raise ValueError("old plan does not pin its actual manifest")
        pin(path, expected)
        for name in ("cache_config.json", "cache_index.json", "cache_summary.json"):
            metadata_path = Path(old[f"{split}_cache"]) / name
            expected_metadata = old["input_sha256"].get(str(metadata_path))
            if expected_metadata is None:
                raise ValueError("old plan does not pin source cache metadata")
            pin(metadata_path, expected_metadata)
        with stage_progress(f"Verify original {split} cache metadata / tensors={verify_features}"):
            caches[split] = load_disk_feature_cache(
                old[f"{split}_cache"],
                expected_split=split,
                source_manifest_path=path,
                vocab_path=old["vocab"],
                verify_sha256=verify_features,
            )
        metadata = read(caches[split].root / "cache_config.json")
        if (
            caches[split].encoder_frozen_parameters != 317477504
            or metadata["tap_module"] != "thinker.audio_tower.ln_post"
        ):
            raise ValueError("unexpected inherited Encoder/tap boundary")
        for name, identity in (
            ("config.json", "config_sha256"),
            ("model.safetensors.index.json", "weight_index_sha256"),
        ):
            pin(Path(old["model"]["path"]) / name, metadata["model"][identity])
        if caches[split].ctc_time_upsampling_factor != 2:
            raise ValueError("expected the existing 2x temporal Head boundary")
    a, b = (read(caches[s].root / "cache_config.json") for s in ("train", "validation"))
    if any(a[k] != b[k] for k in ("model", "tap_module", "hidden_size", "feature_dtype", "vocab")):
        raise ValueError("clean Encoder cache identities differ")
    return old, caches


def old_noise_part(
    root: Path, cfg: dict[str, Any], old: dict[str, Any], part: dict[str, Any]
) -> DiskFeatureCache:
    path = root / cfg["noise_source"] / part["manifest"]
    pin(path, part["sha256"])
    return load_disk_feature_cache(
        path.parent / "features",
        expected_split=part["split"],
        source_manifest_path=path,
        vocab_path=old["vocab"],
        verify_sha256=False,
    )
