"""Freeze the SD-only targets plus deterministic legacy Portuguese fillers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.wave_keywords import (
    _check_language,
    _entry,
    _json,
    _old_entry_issues,
)
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.training.spanish_capacity import _sha


def seal(output: Path) -> None:
    lines = "".join(
        f"{_sha(p)}  {p.relative_to(output).as_posix()}\n"
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "sha256.txt"
    )
    path = output / "sha256.txt"
    if path.exists():
        if path.read_text() != lines:
            raise ValueError("completed output checksum manifest changed")
    else:
        external._atomic_write_text(path, lines)


def build_sd_table(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    cfg = _json(config_path)
    identities: dict[str, Any] = {}

    def checked(path: Path, expected: str | None = None) -> Path:
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        identities[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
        return path

    checked(config_path)
    vocab_path = checked(root / cfg["vocab"], cfg["vocab_sha256"])
    vocab = load_phoneme_vocab(vocab_path)
    primary_path = checked(root / cfg["primary"])
    # The existing loader strips the known 'Dra. ' surface before phoneme lookup.
    primary = external.load_keyword_bias(primary_path, vocab=vocab, keyword_set="all_keywords")
    if len(primary.entries) != cfg["expected_primary"]:
        raise ValueError("SD target count changed")
    targets = {e.normalized: e for e in primary.entries}
    old_path = checked(root / cfg["fillers"], cfg["filler_sha256"])
    pool: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    decisions: Counter[str] = Counter()
    for line in old_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        _check_language(row.get("language"), "pt", "old table")
        issues = _old_entry_issues(row, vocab)
        if issues:
            raise ValueError(f"invalid old entry {row.get('surface')}: {issues}")
        entry = _entry(row["surface"], row["pronunciation"], "pt", vocab)
        key = entry["normalized"]
        if key in targets:
            decisions["target_precedence"] += 1
            continue
        if key in pool and pool[key]["token_ids"] != entry["token_ids"]:
            conflicts.add(key)
        pool.setdefault(key, entry)
    for key in conflicts:
        del pool[key]
    needed = cfg["target_size"] - len(targets)
    if needed < 0 or len(pool) < needed:
        raise ValueError(f"insufficient fillers: need {needed}, eligible {len(pool)}")
    ranked = sorted(
        pool, key=lambda k: (hashlib.sha256(f"{cfg['seed']}:{k}".encode()).hexdigest(), k)
    )
    selected = [pool[k] for k in ranked[:needed]]
    surfaces = [e.surface for e in primary.entries] + [e["surface"] for e in selected]
    phones = dict(primary.phonemes)
    phones.update({e["surface"]: e["pronunciation"] for e in selected})
    table = {
        "language": "Portuguese",
        "keyword_sets": {"all_keywords": surfaces},
        "keyword_phonemes": phones,
    }
    for path, identity in identities.items():
        if _sha(Path(path)) != identity["sha256"]:
            raise ValueError(f"input changed during build: {path}")
    output.mkdir(parents=True, exist_ok=False)
    external._write_json(output / "keyword_bias_phoneme.json", table)
    bundle = external.load_keyword_bias(
        output / "keyword_bias_phoneme.json", vocab=vocab, keyword_set="all_keywords"
    )
    if len(bundle.entries) != cfg["target_size"]:
        raise ValueError("final table count mismatch")
    external._write_json(output / "targets.json", sorted(targets))
    external._write_jsonl(
        output / "selection.jsonl",
        [
            {"surface": e.surface, "normalized": e.normalized, "source": "sd_target"}
            for e in primary.entries
        ]
        + [
            {"surface": e["surface"], "normalized": e["normalized"], "source": "old_table"}
            for e in selected
        ],
    )
    report = {
        "status": "completed",
        "tables_created": True,
        "output_dir": str(output),
        "total": len(bundle.entries),
        "mandatory": len(targets),
        "old_table_fillers": needed,
        "eligible_fillers": len(pool),
        "excluded_conflicting_surfaces": len(conflicts),
        "decisions": dict(decisions),
        "inputs": identities,
        "seed": cfg["seed"],
        "policy": "SD all_keywords only; old_table fill; no generated-neighbor source",
        "all_targets_retained": True,
        "oov_count": 0,
        "audio_or_transcripts_used": False,
        "model_loaded": False,
        "limitations": ["Legacy fillers may naturally be phonetically similar to targets."],
    }
    external._write_json(output / "report.json", report)
    seal(output)
    return report
