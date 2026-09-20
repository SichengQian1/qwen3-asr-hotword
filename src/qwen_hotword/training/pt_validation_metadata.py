"""Metadata evidence only: no quality certification, selection, or model inference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.training.g2p_prep import digit_fragments, extract_word_tokens


def inspect_mfa_environment(repo_root: Path, *, mfa_root: Path | None = None) -> dict[str, Any]:
    """Inspect package metadata and Portuguese assets without importing/running MFA."""
    versions: dict[str, str | None] = {}
    for package in ("montreal-forced-aligner", "kalpy", "pynini"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = mfa_root or Path(os.environ.get("MFA_ROOT_DIR") or Path.home() / "Documents/MFA")
    roots = [repo_root / "models/mfa", root.expanduser() / "pretrained_models"]
    directories: list[dict[str, object]] = []
    assets: list[dict[str, object]] = []
    seen: set[Path] = set()
    found = 0
    for base in roots:
        for kind in ("acoustic", "dictionary", "g2p"):
            directory = base / kind
            directories.append({"path": str(directory), "exists": directory.is_dir()})
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if not path.name.casefold().startswith("portug") or path.resolve() in seen:
                    continue
                seen.add(path.resolve())
                found += 1
                if len(assets) < 20:
                    assets.append({
                        "kind_from_directory": kind, "path": str(path),
                        "is_file": path.is_file(),
                        "sha256": _sha256(path) if path.is_file() else None,
                    })
    return {
        "scope": "mfa_package_and_asset_inventory_only",
        "python": sys.executable, "versions": versions,
        "executables": {name: shutil.which(name) for name in ("mfa", "sox", "ffmpeg")},
        "directories_searched": directories,
        "portuguese_assets": assets, "asset_count": found, "assets_truncated": found > 20,
        "limitations": [
            "Only repository models/mfa and MFA_ROOT_DIR/default pretrained_models searched.",
            "Presence is not runtime readiness, phoneset compatibility, or model validation.",
        ],
        "mfa_imported_or_run": False, "model_loaded": False, "files_written": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _preview(text: str) -> dict[str, object]:
    return {"text": text[:240], "characters": len(text), "truncated": len(text) > 240}


def _difference_kind(current: str, original: str) -> str:
    if _normalized(current.replace('"', "")) == _normalized(original.replace('"', "")):
        return "ascii_double_quotes_case_space_only"
    if (extract_word_tokens(current) == extract_word_tokens(original)
            and digit_fragments(current) == digit_fragments(original)):
        return "same_g2p_words_and_digit_fragments_other_text_difference"
    return "g2p_words_or_digit_fragments_differ"


def _text_state(current: str, original: str) -> str:
    if not original.strip():
        return "empty_original_text"
    if current == original:
        return "exact_text_match"
    if _normalized(current) == _normalized(original):
        return "case_space_unicode_only"
    return "text_difference"


def _audit_fleurs(candidates: list[dict[str, str]], path: Path) -> dict[str, Any]:
    meta: dict[str, list[str]] = defaultdict(list)
    wanted = {r["audio_path"] for r in candidates}
    if len(wanted) != len(candidates) or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate FLEURS candidate ID/audio path")
    with path.open(encoding="utf-8-sig") as handle:
        header = next(handle, "").rstrip("\r\n").split("\t")
        if header != ["path", "transcription"]:
            raise ValueError("FLEURS TSV: expected path/transcription header")
        for number, line in enumerate(handle, 2):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"FLEURS TSV line {number}: expected 2 columns; no rows skipped")
            relative = Path(fields[0])
            if not fields[0] or relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(f"FLEURS TSV line {number}: expected audio filename")
            key = str((path.parent / "train" / relative).resolve())
            if key in wanted:
                meta[key].append(fields[1])
    counts: Counter[str] = Counter()
    differences: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for row in candidates:
        matches = meta[row["audio_path"]]
        if len(matches) != 1:
            counts["missing_metadata" if not matches else "duplicate_metadata"] += 1
            continue
        original = matches[0]
        state = _text_state(row["text"], original)
        counts[state] += 1
        if state in {"text_difference", "empty_original_text"}:
            kind = _difference_kind(row["text"], original)
            differences[kind] += 1
            if len(examples) < 8:
                examples.append({
                    "id": row["id"], "kind": kind, "current": _preview(row["text"]),
                    "original": _preview(original),
                })
    return {
        "candidates": len(candidates), "checks": dict(counts),
        "difference_kinds": dict(differences), "difference_examples": examples,
        "join_policy": "resolved_path_under_tsv_parent_train_no_basename_fallback",
    }


def audit_pt_validation_metadata(
    *,
    validation_manifest: Path,
    expected_sha256: str,
    common_voice_root: Path,
    fleurs_train_tsv: Path,
) -> dict[str, Any]:
    """Join CV and FLEURS candidates to their original metadata by resolved path.

    CV is read as literal, single-line TSV. Quotes remain part of the text, not
    CSV delimiters. Unexpected column counts fail rather than silently dropping
    rows. Differences may still be serialization differences, not label errors.
    """
    manifest_sha = _sha256(validation_manifest)
    if manifest_sha != expected_sha256:
        raise ValueError(f"Validation SHA256 mismatch: {manifest_sha}")

    candidates: list[dict[str, str]] = []
    fleurs_candidates: list[dict[str, str]] = []
    total = 0
    with validation_manifest.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("split") != "validation":
                raise ValueError(f"Manifest line {number}: expected validation record")
            total += 1
            if row.get("source_corpus") not in {"common_voice", "fleurs"}:
                continue
            if any(not isinstance(row.get(k), str) for k in ("id", "audio_path", "text")):
                raise ValueError(f"Manifest line {number}: invalid ID/audio/text")
            if not row["id"] or not row["audio_path"]:
                raise ValueError(f"Manifest line {number}: empty ID/audio")
            destination = (
                candidates if row["source_corpus"] == "common_voice" else fleurs_candidates
            )
            destination.append({
                "id": row["id"],
                "audio_path": str(Path(row["audio_path"]).resolve()),
                "text": row["text"],
            })
    if not candidates:
        raise ValueError("No Common Voice validation candidates")
    wanted = {r["audio_path"] for r in candidates}
    if len(wanted) != len(candidates) or len({r["id"] for r in candidates}) != len(candidates):
        raise ValueError("Duplicate Common Voice candidate ID/audio path")

    metadata_path = common_voice_root / "validated.tsv"
    metadata_sha = _sha256(metadata_path)
    meta: dict[str, list[dict[str, str]]] = defaultdict(list)
    metadata_rows = 0
    clips = (common_voice_root / "clips").resolve()
    with metadata_path.open(encoding="utf-8-sig") as handle:
        header = next(handle, "").rstrip("\r\n").split("\t")
        required = {"path", "sentence", "client_id", "up_votes", "down_votes"}
        if len(header) != len(set(header)) or not required <= set(header):
            raise ValueError("Common Voice TSV: missing required or duplicate columns")
        for number, line in enumerate(handle, 2):
            values = line.rstrip("\r\n").split("\t")
            if len(values) != len(header):
                raise ValueError(
                    f"Common Voice TSV line {number}: {len(values)} columns; "
                    f"expected {len(header)}. No rows skipped; inspect TSV format."
                )
            entry = dict(zip(header, values, strict=True))
            relative = Path(entry["path"])
            if not entry["path"] or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Common Voice TSV line {number}: invalid clip path")
            key = str((clips / relative).resolve())
            metadata_rows += 1
            if key in wanted:
                meta[key].append(entry)

    counts: Counter[str] = Counter()
    votes: Counter[str] = Counter()
    accents: Counter[str] = Counter()
    locales: Counter[str] = Counter()
    differences: Counter[str] = Counter()
    text_vote_groups: Counter[str] = Counter()
    speakers: set[str] = set()
    examples: list[dict[str, object]] = []
    for candidate in candidates:
        matches = meta[candidate["audio_path"]]
        if len(matches) != 1:
            counts["not_in_validated" if not matches else "duplicate_metadata"] += 1
            continue
        entry = matches[0]
        current, original = candidate["text"], entry["sentence"]
        state = _text_state(current, original)
        counts[state] += 1
        vote_group = "unknown_votes"
        if entry["up_votes"].isdigit() and entry["down_votes"].isdigit():
            vote_group = "zero_down_votes" if int(entry["down_votes"]) == 0 else "has_down_votes"
        text_vote_groups[f"{state}::{vote_group}"] += 1
        votes[f"up={entry['up_votes']},down={entry['down_votes']}"] += 1
        accents[entry.get("accents") or entry.get("accent") or "unknown"] += 1
        locales[entry.get("locale") or "unknown"] += 1
        if entry["client_id"]:
            speakers.add(entry["client_id"])
        if state in {"text_difference", "empty_original_text"}:
            kind = _difference_kind(current, original)
            differences[kind] += 1
            if len(examples) < 8:
                examples.append({
                    "id": candidate["id"], "kind": kind, "current": _preview(current),
                    "original": _preview(original),
                })

    fleurs = _audit_fleurs(fleurs_candidates, fleurs_train_tsv)
    return {
        "schema_version": 2,
        "status": "completed",
        "scope": "metadata_only_not_label_accuracy_certification",
        "inputs": {
            "validation": {"path": str(validation_manifest), "sha256": manifest_sha},
            "cv_metadata": {"path": str(metadata_path), "sha256": metadata_sha},
            "fleurs_train_tsv": {
                "path": str(fleurs_train_tsv), "sha256": _sha256(fleurs_train_tsv),
            },
        },
        "parser": "literal_single_line_tsv_strict_column_count_quotes_preserved",
        "validation_records": total,
        "cv_metadata_records": metadata_rows,
        "cv_candidates": len(candidates),
        "cv_checks": dict(sorted(counts.items())),
        "cv_votes_top20": votes.most_common(20),
        "cv_vote_bin_count": len(votes),
        "cv_accents_top10": accents.most_common(10),
        "cv_locales": dict(sorted(locales.items())),
        "cv_known_speakers": len(speakers),
        "cv_difference_kinds": dict(differences),
        "cv_text_vote_groups": dict(text_vote_groups),
        "text_difference_examples": examples,
        "fleurs": fleurs,
        "limitations": [
            "Text differences are flags, not confirmed annotation errors.",
            "No audio, phoneme labels, or pronunciation accuracy was evaluated.",
            "Matching G2P input words does not certify spoken pronunciation or saved labels.",
            "TSV quotes are preserved; serialization may explain text differences.",
        ],
        "model_loaded": False,
        "test_set_used": False,
        "files_written": False,
    }
