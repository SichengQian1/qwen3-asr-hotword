"""Verify approved pronunciation assets and inspect bounded corpus schemas on CPU."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.en_ar_source_audit import read_records
from qwen_hotword.training.stage_progress import stage_progress

TEXT_SUFFIXES = {".txt", ".json", ".jsonl", ".tsv", ".csv", ".yaml", ".yml"}
HELDOUT = re.compile(r"(^|[/_.-])(test|dev|validation|valid|eval)([/_.-]|$)", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    vocab_path = root / config["vocab"]
    if sha256(vocab_path) != config["vocab_sha256"]:
        raise ValueError("approved vocabulary SHA256 mismatch")
    vocab = load_phoneme_vocab(vocab_path)
    if len(vocab.tokens) != config["num_classes"]:
        raise ValueError("vocabulary class count mismatch")
    details = {}
    with stage_progress("Verify fixed Chinese / US / UK MFA assets", total=3, every=1) as tick:
        for number, (lang, model) in enumerate(config["models"].items(), 1):
            folder = root / model["directory"]
            for filename, expected in model["files_sha256"].items():
                if sha256(folder / filename) != expected:
                    raise ValueError(f"{lang}/{filename}: asset SHA256 mismatch")
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            if meta["version"] != model["g2p_version"]:
                raise ValueError(f"{lang}: G2P version mismatch")
            count = 0
            for row in (folder / "phones.txt").read_text(encoding="utf-8").splitlines():
                unit, _ = row.rsplit(None, 1)
                if unit == "<eps>":
                    continue
                result = tokenize_ipa_to_vocab(unit, vocab)
                if result.oov_units or len(result.token_ids) != 1:
                    raise ValueError(f"{lang}: unmapped MFA unit {unit!r}")
                count += 1
            details[lang] = {
                "model": model["name"],
                "g2p_version": meta["version"],
                "verified_files": len(model["files_sha256"]),
                "units": count,
            }
            tick(number)
    return {"status": "verified", "num_classes": len(vocab.tokens), "models": details}


def preview(value: Any, depth: int = 0) -> Any:
    """Keep schema and readable examples without returning full manifests."""
    if isinstance(value, str):
        return value if len(value) <= 800 else value[:800] + "...[truncated]"
    if depth >= 4:
        return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        items = list(value.items())[:16]
        return {str(k): preview(v, depth + 1) for k, v in items}
    if isinstance(value, list):
        return [preview(v, depth + 1) for v in value[:4]]
    return value


def inspect_json(path: Path, rows: int = 8) -> dict[str, Any]:
    fmt, iterator = read_records(path)
    try:
        examples = list(islice(iterator, rows))
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    paths = []
    for row in examples:
        raw = row.get("audios", row.get("audio_path", row.get("audio")))
        for text in (raw if isinstance(raw, list) else [raw])[:2]:
            if isinstance(text, str):
                audio = Path(text)
                resolved = audio if audio.is_absolute() else path.parent / audio
                paths.append(
                    {"original": text, "candidate": str(resolved), "exists": resolved.is_file()}
                )
    return {
        "format": fmt,
        "sampled_rows": len(examples),
        "total_rows": None,
        "fields": dict(Counter(key for row in examples for key in row)),
        "examples": [preview(row) for row in examples],
        "audio_paths": paths,
        "accent_inferred_from_filename": False,
    }


def inspect_tar(path: Path, limit: int = 64) -> dict[str, Any]:
    members = []
    groups: dict[str, list[str]] = defaultdict(list)
    payload = 0
    exhausted = False
    # Stream archive members in place. No extract/extractall or audio loading.
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            heldout = bool(HELDOUT.search(member.name))
            row: dict[str, Any] = {
                "name": member.name,
                "bytes": member.size,
                "regular_file": member.isfile(),
                "heldout_name": heldout,
            }
            suffix = Path(member.name).suffix.lower()
            if member.isfile():
                groups[str(Path(member.name).with_suffix(""))].append(suffix)
            if member.isfile() and not heldout and suffix in TEXT_SUFFIXES:
                handle = archive.extractfile(member)
                if handle is not None:
                    with handle:
                        data = handle.read(4096)
                    row["text_preview"] = data.decode("utf-8-sig", errors="replace")
                    row["preview_truncated"] = member.size > len(data)
            members.append(row)
            payload += max(0, member.size)
            if len(members) >= limit or payload >= 32 * 1024 * 1024:
                break
        else:
            exhausted = True
    paired = {
        key: suffixes
        for key, suffixes in groups.items()
        if any(s in TEXT_SUFFIXES for s in suffixes)
        and any(s in {".wav", ".flac", ".mp3", ".ogg"} for s in suffixes)
    }
    return {
        "format": "tar",
        "members_inspected": len(members),
        "archive_fully_enumerated": exhausted,
        "members": members,
        "same_stem_audio_text_examples": dict(islice(paired.items(), 8)),
        "webdataset_compatible_pairs_observed": bool(paired),
        "format_contract_confirmed": False,
    }


def inspect_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        result["bytes"] = path.stat().st_size
        if path.name.lower().endswith((".tar", ".tar.gz", ".tgz")):
            result.update(inspect_tar(path))
        elif path.suffix.lower() in {".json", ".jsonl"}:
            try:
                result.update(inspect_json(path))
            except ValueError as error:
                result["schema_parse_error"] = str(error)
                with path.open("rb") as stream:
                    result["text_preview"] = stream.read(4096).decode("utf-8-sig", "replace")
        elif path.suffix.lower() in TEXT_SUFFIXES:
            with path.open("rb") as stream:
                result["text_preview"] = stream.read(4096).decode("utf-8-sig", "replace")
            result["format"] = "text_preview_only"
        else:
            with path.open("rb") as stream:
                result["header_hex"] = stream.read(32).hex()
            result["format"] = "unresolved_binary_or_audio"
    except (OSError, ValueError, tarfile.TarError) as error:
        result["error"] = str(error)
    return result


def inspect_chinese(root: Path, limit: int = 10000) -> dict[str, Any]:
    if not root.is_dir():
        return {"root": str(root), "error": "directory missing or not mounted"}
    counts: Counter[str] = Counter()
    selected: dict[str, list[Path]] = defaultdict(list)
    skipped = 0
    examined = 0
    truncated = False
    walk_errors: list[OSError] = []
    with stage_progress("Inspect Chinese shard directory (bounded; no extraction)") as tick:
        for folder, dirs, files in os.walk(root, followlinks=False, onerror=walk_errors.append):
            dirs[:] = sorted(d for d in dirs if not HELDOUT.search(d))
            for name in sorted(files):
                path = Path(folder) / name
                if path.is_symlink() or HELDOUT.search(str(path.relative_to(root))):
                    skipped += 1
                    continue
                kind = (
                    "tar"
                    if name.lower().endswith((".tar", ".tar.gz", ".tgz"))
                    else (path.suffix.lower() or "no_extension")
                )
                counts[kind] += 1
                if len(selected[kind]) < 2:
                    selected[kind].append(path)
                examined += 1
                tick(examined)
                if examined >= limit:
                    truncated = True
                    break
            if truncated:
                break
    # Limit total selected files even if an unexpected tree has many suffix kinds.
    paths = sorted(
        (p for group in selected.values() for p in group),
        key=lambda p: (
            not p.name.lower().endswith((".tar", ".tar.gz", ".tgz")),
            p.suffix.lower() not in TEXT_SUFFIXES,
            str(p),
        ),
    )[:12]
    inspections = []
    with stage_progress(
        "Read Chinese shard/metadata schema samples", total=len(paths), every=1
    ) as tick:
        for number, path in enumerate(paths, 1):
            inspections.append(inspect_file(path))
            tick(number)
    return {
        "root": str(root),
        "files_observed": examined,
        "scan_truncated": truncated,
        "file_types_observed": dict(counts),
        "heldout_or_link_files_skipped": skipped,
        "walk_errors": [str(error) for error in walk_errors],
        "examples": inspections,
        "split_provenance_verified": False,
        "corpus_hours_verified": False,
    }
