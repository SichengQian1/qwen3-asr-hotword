"""Metadata evidence only: no quality certification, selection, or model inference."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def audit_pt_validation_metadata(
    *,
    validation_manifest: Path,
    expected_sha256: str,
    common_voice_root: Path,
    fleurs_train_tsv: Path,
) -> dict[str, Any]:
    """Join CV candidates by resolved path; inspect two FLEURS train metadata rows.

    CV is read as literal, single-line TSV. Quotes remain part of the text, not
    CSV delimiters. Unexpected column counts fail rather than silently dropping
    rows. Differences may still be serialization differences, not label errors.
    """
    manifest_sha = _sha256(validation_manifest)
    if manifest_sha != expected_sha256:
        raise ValueError(f"Validation SHA256 mismatch: {manifest_sha}")

    candidates: list[dict[str, str]] = []
    total = 0
    with validation_manifest.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("split") != "validation":
                raise ValueError(f"Manifest line {number}: expected validation record")
            total += 1
            if row.get("source_corpus") != "common_voice":
                continue
            if any(not isinstance(row.get(k), str) for k in ("id", "audio_path", "text")):
                raise ValueError(f"Manifest line {number}: invalid ID/audio/text")
            if not row["id"] or not row["audio_path"]:
                raise ValueError(f"Manifest line {number}: empty ID/audio")
            candidates.append({
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
    speakers: set[str] = set()
    examples: list[dict[str, object]] = []
    for candidate in candidates:
        matches = meta[candidate["audio_path"]]
        if len(matches) != 1:
            counts["not_in_validated" if not matches else "duplicate_metadata"] += 1
            continue
        entry = matches[0]
        current, original = candidate["text"], entry["sentence"]
        if not original.strip():
            state = "empty_original_text"
        elif current == original:
            state = "exact_text_match"
        elif _normalized(current) == _normalized(original):
            state = "case_space_unicode_only"
        else:
            state = "text_difference"
        counts[state] += 1
        votes[f"up={entry['up_votes']},down={entry['down_votes']}"] += 1
        accents[entry.get("accents") or entry.get("accent") or "unknown"] += 1
        locales[entry.get("locale") or "unknown"] += 1
        if entry["client_id"]:
            speakers.add(entry["client_id"])
        if state in {"text_difference", "empty_original_text"} and len(examples) < 3:
            examples.append({
                "id": candidate["id"], "current": _preview(current),
                "original": _preview(original),
            })

    fleurs_preview: list[dict[str, object]] = []
    with fleurs_train_tsv.open(encoding="utf-8-sig") as handle:
        for _ in range(2):
            preview_line = next(handle, None)
            if preview_line is None:
                break
            fields = preview_line.rstrip("\r\n").split("\t")
            fleurs_preview.append({
                "column_count": len(fields),
                "first_12_fields": [_preview(field) for field in fields[:12]],
            })
    return {
        "status": "completed",
        "scope": "metadata_only_not_label_accuracy_certification",
        "inputs": {
            "validation": {"path": str(validation_manifest), "sha256": manifest_sha},
            "cv_metadata": {"path": str(metadata_path), "sha256": metadata_sha},
            "fleurs_train_tsv": {"path": str(fleurs_train_tsv)},
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
        "text_difference_examples": examples,
        "fleurs_train_first_two_rows": fleurs_preview,
        "limitations": [
            "Text differences are flags, not confirmed annotation errors.",
            "No audio, phoneme labels, or pronunciation accuracy was evaluated.",
            "FLEURS rows are schema previews only; no FLEURS transcript join yet.",
            "TSV quotes are preserved; serialization may explain text differences.",
        ],
        "model_loaded": False,
        "test_set_used": False,
        "files_written": False,
    }
