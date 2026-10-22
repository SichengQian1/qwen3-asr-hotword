"""Package saved full-table G2P differences without rerunning any inference."""

from __future__ import annotations

import base64
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.mmservice_full import LANGUAGES, summarize
from qwen_hotword.phonemes.mmservice_smoke import _save, _sha
from qwen_hotword.training.stage_progress import stage_progress

REQUIRED = (
    "report.json",
    "input_inventory.jsonl",
    "case_comparison.jsonl",
    "word_engine_comparison.jsonl",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _parts(archive: Path, limit: int) -> list[dict[str, Any]]:
    """Make bounded, independently uploadable text pieces of one ZIP."""
    payload = base64.b64encode(archive.read_bytes()).decode("ascii")
    capacity = limit - 512
    count = (len(payload) + capacity - 1) // capacity
    digest = _sha(archive)
    result = []
    for index in range(count):
        header = {
            "encoding": "base64",
            "archive": archive.name,
            "archive_sha256": digest,
            "part": index + 1,
            "parts": count,
        }
        path = archive.parent / f"return_part_{index + 1:03d}.txt"
        path.write_text(
            json.dumps(header, separators=(",", ":"))
            + "\n"
            + payload[index * capacity : (index + 1) * capacity]
            + "\n",
            encoding="ascii",
        )
        if path.stat().st_size > limit:
            raise ValueError("text part exceeds the requested byte limit")
        result.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)})
    return result


def package(source: Path, output: Path, *, part_kib: int = 32) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("choose a new output directory outside the original run")
    if not 4 <= part_kib <= 1024:
        raise ValueError("part-kib must be between 4 and 1024")
    with stage_progress("Verify saved G2P evidence hashes (no models or audio)"):
        manifest = source / "sha256.txt"
        manifest_sha = _sha(manifest)
        members = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            if name in members or not (source / name).resolve().is_relative_to(source):
                raise ValueError("invalid/duplicate source evidence member")
            members[name] = digest
        identities = {}
        for name in REQUIRED:
            digest = _sha(source / name)
            if digest != members.get(name):
                raise ValueError(f"source SHA256 mismatch: {name}")
            identities[name] = digest
        report = json.loads((source / "report.json").read_text(encoding="utf-8"))
        if (
            report.get("status") != "completed"
            or report.get("scope") != "frozen_joint_4000_all_retained_variants"
        ):
            raise ValueError("a completed full-table diagnostic run is required")
        cases = _rows(source / "case_comparison.jsonl")
        words = _rows(source / "word_engine_comparison.jsonl")
        saved = _rows(source / "input_inventory.jsonl")
        if summarize(cases, words, saved) != report["totals"]:
            raise ValueError("saved rows do not reproduce report totals")
        for lang in LANGUAGES:
            counts = summarize(
                [r for r in cases if r["language"] == lang],
                [r for r in words if r["language"] == lang],
                [r for r in saved if r["language"] == lang],
            )
            if any(report["by_language"][lang].get(key) != value for key, value in counts.items()):
                raise ValueError(f"saved rows do not reproduce {lang} report counts")
    selected = []
    controls: Counter[str] = Counter()
    for row in cases:
        difference = bool(row["difference_flags"]) or any(
            not v["after_es_cleanup_ids_equal"] for v in row["saved_variants"]
        )
        if not difference and controls[row["language"]] >= 3:
            continue
        if not difference:
            controls[row["language"]] += 1
        selected.append({**row, "compact_selection": "difference" if difference else "control"})
    indices = {v["variant_index"] for r in selected for v in r["saved_variants"]}
    word_keys = {(r["language"], w) for r in selected for w in r["project_words"]}
    selected_words = [
        row for row in words if not row["raw_equal"] or (row["language"], row["word"]) in word_keys
    ]
    output.mkdir(parents=True)
    with stage_progress("Package all differences and three matching controls per language"):
        # Copy the original report verbatim; retain its original identity and limitations.
        (output / "report.json").write_bytes((source / "report.json").read_bytes())
        _write_rows(output / "cases.jsonl", selected)
        _write_rows(
            output / "saved_labels.jsonl", [r for r in saved if r["variant_index"] in indices]
        )
        _write_rows(output / "words.jsonl", selected_words)
        metadata = {
            "schema_version": 1,
            "source_run": str(source),
            "source_sha256_manifest": manifest_sha,
            "verified_source_files": identities,
            "recounted_totals_equal_report": True,
            "recounted_languages_equal_report": True,
            "selected_cases": len(selected),
            "selected_by_language": dict(Counter(r["language"] for r in selected)),
            "controls_by_language": dict(controls),
            "selected_words": len(selected_words),
            "selection": "all flagged cases and all post-cleanup saved-variant mismatches; "
            "first three remaining controls per language; all failing/mismatched word rewrites "
            "and component words of selected cases",
            "omitted": "other successful cases, unused successful words, logs and intermediates",
            "identity_scope": "verifies four saved evidence files against the original manifest; "
            "does not reopen original models, binaries, dictionaries or source data",
            "inference_rerun": False,
            "replacement_verified": False,
        }
        _save(output / "compact_report.json", metadata)
        files = sorted(output.iterdir())
        (output / "sha256.txt").write_text(
            "".join(f"{_sha(path)}  {path.name}\n" for path in files), encoding="utf-8"
        )
        archive = output / "return_compact.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
            for path in [*files, output / "sha256.txt"]:
                handle.write(path, path.name)
        for name, digest in {**identities, "sha256.txt": manifest_sha}.items():
            if _sha(source / name) != digest:
                raise ValueError(f"source changed while packaging: {name}")
        parts = _parts(archive, part_kib * 1024)
    result = {
        "status": "completed",
        "inference_rerun": False,
        "source_run": str(source),
        "selected_cases": len(selected),
        "return_file": str(archive),
        "zip_bytes": archive.stat().st_size,
        "zip_sha256": _sha(archive),
        "text_part_byte_limit": part_kib * 1024,
        "text_parts": parts,
        "transfer": "send the compact ZIP, OR every numbered text part; do not send both",
    }
    _save(output / "transfer.json", result)
    return result
