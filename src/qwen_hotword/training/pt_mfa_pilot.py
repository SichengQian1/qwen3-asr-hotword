"""Frozen, prediction-independent sampling and descriptive MFA diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def select_pilot(
    manifest: Path,
    expected_sha256: str,
    sources: list[str],
    per_source: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Equal-source pilot, water-filled release/density strata, stable hash draw.

    Density tertiles are sample-count tertiles within source AND release, sorted
    by label/frame then ID; not cross-language or corpus-wide pressure bins.
    This is exploratory oversampling, not a prevalence estimate or clean set.
    """
    if per_source < 1 or not sources or len(set(sources)) != len(sources):
        raise ValueError("Need distinct sources and positive samples_per_source")
    if sha256_file(manifest) != expected_sha256:
        raise ValueError("Validation SHA256 mismatch")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    audio: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "validation":
                raise ValueError("Only the existing validation pool may be read")
            for key in ("id", "audio_path", "text", "source_corpus", "release_source"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Missing/invalid {key}")
            path = str(Path(row["audio_path"]).resolve())
            if row["id"] in ids or path in audio:
                raise ValueError("Duplicate validation ID/audio path")
            ids.add(row["id"])
            audio.add(path)
            if row["source_corpus"] not in sources:
                continue
            for key in ("duration_seconds", "label_length", "effective_ctc_input_length"):
                value = row.get(key)
                if isinstance(value, bool) or not isinstance(value, (float, int)):
                    raise ValueError(f"Invalid {key}")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid {key}")
            if row.get("ctc_time_upsampling_factor") != 2:
                raise ValueError("Expected temporal-2x candidate metadata")
            if row["release_source"] not in {"original_ready", "temporal_2x_recovery"}:
                raise ValueError("Unknown release category")
            # Keep only diagnostic inputs: predictions/errors never enter selection or outputs.
            item = {
                key: row[key]
                for key in (
                    "id",
                    "text",
                    "source_corpus",
                    "release_source",
                    "duration_seconds",
                    "label_length",
                    "effective_ctc_input_length",
                )
            }
            item["audio_path"] = path
            item["ref_per_frame"] = row["label_length"] / row["effective_ctc_input_length"]
            groups[(row["source_corpus"], row["release_source"])].append(item)

    selected: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for source in sorted(sources):
        strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for (corpus, release), items in sorted(groups.items()):
            if corpus != source:
                continue
            ordered = sorted(items, key=lambda r: (r["ref_per_frame"], r["id"]))
            for index, row in enumerate(ordered):
                bucket = ("low", "medium", "high")[min(2, index * 3 // len(ordered))]
                strata[f"{release}::{bucket}"].append(row)
        if sum(map(len, strata.values())) < per_source:
            raise ValueError(f"Insufficient validation candidates for {source}")
        quotas = dict.fromkeys(sorted(strata), 0)
        remaining = per_source
        while remaining:
            for key in quotas:
                if remaining and quotas[key] < len(strata[key]):
                    quotas[key] += 1
                    remaining -= 1
        for key, count in quotas.items():
            pool = strata[key]
            ranked = sorted(
                pool,
                key=lambda r: (hashlib.sha256(f"{seed}\0{r['id']}".encode()).hexdigest(), r["id"]),
            )
            summary.append(
                {
                    "source": source,
                    "stratum": key,
                    "population": len(pool),
                    "selected": count,
                    "inclusion_fraction": count / len(pool),
                    "min_ref_per_frame": min(r["ref_per_frame"] for r in pool),
                    "max_ref_per_frame": max(r["ref_per_frame"] for r in pool),
                }
            )
            for row in ranked[:count]:
                selected.append({**row, "stratum": key, "inclusion_fraction": count / len(pool)})
    selected.sort(key=lambda r: r["id"])
    for index, row in enumerate(selected):
        row["pilot_id"] = f"pt_{index:04d}"
    return selected, summary


def inspect_alignment(path: Path, audio_duration: float) -> dict[str, Any]:
    """Parse MFA JSON export; thresholds flag review, never certify labels."""
    if not path.is_file():
        return {"status": "missing_alignment", "flags": ["missing_alignment"]}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tiers"), dict):
        raise ValueError(f"Unsupported MFA JSON tiers: {path.name}")
    phones: list[tuple[float, float, str]] = []
    word_labels: list[str] = []
    for name, tier in data["tiers"].items():
        kind = name.rsplit(" - ", 1)[-1]
        if kind not in {"phones", "words"}:
            continue
        previous = -1.0
        for entry in tier["entries"]:
            if len(entry) != 3 or not isinstance(entry[2], str):
                raise ValueError(f"Invalid interval: {path.name}")
            begin, end, label = float(entry[0]), float(entry[1]), entry[2]
            if (
                not math.isfinite(begin)
                or not math.isfinite(end)
                or begin < -0.001
                or end <= begin
                or end > audio_duration + 0.05
                or begin < previous - 0.001
            ):
                raise ValueError(f"Invalid interval timing: {path.name}")
            previous = end
            if kind == "phones":
                phones.append((begin, end, label))
            elif label:
                word_labels.append(label)
    unknown = sum(label in {"spn", "<unk>", "unk"} for _, _, label in phones)
    speech = [(a, b, p) for a, b, p in phones if p not in {"", "sil", "sp", "spn", "<unk>", "unk"}]
    if not speech or not word_labels:
        return {"status": "incomplete_alignment", "flags": ["missing_word_or_phone_tier"]}
    durations = [b - a for a, b, _ in speech]
    short = sum(d <= 0.0101 for d in durations)
    long = sum(d >= 0.300 for d in durations)
    edge = min(a for a, _, _ in speech) <= 0.020 or (
        audio_duration - max(b for _, b, _ in speech) <= 0.020
    )
    flags = []
    if unknown or "<unk>" in word_labels:
        flags.append("unknown_phone_or_word")
    if short / len(speech) >= 0.25:
        flags.append("many_minimum_duration_phones")
    if long:
        flags.append("long_phone")
    if edge:
        flags.append("alignment_touches_audio_edge")
    return {
        "status": "aligned",
        "flags": flags,
        "phone_count": len(speech),
        "minimum_duration_phone_count": short,
        "long_phone_count": long,
        "unknown_phone_count": unknown,
        "word_count": len(word_labels),
        "minimum_duration_phone_fraction": short / len(speech),
        "aligned_word_labels": word_labels,
        "alignment_sha256": sha256_file(path),
    }
