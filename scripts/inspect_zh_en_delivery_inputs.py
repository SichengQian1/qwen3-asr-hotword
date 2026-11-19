"""Inspect work-zone retrieval assets without loading a model or opening GlusterFS 103."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def english(index: Path, audio: Path, expected: str) -> dict[str, Any]:
    # Compare both naming conventions to actual files before selecting a mapping.
    files = {p.name for p in audio.glob("*.wav")}
    matches: dict[str, set[str]] = {"basename": set(), "md5_source_path": set()}
    words: dict[str, set[str]] = {mode: set() for mode in matches}
    count = 0
    for row in records(index):
        count += 1
        src = row["audio_path"]
        names = {
            "basename": Path(src).name,
            "md5_source_path": hashlib.md5(src.encode()).hexdigest()[:16] + ".wav",
        }
        for mode, name in names.items():
            if name in files:
                matches[mode].add(name)
                words[mode].update(e["surface"] for e in row["entities"])
    return {
        "index_rows": count,
        "directory_wavs": len(files),
        "mapping": {
            mode: {
                "unique_wavs_joined": len(found),
                "directory_wavs_unmatched": len(files - found),
                "distinct_target_words": len(words[mode]),
                "exact_583_surface_set": hashlib.sha256(
                    "\n".join(sorted(words[mode])).encode()
                ).hexdigest()
                == expected,
            }
            for mode, found in matches.items()
        },
    }


def home(path: Path) -> dict[str, Any]:
    groups: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    seen: set[str] = set()
    for row in records(path):
        uid = row["id"]
        if not uid.startswith("val_"):
            continue
        key = re.fullmatch(r"(val_(?:v1|v2|n1|n2))_(zh|en)_\d+", uid)
        if not key:
            issues["other_val_id"] += 1
            continue
        groups[key[2] + "/" + key[1]] += 1
        if uid in seen:
            issues["duplicate_id"] += 1
        seen.add(uid)
        if row.get("status") != "ok" or not Path(row["wav"]).is_file():
            issues["bad_status_or_missing_wav"] += 1
        targets = row.get("keywords")
        if not isinstance(targets, list) or bool(targets) != (key[1] in {"val_v1", "val_v2"}):
            issues["unexpected_keyword_field_or_polarity"] += 1
    return {"groups": groups, "issues": issues}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Known readable alternate Qwen snapshot")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/zh_en_stream2351_probe.workzone.json").read_text())
    ais, work, train = Path(cfg["aishell"]), Path(cfg["work"]), ROOT / cfg["train"]

    def subset() -> dict[str, Any]:
        p = ais / "val/kw_subset/kw.jsonl"
        if p.is_file():
            ids = {r["utt_id"]: {k["word"] for k in r["keywords"]} for r in records(p)}
        else:
            p = ROOT / cfg["previous_subset"]
            ids = {r["sample_id"]: set(r["targets"]) for r in records(p)}
        return {
            "source": str(p),
            "utterances": len(ids),
            "surfaces": len(set().union(*ids.values())),
            "pairs": sum(map(len, ids.values())),
        }

    def base() -> dict[str, Any]:
        model_files = json.loads((train / "model_identity.json").read_text())
        paths = [args.model] if args.model else [Path(s) for s in cfg["base_candidates"]]
        return {str(p): [f for f in model_files if not (p / f).is_file()] for p in paths if p}

    def names() -> list[str]:
        # Limit discovery to project metadata, keep clear of corpus/large shared mounts.
        from os import walk

        found = []
        for directory, dirs, files in walk(work, followlinks=False):
            if len(Path(directory).relative_to(work).parts) >= 2:
                dirs.clear()
            for name in files:
                if re.search(r"ner|entit|step|index|testonly", name, re.I) and Path(
                    name
                ).suffix in {".json", ".jsonl", ".tsv", ".txt", ".csv", ".bio", ".conll"}:
                    found.append(str(Path(directory) / name))
        return sorted(found)[:120]

    for title, fn in (
        ("AISHELL subset", subset),
        ("AISHELL WAV count", lambda: sum(1 for _ in Path(cfg["aishell_wav"]).glob("*.wav"))),
        ("Smart home", lambda: home(Path(cfg["home_meta"]))),
        (
            "English probe",
            lambda: english(work / cfg["en_index"], Path(cfg["en_wav"]), cfg["en_surface_sha256"]),
        ),
        ("Candidate label and index files", names),
        ("Qwen snapshots: missing files", base),
    ):
        print(title, flush=True)
        try:
            print(json.dumps(fn(), ensure_ascii=False), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
