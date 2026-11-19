"""Additional gold sources for the 480h / joint-2351 evaluation."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.inference.stream_context_contract import boundaries
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import rows

FULL = "zh-aishell-test"
PROBE = "en-tts-583"
DATASETS = (FULL, *base.COUNTS, PROBE)
NER_SHA = "0b2dfdf9d4196b4f3ce089937f714267ab6b0eee54513ac8d26221012c53ba53"
NER_URL = (
    "https://raw.githubusercontent.com/Alibaba-NLP/AISHELL-NER/"
    "fb11b6495281bc9bbc2bea7b8bde9eb3c19a4605/data/aishell_ner_transcript.test.txt"
)
MARKS = re.compile(r"\[([^\[\]<>()]+)\]|\(([^\[\]<>()]+)\)|<([^\[\]<>()]+)>")


def official_file(out: Path, local: Path | None) -> Path:
    if local:
        pin(local, NER_SHA)
        return local
    path = out / "aishell_ner_transcript.test.txt"
    if not path.exists():
        part = path.with_suffix(".part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--insecure",
                "--retry",
                "2",
                "--connect-timeout",
                "15",
                "--max-time",
                "90",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--output",
                str(part),
                NER_URL,
            ],
            check=True,
        )
        pin(part, NER_SHA)
        part.replace(path)
    pin(path, NER_SHA)
    return path


def keyed_text(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            uid, text = line.split(maxsplit=1)
            if uid in result or not re.fullmatch(r"[A-Za-z0-9_-]+", uid):
                raise ValueError(f"invalid/duplicate utterance: {uid} in {path}")
            result[uid] = "".join(text.split())
    return result


def full_gold(official: Path, reference: Path, words: set[str]) -> tuple[dict[str, list[str]], int]:
    annotated, plain = keyed_text(official), keyed_text(reference)
    if annotated.keys() != plain.keys():
        raise ValueError("AISHELL-NER IDs differ from work-zone test transcript")
    targets = {}
    mentions = 0
    for uid, text in annotated.items():
        clean = MARKS.sub(lambda m: m[m.lastindex or 1], text)
        if re.search(r"[\[\]<>()]", clean) or clean not in {
            plain[uid],
            MARKS.sub(lambda m: m[m.lastindex or 1], plain[uid]),
        }:
            raise ValueError(f"AISHELL text/markers differ: {uid}")
        found = [m[m.lastindex or 1] for m in MARKS.finditer(text) if m[m.lastindex or 1] in words]
        mentions += len(found)
        targets[uid] = sorted(set(found))
    return targets, mentions


def probe_gold(
    index: Path, directory: Path, expected: set[str]
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    files = {p.name for p in directory.glob("*.wav") if p.is_file()}
    if not files:
        raise ValueError(f"no English probe WAVs in {directory}")
    lookup = {base.canonical(w): w for w in expected}
    by_name: dict[str, tuple[str, tuple[str, ...]]] = {}
    missing_files, duplicate_rows = 0, 0
    extras: Counter[str] = Counter()
    for row in rows(index):
        src = row["audio_path"]
        name = hashlib.md5(src.encode()).hexdigest()[:16] + ".wav"
        if name not in files:
            missing_files += 1
            continue
        if str(row.get("language", "en")).casefold().split("-")[0] not in {"en", "english"}:
            raise ValueError(f"unexpected language for English probe WAV: {name}")
        ents = row["entities"]
        if not isinstance(ents, list) or any(
            not isinstance(e["surface"], str) or not e["surface"].strip() for e in ents
        ):
            raise ValueError(f"invalid entities for {name}")
        all_words = tuple(sorted({base.canonical(e["surface"]) for e in ents}))
        if name in by_name:
            if by_name[name] != (src, all_words):
                raise ValueError(f"conflicting index labels/source path: {name}")
            duplicate_rows += 1
            continue
        by_name[name] = (src, all_words)
        extras.update(w for w in all_words if w not in lookup)
    if files != by_name.keys():
        raise ValueError(
            f"unmatched English WAVs: {len(files - by_name.keys())}; "
            f"e.g. {sorted(files - by_name.keys())[:3]}"
        )
    target = {
        name[:-4]: [w for w in words if w in lookup] for name, (_, words) in sorted(by_name.items())
    }
    covered = {w for ws in target.values() for w in ws}
    if not covered:
        raise ValueError("English probe has no verified targets in the declared list")
    return target, {
        "directory_wavs": len(files),
        "scored_audio": sum(bool(ws) for ws in target.values()),
        "with_no_declared_target": sum(not ws for ws in target.values()),
        "index_rows_without_directory_wav": missing_files,
        "duplicate_index_rows": duplicate_rows,
        "declared_words": len(expected),
        "covered_words": len(covered),
        "missing_words": sorted(lookup[w] for w in lookup.keys() - covered),
        "annotations_outside_probe": dict(extras),
        "index_sha256": sha(index),
    }


def extra_inventory(
    cfg: dict[str, Any], table: Path, official: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import soundfile as sf

    word_lists: dict[str, set[str]] = {name: set() for name in (FULL, PROBE)}
    for row in rows(table):
        sources = {s for p in row["pronunciations"] for s in p["sources"]}
        for name in sources & word_lists.keys():
            word_lists[name].add(row["surface"])
    full, mentions = full_gold(official, Path(cfg["aishell"]) / "data/test/text", word_lists[FULL])
    full_counts = (
        len(full),
        len({w for words in full.values() for w in words}),
        mentions,
        sum(map(len, full.values())),
    )
    if full_counts != (7176, 1186, 3082, 3064):
        raise ValueError(f"AISHELL gold counts differ: {full_counts}")
    if (
        hashlib.sha256("\n".join(sorted(word_lists[PROBE])).encode()).hexdigest()
        != cfg["en_surface_sha256"]
    ):
        raise ValueError("declared English probe surfaces differ from source TSV")
    probe, audit = probe_gold(
        Path(cfg["work"]) / cfg["en_index"], Path(cfg["en_wav"]), word_lists[PROBE]
    )
    samples = []
    for name, gold, audio in (
        (FULL, full, Path(cfg["aishell_wav"])),
        (PROBE, probe, Path(cfg["en_wav"])),
    ):
        lang = name[:2]
        with stage_progress(f"Verify audio/targets: {name}", total=len(gold), every=250) as tick:
            for i, (uid, words) in enumerate(gold.items(), 1):
                wav = audio / (uid + ".wav")
                info = sf.info(str(wav))
                samples.append(
                    dict(
                        group=name,
                        key=f"{name}@@{lang}",
                        language=lang,
                        sample_id=uid,
                        audio_path=str(wav),
                        audio_sha256=sha(wav),
                        frames=info.frames,
                        sample_rate=info.samplerate,
                        channels=info.channels,
                        duration_seconds=info.frames / info.samplerate,
                        step_count=len(boundaries(info.frames, info.samplerate)),
                        targets=sorted(base.canonical(w) for w in words),
                    )
                )
                tick(i)
    return samples, {
        FULL: {
            "reference_url": NER_URL,
            "reference_sha256": NER_SHA,
            "mentions": mentions,
            "target_pairs": full_counts[-1],
            "positive_audio": sum(bool(ws) for ws in full.values()),
        },
        PROBE: audit,
    }
