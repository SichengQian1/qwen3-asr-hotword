"""Streaming source audit, duplicate-group isolation and bounded training manifests."""

from __future__ import annotations

import hashlib
import io
import math
import re
import sqlite3
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

from qwen_hotword.phonemes.mfa_v03_assets import sha, write_json
from qwen_hotword.training.en_ar_source_audit import read_records
from qwen_hotword.training.noise_audio import rank
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import completed, digest, finish, read, rows, save_rows


def waveform(data: bytes) -> Any:
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("empty_or_nonfinite_audio")
    return np.asarray(audio, dtype="<f4")


def audio_bytes(row: dict[str, Any]) -> bytes:
    ref = row["audio"]
    if "member" in ref:
        # Offsets come from a verified regular member of an uncompressed tar, never extraction.
        with Path(ref["path"]).open("rb") as stream:
            stream.seek(ref["offset"])
            data = stream.read(ref["bytes"])
        if len(data) != ref["bytes"]:
            raise ValueError("truncated_tar_member")
    else:
        data = Path(ref["path"]).read_bytes()
    if row.get("audio_sha256") and hashlib.sha256(data).hexdigest() != row["audio_sha256"]:
        raise ValueError("source audio changed: " + row["id"])
    return data


def inspect_audio(row: dict[str, Any]) -> dict[str, Any]:
    if row["issues"]:
        return row
    try:
        data = audio_bytes(row)
        audio = waveform(data)
        row.update(
            audio_sha256=hashlib.sha256(data).hexdigest(),
            pcm_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
            samples_16k=len(audio),
            seconds=len(audio) / 16000,
        )
    except Exception as error:
        row["issues"].append(f"audio:{type(error).__name__}:{error}")
    return row


def grouping(group: str, name: str, metadata: dict[str, Any]) -> tuple[str, str]:
    for field in ("speaker_id", "session_id"):
        if metadata.get(field):
            return group + ":" + field + ":" + str(metadata[field]), field
    stem = Path(name).stem
    # Filename families are conservative proxies, not verified speaker identities.
    prefix = re.sub(r"[-_]\d+$", "", stem)
    return group + ":filename_family:" + prefix, "filename_family_unverified"


def english_rows(path: Path, group: str, fallbacks: dict[str, str]) -> Iterator[dict[str, Any]]:
    _, records = read_records(path)
    for number, raw in enumerate(records, 1):
        issues = []
        audios = raw.get("audios")
        original = audios[0] if isinstance(audios, list) and len(audios) == 1 else ""
        audio = Path(original) if isinstance(original, str) and original else None
        if audio is None or not audio.is_absolute():
            issues.append("expected_one_absolute_audio")
        elif not audio.is_file():
            for old, new in fallbacks.items():
                if audio.is_relative_to(Path(old)):
                    candidate = Path(new) / audio.relative_to(Path(old))
                    if candidate.is_file():
                        audio = candidate
                        break
        response = raw.get("response", "")
        match = (
            re.fullmatch(r"\s*language\s+English\s*<asr_text>(.*)", response, re.S)
            if isinstance(response, str)
            else None
        )
        text = match.group(1).strip() if match else ""
        assistant = [
            m.get("content")
            for m in raw.get("messages", [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        if not text or not assistant or assistant[-1] != response:
            issues.append("missing_or_inconsistent_transcript")
        if str(raw.get("language", "")).casefold() != "english":
            issues.append("unexpected_language")
        family, evidence = grouping(group, str(audio or original), raw)
        yield {
            "id": f"{group}_{number:09d}",
            "source": str(path),
            "row": number,
            "group": group,
            "text": text,
            "audio": {"path": str(audio or original)},
            "original_audio": original,
            "family": family,
            "grouping_evidence": evidence,
            "issues": issues,
        }


def chinese_rows(path: Path, index: int) -> Iterator[dict[str, Any]]:
    with tarfile.open(path, "r:") as archive:
        pairs: dict[str, dict[str, list[tarfile.TarInfo]]] = defaultdict(lambda: defaultdict(list))
        for member in archive:
            suffix = Path(member.name).suffix.lower()
            if suffix in {".wav", ".txt"}:
                pairs[str(Path(member.name).with_suffix(""))][suffix].append(member)
        for number, (name, pair) in enumerate(sorted(pairs.items()), 1):
            issues = []
            if any(len(pair[s]) != 1 for s in (".wav", ".txt")):
                issues.append("missing_or_duplicate_tar_pair")
            if any(not m.isfile() for items in pair.values() for m in items):
                issues.append("nonregular_tar_member")
            text = ""
            ref: dict[str, Any] = {"path": str(path)}
            if not issues:
                wav, txt = pair[".wav"][0], pair[".txt"][0]
                if txt.size > 1024 * 1024:
                    issues.append("oversized_text_member")
                else:
                    handle = archive.extractfile(txt)
                    if handle is None:
                        raise ValueError("missing text member stream")
                    with handle:
                        try:
                            text = handle.read().decode("utf-8-sig").strip()
                        except UnicodeDecodeError:
                            issues.append("text_not_utf8")
                ref.update(member=wav.name, offset=wav.offset_data, bytes=wav.size)
            family, evidence = grouping("zh", name, {})
            yield {
                "id": f"zh_{index:05d}_{number:07d}",
                "group": "zh",
                "source": str(path),
                "row": number,
                "text": text,
                "audio": ref,
                "family": family,
                "grouping_evidence": evidence,
                "issues": issues,
            }


def verify_source_files(output: Path, foundation: dict[str, Any]) -> dict[str, str]:
    source_files = [Path(p) for p in foundation["english_sources"].values()]
    chinese = sorted(Path(foundation["chinese_root"]).glob("*.tar"))
    if not chinese:
        raise ValueError("no uncompressed Chinese tar shards")
    source_files += chinese
    identities = {}
    with stage_progress(
        "Hash original English JSON and Chinese tar sources", total=len(source_files), every=1
    ) as tick:
        for i, p in enumerate(source_files, 1):
            identities[str(p)] = sha(p)
            tick(i)
    frozen = output / "source_identity.json"
    if frozen.exists() and read(frozen) != identities:
        raise ValueError("source files changed; use a new run")
    write_json(frozen, identities)
    return identities


def inventory(output: Path, foundation: dict[str, Any], cfg: dict[str, Any]) -> list[Path]:
    identities = verify_source_files(output, foundation)
    chinese = sorted(Path(foundation["chinese_root"]).glob("*.tar"))
    result = []

    def process(
        name: str, batch: list[dict[str, Any]], source: Path, pool: ThreadPoolExecutor
    ) -> None:
        folder = output / "inventory" / name
        identity = {"source": identities[str(source)], "rows": digest(batch)}
        if completed(folder, identity) is None:
            target = folder / "rows.jsonl"
            save_rows(target, pool.map(inspect_audio, batch))
            finish(folder, identity, [target], {"rows": len(batch)})
        result.append(folder / "rows.jsonl")

    with (
        ThreadPoolExecutor(max_workers=cfg["cpu_workers"]) as pool,
        stage_progress("Audit all English WAVs / Chinese tar audio (resumable parts)") as tick,
    ):
        total = 0
        for group, path in foundation["english_sources"].items():
            source, batch, part = Path(path), [], 0
            for row in english_rows(source, group, cfg["audio_fallbacks"]):
                batch.append(row)
                if len(batch) >= cfg["part_records"]:
                    process(f"{group}-{part:06d}", batch, source, pool)
                    total += len(batch)
                    tick(total)
                    batch, part = [], part + 1
            if batch:
                process(f"{group}-{part:06d}", batch, source, pool)
                total += len(batch)
                tick(total)
        for index, source in enumerate(chinese):
            batch = []
            part = 0
            for row in chinese_rows(source, index):
                batch.append(row)
                if len(batch) >= cfg["part_records"]:
                    process(f"zh-{index:05d}-{part:05d}", batch, source, pool)
                    total += len(batch)
                    tick(total)
                    batch, part = [], part + 1
            if batch:
                process(f"zh-{index:05d}-{part:05d}", batch, source, pool)
                total += len(batch)
                tick(total)
    write_json(output / "inventory_parts.json", [str(p) for p in result])
    return result


def split_inventory(output: Path, sources: list[Path], seed: int) -> list[Path]:
    """Union filename/metadata families connected by identical canonical PCM."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        cursor = value
        while parent[cursor] != cursor:
            cursor = parent[cursor]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = cursor
            value = next_value
        return cursor

    database = output / "duplicate_groups.sqlite"
    db = sqlite3.connect(database)
    db.execute("DROP TABLE IF EXISTS pcm")
    db.execute(
        "CREATE TABLE pcm (sha TEXT PRIMARY KEY, family TEXT, id TEXT, text TEXT, conflict INTEGER)"
    )
    counts: Counter[str] = Counter()
    with stage_progress("Isolate duplicate PCM and source families across splits") as tick:
        n = 0
        for source in sources:
            for row in rows(source):
                n += 1
                tick(n)
                if row["issues"]:
                    continue
                family = row["family"]
                find(family)
                old = db.execute(
                    "SELECT family,text FROM pcm WHERE sha=?", (row["pcm_sha256"],)
                ).fetchone()
                if old:
                    a, b = find(family), find(old[0])
                    parent[max(a, b)] = min(a, b)
                    if old[1] != row["text"]:
                        db.execute("UPDATE pcm SET conflict=1 WHERE sha=?", (row["pcm_sha256"],))
                else:
                    db.execute(
                        "INSERT INTO pcm VALUES (?,?,?,?,0)",
                        (row["pcm_sha256"], family, row["id"], row["text"]),
                    )
        db.commit()
    destinations = []
    for i, source in enumerate(sources):
        destination = output / "split_rows" / f"part-{i:06d}.jsonl"

        def converted(source: Path = source) -> Iterator[dict[str, Any]]:
            for row in rows(source):
                row["split"] = "quarantine"
                if not row["issues"]:
                    family = find(row["family"])
                    portion = int(rank(seed, family)[:8], 16) / 2**32
                    row["split"] = (
                        "test" if portion < 0.02 else "validation" if portion < 0.04 else "train"
                    )
                    row["split_group"] = family
                    first = db.execute(
                        "SELECT id,text,conflict FROM pcm WHERE sha=?", (row["pcm_sha256"],)
                    ).fetchone()
                    if first[2]:
                        row["issues"].append("duplicate_pcm_conflicting_text")
                    elif first[0] != row["id"]:
                        row["issues"].append("duplicate_pcm")
                        row["duplicate_of"] = first[0]
                counts[row["group"] + "/" + row["split"]] += 1
                yield row

        save_rows(destination, converted())
        destinations.append(destination)
    db.close()
    write_json(
        output / "split_report.json",
        {
            "counts": counts,
            "group_policy": (
                "metadata speaker/session else filename family; union exact 16k mono PCM"
            ),
            "speaker_disjoint_verified": False,
            "acoustic_near_duplicates_verified": False,
            "test_labels_generated": False,
            "split_ratios": {"train": 0.96, "validation": 0.02, "test": 0.02},
        },
    )
    return destinations


def manifests(output: Path, sources: list[Path], cfg: dict[str, Any]) -> dict[str, Any]:
    folder = output / "prepared"
    identity = {"labels": [sha(p) for p in sources], "config": digest(cfg)}
    done = completed(folder, identity)
    if done is not None:
        return done
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "selection.sqlite")
    db.execute("DROP TABLE IF EXISTS records")
    db.execute(
        "CREATE TABLE records (id TEXT PRIMARY KEY, split TEXT, grp TEXT, "
        "ranking TEXT, seconds REAL, payload TEXT)"
    )
    import json

    rejects: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    quarantine = folder / "quarantine.jsonl"
    sealed = folder / "sealed_test.jsonl"
    with (
        quarantine.open("w", encoding="utf-8") as rejected,
        sealed.open("w", encoding="utf-8") as test,
    ):
        for source in sources:
            for row in rows(source):
                totals[row["group"]] += 1
                if row["split"] == "test":
                    test.write(json.dumps(row, ensure_ascii=False) + "\n")
                elif row["issues"]:
                    rejects[row["group"]] += 1
                    reasons.update(row["group"] + "/" + i.split(":", 1)[0] for i in row["issues"])
                    if len(examples) < 30:
                        examples.append({k: row[k] for k in ("id", "group", "text", "issues")})
                    rejected.write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    db.execute(
                        "INSERT INTO records VALUES (?,?,?,?,?,?)",
                        (
                            row["id"],
                            row["split"],
                            row["group"],
                            rank(cfg["seed"], row["id"]),
                            row["seconds"],
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
    db.commit()
    report: dict[str, Any] = {
        "totals": totals,
        "quarantined_rows": rejects,
        "failure_reasons": reasons,
        "failure_examples": examples,
        "groups": {},
        "parts": [],
    }
    phone_counts: Counter[int] = Counter()
    files = [quarantine, sealed]

    def emit(items: list[dict[str, Any]], split: str, group: str) -> None:
        views = ("mixed",) if split == "train" else ("original", "noise10db")
        for view in views:
            for start in range(0, len(items), cfg["part_records"]):
                path = (
                    folder / split / group / view / f"part-{start // cfg['part_records']:06d}.jsonl"
                )
                chunk = []
                for original in items[start : start + cfg["part_records"]]:
                    row = {**original}
                    if split == "validation":
                        row.update(id=row["id"] + ":" + view, noisy=view == "noise10db")
                    row.update(condition=view, base_id=original["id"])
                    chunk.append(row)
                save_rows(path, chunk)
                files.append(path)
                report["parts"].append(
                    {
                        "path": str(path),
                        "sha256": sha(path),
                        "split": split,
                        "group": group,
                        "condition": view,
                        "records": len(chunk),
                    }
                )

    for split in ("train", "validation"):
        for group in ("zh", "us", "uk"):
            target = cfg["hours"][split][group]
            eligible = db.execute(
                "SELECT count(*),coalesce(sum(seconds),0) FROM records WHERE split=? AND grp=?",
                (split, group),
            ).fetchone()
            selected_rows = []
            seconds = 0.0
            for (payload,) in db.execute(
                "SELECT payload FROM records WHERE split=? AND grp=? ORDER BY ranking,id",
                (split, group),
            ):
                if seconds >= target * 3600:
                    break
                row = json.loads(payload)
                selected_rows.append(row)
                seconds += row["seconds"]
            if not selected_rows:
                write_json(output / "label_report.json", report)
                db.close()
                raise ValueError(f"no usable {split}/{group}; inspect quarantine and grouping")
            # Report shortfalls rather than silently duplicating data or leaking holdout rows.
            report["groups"][split + "/" + group] = {
                "records": len(selected_rows),
                "hours": seconds / 3600,
                "target_hours": target,
                "available_hours": eligible[1] / 3600,
                "shortfall_hours": max(0, target - seconds / 3600),
            }
            if seconds < cfg["minimum_hours"][split][group] * 3600:
                write_json(output / "label_report.json", report)
                db.close()
                raise ValueError(
                    f"insufficient usable {split}/{group}: {seconds / 3600:.3f}h; "
                    f"minimum {cfg['minimum_hours'][split][group]}h"
                )
            noisy = set()
            if split == "train":
                n = math.floor(len(selected_rows) * 0.7 + 0.5)
                noisy = {
                    r["id"]
                    for r in sorted(selected_rows, key=lambda r: rank(cfg["seed"] + 1, r["id"]))[:n]
                }
            report["groups"][split + "/" + group]["noise_records"] = len(noisy)
            report["groups"][split + "/" + group]["noise_hours"] = (
                sum(r["seconds"] for r in selected_rows if r["id"] in noisy) / 3600
            )
            for row in selected_rows:
                if split == "train":
                    phone_counts.update(row["phoneme_token_ids"])
                row["noisy"] = row["id"] in noisy
            emit(selected_rows, split, group)
    db.close()
    report.update(
        status="new_183_class_labels_prepared",
        training_token_counts={str(k): v for k, v in phone_counts.items()},
        zero_frequency_tokens=[i for i in range(2, 183) if not phone_counts[i]],
        test_set_used_for_training=False,
        noise_percent=70,
        ratio_unit="nearest_records_per_source_group",
        labels_are_automatic=True,
        pronunciation_accuracy_verified=False,
    )
    write_json(output / "label_report.json", {k: v for k, v in report.items() if k != "parts"})
    return cast(dict[str, Any], finish(folder, identity, files, report))
