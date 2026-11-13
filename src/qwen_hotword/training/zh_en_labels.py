"""Pinned Chinese text frontend and unchanged official MFA Top1 rewriting."""

from __future__ import annotations

import importlib
import re
import sqlite3
import ssl
import sys
import tarfile
import unicodedata
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from qwen_hotword.phonemes.mfa_v03_assets import environment, pin, run_worker, sha, write_json
from qwen_hotword.training.g2p_prep import CONNECTOR_REPLACEMENTS
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import completed, digest, finish, rows, save_rows

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]+")
LEXEME = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]+|[A-Za-z\u00c0-\u02af\u0300-\u036f]+(?:['-][A-Za-z\u00c0-\u02af\u0300-\u036f]+)*"
)


def install_frontend(output: Path, packages: list[dict[str, str]]) -> None:
    """Unpack two hash-pinned pure-Python packages privately; no pip/Conda mutation."""
    folder = output / "frontend_runtime"
    identity = packages
    if completed(folder, identity) is None:
        folder.mkdir(parents=True, exist_ok=True)
        written = []
        with stage_progress(
            "Prepare pinned Chinese frontend (private directory; no environment changes)"
        ):
            for spec in packages:
                archive = folder / spec["filename"]
                if not archive.exists() or sha(archive) != spec["sha256"]:
                    # Work-zone TLS bypass is explicitly authorized; SHA remains mandatory.
                    context = ssl._create_unverified_context()
                    with urllib.request.urlopen(
                        spec["url"], context=context, timeout=120
                    ) as response:
                        data = response.read()
                    import hashlib

                    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                        raise ValueError("frontend download checksum mismatch")
                    archive.write_bytes(data)
                pin(archive, spec["sha256"])
                if archive.suffix == ".whl":
                    with zipfile.ZipFile(archive) as zipped:
                        files = [
                            (name, zipped.read(name))
                            for name in zipped.namelist()
                            if name.startswith(spec["package"] + "/") and not name.endswith("/")
                        ]
                else:
                    with tarfile.open(archive) as zipped:
                        files = []
                        for member in zipped:
                            parts = PurePosixPath(member.name).parts
                            if member.isfile() and len(parts) > 2 and parts[1] == spec["package"]:
                                stream = zipped.extractfile(member)
                                if stream is None:
                                    raise ValueError("missing package member")
                                with stream:
                                    files.append(("/".join(parts[1:]), stream.read()))
                for name, data in files:
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError("unsafe package member")
                    target = folder / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    written.append(target)
                if not files:
                    raise ValueError("empty frontend package")
        finish(folder, identity, written, {"packages": packages})
    sys.path.insert(0, str(folder))
    importlib.invalidate_caches()


def frontend(text: str, group: str) -> list[dict[str, str]]:
    """Keep dictionary tones and phrase-aware polyphones; reject unresolved characters."""
    normalized = unicodedata.normalize("NFC", text)
    for old, new in CONNECTOR_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)
    if any(c.isdigit() for c in normalized):
        raise ValueError("number_requires_verbalization")
    spans = list(LEXEME.finditer(normalized))
    covered = set(i for match in spans for i in range(match.start(), match.end()))
    for i, char in enumerate(normalized):
        if (
            i not in covered
            and not char.isspace()
            and (not unicodedata.category(char).startswith("P") or char in "&@#%*_")
        ):
            raise ValueError("unsupported_symbol_or_character:" + char)
    result = []
    for match in spans:
        word = match.group()
        if HAN.fullmatch(word):
            if group != "zh":
                raise ValueError("hanzi_in_english_source")
            import jieba  # type: ignore[import-not-found]
            from pypinyin import Style, lazy_pinyin  # type: ignore[import-not-found]

            syllables = lazy_pinyin(
                word, style=Style.TONE, errors=lambda s: list(s), strict=True, tone_sandhi=False
            )
            if len(syllables) != len(word) or any(HAN.search(s) for s in syllables):
                raise ValueError("unresolved_hanzi_reading")
            offset = 0
            for segment in jieba.lcut(word, HMM=False):
                phones = syllables[offset : offset + len(segment)]
                # Pinyin apostrophe disambiguates vowel-initial internal syllables.
                spelling = "".join(
                    ("'" if i and unicodedata.normalize("NFD", s)[0] in "aeo" else "") + s
                    for i, s in enumerate(phones)
                )
                result.append({"language": "zh", "word": spelling, "surface": segment})
                offset += len(segment)
            if offset != len(word):
                raise ValueError("Chinese segmentation lost characters")
        else:
            if not all(
                "LATIN" in unicodedata.name(c, "") or c in "'-" or unicodedata.combining(c)
                for c in word
            ):
                raise ValueError("unsupported_script:" + word)
            language = "us" if group == "zh" else group
            if len(word) > 1 and word.isupper():
                raise ValueError("uppercase_reading_requires_policy:" + word)
            result.append({"language": language, "word": word.casefold(), "surface": word})
    if not result:
        raise ValueError("empty_text")
    return result


def make_worker(
    root: Path, output: Path, foundation: dict[str, Any], cfg: dict[str, Any]
) -> tuple[dict[str, Any], Path, list[str]]:
    assets = output / "g2p_assets"
    models = {}
    for lang, spec in foundation["models"].items():
        target = assets / lang / "g2p" / spec["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        source = (root / spec["directory"]).resolve()
        if target.is_symlink():
            if target.resolve() != source:
                raise ValueError("MFA asset link changed")
        elif target.exists():
            raise ValueError("unexpected MFA asset view")
        else:
            target.symlink_to(source, target_is_directory=True)
        models[lang] = {
            "repo": spec["repo"],
            "revision": spec["revision"],
            "dialect": spec["name"],
            "symbol_suffix": "txt",
            "files": {
                f"g2p/{spec['name']}/{name}": {"sha256": value}
                for name, value in spec["files_sha256"].items()
            },
        }
    worker = {
        "models": models,
        "vocab": str(root / foundation["vocab"]),
        "vocab_sha256": foundation["vocab_sha256"],
        "g2p_num_pronunciations": 1,
        "environment": {
            "mode": "existing_readonly",
            "name": cfg["mfa_environment"],
            "allowed_mfa_versions": ["3.4.0", "3.4.3"],
        },
    }
    path = output / "g2p_worker_config.json"
    write_json(path, worker)
    write_json(assets / "verified.json", foundation["models"])
    worker["_config_path"] = str(path)
    return worker, assets, environment(worker, assets, output / "logs")


def text_parts(
    output: Path, sources: list[Path], tick: Callable[[int], None] | None = None
) -> list[Path]:
    result = []
    for index, source in enumerate(sources):
        folder = output / "text_parts" / f"part-{index:06d}"
        identity = sha(source)
        if completed(folder, identity) is None:
            values = list(rows(source))
            for row in values:
                if row["issues"] or row["split"] == "test":
                    continue
                try:
                    row["frontend"] = frontend(row["text"], row["group"])
                except ValueError as error:
                    row["issues"].append(str(error))
            save_rows(folder / "rows.jsonl", values)
            finish(folder, identity, [folder / "rows.jsonl"], {"rows": len(values)})
        result.append(folder / "rows.jsonl")
        if tick is not None:
            tick(index + 1)
    return result


def generate_unique(
    root: Path,
    output: Path,
    sources: list[Path],
    worker: dict[str, Any],
    assets: Path,
    prefix: list[str],
    workers: int,
) -> Path:
    import json
    from concurrent.futures import ThreadPoolExecutor

    path = output / "pronunciations.sqlite"
    db = sqlite3.connect(path)
    db.execute("DROP TABLE IF EXISTS words")
    db.execute("CREATE TABLE words (lang TEXT, word TEXT, payload TEXT, PRIMARY KEY(lang,word))")
    with stage_progress("Collect unique pronunciation requests; sealed test excluded") as tick:
        for index, source in enumerate(sources, 1):
            for row in rows(source):
                if not row["issues"] and row["split"] != "test":
                    db.executemany(
                        "INSERT OR IGNORE INTO words VALUES (?,?,NULL)",
                        ((u["language"], u["word"]) for u in row["frontend"]),
                    )
            tick(index)
    db.commit()
    jobs: list[tuple[Path, Path]] = []
    cursor = db.execute("SELECT lang,word FROM words ORDER BY lang,word")
    while batch := cursor.fetchmany(2000):
        folder = output / "g2p" / f"part-{len(jobs):06d}"
        request = folder / "request.jsonl"
        save_rows(request, ({"language": lang, "word": word} for lang, word in batch))
        jobs.append((request, folder / "predictions.jsonl"))

    def task(job: tuple[Path, Path]) -> Path:
        run_worker(root, worker, assets, prefix, *job)
        return job[1]

    counts: dict[str, int] = {}
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        stage_progress("Generate all unique MFA Top1 labels", total=len(jobs), every=1) as tick,
    ):
        for index, prediction in enumerate(pool.map(task, jobs), 1):
            for row in rows(prediction):
                key = row["language"] + "/" + row["status"]
                counts[key] = counts.get(key, 0) + 1
                cur = db.execute(
                    "UPDATE words SET payload=? WHERE lang=? AND word=?",
                    (json.dumps(row, ensure_ascii=False), row["language"], row["word"]),
                )
                if cur.rowcount != 1:
                    raise ValueError("unexpected G2P output word")
            db.commit()
            tick(index)
    if db.execute("SELECT count(*) FROM words WHERE payload IS NULL").fetchone()[0]:
        raise ValueError("missing G2P outputs")
    db.close()
    write_json(
        output / "g2p_report.json",
        {
            "counts": counts,
            "jobs": len(jobs),
            "test_used": False,
            "prediction_sha256": digest([sha(path) for _, path in jobs]),
        },
    )
    return path


def label_part(output: Path, source: Path, index: int, predictions_path: Path, factor: int) -> Path:
    folder = output / "labels" / f"part-{index:06d}"
    identity = {
        "source_sha256": sha(source),
        "g2p_report": sha(output / "g2p_report.json"),
        "factor": factor,
    }
    if completed(folder, identity) is not None:
        return folder / "rows.jsonl"
    folder.mkdir(parents=True, exist_ok=True)
    originals = list(rows(source))
    import json

    db = sqlite3.connect(f"file:{predictions_path}?mode=ro", uri=True)
    from qwen_hotword.training.experiment_a import (
        AudioMetadata,
        ctc_minimum_input_length,
        estimate_qwen_lengths,
    )

    for row in originals:
        if row.get("issues") or row["split"] == "test":
            continue
        ids = []
        for unit in row["frontend"]:
            stored = db.execute(
                "SELECT payload FROM words WHERE lang=? AND word=?",
                (unit["language"], unit["word"]),
            ).fetchone()
            if stored is None or stored[0] is None:
                raise ValueError("missing G2P word")
            result = json.loads(stored[0])
            candidates = result["candidates"]
            if result["status"] != "ok" or not candidates or not candidates[0]["usable"]:
                row["issues"].append(f"g2p:{unit['language']}:{unit['word']}:{result['status']}")
            else:
                ids.extend(candidates[0]["token_ids"])
        if row["issues"]:
            continue
        minimum = ctc_minimum_input_length(ids)
        _, frames = estimate_qwen_lengths(AudioMetadata(row["samples_16k"], 16000, row["seconds"]))
        if not ids or any(i < 2 or i >= 183 for i in ids):
            row["issues"].append("invalid_complete_label")
        elif minimum > frames * factor:
            row["issues"].append("ctc_length_infeasible")
        else:
            row.update(
                phoneme_token_ids=ids,
                ctc_minimum_input_length=minimum,
                ctc_time_upsampling_factor=factor,
            )
    target = folder / "rows.jsonl"
    db.close()
    save_rows(target, originals)
    finish(folder, identity, [target], {"rows": len(originals)})
    return target
