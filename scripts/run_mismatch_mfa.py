#!/usr/bin/env python3
"""Generate raw MFA word pronunciations and reconstruct the user's phrase lists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

MODELS = {
    "en": "english_us_mfa.zip",
    "es": "spanish_latin_america_mfa.zip",
    "pt": "portuguese_brazil_mfa.zip",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_terms(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        words = unicodedata.normalize("NFC", text).casefold().split()
        if words:
            rows.append({"input_line": number, "original": text, "words": words})
    if not rows:
        raise ValueError(f"empty term list: {path}")
    return rows


def reconstruct(rows: list[dict[str, Any]], dictionary: Path) -> list[dict[str, Any]]:
    pronunciations: dict[str, set[str]] = {}
    for line in dictionary.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"invalid MFA dictionary line: {line!r}")
        pronunciations.setdefault(parts[0], set()).add(parts[1].strip())
    result = []
    for row in rows:
        per_word = [
            {"word": w, "pronunciations": sorted(pronunciations.get(w, set()))}
            for w in row["words"]
        ]
        missing = sorted({p["word"] for p in per_word if not p["pronunciations"]})
        ambiguous = sorted({p["word"] for p in per_word if len(p["pronunciations"]) > 1})
        complete = not missing and not ambiguous
        result.append(
            {
                **row,
                "word_results": per_word,
                "missing_words": missing,
                "ambiguous_words": ambiguous,
                "status": "generated" if complete else "unresolved",
                "raw_phrase_pronunciation": " ".join(p["pronunciations"][0] for p in per_word)
                if complete
                else None,
            }
        )
    return result


def run(root: Path, output: Path | None, jobs: int, mfa: str) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    sources, models, terms = {}, {}, {}
    for lang, filename in MODELS.items():
        sources[lang] = root / f"mismatch_words_{lang}.txt"
        models[lang] = root / "models/mfa/g2p" / filename
        terms[lang] = read_terms(sources[lang])
        if not models[lang].is_file():
            raise FileNotFoundError(f"missing local model (no download): {models[lang]}")
    if output is None:
        (root / "outputs").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="mismatch_mfa_phrases_", dir=root / "outputs"))
    else:
        output.mkdir(parents=True, exist_ok=False)
    print(f"Output directory: {output}", flush=True)
    try:
        mfa_version = importlib.metadata.version("montreal-forced-aligner")
    except importlib.metadata.PackageNotFoundError:
        mfa_version = None
    report: dict[str, Any] = {
        "status": "running",
        "output_dir": str(output),
        "inputs": {},
        "by_language": {},
        "runtime": {
            "python": sys.version,
            "mfa_version": mfa_version,
            "mfa_executable": mfa,
            "script_sha256": sha(Path(__file__)),
        },
        "policy": "NFC + casefold + whitespace split; top1 per word; join in original order",
        "limitations": [
            "Language model follows the input filename; no language detection or code-switching.",
            "Original text, accents and punctuation retained; no apostrophe/spelling repair.",
            "Phrase output concatenates isolated-word G2P, not contextual or audio alignment.",
            "Generated pronunciation is not correctness certification; missing words stay null.",
            "Raw MFA phones are not normalized to the CTC vocabulary or installed in any table.",
        ],
        "audio_read": False,
        "keyword_table_modified": False,
    }
    try:
        for lang in MODELS:
            directory = output / lang
            directory.mkdir()
            shutil.copyfile(sources[lang], directory / "input.txt")
            if read_terms(directory / "input.txt") != terms[lang]:
                raise ValueError(f"input changed since preflight: {sources[lang]}")
            report["inputs"][lang] = {
                "source": str(sources[lang]),
                "source_sha256": sha(directory / "input.txt"),
                "model": str(models[lang]),
                "model_sha256": sha(models[lang]),
            }
            words = sorted({w for row in terms[lang] for w in row["words"]})
            word_file, dictionary = directory / "words.txt", directory / "raw.dict"
            word_file.write_text("\n".join(words) + "\n", encoding="utf-8")
            command = [
                mfa,
                "g2p",
                "--num_jobs",
                str(jobs),
                "--num_pronunciations",
                "1",
                str(word_file),
                str(models[lang]),
                str(dictionary),
            ]
            report["by_language"][lang] = {"unique_words": len(words), "command": command}
            write_json(output / "report.json", report)
            print(
                f"{lang}: {len(terms[lang])} terms -> {len(words)} words; "
                f"log: {directory / 'mfa.log'}",
                flush=True,
            )
            with (directory / "mfa.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work" / lang)},
                )
            result = reconstruct(terms[lang], dictionary)
            write_json(directory / "phrases.json", result)
            with (directory / "phrases.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(["input_line", "original", "status", "raw_mfa_pronunciation"])
                writer.writerows(
                    [r["input_line"], r["original"], r["status"], r["raw_phrase_pronunciation"]]
                    for r in result
                )
            missing = sorted({w for r in result for w in r["missing_words"]})
            ambiguous = sorted({w for r in result for w in r["ambiguous_words"]})
            report["by_language"][lang].update(
                input_terms=len(result),
                generated_terms=sum(r["status"] == "generated" for r in result),
                unresolved_terms=sum(r["status"] != "generated" for r in result),
                missing_words=missing,
                ambiguous_words=ambiguous,
            )
        unresolved = sum(v["unresolved_terms"] for v in report["by_language"].values())
        report["status"] = "completed_with_unresolved" if unresolved else "completed"
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=str(error))
    write_json(output / "report.json", report)
    files = [output / "report.json"] + sorted(
        p for lang in MODELS for p in (output / lang).glob("*") if p.is_file()
    )
    (output / "sha256.txt").write_text(
        "".join(f"{sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    try:
        mfa = shutil.which("mfa")
        if not mfa:
            raise FileNotFoundError("mfa not found; run this script in the aligner environment")
        report = run(
            args.root.resolve(),
            args.output_dir.resolve() if args.output_dir else None,
            args.jobs,
            mfa,
        )
    except (OSError, UnicodeError, ValueError) as error:
        report = {"status": "failed", "error": str(error)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
