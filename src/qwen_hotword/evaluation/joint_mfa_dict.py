"""Fresh G2P for a sealed joint table; export one human-readable dictionary."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import LANGUAGES, _checked
from qwen_hotword.evaluation.wave_keywords import _json, _key
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_capacity import _sha, _verified

ALIASES = {"en": "en", "es": "es", "pt": "pt", "pt-BR": "pt"}


def table_terms(table: dict[str, Any], expected: int) -> list[dict[str, Any]]:
    """Keep each language/surface once, including fillers and cross-language homographs."""
    seen: set[str] = set()
    terms: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in table["entries"]:
        key = entry["normalized"]
        if key != _key(entry["surface"]) or key in seen or not entry["variants"]:
            raise ValueError("invalid/duplicate source surface or empty variants")
        seen.add(key)
        for variant in entry["variants"]:
            surface = variant["surface"]
            language = ALIASES.get(variant["language"])
            if (
                language is None
                or variant["normalized"] != key
                or _key(surface) != key
                or any(c in surface for c in "\t\r\n")
                or variant["decision"] != "retained"
                or variant.get("role", "target") not in {"target", "filler"}
            ):
                raise ValueError(f"invalid source variant: {surface}")
            words = extract_word_tokens(surface)
            terms.setdefault(
                (language, key),
                {
                    "language": language,
                    "surface": surface,
                    "normalized": key,
                    "words": words,
                    "blocked": any(c.isdigit() for c in surface) or not words,
                },
            )
    if len(seen) != expected or {t["language"] for t in terms.values()} != set(LANGUAGES):
        raise ValueError("source surface count or language coverage differs")
    return [terms[k] for k in sorted(terms)]


def reconstruct(
    terms: list[dict[str, Any]], dictionary: Path
) -> tuple[list[str], dict[str, Any]]:
    """Use exact normalized word keys, never strip accents or substitute saved labels."""
    phones: dict[str, set[str]] = {}
    for word, values in load_mfa_dictionary(dictionary).items():
        phones.setdefault(normalize_training_text(word), set()).update(values)
    lines: list[str] = []
    counts: Counter[str] = Counter()
    input_words = {w for t in terms if not t["blocked"] for w in t["words"]}
    for term in terms:
        missing = sorted({w for w in term["words"] if not phones.get(w)})
        ambiguous = sorted({w for w in term["words"] if len(phones.get(w, ())) > 1})
        reasons = []
        if term["blocked"]:
            reasons.append("numeric_or_empty_lexical_input")
        if missing:
            reasons.append("missing_words=" + ",".join(missing))
        if ambiguous:
            reasons.append("ambiguous_words=" + ",".join(ambiguous))
        prefix = term["language"] + "\t" + term["surface"]
        if reasons:
            lines.append("# UNRESOLVED\t" + prefix + "\t" + ";".join(reasons) + "\n")
            counts["unresolved"] += 1
        else:
            raw = " ".join(next(iter(phones[w])) for w in term["words"])
            lines.append(prefix + "\t" + raw + "\n")
            counts["generated"] += 1
    return lines, {
        "terms": len(terms),
        "generated": counts["generated"],
        "unresolved": counts["unresolved"],
        "input_unique_words": len(input_words),
        "output_unique_words": len(phones),
        "missing_input_words": sorted(input_words - phones.keys()),
        "unexpected_output_words": sorted(phones.keys() - input_words),
    }


def _write_report(output: Path, report: dict[str, Any]) -> None:
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(root: Path, config: Path, output: Path, jobs: int, mfa: str) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}; use a new directory")
    identities: dict[str, Any] = {}
    cfg = _json(_checked(config, identities))
    source = (root / cfg["source_run"]).resolve()
    _checked(source / "sha256.txt", identities)
    table_path = _checked(
        _verified(source, "reviewed_keyword_table.json", identities),
        identities,
        cfg["table_sha256"],
    )
    source_report = _json(_verified(source, "report.json", identities))
    review = source_report["review"]
    if (
        source_report["status"] != "completed"
        or source_report["candidate_scope"] != "joint_multilingual"
        or review["reviewed_table_sha256"] != cfg["table_sha256"]
        or review["active_unique_surfaces_per_audio"] != cfg["expected_surfaces"]
    ):
        raise ValueError("completed sealed 4000-table source identity differs")
    terms = table_terms(_json(table_path), cfg["expected_surfaces"])
    if set(cfg["models"]) != set(LANGUAGES):
        raise ValueError("exactly en/es/pt local G2P models required")
    models = {
        lang: _checked(
            (root / asset["path"]).resolve(), identities, asset["sha256"]
        )
        for lang, asset in cfg["models"].items()
    }
    if output.is_relative_to(source) or any(
        output.is_relative_to(Path(p).parent) or Path(p).is_relative_to(output)
        for p in identities
    ):
        raise ValueError("output must be separate from source/config/model assets")
    output.mkdir(parents=True, exist_ok=False)
    try:
        version = importlib.metadata.version("montreal-forced-aligner")
    except importlib.metadata.PackageNotFoundError:
        version = None
    report: dict[str, Any] = {
        "status": "running",
        "output_dir": str(output),
        "dictionary": str(output / "joint_4000_mfa.dict"),
        "unique_surfaces": cfg["expected_surfaces"],
        "language_surface_rows": len(terms),
        "inputs": identities,
        "by_language": {},
        "runtime": {"python": sys.version, "mfa_version": version, "mfa": mfa},
        "code_sha256": {
            "joint_mfa_dict.py": _sha(Path(__file__)),
            "run_joint_4000_mfa.py": _sha(
                Path(__file__).parents[3] / "scripts/run_joint_4000_mfa.py"
            ),
            "g2p_prep.py": _sha(Path(__file__).parents[1] / "training/g2p_prep.py"),
            "mfa_audit.py": _sha(Path(__file__).parents[1] / "training/mfa_audit.py"),
        },
        "audio_read": False,
        "keyword_table_modified": False,
        "policy": "training word extraction; top1 isolated words joined in original order",
        "limitations": [
            "Inspection TSV dictionary, not an importable MFA or runtime keyword dictionary.",
            "Language tag selects model; no spoken-language detection or code-switch repair.",
            "Fresh G2P is not acoustic alignment or a pronunciation correctness certificate.",
            "No accent stripping, spelling repair, CTC normalization or saved-label fallback.",
            "Unresolved terms are comments; phrases do not model connected speech.",
        ],
    }
    print(f"New output: {output}", flush=True)
    _write_report(output, report)
    try:
        merged = [
            "# Inspection only: language<TAB>original surface<TAB>raw MFA phones\n",
            "# Phrases concatenate isolated words; unresolved terms are explicit comments.\n",
        ]
        for lang in LANGUAGES:
            directory = output / "work" / lang
            directory.mkdir(parents=True)
            selected = [t for t in terms if t["language"] == lang]
            words = sorted({w for t in selected if not t["blocked"] for w in t["words"]})
            word_file, dictionary = directory / "words.txt", directory / "raw.dict"
            word_file.write_text("\n".join(words) + "\n", encoding="utf-8")
            command = [
                mfa, "g2p", "--num_jobs", str(jobs), "--num_pronunciations", "1",
                str(word_file), str(models[lang]), str(dictionary),
            ]
            print(f"{lang}: {len(selected)} terms -> {len(words)} words", flush=True)
            if words:
                with (directory / "mfa.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT,
                        env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work" / lang)},
                    )
            else:
                dictionary.write_text("", encoding="utf-8")
            lines, metrics = reconstruct(selected, dictionary)
            merged.extend(lines)
            report["by_language"][lang] = {**metrics, "command": command if words else None}
            _write_report(output, report)
        for path, identity in identities.items():
            if _sha(Path(path)) != identity["sha256"]:
                raise ValueError(f"input changed during G2P: {path}")
        # Publish only after every language and input identity check succeeded.
        pending = output / "joint_4000_mfa.dict.pending"
        pending.write_text("".join(merged), encoding="utf-8")
        pending.replace(output / "joint_4000_mfa.dict")
        report["status"] = (
            "completed_with_unresolved"
            if any(v["unresolved"] for v in report["by_language"].values())
            else "completed"
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=str(error))
    _write_report(output, report)
    files = [p for p in output.rglob("*") if p.is_file() and "mfa_work" not in p.parts]
    (output / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in sorted(files)),
        encoding="utf-8",
    )
    return report
