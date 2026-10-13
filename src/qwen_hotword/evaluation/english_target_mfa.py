"""Read-only pronunciation consistency audit; never install generated labels."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json, _key
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.edit_distance import sequence_editops
from qwen_hotword.training.g2p_prep import extract_word_tokens, normalize_training_text
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_capacity import _sha, _verified


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def english_targets(table: dict[str, Any], vocab: PhonemeVocab) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for entry in table["entries"]:
        for variant in entry["variants"]:
            if variant["language"] != "en":
                continue
            if variant.get("role", "target") != "target" or variant["decision"] != "retained":
                raise ValueError("target-only source required; fillers/removals are not audited")
            surface, normalized = variant["surface"], variant["normalized"]
            phones = tokenize_ipa_to_vocab(variant["pronunciation"], vocab)
            if (
                normalized in seen
                or normalized != _key(surface)
                or phones.oov_units
                or not phones.token_ids
                or phones.token_ids != variant["token_ids"]
                or phones.tokens != variant["phoneme_tokens"]
            ):
                raise ValueError(f"duplicate/invalid saved English target: {surface}")
            seen.add(normalized)
            words = extract_word_tokens(surface)
            flags = []
            if any(c.isdigit() for c in surface) or not words:
                flags.append("numeric_or_empty_lexical_input_requires_review")
            if not surface.isascii():
                flags.append("non_ascii_surface_not_language_identification")
            if " " in surface or any(not c.isalpha() for c in surface):
                flags.append("phrase_or_punctuation_check_lexical_interpretation")
            rows.append(
                {
                    "entry_id": entry["id"],
                    **variant,
                    "words": words,
                    "flags": flags,
                }
            )
    return sorted(rows, key=lambda r: r["normalized"])


def phrase_label(
    words: list[str],
    dictionary: dict[str, list[str]],
    vocab: PhonemeVocab,
    *,
    blocked: bool = False,
) -> dict[str, Any]:
    per_word: list[dict[str, Any]] = [
        {"word": w, "raw_pronunciations": sorted(set(dictionary.get(w, [])))} for w in words
    ]
    missing = sorted({r["word"] for r in per_word if not r["raw_pronunciations"]})
    ambiguous = sorted({r["word"] for r in per_word if len(r["raw_pronunciations"]) > 1})
    result: dict[str, Any] = {
        "word_results": per_word,
        "missing_words": missing,
        "ambiguous_words": ambiguous,
        "raw_pronunciation": None,
        "phoneme_tokens": None,
        "token_ids": None,
        "oov_units": [],
        "status": "unresolved",
    }
    if blocked or not words or missing or ambiguous:
        return result
    raw = " ".join(r["raw_pronunciations"][0] for r in per_word)
    phones = tokenize_ipa_to_vocab(raw, vocab)
    result.update(
        raw_pronunciation=raw,
        phoneme_tokens=phones.tokens,
        token_ids=phones.token_ids,
        oov_units=phones.oov_units,
    )
    if not phones.oov_units and phones.token_ids:
        result["status"] = "available"
    return result


def _dictionary(path: Path) -> dict[str, list[str]]:
    # Same key normalization as full-manifest normalized_dictionary, preserving ambiguity.
    normalized: dict[str, list[str]] = {}
    for word, pronunciations in load_mfa_dictionary(path).items():
        values = normalized.setdefault(normalize_training_text(word), [])
        values.extend(p for p in pronunciations if p not in values)
    return normalized


def compare_label(current: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    if other["status"] != "available":
        return {"status": "unresolved", "edit_distance": None, "edits": []}
    a, b = tuple(current["token_ids"]), tuple(other["token_ids"])
    edits = sequence_editops(a, b)
    return {
        "status": "ctc_equal" if a == b else "ctc_different",
        "raw_equal": current["pronunciation"].strip() == other["raw_pronunciation"].strip(),
        "edit_distance": len(edits),
        "edits": [
            {
                "operation": op,
                "current_phone": current["phoneme_tokens"][i] if op != "insert" else None,
                "other_phone": other["phoneme_tokens"][j] if op != "delete" else None,
                "current_index": i,
                "other_index": j,
            }
            for op, i, j in edits
        ],
    }


def compare_targets(
    targets: list[dict[str, Any]],
    fresh: dict[str, list[str]],
    training: dict[str, list[str]],
    vocab: PhonemeVocab,
) -> list[dict[str, Any]]:
    rows = []
    for target in targets:
        blocked = "numeric_or_empty_lexical_input_requires_review" in target["flags"]
        new = phrase_label(target["words"], fresh, vocab, blocked=blocked)
        old = phrase_label(target["words"], training, vocab, blocked=blocked)
        rows.append(
            {
                "target": target,
                "fresh_mfa": new,
                "training_dictionary": old,
                "current_vs_fresh": compare_label(target, new),
                "current_vs_training": compare_label(target, old),
                "fresh_vs_training": compare_label(
                    {
                        "token_ids": new["token_ids"],
                        "phoneme_tokens": new["phoneme_tokens"],
                        "pronunciation": new["raw_pronunciation"],
                    },
                    old,
                )
                if new["status"] == "available"
                else {
                    "status": "unresolved",
                    "edit_distance": None,
                    "edits": [],
                },
            }
        )
    return rows


def _deliver(output: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    _write(output / "report.json", report)
    if rows:
        _write(output / "comparisons.json", rows)
        with (output / "comparisons.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "surface",
                    "current_phones",
                    "fresh_mfa_raw",
                    "fresh_ctc_phones",
                    "vs_fresh",
                    "training_raw",
                    "vs_training",
                    "flags",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["target"]["surface"],
                        " ".join(r["target"]["phoneme_tokens"]),
                        r["fresh_mfa"]["raw_pronunciation"],
                        " ".join(r["fresh_mfa"]["phoneme_tokens"] or []),
                        r["current_vs_fresh"]["status"],
                        r["training_dictionary"]["raw_pronunciation"],
                        r["current_vs_training"]["status"],
                        "; ".join(r["target"]["flags"]),
                    ]
                )
    files = sorted(
        p
        for p in output.rglob("*")
        if p.is_file() and "mfa_work" not in p.relative_to(output).parts
    )
    (output / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files),
        encoding="utf-8",
    )
    with zipfile.ZipFile(output / "return_bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in [*files, output / "sha256.txt"]:
            bundle.write(path, path.relative_to(output).as_posix())


def run_audit(root: Path, config: Path, output: Path, jobs: int, mfa: str) -> dict[str, Any]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}; use a new directory")
    cfg = _json(config)
    identities: dict[str, Any] = {}
    _checked(config, identities)
    source = (root / cfg["source_run"]).resolve()
    _checked(source / "sha256.txt", identities)
    table_path = _checked(
        _verified(source, "reviewed_keyword_table.json", identities),
        identities,
        cfg["table_sha256"],
    )
    source_report = _json(_verified(source, "report.json", identities))
    plan = _json(_verified(source, "run_config.json", identities))
    if (
        source_report.get("status") != "completed"
        or source_report.get("candidate_scope") != "joint_multilingual"
        or source_report["review"]["filler_entries_added"] != 0
        or source_report["review"]["reviewed_table_sha256"] != cfg["table_sha256"]
    ):
        raise ValueError("completed reviewed target-only run with expected table required")
    vocab_path = _checked((root / cfg["vocab"]).resolve(), identities, cfg["vocab_sha256"])
    coverage_path = Path(__file__).parents[1] / "phonemes/coverage.py"
    _checked(coverage_path, identities, plan["code_sha256"]["phonemes/coverage.py"])
    model = _checked((root / cfg["model"]).resolve(), identities, cfg["model_sha256"])
    dictionary = _checked(
        (root / cfg["training_dictionary"]).resolve(), identities, cfg["training_dictionary_sha256"]
    )
    if output.is_relative_to(source) or any(
        output.is_relative_to(Path(p).parent) for p in identities
    ):
        raise ValueError("output must be outside existing source/asset directories")
    vocab = load_phoneme_vocab(vocab_path)
    table = _json(table_path)
    targets = english_targets(table, vocab)
    if (
        len(targets) != cfg["expected_english_targets"]
        or len(targets) != source_report["review"]["keywords_by_language"]["en"]
    ):
        raise ValueError("English target count differs from frozen scope")
    training = _dictionary(dictionary)
    output.mkdir(parents=True, exist_ok=False)
    (output / "mfa_en").mkdir()
    _write(output / "targets.json", targets)
    _write(output / "configuration.json", cfg)
    words = sorted(
        {
            w
            for r in targets
            for w in r["words"]
            if "numeric_or_empty_lexical_input_requires_review" not in r["flags"]
        }
    )
    word_file, raw_dict = output / "words.txt", output / "mfa_en/raw.dict"
    word_file.write_text("\n".join(words) + "\n", encoding="utf-8")
    command = [
        mfa,
        "g2p",
        "--num_jobs",
        str(jobs),
        "--num_pronunciations",
        "1",
        str(word_file),
        str(model),
        str(raw_dict),
    ]
    try:
        version = importlib.metadata.version("montreal-forced-aligner")
    except importlib.metadata.PackageNotFoundError:
        version = None
    report: dict[str, Any] = {
        "status": "running",
        "scope": "english_target_pronunciation_consistency_only",
        "output_dir": str(output),
        "inputs": identities,
        "target_entries": len(targets),
        "unique_mfa_input_words": len(words),
        "runtime": {
            "python": sys.version,
            "mfa_version": version,
            "command": command,
            "code_sha256": {
                p.name: _sha(p)
                for p in [
                    Path(__file__),
                    coverage_path,
                    Path(__file__).parents[3] / "scripts/audit_english_target_mfa.py",
                    Path(__file__).parents[1] / "training/g2p_prep.py",
                    Path(__file__).parents[1] / "training/mfa_audit.py",
                    Path(__file__).parents[1] / "training/edit_distance.py",
                ]
            },
        },
        "audio_read": False,
        "asr_or_ctc_model_loaded": False,
        "keyword_table_modified": False,
        "retrieval_run": False,
        "fillers_audited": False,
        "word_policy": "training extract_word_tokens; isolated top1 words joined in order",
        "limitations": [
            "Differences are consistency flags, not confirmed pronunciation or model errors.",
            "Fresh MFA top1 is not a correctness certificate or an exhaustive variant dictionary.",
            "Phrases concatenate isolated words; connected speech is not reconstructed.",
            "English source tags do not identify spoken language; no automatic language switching.",
            "Numeric terms remain unresolved; spelling, apostrophes and accents are not repaired.",
            "Training dictionary comparison may be unavailable for missing/ambiguous words.",
            "No listening, acoustic alignment, recall experiment, or production label change.",
        ],
    }
    _write(output / "report.json", report)
    print(f"New audit output: {output}\nMFA English G2P: {len(words)} words", flush=True)
    rows: list[dict[str, Any]] = []
    try:
        with (output / "mfa_en/mfa.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work/en")},
            )
        rows = compare_targets(targets, _dictionary(raw_dict), training, vocab)
        for path, identity in identities.items():
            if _sha(Path(path)) != identity["sha256"]:
                raise ValueError(f"input changed during audit: {path}")
        report["comparisons"] = {
            key: dict(Counter(r[key]["status"] for r in rows))
            for key in ("current_vs_fresh", "current_vs_training", "fresh_vs_training")
        }
        report["ctc_equal_with_raw_difference"] = {
            key: sum(r[key]["status"] == "ctc_equal" and not r[key]["raw_equal"] for r in rows)
            for key in ("current_vs_fresh", "current_vs_training", "fresh_vs_training")
        }
        report["flag_counts"] = dict(Counter(f for r in targets for f in r["flags"]))
        report["status"] = (
            "completed_with_unresolved"
            if any(r["current_vs_fresh"]["status"] == "unresolved" for r in rows)
            else "completed"
        )
        report["training_comparison_unresolved"] = sum(
            r["current_vs_training"]["status"] == "unresolved" for r in rows
        )
        report["examples_different_from_fresh"] = [
            {
                "surface": r["target"]["surface"],
                "current": r["target"]["pronunciation"],
                "fresh_mfa": r["fresh_mfa"]["raw_pronunciation"],
            }
            for r in rows
            if r["current_vs_fresh"]["status"] == "ctc_different"
        ][:20]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", error=str(error))
        rows = []
    report["return_file"] = str(output / "return_bundle.zip")
    _deliver(output, report, rows)
    return report
