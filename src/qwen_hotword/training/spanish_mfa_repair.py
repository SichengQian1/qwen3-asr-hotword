from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.g2p_prep import normalize_training_text
from qwen_hotword.training.mfa_audit import (
    load_mfa_dictionary,
    load_word_counts,
    load_words,
)

COMBINING_ACUTE = "\u0301"
COMBINING_TILDE = "\u0303"
PLAN_FIELDS = ("word", "corpus_count", "resolution", "proxy", "rules", "detail")


def prepare_shared_spanish_mfa_repair(
    corpora: Mapping[str, tuple[str | Path, str | Path]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Prepare per-corpus plans and one deduplicated proxy list for MFA."""

    destination = Path(output_dir).expanduser()
    _validate_corpus_names(corpora)
    _require_empty_directory(destination)
    destination.mkdir(parents=True)
    reports: dict[str, dict[str, Any]] = {}
    shared_proxies: set[str] = set()
    for name, (g2p_dir_value, dictionary_value) in corpora.items():
        g2p_dir = Path(g2p_dir_value).expanduser()
        dictionary = Path(dictionary_value).expanduser()
        report = prepare_spanish_mfa_repair(
            g2p_dir / "words.txt",
            g2p_dir / "word_counts.tsv",
            dictionary,
            destination / name,
        )
        reports[name] = report
        shared_proxies.update(
            line.strip()
            for line in Path(str(report["proxy_words_path"])).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )

    proxy_path = destination / "proxy_words.txt"
    _write_text_atomic(proxy_path, "".join(f"{word}\n" for word in sorted(shared_proxies)))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "corpora": reports,
        "corpus_count": len(reports),
        "shared_proxy_words": len(shared_proxies),
        "proxy_words_path": str(proxy_path),
    }
    _write_json_atomic(destination / "prepare_summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def finalize_shared_spanish_mfa_repair(
    corpora: Mapping[str, tuple[str | Path, str | Path]],
    repair_root: str | Path,
    proxy_dictionary_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Finalize and audit repaired dictionaries for all prepared corpora."""

    from qwen_hotword.training.mfa_audit import audit_mfa_dictionary

    root = Path(repair_root).expanduser()
    proxy_dictionary = Path(proxy_dictionary_path).expanduser()
    vocab = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    _validate_corpus_names(corpora)
    _require_files(root / "prepare_summary.json", proxy_dictionary, vocab)
    _require_empty_directory(destination)
    destination.mkdir(parents=True)

    reports: dict[str, dict[str, Any]] = {}
    for name, (g2p_dir_value, dictionary_value) in corpora.items():
        g2p_dir = Path(g2p_dir_value).expanduser()
        dictionary = Path(dictionary_value).expanduser()
        corpus_output = destination / name
        finalized = finalize_spanish_mfa_repair(
            g2p_dir / "words.txt",
            g2p_dir / "word_counts.tsv",
            dictionary,
            root / name / "repair_plan.tsv",
            proxy_dictionary,
            vocab,
            corpus_output,
            dictionary_name=f"{name}_spanish_latin_america_repaired.v1.dict",
        )
        audit = audit_mfa_dictionary(
            g2p_dir / "words.txt",
            Path(str(finalized["dictionary_path"])),
            vocab,
            corpus_output / "mfa_audit",
            word_counts_path=g2p_dir / "word_counts.tsv",
        )
        reports[name] = {
            "finalize": finalized,
            "audit": audit.to_dict(),
        }
        _write_sha256_manifest(corpus_output)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "training_labels_ready": all(
            bool(report["audit"]["training_labels_ready"])
            for report in reports.values()
        ),
        "corpus_count": len(reports),
        "corpora": reports,
        "repair_root": str(root),
        "proxy_dictionary_path": str(proxy_dictionary),
        "vocab_path": str(vocab),
        "output_dir": str(destination),
    }
    _write_json_atomic(destination / "summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def prepare_spanish_mfa_repair(
    words_path: str | Path,
    word_counts_path: str | Path,
    dictionary_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Plan conservative Spanish spelling proxies for words missing from MFA output."""

    words_file = Path(words_path).expanduser()
    counts_file = Path(word_counts_path).expanduser()
    dictionary_file = Path(dictionary_path).expanduser()
    destination = Path(output_dir).expanduser()
    _require_files(words_file, counts_file, dictionary_file)
    _require_empty_directory(destination)

    words = load_words(words_file)
    counts = load_word_counts(counts_file)
    dictionary = _normalized_unique_dictionary(dictionary_file)
    rows: list[dict[str, str | int]] = []
    proxy_words: set[str] = set()
    resolution_counts: Counter[str] = Counter()
    weighted_resolution_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()

    for word in sorted(words):
        corpus_count = counts.get(word, 0)
        exact = dictionary.get(word, ())
        if len(exact) == 1:
            resolution = "base_exact"
            proxy = word
            rules: tuple[str, ...] = ()
            detail = ""
        elif len(exact) > 1:
            resolution = "unresolved"
            proxy = ""
            rules = ()
            detail = "ambiguous_base_pronunciation"
        else:
            proxy, rules, unsafe_reason = spanish_g2p_proxy(word)
            if unsafe_reason is not None:
                resolution = "unresolved"
                proxy = ""
                detail = unsafe_reason
            else:
                proxy_pronunciations = dictionary.get(proxy, ())
                if len(proxy_pronunciations) == 1:
                    resolution = "base_proxy"
                    detail = ""
                elif len(proxy_pronunciations) > 1:
                    resolution = "unresolved"
                    detail = "ambiguous_base_proxy_pronunciation"
                else:
                    resolution = "mfa_proxy"
                    detail = ""
                    proxy_words.add(proxy)

        rows.append(
            {
                "word": word,
                "corpus_count": corpus_count,
                "resolution": resolution,
                "proxy": proxy,
                "rules": ",".join(rules),
                "detail": detail,
            }
        )
        resolution_counts[resolution] += 1
        weighted_resolution_counts[resolution] += corpus_count
        rule_counts.update(rules)

    destination.mkdir(parents=True)
    plan_path = destination / "repair_plan.tsv"
    proxy_path = destination / "proxy_words.txt"
    unresolved_path = destination / "unresolved_words.tsv"
    _write_plan(plan_path, rows)
    _write_text_atomic(proxy_path, "".join(f"{word}\n" for word in sorted(proxy_words)))
    _write_plan(
        unresolved_path,
        [row for row in rows if row["resolution"] == "unresolved"],
    )
    total_tokens = sum(counts.get(word, 0) for word in words)
    unresolved_tokens = weighted_resolution_counts["unresolved"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "training_labels_ready": resolution_counts["unresolved"] == 0,
        "words_path": str(words_file),
        "word_counts_path": str(counts_file),
        "base_dictionary_path": str(dictionary_file),
        "output_dir": str(destination),
        "input_unique_words": len(words),
        "input_corpus_tokens": total_tokens,
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "weighted_resolution_counts": dict(sorted(weighted_resolution_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "proxy_words_for_mfa": len(proxy_words),
        "unresolved_unique_words": resolution_counts["unresolved"],
        "unresolved_corpus_tokens": unresolved_tokens,
        "planned_corpus_token_coverage": (
            1.0 if total_tokens == 0 else (total_tokens - unresolved_tokens) / total_tokens
        ),
        "repair_plan_path": str(plan_path),
        "proxy_words_path": str(proxy_path),
        "unresolved_words_path": str(unresolved_path),
        "input_sha256": {
            "words": _sha256_file(words_file),
            "word_counts": _sha256_file(counts_file),
            "base_dictionary": _sha256_file(dictionary_file),
        },
    }
    _write_json_atomic(destination / "prepare_summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def finalize_spanish_mfa_repair(
    words_path: str | Path,
    word_counts_path: str | Path,
    base_dictionary_path: str | Path,
    repair_plan_path: str | Path,
    proxy_dictionary_path: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    *,
    dictionary_name: str = "spanish_latin_america_repaired.v1.dict",
) -> dict[str, Any]:
    """Resolve a repair plan into a one-pronunciation-per-word CTC dictionary."""

    words_file = Path(words_path).expanduser()
    counts_file = Path(word_counts_path).expanduser()
    base_file = Path(base_dictionary_path).expanduser()
    plan_file = Path(repair_plan_path).expanduser()
    proxy_file = Path(proxy_dictionary_path).expanduser()
    vocab_file = Path(vocab_path).expanduser()
    destination = Path(output_dir).expanduser()
    _require_files(words_file, counts_file, base_file, plan_file, proxy_file, vocab_file)
    _require_empty_directory(destination)
    if not dictionary_name or Path(dictionary_name).name != dictionary_name:
        raise ValueError("dictionary_name must be a plain file name")

    words = load_words(words_file)
    counts = load_word_counts(counts_file)
    base_dictionary = _normalized_unique_dictionary(base_file)
    proxy_dictionary = _normalized_unique_dictionary(proxy_file)
    plan_rows = _read_plan(plan_file)
    planned_words = {row["word"] for row in plan_rows}
    if planned_words != words or len(plan_rows) != len(planned_words):
        raise ValueError("repair plan words do not exactly match words.txt")
    vocab = load_phoneme_vocab(vocab_file)

    output_rows: list[tuple[str, str]] = []
    unresolved_rows: list[dict[str, str | int]] = []
    resolution_counts: Counter[str] = Counter()
    weighted_resolution_counts: Counter[str] = Counter()
    cleanup_counts: Counter[str] = Counter()

    for row in sorted(plan_rows, key=lambda value: value["word"]):
        word = row["word"]
        corpus_count = counts.get(word, 0)
        resolution = row["resolution"]
        proxy = row["proxy"]
        rules = tuple(item for item in row["rules"].split(",") if item)
        if resolution == "unresolved":
            unresolved_rows.append(
                _unresolved_row(word, corpus_count, proxy, rules, row["detail"])
            )
            continue
        if resolution == "base_exact":
            candidates = base_dictionary.get(word, ())
        elif resolution == "base_proxy":
            candidates = base_dictionary.get(proxy, ())
        elif resolution == "mfa_proxy":
            candidates = proxy_dictionary.get(proxy, ())
        else:
            raise ValueError(f"unknown repair resolution: {resolution}")
        if len(candidates) != 1:
            detail = "missing_proxy_pronunciation" if not candidates else "ambiguous_pronunciation"
            unresolved_rows.append(
                _unresolved_row(word, corpus_count, proxy, rules, detail)
            )
            continue

        pronunciation, cleanup = repair_spanish_pronunciation(candidates[0], rules)
        tokenized = tokenize_ipa_to_vocab(pronunciation, vocab)
        if not tokenized.tokens:
            unresolved_rows.append(
                _unresolved_row(word, corpus_count, proxy, rules, "empty_pronunciation")
            )
            continue
        if tokenized.oov_units:
            unresolved_rows.append(
                _unresolved_row(
                    word,
                    corpus_count,
                    proxy,
                    rules,
                    "oov_phone:" + " ".join(tokenized.oov_units),
                )
            )
            continue
        output_rows.append((word, pronunciation))
        resolution_counts[resolution] += 1
        weighted_resolution_counts[resolution] += corpus_count
        cleanup_counts.update(cleanup)

    destination.mkdir(parents=True)
    dictionary_path = destination / dictionary_name
    unresolved_path = destination / "unresolved_words.tsv"
    _write_text_atomic(
        dictionary_path,
        "".join(f"{word}\t{pronunciation}\n" for word, pronunciation in output_rows),
    )
    _write_unresolved(unresolved_path, unresolved_rows)

    total_tokens = sum(counts.get(word, 0) for word in words)
    unresolved_tokens = sum(int(row["corpus_count"]) for row in unresolved_rows)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "training_labels_ready": not unresolved_rows,
        "words_path": str(words_file),
        "word_counts_path": str(counts_file),
        "base_dictionary_path": str(base_file),
        "repair_plan_path": str(plan_file),
        "proxy_dictionary_path": str(proxy_file),
        "vocab_path": str(vocab_file),
        "output_dir": str(destination),
        "dictionary_path": str(dictionary_path),
        "input_unique_words": len(words),
        "input_corpus_tokens": total_tokens,
        "dictionary_unique_words": len(output_rows),
        "dictionary_pronunciations": len(output_rows),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "weighted_resolution_counts": dict(sorted(weighted_resolution_counts.items())),
        "phone_cleanup_counts": dict(sorted(cleanup_counts.items())),
        "unresolved_unique_words": len(unresolved_rows),
        "unresolved_corpus_tokens": unresolved_tokens,
        "corpus_token_coverage": (
            1.0 if total_tokens == 0 else (total_tokens - unresolved_tokens) / total_tokens
        ),
        "unresolved_words_path": str(unresolved_path),
        "input_sha256": {
            "words": _sha256_file(words_file),
            "word_counts": _sha256_file(counts_file),
            "base_dictionary": _sha256_file(base_file),
            "repair_plan": _sha256_file(plan_file),
            "proxy_dictionary": _sha256_file(proxy_file),
            "vocab": _sha256_file(vocab_file),
        },
        "dictionary_sha256": _sha256_file(dictionary_path),
    }
    _write_json_atomic(destination / "finalize_summary.json", summary)
    _write_sha256_manifest(destination)
    return summary


def spanish_g2p_proxy(word: str) -> tuple[str, tuple[str, ...], str | None]:
    normalized = unicodedata.normalize("NFC", normalize_training_text(word))
    proxy = normalized
    rules: list[str] = []
    if "gü" in proxy:
        proxy = proxy.replace("gü", "gw")
        rules.append("diaeresis_via_gw")
    if "ü" in proxy:
        return "", tuple(rules), "unsupported_diaeresis_context"
    if "ñ" in proxy:
        proxy = proxy.replace("ñ", "ni")
        rules.append("enye_via_ni")
    without_acute = _strip_acute(proxy)
    if without_acute != proxy:
        proxy = without_acute
        rules.append("strip_acute")
    if not rules or proxy == normalized:
        return "", tuple(rules), "no_safe_proxy"
    return proxy, tuple(rules), None


def repair_spanish_pronunciation(
    pronunciation: str,
    rules: Iterable[str],
) -> tuple[str, Counter[str]]:
    active_rules = set(rules)
    decomposed = unicodedata.normalize("NFD", pronunciation)
    tilde_count = decomposed.count(COMBINING_TILDE)
    without_tilde = decomposed.replace(COMBINING_TILDE, "")
    phones = unicodedata.normalize("NFC", without_tilde).split()
    cleanup: Counter[str] = Counter()
    if tilde_count:
        cleanup["combining_tilde_removed"] = tilde_count
    if "enye_via_ni" in active_rules:
        collapsed: list[str] = []
        index = 0
        while index < len(phones):
            if index + 1 < len(phones) and phones[index] == "ɲ" and phones[index + 1] == "j":
                collapsed.append("ɲ")
                cleanup["enye_proxy_glide_removed"] += 1
                index += 2
            else:
                collapsed.append(phones[index])
                index += 1
        phones = collapsed
    return " ".join(phones), cleanup


def _strip_acute(text: str) -> str:
    return unicodedata.normalize(
        "NFC",
        "".join(
            character
            for character in unicodedata.normalize("NFD", text)
            if character != COMBINING_ACUTE
        ),
    )


def _normalized_unique_dictionary(path: Path) -> dict[str, tuple[str, ...]]:
    raw = load_mfa_dictionary(path)
    normalized: dict[str, list[str]] = defaultdict(list)
    for word, pronunciations in raw.items():
        normalized[normalize_training_text(word)].extend(pronunciations)
    return {
        word: tuple(dict.fromkeys(pronunciations))
        for word, pronunciations in normalized.items()
    }


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PLAN_FIELDS:
            raise ValueError(f"repair plan must contain fields: {PLAN_FIELDS}")
        return [dict(row) for row in reader]


def _write_plan(path: Path, rows: list[dict[str, str | int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _unresolved_row(
    word: str,
    corpus_count: int,
    proxy: str,
    rules: Iterable[str],
    detail: str,
) -> dict[str, str | int]:
    return {
        "word": word,
        "corpus_count": corpus_count,
        "proxy": proxy,
        "rules": ",".join(rules),
        "detail": detail,
    }


def _write_unresolved(path: Path, rows: list[dict[str, str | int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("word", "corpus_count", "proxy", "rules", "detail"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (-int(row["corpus_count"]), str(row["word"])),
            )
        )
    temporary.replace(path)


def _require_files(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required file does not exist: {path}")


def _require_empty_directory(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output directory must be absent or empty: {path}")


def _validate_corpus_names(
    corpora: Mapping[str, tuple[str | Path, str | Path]],
) -> None:
    if not corpora:
        raise ValueError("at least one corpus is required")
    for name in corpora:
        if (
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or any(character.isspace() for character in name)
        ):
            raise ValueError(f"invalid corpus name: {name!r}")


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sha256_manifest(directory: Path) -> None:
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "sha256.txt"
    )
    _write_text_atomic(
        directory / "sha256.txt",
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(directory)}\n" for path in paths
        ),
    )
