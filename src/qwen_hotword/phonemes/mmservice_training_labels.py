"""Source-aware training-label compatibility; never infer word boundaries from CTC IDs."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.phonemes.mmservice_pipeline import _dictionary
from qwen_hotword.phonemes.mmservice_smoke import _phones
from qwen_hotword.training.g2p_prep import extract_word_tokens
from qwen_hotword.training.spanish_mfa_repair import repair_spanish_pronunciation
from qwen_hotword.training.stage_progress import stage_progress

LANGUAGES = ("en", "es", "pt")
ALIASES = {"en": "en", "en-US": "en", "es": "es", "es-419": "es", "pt": "pt", "pt-BR": "pt"}


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"invalid JSON: {path}:{number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object: {path}:{number}")
            yield row


def source_key(value: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"absolute source TSV path required: {value}")
    return os.path.normpath(value)


def sample(row: dict[str, Any], vocab_size: int) -> tuple[str, list[str], list[int]]:
    lang = ALIASES.get(row["language"])
    ids = row["phoneme_token_ids"]
    if (
        lang is None
        or row.get("split") != "train"
        or row.get("balanced_language_bucket", lang) != lang
        or not isinstance(ids, list)
        or not ids
        or any(type(i) is not int or not 2 <= i < vocab_size for i in ids)
        or row.get("label_length", len(ids)) != len(ids)
        or not isinstance(row.get("id"), str)
        or not row["id"]
    ):
        raise ValueError("invalid training language/split/IDs; do not read validation/test")
    words = extract_word_tokens(row["text"])
    if not words or any(c.isdigit() for c in row["text"]):
        raise ValueError("training row contains unresolved numeric or empty text")
    return lang, words, ids


def dictionary_labels(path: Path, vocab: PhonemeVocab) -> dict[str, dict[str, Any]]:
    result = {}
    for word, candidates in _dictionary(path).items():
        result[word] = {
            **_phones(next(iter(candidates)) if len(candidates) == 1 else None, vocab),
            "pronunciation_candidates": sorted(candidates),
            "ambiguous": len(candidates) != 1,
        }
    return result


def inventory(
    manifest: Path, sources: dict[str, dict[str, Any]], vocab: PhonemeVocab, total: int
) -> dict[str, Any]:
    frequencies: Counter[tuple[str, str]] = Counter()
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    with stage_progress(
        "Scan ALL actual training text and stored labels (no WAV reads)", total=total
    ) as update:
        for number, row in enumerate(rows(manifest), 1):
            lang, words, stored = sample(row, len(vocab.tokens))
            key = source_key(row["source_tsv"])
            if key not in sources or sources[key]["language"] != lang:
                raise ValueError(f"training source TSV has no same-language build config: {key}")
            source = sources[key]
            metrics = stats[lang]
            metrics["samples"] += 1
            metrics["word_tokens"] += len(words)
            labels = [source["labels"].get(word) for word in words]
            ready = all(label is not None and label["usable"] for label in labels)
            same = ready and [i for label in labels if label for i in label["token_ids"]] == stored
            metrics["baseline_reconstructed_equal" if same else "baseline_not_verified"] += 1
            if not same and len(examples) < 20:
                examples.append(
                    {
                        "id": row["id"],
                        "language": lang,
                        "source_tsv": key,
                        "text": row["text"],
                        "dictionary_labels_available": ready,
                    }
                )
            for word in words:
                frequencies[(key, word)] += 1
            update(number)
    if sum(s["samples"] for s in stats.values()) != total:
        raise ValueError("training sample count differs from frozen expected total")
    return {"frequencies": frequencies, "stats": dict(stats), "baseline_examples": examples}


def candidates(cpp: dict[str, Any], vocab: PhonemeVocab, lang: str) -> dict[str, dict[str, Any]]:
    native = _phones(cpp["ipa"] if cpp["ok"] else None, vocab)
    cleaned = native["raw_ipa"]
    if lang == "es" and cleaned is not None:
        cleaned, _ = repair_spanish_pronunciation(cleaned, ())
    return {"native": native, "existing_es_cleanup": _phones(cleaned, vocab)}


def status(reference: dict[str, Any] | None, candidate: dict[str, Any]) -> str:
    if reference is None or not reference["usable"]:
        return "reference_unusable"
    if not candidate["usable"]:
        return "cpp_failed" if candidate["raw_ipa"] is None else "cpp_label_oov_or_empty"
    return "equal" if reference["token_ids"] == candidate["token_ids"] else "different"


def compare_words(
    frequencies: Counter[tuple[str, str]],
    sources: dict[str, dict[str, Any]],
    decoded: dict[str, dict[str, dict[str, dict[str, Any]]]],
    output: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: dict[str, Any] = {
        lang: {
            mode: {"source_word_pairs": Counter(), "word_occurrences": Counter()}
            for mode in ("native", "existing_es_cleanup")
        }
        for lang in LANGUAGES
    }
    selected: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    # Frequent differences come first, making the small returned examples useful.
    with (
        output.open("x", encoding="utf-8") as handle,
        stage_progress(
            "Compare source-specific words and occurrence weights",
            total=len(frequencies),
            every=10000,
        ) as update,
    ):
        for number, ((key, word), count) in enumerate(
            sorted(frequencies.items(), key=lambda item: (-item[1], item[0])), 1
        ):
            source = sources[key]
            lang = source["language"]
            ref = source["labels"].get(word)
            result = decoded[lang][word]
            statuses = {mode: status(ref, candidate) for mode, candidate in result.items()}
            row = {
                "language": lang,
                "source_build_config": source["build_config"],
                "source_tsv": key,
                "word": word,
                "occurrences": count,
                "saved_reference": ref,
                "cpp": result,
                "status": statuses,
            }
            for mode, value in statuses.items():
                counts[lang][mode]["source_word_pairs"][value] += 1
                counts[lang][mode]["word_occurrences"][value] += count
                if value != "equal" and len(selected[(lang, mode)]) < 20:
                    selected[(lang, mode)].append(row)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            update(number)
    # Deduplicate examples selected for both branches without merging dictionary references.
    examples = {
        (row["source_tsv"], row["word"]): row for group in selected.values() for row in group
    }
    for modes in counts.values():
        for comparison in modes.values():
            for denominator in ("source_word_pairs", "word_occurrences"):
                values = comparison[denominator]
                usable = sum(values.values()) - values["reference_unusable"]
                changed = usable - values["equal"]
                comparison[f"{denominator}_reference_usable"] = usable
                comparison[f"{denominator}_non_equal"] = changed
                comparison[f"{denominator}_non_equal_rate"] = changed / usable if usable else None
    return counts, list(examples.values())


def sentence_effect(
    manifest: Path,
    sources: dict[str, dict[str, Any]],
    decoded: dict[str, dict[str, dict[str, dict[str, Any]]]],
    vocab: PhonemeVocab,
    total: int,
    output: Path,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        lang: {"samples": 0, "hours": 0.0, "native": Counter(), "existing_es_cleanup": Counter()}
        for lang in LANGUAGES
    }
    with (
        output.open("x", encoding="utf-8") as handle,
        stage_progress(
            "Compare ALL sentence labels with actual frozen training IDs", total=total
        ) as update,
    ):
        for number, row in enumerate(rows(manifest), 1):
            lang, words, stored = sample(row, len(vocab.tokens))
            key = source_key(row["source_tsv"])
            reference = [sources[key]["labels"].get(word) for word in words]
            valid = all(r is not None and r["usable"] for r in reference)
            verified = valid and [i for r in reference if r for i in r["token_ids"]] == stored
            seconds = float(row["duration_seconds"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid stored training duration")
            metrics[lang]["samples"] += 1
            metrics[lang]["hours"] += seconds / 3600
            statuses = {}
            for mode in ("native", "existing_es_cleanup"):
                labels = [decoded[lang][word][mode] for word in words]
                available = all(label["usable"] for label in labels)
                equal = available and [i for label in labels for i in label["token_ids"]] == stored
                value = "equal" if equal else ("different" if available else "cpp_unusable")
                statuses[mode] = value
                metrics[lang][mode][value] += 1
                metrics[lang][mode][
                    f"baseline_verified_{value}" if verified else "baseline_not_verified"
                ] += 1
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "language": lang,
                        "source_corpus": row.get("source_corpus"),
                        "baseline_verified": verified,
                        "status": statuses,
                    }
                )
                + "\n"
            )
            update(number)
    for group in metrics.values():
        for mode in ("native", "existing_es_cleanup"):
            compared = group[mode]
            denominator = group["samples"]
            compared["non_equal_samples"] = denominator - compared["equal"]
            compared["non_equal_rate"] = (
                compared["non_equal_samples"] / denominator if denominator else None
            )
    return metrics
