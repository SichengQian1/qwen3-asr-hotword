from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from qwen_hotword.training.mfa_audit import load_mfa_dictionary


def export_manifest_mfa_dictionary(
    manifest_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    language: str,
) -> dict[str, object]:
    manifests = tuple(Path(path).expanduser() for path in manifest_paths)
    if not manifests:
        raise ValueError("at least one train or validation manifest is required")
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {path}")
    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(f"manifest dictionary output already exists: {destination}")

    expected_language = _normalize_language(language)
    pronunciations: dict[str, Counter[str]] = defaultdict(Counter)
    split_counts: Counter[str] = Counter()
    row_count = 0
    word_token_count = 0
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise ValueError(f"manifest row is not an object: {manifest}:{line_number}")
                split = str(row.get("split", ""))
                if split not in {"train", "validation"}:
                    raise ValueError(
                        f"only train/validation rows are allowed: {manifest}:{line_number}"
                    )
                if _normalize_language(str(row.get("language", ""))) != expected_language:
                    raise ValueError(f"manifest language mismatch at {manifest}:{line_number}")
                raw_words = row.get("word_pronunciations")
                if not isinstance(raw_words, list) or not raw_words:
                    raise ValueError(f"missing word_pronunciations at {manifest}:{line_number}")
                for item in raw_words:
                    if not isinstance(item, Mapping) or item.get("resolution") != "exact":
                        raise ValueError(
                            f"non-exact word pronunciation at {manifest}:{line_number}"
                        )
                    word = str(item.get("word", "")).strip()
                    pronunciation = str(item.get("mfa_pronunciation", "")).strip()
                    if not word or not pronunciation:
                        raise ValueError(f"empty word or pronunciation at {manifest}:{line_number}")
                    pronunciations[word][pronunciation] += 1
                    word_token_count += 1
                split_counts[split] += 1
                row_count += 1

    selected = {
        word: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for word, counts in pronunciations.items()
    }
    ambiguous = {word: counts for word, counts in pronunciations.items() if len(counts) > 1}
    destination.mkdir(parents=True)
    dictionary_path = destination / "manifest_mfa_dictionary.dict"
    dictionary_path.write_text(
        "".join(f"{word}\t{selected[word]}\n" for word in sorted(selected)),
        encoding="utf-8",
    )
    ambiguity_path = destination / "ambiguous_words.tsv"
    with ambiguity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("word", "selected_pronunciation", "pronunciation_counts"))
        for word in sorted(ambiguous):
            writer.writerow(
                (
                    word,
                    selected[word],
                    json.dumps(dict(sorted(ambiguous[word].items())), ensure_ascii=False),
                )
            )

    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "language": expected_language,
        "manifest_count": len(manifests),
        "rows": row_count,
        "split_counts": dict(sorted(split_counts.items())),
        "word_tokens": word_token_count,
        "unique_words": len(selected),
        "ambiguous_words": len(ambiguous),
        "selection_policy": "most_frequent_manifest_mfa_pronunciation_then_lexicographic",
        "dictionary_file": dictionary_path.name,
        "dictionary_sha256": _sha256(dictionary_path),
        "test_set_used": False,
    }
    config = {
        "schema_version": 1,
        "language": expected_language,
        "inputs": [_file_identity(path) for path in manifests],
        "test_set_used": False,
    }
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "run_config.json", config)
    _write_hashes(destination)
    return summary


def export_source_mfa_dictionary(
    dictionary_paths: Sequence[str | Path],
    build_config_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    language: str,
) -> dict[str, object]:
    dictionaries = [Path(path).expanduser() for path in dictionary_paths]
    configs = [Path(path).expanduser() for path in build_config_paths]
    if not dictionaries and not configs:
        raise ValueError("at least one dictionary or full-manifest build config is required")
    for config_path in configs:
        if not config_path.is_file():
            raise FileNotFoundError(f"full-manifest build config does not exist: {config_path}")
        value = json.loads(config_path.read_text(encoding="utf-8"))
        dictionary = value.get("dictionary") if isinstance(value, Mapping) else None
        raw_path = dictionary.get("path") if isinstance(dictionary, Mapping) else None
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"build config has no dictionary.path: {config_path}")
        dictionaries.append(Path(raw_path).expanduser())
    if len({path.resolve() for path in dictionaries}) != len(dictionaries):
        raise ValueError("source MFA dictionaries must be unique")
    for dictionary in dictionaries:
        if not dictionary.is_file():
            raise FileNotFoundError(f"source MFA dictionary does not exist: {dictionary}")

    destination = Path(output_dir).expanduser()
    if destination.exists():
        raise FileExistsError(f"manifest dictionary output already exists: {destination}")
    expected_language = _normalize_language(language)
    pronunciations: dict[str, Counter[str]] = defaultdict(Counter)
    source_pronunciations = 0
    for dictionary in dictionaries:
        for word, values in load_mfa_dictionary(dictionary).items():
            for pronunciation in values:
                pronunciations[word][pronunciation] += 1
                source_pronunciations += 1
    selected = _select_pronunciations(pronunciations)
    ambiguous = _ambiguous_pronunciations(pronunciations)
    config = {
        "schema_version": 1,
        "language": expected_language,
        "input_mode": "source_mfa_dictionaries_from_full_manifest_build_configs",
        "dictionary_inputs": [_file_identity(path) for path in dictionaries],
        "build_config_inputs": [_file_identity(path) for path in configs],
        "test_set_used": False,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "language": expected_language,
        "source_dictionary_count": len(dictionaries),
        "source_dictionary_pronunciations": source_pronunciations,
        "unique_words": len(selected),
        "ambiguous_words": len(ambiguous),
        "selection_policy": "most_frequent_across_source_dictionaries_then_lexicographic",
        "test_set_used": False,
    }
    _write_outputs(destination, selected, ambiguous, config, summary)
    return summary


def _select_pronunciations(
    pronunciations: Mapping[str, Counter[str]],
) -> dict[str, str]:
    return {
        word: sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for word, counts in pronunciations.items()
    }


def _ambiguous_pronunciations(
    pronunciations: Mapping[str, Counter[str]],
) -> dict[str, Counter[str]]:
    return {word: counts for word, counts in pronunciations.items() if len(counts) > 1}


def _write_outputs(
    destination: Path,
    selected: Mapping[str, str],
    ambiguous: Mapping[str, Counter[str]],
    config: Mapping[str, object],
    summary: dict[str, object],
) -> None:
    destination.mkdir(parents=True)
    dictionary_path = destination / "manifest_mfa_dictionary.dict"
    dictionary_path.write_text(
        "".join(f"{word}\t{selected[word]}\n" for word in sorted(selected)),
        encoding="utf-8",
    )
    ambiguity_path = destination / "ambiguous_words.tsv"
    with ambiguity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("word", "selected_pronunciation", "pronunciation_counts"))
        for word in sorted(ambiguous):
            writer.writerow(
                (
                    word,
                    selected[word],
                    json.dumps(dict(sorted(ambiguous[word].items())), ensure_ascii=False),
                )
            )
    summary.update(
        {
            "dictionary_file": dictionary_path.name,
            "dictionary_sha256": _sha256(dictionary_path),
        }
    )
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "run_config.json", config)
    _write_hashes(destination)


def _normalize_language(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    aliases = {
        "en": "en",
        "en-us": "en",
        "english": "en",
        "es": "es",
        "es-419": "es",
        "es-ar": "es",
        "spanish": "es",
        "pt": "pt",
        "pt-br": "pt",
        "portuguese": "pt",
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported language: {value}")
    return aliases[normalized]


def _file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_hashes(root: Path) -> None:
    names = (
        "manifest_mfa_dictionary.dict",
        "ambiguous_words.tsv",
        "summary.json",
        "run_config.json",
    )
    (root / "sha256.txt").write_text(
        "".join(f"{_sha256(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )
