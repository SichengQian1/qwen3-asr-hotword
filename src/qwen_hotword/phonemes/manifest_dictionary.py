from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


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
