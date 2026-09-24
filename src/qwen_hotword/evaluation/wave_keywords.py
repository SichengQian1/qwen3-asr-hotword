"""Freeze fixed wave keyword tables, without audio, predictions, or model loading."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.wave_inventory import _unique_object
from qwen_hotword.hotwords.capacity_assets import _normalize_language
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.phonemes.coverage import (
    PhonemeVocab,
    load_phoneme_vocab,
    normalization_key,
    tokenize_ipa_to_vocab,
)
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.spanish_mfa_repair import repair_spanish_pronunciation

WAVES = ("wave1", "wave2", "wave3", "wave4")


class InsufficientWaveCapacity(ValueError):
    """A complete scan found too few optional words; never publish a partial table."""

    def __init__(self, message: str, summary: dict[str, Any]) -> None:
        super().__init__(message)
        self.summary = summary


def _check_language(value: Any, expected: str, context: str) -> None:
    """Use the producer's aliases for every input, without accepting unknown dialects."""
    try:
        if not isinstance(value, str) or _normalize_language(value) != _normalize_language(
            expected
        ):
            raise ValueError("different language")
    except ValueError as error:
        raise ValueError(
            f"{context} language mismatch: got {value!r}; expected {expected} "
            "or its supported capacity language alias"
        ) from error


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def _key(surface: Any) -> str:
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("missing/invalid surface")
    key = " ".join(normalize_match_words(surface))
    if not key:
        raise ValueError("empty normalized surface")
    return key


def _entry(surface: Any, phone: Any, lang: str, vocab: PhonemeVocab) -> dict[str, Any]:
    key = _key(surface)
    if not isinstance(phone, str) or not phone.strip():
        raise ValueError("missing/non-string pronunciation")
    cleaned, cleanup = repair_spanish_pronunciation(phone, ()) if lang == "es" else (phone, {})
    tokens = tokenize_ipa_to_vocab(cleaned, vocab)
    if tokens.oov_units or not tokens.token_ids:
        raise ValueError(f"OOV/empty pronunciation: {tokens.oov_units}")
    # Serialize vocabulary tokens, not a partial tokenizer result or original OOV IPA.
    canonical = [vocab.tokens[i] for i in tokens.token_ids]
    return {
        "surface": surface,
        "normalized": key,
        "language": lang,
        "words": list(normalize_match_words(surface)),
        "pronunciation": " ".join(canonical),
        "phoneme_tokens": canonical,
        "token_ids": tokens.token_ids,
        "original_pronunciation": phone,
        "cleaned_pronunciation": cleaned,
        "cleanup": dict(cleanup),
    }


def _old_entry_issues(row: dict[str, Any], vocab: PhonemeVocab) -> list[str]:
    """Compare identities in their own domains; legacy surface punctuation is valid."""
    issues = []
    try:
        # Capacity assets preserve intra-word apostrophes/hyphens. External matching
        # separates punctuation. Compare both surfaces under the same matching rule.
        if _key(row.get("surface")) != _key(row.get("normalized")):
            issues.append("normalized_surface_mismatch")
    except ValueError:
        issues.append("invalid_surface_or_normalized")
    phone = row.get("pronunciation")
    if not isinstance(phone, str) or not phone.strip():
        issues.append("missing_pronunciation")
        return issues
    original = tokenize_ipa_to_vocab(phone, vocab)
    if original.oov_units or not original.token_ids:
        issues.append("pronunciation_oov_or_empty")
    ids = row.get("token_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(i) is not int for i in ids)
        or ids != original.token_ids
    ):
        issues.append("pronunciation_token_ids_mismatch")
    phones = row.get("phoneme_tokens")
    if (
        not isinstance(phones, list)
        or not phones
        or any(not isinstance(p, str) for p in phones)
        or [normalization_key(p) for p in phones] != [normalization_key(p) for p in original.tokens]
    ):
        # Match registry._entry_from_dict: NFC/NFD-equivalent phones are identical.
        issues.append("phoneme_tokens_mismatch")
    return issues


def build_language(
    lang: str,
    primary: dict[str, Any],
    neighbors: dict[str, Any],
    fillers: list[dict[str, Any]],
    vocab: PhonemeVocab,
    *,
    target_size: int,
    seed: str,
    expected_primary: int,
    filler_policy: str = "mixed_neighbors",
) -> dict[str, Any]:
    """Mandatory targets first; conflicting optional pronunciations are quarantined."""
    if lang not in {"es", "pt"} or target_size < 1:
        raise ValueError("invalid language/target size")
    if filler_policy not in {"mixed_neighbors", "old_only_exclude_supplied"}:
        raise ValueError("unknown filler policy")
    if set(primary) != set(WAVES) or set(neighbors) != set(WAVES):
        raise ValueError("all four waves are required")
    mandatory: dict[str, dict[str, Any]] = {}
    by_wave: dict[str, list[str]] = {}
    audit: list[dict[str, Any]] = []
    pools: dict[str, dict[str, dict[str, Any]]] = {"neighbor": {}, "old_table": {}}
    conflicts: set[str] = set()
    supplied_neighbors: set[str] = set()

    for wave in WAVES:
        raw = primary[wave]
        if not isinstance(raw, dict):
            raise ValueError(f"{wave}: primary must be an object")
        declared_language = raw.get("language", lang)
        _check_language(declared_language, lang, f"{wave}/{lang}: primary")
        sets, phones = raw.get("keyword_sets"), raw.get("keyword_phonemes")
        if not isinstance(sets, dict) or not isinstance(phones, dict):
            raise ValueError(f"{wave}: invalid primary schema")
        surfaces: set[str] = set()
        for entries in sets.values():
            if not isinstance(entries, list) or any(not isinstance(s, str) for s in entries):
                raise ValueError(f"{wave}: invalid keyword set")
            surfaces.update(entries)
        if not surfaces:
            raise ValueError(f"{wave}: empty primary table")
        by_wave[wave] = sorted({_key(s) for s in surfaces})
        for surface in sorted(surfaces):
            try:
                entry = _entry(surface, phones.get(surface), lang, vocab)
            except ValueError as error:
                raise ValueError(f"mandatory {wave}/{lang}/{surface}: {error}") from error
            key = entry["normalized"]
            origin = {"source": "primary", "wave": wave, **entry}
            audit.append({**origin, "decision": "mandatory"})
            if key in mandatory and mandatory[key]["token_ids"] != entry["token_ids"]:
                raise ValueError(f"mandatory pronunciation conflict: {lang}/{key}")
            if key not in mandatory:
                mandatory[key] = {**entry, "source": "primary", "origins": []}
            mandatory[key]["origins"].append(origin)
    if len(mandatory) != expected_primary or len(mandatory) > target_size:
        raise ValueError(f"primary union count mismatch: {lang}={len(mandatory)}")

    def add_optional(surface: Any, phone: Any, source: str, origin: dict[str, Any]) -> None:
        record = {"source": source, "surface": surface, "original_pronunciation": phone, **origin}
        try:
            entry = _entry(surface, phone, lang, vocab)
        except ValueError as error:
            audit.append({**record, "decision": "excluded_invalid", "reason": str(error)})
            return
        key = entry["normalized"]
        record.update(entry)
        if key in mandatory:
            audit.append(
                {
                    **record,
                    "decision": "mandatory_precedence",
                    "same_tokens": mandatory[key]["token_ids"] == entry["token_ids"],
                }
            )
            return
        existing = [pool[key] for pool in pools.values() if key in pool]
        if any(e["token_ids"] != entry["token_ids"] for e in existing):
            conflicts.add(key)
        pool = pools[source]
        if key not in pool:
            pool[key] = {**entry, "source": source, "origins": []}
        pool[key]["origins"].append(record)
        audit.append({**record, "decision": "candidate"})

    ignored_parents = 0
    for wave in WAVES:
        raw = neighbors[wave]
        if not isinstance(raw, dict):
            raise ValueError(f"{wave}/{lang}: neighbor must be an object")
        _check_language(raw.get("language"), lang, f"{wave}/{lang}: neighbor")
        mapping = raw.get("neighbors")
        if not isinstance(mapping, dict):
            raise ValueError(f"{wave}: expected neighbors[parent] lists")
        for parent, candidates in sorted(mapping.items()):
            if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
                raise ValueError(f"{wave}/{parent}: malformed neighbor list")
            for item in candidates:
                supplied_neighbors.add(_key(item.get("word")))
            if filler_policy != "mixed_neighbors":
                # All supplied neighbor surfaces are excluded, regardless of parent or IPA.
                continue
            if _key(parent) not in mandatory:
                ignored_parents += 1
                continue
            for item in candidates:
                add_optional(
                    item.get("word"),
                    item.get("phoneme"),
                    "neighbor",
                    {
                        "wave": wave,
                        "parent": parent,
                        "supplied_similarity": item.get("sim"),
                        "supplied_count": item.get("count"),
                    },
                )
    # Scan all legacy language declarations first. The capacity producer retains
    # base-entry tags and train-candidate tags; pt and pt-BR may coexist legitimately.
    language_counts: Counter[str] = Counter()
    language_issues = []
    for index, row in enumerate(fillers):
        value = row.get("language") if isinstance(row, dict) else None
        language_counts[str(value)] += 1
        try:
            _check_language(value, lang, f"old table {lang} row {index + 1}")
        except ValueError as error:
            language_issues.append(str(error))
    if language_issues:
        raise ValueError(
            f"old table language audit failed: {len(language_issues)} rows; "
            f"language_counts={dict(language_counts)}; examples={language_issues[:10]}"
        )
    for index, row in enumerate(fillers):
        origin = {
            "old_hotword_id": row.get("hotword_id"),
            "old_language": row["language"],
            "old_row_number": index + 1,
            "old_normalized": row.get("normalized"),
            "old_phoneme_tokens": row.get("phoneme_tokens"),
            "old_token_ids": row.get("token_ids"),
        }
        # Old entries are optional fillers. Audit every bad row, never silently
        # accept incompatible IDs or stop the entire scan at the first bad filler.
        issues = _old_entry_issues(row, vocab)
        if issues:
            audit.append(
                {
                    **origin,
                    "source": "old_table",
                    "surface": row.get("surface"),
                    "original_pronunciation": row.get("pronunciation"),
                    "decision": "excluded_invalid",
                    "reasons": issues,
                }
            )
            continue
        add_optional(row["surface"], row["pronunciation"], "old_table", origin)

    excluded_supplied = set()
    if filler_policy == "old_only_exclude_supplied":
        excluded_supplied = set(pools["old_table"]) & supplied_neighbors
        for key in excluded_supplied:
            pools["old_table"].pop(key)
    for key in conflicts:
        for pool in pools.values():
            pool.pop(key, None)
    selected = dict(mandatory)

    def ranked(pool: dict[str, Any]) -> list[str]:
        return sorted(
            pool, key=lambda k: (hashlib.sha256(f"{seed}:{lang}:{k}".encode()).hexdigest(), k)
        )

    def take(source: str, limit: int) -> None:
        count = 0
        for key in ranked(pools[source]):
            if count >= limit or len(selected) >= target_size:
                break
            if key not in selected:
                selected[key] = pools[source][key]
                count += 1

    neighbor_target = (
        (target_size - len(mandatory)) // 2 if filler_policy == "mixed_neighbors" else 0
    )
    take("neighbor", neighbor_target)
    take("old_table", target_size - len(selected))
    if filler_policy == "mixed_neighbors":
        take("neighbor", target_size - len(selected))  # never duplicate a surface
    if len(selected) != target_size:
        rejected = Counter(reason for r in audit for reason in r.get("reasons", []))
        raise InsufficientWaveCapacity(
            f"insufficient valid unique words: {lang}={len(selected)}/{target_size}; "
            f"old_table_issues={dict(rejected)}; optional_conflicts={len(conflicts)}; "
            f"missing={target_size - len(selected)}",
            {
                "status": "insufficient_capacity",
                "mandatory": len(mandatory),
                "available_total": len(selected),
                "target_size": target_size,
                "missing": target_size - len(selected),
                "eligible_optional_by_source": {k: len(v) for k, v in pools.items()},
                "excluded_old_supplied_neighbor_surfaces": len(excluded_supplied),
                "mandatory_supplied_neighbor_overlap": len(set(mandatory) & supplied_neighbors),
                "old_table_issues": dict(rejected),
                "optional_conflicts": len(conflicts),
            },
        )
    for record in audit:
        key = record.get("normalized")
        if record["decision"] == "candidate":
            record["decision"] = (
                "excluded_supplied_neighbor"
                if key in excluded_supplied
                else "excluded_conflict"
                if key in conflicts
                else "selected_origin"
                if key in selected and selected[key]["source"] == record["source"]
                else "not_selected"
            )
    entries = []
    homophones: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for key, entry in sorted(selected.items()):
        entry = {
            **entry,
            "hotword_id": f"wave_{lang}_{hashlib.sha256(key.encode()).hexdigest()[:20]}",
        }
        entries.append(entry)
        homophones[tuple(entry["token_ids"])].append(key)
    return {
        "entries": entries,
        "audit": audit,
        "targets_by_wave": by_wave,
        "summary": {
            "total": len(entries),
            "filler_policy": filler_policy,
            "excluded_old_supplied_neighbor_surfaces": len(excluded_supplied),
            "supplied_neighbor_unique_surfaces": len(supplied_neighbors),
            "selected_optional_supplied_neighbor_overlap": len(
                (set(selected) - set(mandatory)) & supplied_neighbors
            ),
            "mandatory_supplied_neighbor_overlap": len(set(mandatory) & supplied_neighbors),
            "mandatory": len(mandatory),
            "selected_by_source": dict(Counter(e["source"] for e in entries)),
            "neighbor_target": neighbor_target,
            "eligible_optional_by_source": {k: len(v) for k, v in pools.items()},
            "ignored_non_target_parent_occurrences": ignored_parents,
            "optional_conflicting_surfaces": len(conflicts),
            "audit_decisions": dict(Counter(r["decision"] for r in audit)),
            "old_table_language_counts": dict(language_counts),
            "old_table_validation_issue_counts": dict(
                Counter(
                    reason
                    for r in audit
                    if r["source"] == "old_table"
                    for reason in r.get("reasons", [])
                )
            ),
            "old_table_validation_issue_examples": [
                r for r in audit if r["source"] == "old_table" and r.get("reasons")
            ][:5],
            "old_table_surface_normalization_updates": sum(
                r["source"] == "old_table"
                and "normalized" in r
                and r["normalized"] != r.get("old_normalized")
                for r in audit
            ),
            "primary_cleanup_occurrences": dict(
                sum((Counter(r["cleanup"]) for r in audit if r["source"] == "primary"), Counter())
            ),
            "homophone_groups_retained": sum(len(v) > 1 for v in homophones.values()),
            "homophone_examples": [v for v in homophones.values() if len(v) > 1][:10],
            "all_targets_retained": set(mandatory) <= set(selected),
            "oov_count": 0,
        },
    }


def freeze_wave_keywords(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    """Verify inventoried input identities; publish only after both languages succeed."""
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    identities: dict[str, Any] = {}

    def checked(path: Path, expected: str | None = None) -> Path:
        digest = _sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        identities[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
        return path

    config = _json(checked(config_path.resolve()))
    inventory_path = _verified(root / config["inventory_dir"], "report.json", identities)
    inventory = _json(inventory_path)
    vocab_path = checked(root / config["vocab"], config["vocab_sha256"])
    if inventory["vocab_sha256"] != config["vocab_sha256"]:
        raise ValueError("inventory vocab identity mismatch")
    vocab = load_phoneme_vocab(vocab_path)
    built = {}
    shortages = {}
    for lang in ("es", "pt"):
        inputs: dict[str, dict[str, Any]] = {
            "keyword_bias_phoneme": {},
            "phonetic_neighbors_phoneme": {},
        }
        for wave in WAVES:
            for kind in inputs:
                info = inventory["json_files"][f"{wave}/{lang}/{kind}"]
                path = checked(root / wave / lang / f"{lang}_{kind}.json", info["sha256"])
                inputs[kind][wave] = _json(path)
        old_path = root / (
            "outputs/en_es_pt_streaming_e2e_4k_formal100_v1/"
            f"capacity_{lang}/representative/size_4000/hotwords.jsonl"
        )
        checked(old_path, config["filler_sha256"][lang])
        fillers = [
            json.loads(s, object_pairs_hook=_unique_object)
            for s in old_path.read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        try:
            built[lang] = build_language(
                lang,
                inputs["keyword_bias_phoneme"],
                inputs["phonetic_neighbors_phoneme"],
                fillers,
                vocab,
                target_size=config["target_size"],
                seed=config["seed"],
                expected_primary=config["expected_primary_counts"][lang],
                filler_policy=config.get("filler_policy", "mixed_neighbors"),
            )
        except InsufficientWaveCapacity as error:
            shortages[lang] = error.summary
    for input_name, identity in identities.items():
        if _sha(Path(input_name)) != identity["sha256"]:
            raise ValueError(f"input changed during build: {input_name}")
    if shortages:
        return {
            "status": "insufficient_capacity",
            "tables_created": False,
            "languages": {**{k: v["summary"] for k, v in built.items()}, **shortages},
            "inputs": identities,
            "files_written": False,
            "model_loaded": False,
            "audio_read": False,
        }
    # No output is opened until validation and selection pass for both languages.
    output.mkdir(parents=True, exist_ok=False)

    def write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_json(output / "config.json", config)
    for lang, result in built.items():
        directory = output / lang
        directory.mkdir()
        entries = result["entries"]
        write_json(
            directory / "keyword_bias_phoneme.json",
            {
                "language": lang,
                "keyword_sets": {"all_keywords": [e["surface"] for e in entries]},
                "keyword_phonemes": {e["surface"]: e["pronunciation"] for e in entries},
            },
        )
        write_json(directory / "targets_by_wave.json", result["targets_by_wave"])
        for name, records in (
            ("hotwords.jsonl", entries),
            ("selection_audit.jsonl", result["audit"]),
        ):
            (directory / name).write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
                ),
                encoding="utf-8",
            )
    report = {
        "status": "completed",
        "tables_created": True,
        "output_dir": str(output),
        "inputs": identities,
        "languages": {k: v["summary"] for k, v in built.items()},
        "policy": (
            (
                "all_targets; old_table_fill_only; "
                "exclude_all_supplied_neighbor_surfaces_except_targets"
            )
            if config.get("filler_policy") == "old_only_exclude_supplied"
            else (
                "all_targets; half_remaining_neighbors_of_targets; "
                "old_table_fill; neighbor_fallback"
            )
        ),
        "seed": config["seed"],
        "model_loaded": False,
        "audio_read": False,
        "transcripts_or_predictions_used_for_selection": False,
        "limitations": [
            "Vocabulary compatibility is not pronunciation accuracy certification.",
            "Similarity values are supplied provenance, not recomputed after ES cleanup.",
            "Homophones with distinct surfaces are retained, not collapsed.",
            "Old-table fillers may have natural phonetic similarities; no acoustic filtering.",
            "Top5/Top7 inference and 16 downstream exports are separate work.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    write_json(output / "report.json", report)
    files = sorted(p for p in output.rglob("*") if p.is_file())
    (output / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in files), encoding="utf-8"
    )
    return report
