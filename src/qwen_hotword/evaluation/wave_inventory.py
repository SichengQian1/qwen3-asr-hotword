"""Read wave inputs without guessing a new keyword/neighbor schema or running models."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.inference.external_keyword_retrieval import _read_transcripts
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.spanish_capacity import _sha


def preview(value: Any, depth: int = 0) -> Any:
    """Bound console/report size even for large nested neighbor maps."""
    if isinstance(value, dict):
        return {
            "type": "object",
            "count": len(value),
            "keys_first20": list(value)[:20],
            "examples": {k: preview(v, depth + 1) for k, v in list(value.items())[:2]}
            if depth < 4
            else {},
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "count": len(value),
            "examples": [preview(v, depth + 1) for v in value[:2]] if depth < 4 else [],
        }
    if isinstance(value, str):
        return {"type": "string", "length": len(value), "example": value[:180]}
    return {"type": type(value).__name__, "example": value}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key[:80]}")
        result[key] = value
    return result


def inspect_waves(root: Path, vocab_path: Path) -> dict[str, Any]:
    vocab = load_phoneme_vocab(vocab_path)
    files: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    unions: dict[str, dict[str, list[dict[str, Any]]]] = {
        lang: defaultdict(list) for lang in ("es", "pt")
    }
    wave_stems: dict[str, dict[str, list[str]]] = {lang: defaultdict(list) for lang in ("es", "pt")}
    parsed_primary: Counter[str] = Counter()
    for wave in ("wave1", "wave2", "wave3", "wave4"):
        for lang in ("es", "pt"):
            base = root / wave / lang
            group = f"{wave}/{lang}"
            for kind in ("keyword_bias_phoneme", "phonetic_neighbors_phoneme"):
                path = base / f"{lang}_{kind}.json"
                info: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
                files[f"{group}/{kind}"] = info
                try:
                    digest = _sha(path)
                    raw = json.loads(
                        path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object
                    )
                    info.update(
                        sha256=digest, size_bytes=path.stat().st_size, structure=preview(raw)
                    )
                    # Also show each root field: neighbor data may follow two metadata fields.
                    if isinstance(raw, dict):
                        info["root_fields"] = {k: preview(v, 1) for k, v in list(raw.items())[:12]}
                    if kind == "keyword_bias_phoneme":
                        sets, phones = raw.get("keyword_sets"), raw.get("keyword_phonemes")
                        if not isinstance(sets, dict) or not isinstance(phones, dict):
                            raise ValueError("primary schema needs review: keyword_sets/phonemes")
                        surfaces: list[str] = []
                        for name, entries in sets.items():
                            if not isinstance(entries, list) or any(
                                not isinstance(s, str) or not s.strip() for s in entries
                            ):
                                raise ValueError(f"invalid keyword set {name}")
                            surfaces.extend(entries)
                        info["set_counts"] = {k: len(v) for k, v in sets.items()}
                        # Inventory all declared sets; never select an empty baseline by default.
                        info["declared_surfaces"] = len(set(surfaces))
                        for surface in sorted(set(surfaces)):
                            key = " ".join(normalize_match_words(surface))
                            phone = phones.get(surface)
                            if not key or not isinstance(phone, str) or not phone.strip():
                                raise ValueError(f"missing surface/phoneme: {surface[:80]}")
                            tokens = tokenize_ipa_to_vocab(phone, vocab)
                            if tokens.oov_units or not tokens.token_ids:
                                raise ValueError(f"OOV/empty phonemes: {surface[:80]}")
                            unions[lang][key].append(
                                {
                                    "wave": wave,
                                    "surface": surface,
                                    "phoneme": phone,
                                    "token_ids": tokens.token_ids,
                                }
                            )
                        parsed_primary[lang] += 1
                    if _sha(path) != digest:
                        raise ValueError("input changed during inspection")
                except (OSError, ValueError, AttributeError) as error:
                    info["error"] = str(error)
                    issues.append({"input": str(path), "error": str(error)})
            audio = base / "wav"
            transcripts = base / "transcripts.txt"
            data: dict[str, Any] = {"audio_dir": str(audio), "transcripts": str(transcripts)}
            datasets[group] = data
            try:
                if not audio.is_dir():
                    raise ValueError("missing wav directory")
                names = Counter(
                    p.stem
                    for p in audio.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".wav", ".flac"}
                )
                refs = _read_transcripts(transcripts)
                data.update(
                    audio_count=sum(names.values()),
                    transcript_count=len(refs),
                    transcripts_sha256=_sha(transcripts),
                    duplicate_stem_count=sum(n > 1 for n in names.values()),
                    audio_without_reference=len(names.keys() - refs.keys()),
                    reference_without_audio=len(refs.keys() - names.keys()),
                    transcript_preview=[(k, v[:180]) for k, v in list(refs.items())[:2]],
                )
                for stem in names:
                    wave_stems[lang][stem].append(wave)
                if (
                    not names
                    or data["duplicate_stem_count"]
                    or (data["audio_without_reference"] or data["reference_without_audio"])
                ):
                    raise ValueError("audio/transcript identity mismatch")
            except (OSError, ValueError) as error:
                data["error"] = str(error)
                issues.append({"input": group, "error": str(error)})
    union_summary = {}
    for lang, words in unions.items():
        conflicts = {
            key: entries
            for key, entries in words.items()
            if len({tuple(e["token_ids"]) for e in entries}) > 1
        }
        union_summary[lang] = {
            "complete_primary_files": parsed_primary[lang],
            "counts_are_partial": parsed_primary[lang] != 4,
            "normalized_union_count": len(words),
            "remaining_to_4000": max(0, 4000 - len(words)),
            "phoneme_conflict_count": len(conflicts),
            "phoneme_conflict_examples": dict(list(conflicts.items())[:5]),
            "cross_wave_shared_stem_count": sum(len(v) > 1 for v in wave_stems[lang].values()),
        }
        if conflicts or len(words) > 4000:
            issues.append({"input": lang, "error": "union conflict or mandatory union exceeds4000"})
    fillers = {}
    for lang in ("es", "pt"):
        path = root / (
            "outputs/en_es_pt_streaming_e2e_4k_formal100_v1/"
            f"capacity_{lang}/representative/size_4000/hotwords.jsonl"
        )
        info = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            try:
                lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
                info.update(sha256=_sha(path), records=len(lines), examples=lines[:2])
            except (OSError, ValueError) as error:
                info["error"] = str(error)
                issues.append({"input": str(path), "error": str(error)})
        fillers[lang] = info
    return {
        "status": "inventory_completed",
        "root": str(root),
        "scope": "input_schema_and_primary_union_inventory_not_4000_table_release",
        "vocab_sha256": _sha(vocab_path),
        "primary_unions": union_summary,
        "datasets": datasets,
        "json_files": files,
        "old_4000_tables": fillers,
        "issues": issues,
        "model_loaded": False,
        "audio_bytes_read": False,
        "tables_created": False,
        "next": "Review actual neighbor schema, then freeze language-specific 4000 tables.",
    }
