"""Three English target sets plus train-only, transcript-absent single-word fillers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.text import tokenize_words
from qwen_hotword.evaluation.wave_keywords import _check_language, _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_capacity import _sha

DATASETS = ("en_chile", "en_colombia", "en_mexico")


def checked(path: Path, identities: dict[str, Any], expected: str | None = None) -> Path:
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch: {path}; actual={digest}")
    identities[str(path)] = {"sha256": digest, "size_bytes": path.stat().st_size}
    return path


def read_targets(
    root: Path, cfg: dict[str, Any], vocab: PhonemeVocab, identities: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, dict[str, str]], list[dict[str, str]]]:
    if set(cfg["sources"]) != set(DATASETS):
        raise ValueError("expected exactly en_chile, en_colombia and en_mexico")
    union: dict[str, Any] = {}
    targets: dict[str, list[str]] = {}
    transcripts: dict[str, dict[str, str]] = {}
    aliases = []
    for name in DATASETS:
        source = cfg["sources"][name]
        primary = checked(root / source["primary"], identities)
        # Reject duplicate JSON keys instead of silently losing a supplied pronunciation.
        _json(primary)
        bundle = external.load_keyword_bias(
            primary, vocab=vocab, keyword_set=cfg["primary_keyword_set"], language="en"
        )
        targets[name] = sorted(e.normalized for e in bundle.entries)
        for entry in bundle.entries:
            previous = union.get(entry.normalized)
            if previous is not None:
                if previous.token_ids != entry.token_ids:
                    raise ValueError(
                        f"conflicting target pronunciation: {entry.surface!r} in {name}; "
                        "all targets are mandatory; resolve the source labels first"
                    )
                if previous.surface != entry.surface:
                    aliases.append(
                        {
                            "dataset": name,
                            "surface": entry.surface,
                            "canonical_surface": previous.surface,
                        }
                    )
            else:
                union[entry.normalized] = entry
        text_path = checked(root / source["transcripts"], identities)
        transcripts[name] = external._read_transcripts(text_path)
    return union, targets, transcripts, aliases


def transcript_words(transcripts: dict[str, dict[str, str]]) -> set[str]:
    return {
        word
        for group in transcripts.values()
        for text in group.values()
        for word in normalize_match_words(text)
    }


def build_english_table(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    cfg = _json(config_path)
    size = cfg["target_size"]
    if type(size) is not int or size <= 0:
        raise ValueError("target_size must be a positive integer")
    identities: dict[str, Any] = {}
    checked(config_path, identities)
    vocab = load_phoneme_vocab(checked(root / cfg["vocab"], identities, cfg["vocab_sha256"]))
    union, targets, transcripts, aliases = read_targets(root, cfg, vocab, identities)
    if len(union) > size:
        raise ValueError(f"mandatory union {len(union)} exceeds target size {size}; no truncation")
    forbidden = transcript_words(transcripts)
    train = checked(root / cfg["training_manifest"], identities, cfg["train_sha256"])
    dictionary_path = checked(root / cfg["dictionary"], identities, cfg["dictionary_sha256"])
    counts: Counter[str] = Counter()
    records = 0
    with train.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"non-train row at {train}:{line_number}")
            _check_language(row.get("language"), "en", f"train row {line_number}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty train text at row {line_number}")
            counts.update(tokenize_words(text))
            records += 1
            if records % 50000 == 0:
                print(f"English training text checked: {records}", flush=True)
    if records == 0:
        raise ValueError("empty training manifest")
    dictionary = load_mfa_dictionary(dictionary_path)
    candidates: dict[str, str] = {}
    decisions: Counter[str] = Counter()
    rejected = []
    for word in sorted(counts):
        reason = ""
        if word in union:
            reason = "mandatory_precedence"
        elif not re.fullmatch(r"[a-z]{2,}", word):
            reason = "not_plain_english_single_word"
        elif word in forbidden:
            reason = "present_in_any_test_transcript"
        else:
            pronunciations = sorted(set(dictionary.get(word, ())))
            if len(pronunciations) != 1:
                reason = "dictionary_missing_or_ambiguous"
            else:
                phone = pronunciations[0]
                tokenized = tokenize_ipa_to_vocab(phone, vocab)
                if tokenized.oov_units or not tokenized.token_ids:
                    reason = "phoneme_oov_or_empty"
                else:
                    candidates[word] = phone
        decisions[reason or "eligible"] += 1
        if reason:
            rejected.append({"word": word, "train_occurrences": counts[word], "reason": reason})
    needed = size - len(union)
    if len(candidates) < needed:
        raise ValueError(
            f"insufficient transcript-absent train fillers: need={needed}, "
            f"eligible={len(candidates)}, decisions={dict(decisions)}; no table published"
        )
    selected = sorted(
        candidates, key=lambda w: (hashlib.sha256(f"{cfg['seed']}:{w}".encode()).hexdigest(), w)
    )[:needed]
    ordered_targets = [union[key] for key in sorted(union)]
    table = {
        "language": "English",
        "keyword_sets": {"all_keywords": [e.surface for e in ordered_targets] + selected},
        "keyword_phonemes": {
            **{e.surface: e.pronunciation for e in ordered_targets},
            **{word: candidates[word] for word in selected},
        },
        "selection_rule": "all_three_target_unions_plus_transcript_absent_train_words",
    }
    if set(selected) & forbidden or set(selected) & set(union):
        raise ValueError("internal filler exclusion failure")
    for filename, identity in identities.items():
        if _sha(Path(filename)) != identity["sha256"]:
            raise ValueError(f"input changed during build: {filename}")
    output.mkdir(parents=True, exist_ok=False)
    external._write_json(output / "keyword_bias_phoneme.json", table)
    final = external.load_keyword_bias(
        output / "keyword_bias_phoneme.json", vocab=vocab, keyword_set="all_keywords", language="en"
    )
    if len(final.entries) != size:
        raise ValueError("final table count mismatch")
    external._write_json(output / "targets_by_dataset.json", targets)
    external._write_json(output / "targets.json", sorted(union))
    external._write_jsonl(
        output / "selection.jsonl",
        [
            {
                "surface": e.surface,
                "normalized": e.normalized,
                "source": "mandatory_target",
                "datasets": [name for name in DATASETS if e.normalized in targets[name]],
            }
            for e in ordered_targets
        ]
        + [
            {
                "surface": word,
                "normalized": word,
                "source": "english_train_dictionary",
                "train_occurrences": counts[word],
            }
            for word in selected
        ],
    )
    external._write_jsonl(output / "filler_rejections.jsonl", rejected)
    homophones: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for entry in final.entries:
        homophones[entry.token_ids].append(entry.surface)
    report = {
        "status": "completed",
        "tables_created": True,
        "output_dir": str(output),
        "total": size,
        "mandatory_union": len(union),
        "targets_by_dataset": {name: len(values) for name, values in targets.items()},
        "transcript_counts": {name: len(rows) for name, rows in transcripts.items()},
        "training_records": records,
        "training_unique_words": len(counts),
        "fillers": needed,
        "eligible_fillers": len(candidates),
        "filler_decisions": dict(decisions),
        "normalization_aliases": aliases,
        "selected_filler_transcript_overlap": 0,
        "all_targets_retained": True,
        "oov_count": 0,
        "homophone_groups_retained": sum(len(v) > 1 for v in homophones.values()),
        "inputs": identities,
        "seed": cfg["seed"],
        "filler_policy": "unique ASCII single words of length >=2; exact existing MFA dictionary",
        "exclusion_policy": "NFKC+casefold+punctuation to spaces; whole token in any transcript",
        "table_filler_exclusion_used_transcripts": True,
        "predictions_used": False,
        "audio_read": False,
        "model_loaded": False,
        "limitations": [
            "This is a transcript-conditioned distractor experiment, not a blind vocabulary test.",
            "Absence refers to normalized supplied transcripts, not verified spoken audio.",
            "Fillers are single words; target phrases and all supplied target labels are retained.",
            "Natural homophones are retained; vocabulary coverage does not certify pronunciation.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    external._write_json(output / "report.json", report)
    seal(output)
    return report
