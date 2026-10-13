"""Frozen, transcript-absent train-word distractors for the reviewed joint table."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.reviewed_targets import import_review
from qwen_hotword.evaluation.sd_keywords import seal
from qwen_hotword.evaluation.target_review import (
    LANGUAGES,
    _checked,
    combine_variants,
    compile_policy,
)
from qwen_hotword.evaluation.text import tokenize_words
from qwen_hotword.evaluation.wave_keywords import _check_language, _json, _key
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_capacity import _sha, _verified

SCOPE = "reviewed_joint_with_transcript_absent_train_fillers"


def lexical_rejection(word: str, forbidden: set[str], basic: set[str], minimum: int) -> str:
    if (
        not word.isalpha()
        or _key(word) != word
        or any("LATIN" not in unicodedata.name(c, "") for c in word)
    ):
        return "not_plain_latin_single_word"
    if len(word) < minimum:
        return "short_word"
    if word in basic:
        return "basic_word"
    if word in forbidden:
        return "present_in_any_test_transcript"
    return ""


def _reference_words(
    root: Path, cfg: dict[str, Any], identities: dict[str, Any]
) -> tuple[set[str], dict[str, int]]:
    directory = (root / cfg["reference_run"]).resolve()
    _checked(directory / "sha256.txt", identities)
    report = _json(_verified(directory, "report.json", identities))
    plan = _json(_verified(directory, "run_config.json", identities))
    reference_cfg = _json(_checked((root / cfg["reference_config"]).resolve(), identities))
    if (
        report.get("status") != "completed"
        or report.get("candidate_scope") != "joint_multilingual"
        or report["review"]["reviewed_table_sha256"] != cfg["reviewed_table_sha256"]
        or plan["configuration"] != reference_cfg
        or set(plan["groups"]) != set(reference_cfg["datasets"])
    ):
        raise ValueError("completed reviewed clean reference run identity differs")
    forbidden: set[str] = set()
    counts = {}
    for name, group in plan["groups"].items():
        source = group["config"]["sources"][0]["transcripts"]
        path = _checked(Path(source["path"]), identities, source["sha256"])
        rows = external._read_transcripts(path)
        if len(rows) != reference_cfg["datasets"][name]["expected_samples"] or rows != {
            r["sample_id"]: r["reference_text"] for r in group["records"]
        }:
            raise ValueError(f"reference sample/text identity differs: {name}")
        forbidden.update(w for text in rows.values() for w in normalize_match_words(text))
        counts[name] = len(rows)
    return forbidden, counts


def _pool(
    root: Path,
    lang: str,
    source: dict[str, Any],
    cfg: dict[str, Any],
    vocab: PhonemeVocab,
    excluded_targets: set[str],
    forbidden: set[str],
    basic: set[str],
    identities: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = _checked(
        (root / source["manifest"]).resolve(), identities, source["manifest_sha256"]
    )
    dictionary = _checked(
        (root / source["dictionary"]).resolve(), identities, source.get("dictionary_sha256")
    )
    counts: Counter[str] = Counter()
    records = 0
    with manifest.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train" or not isinstance(row.get("text"), str):
                raise ValueError(f"only train text allowed: {manifest}")
            _check_language(row.get("language"), lang, str(manifest))
            counts.update(tokenize_words(row["text"]))
            records += 1
    if not records:
        raise ValueError(f"empty training source: {lang}")
    pronunciations = load_mfa_dictionary(dictionary)
    pool: dict[str, dict[str, Any]] = {}
    decisions: Counter[str] = Counter()
    for word, count in sorted(counts.items()):
        reason = (
            "original_target_or_deleted_target"
            if _key(word) in excluded_targets
            else lexical_rejection(word, forbidden, basic, cfg["min_letters"])
        )
        if not reason and count < cfg["min_train_occurrences"]:
            reason = "low_train_frequency"
        phones = sorted(set(pronunciations.get(word, ())))
        if not reason and len(phones) != 1:
            reason = "missing_or_ambiguous_dictionary"
        if not reason:
            tokenized = tokenize_ipa_to_vocab(phones[0], vocab)
            if tokenized.oov_units or not tokenized.token_ids:
                reason = "phoneme_oov_or_empty"
            elif len(tokenized.token_ids) < cfg["min_phonemes"]:
                reason = "short_phoneme_sequence"
            else:
                pool[word] = {
                    "language": lang,
                    "surface": word,
                    "normalized": word,
                    "pronunciation": phones[0],
                    "phoneme_tokens": tokenized.tokens,
                    "token_ids": tokenized.token_ids,
                    "source_groups": [],
                    "decision": "retained",
                    "source_manifest": str(manifest),
                    "source_dictionary": str(dictionary),
                    "train_occurrences": count,
                    "role": "filler",
                }
        decisions[reason or "eligible"] += 1
    print(f"{lang}: {records} train rows; {len(pool)} eligible fillers", flush=True)
    return pool, {"train_records": records, "train_unique_words": len(counts), **decisions}


def balanced_select(
    pools: dict[str, dict[str, Any]], needed: int, seed: str
) -> list[dict[str, Any]]:
    """Equal added counts (EN/ES/PT remainder order), globally unique surfaces."""
    queues = {
        lang: iter(
            sorted(
                pool, key=lambda w: (hashlib.sha256(f"{seed}:{lang}:{w}".encode()).hexdigest(), w)
            )
        )
        for lang, pool in pools.items()
    }
    selected, used = [], set()
    for n in range(needed):
        lang = LANGUAGES[n % len(LANGUAGES)]
        word = next((w for w in queues[lang] if w not in used), None)
        if word is None:
            raise ValueError(
                f"insufficient distinct {lang} fillers for balanced quotas; "
                "no relaxation, repetition or partial table"
            )
        used.add(word)
        selected.append(pools[lang][word])
    return selected


def _table(reviewed: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, Any]:
    fillers = combine_variants(selected)
    for entry in fillers:
        entry["id"] = entry["id"].replace("target_", "filler_", 1)
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "entries": reviewed["entries"] + fillers,
    }


def build_joint_fillers(root: Path, config: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    cfg = _json(config)
    identities: dict[str, Any] = {}
    _checked(config, identities)
    for key in ("target_size", "min_letters", "min_phonemes", "min_train_occurrences"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    if set(cfg["training_sources"]) != set(LANGUAGES):
        raise ValueError("exactly en/es/pt training sources required")
    reviewed_path = _checked(
        (root / cfg["reviewed_table"]).resolve(), identities, cfg["reviewed_table_sha256"]
    )
    vocab = load_phoneme_vocab(
        _checked((root / cfg["vocab"]).resolve(), identities, cfg["vocab_sha256"])
    )
    _, _, review = import_review(reviewed_path, vocab, identities)
    reviewed = _json(reviewed_path)
    needed = cfg["target_size"] - len(reviewed["entries"])
    if needed < 0:
        raise ValueError("target_size cannot truncate reviewed targets")
    policy = _json(_checked((root / cfg["basic_word_policy"]).resolve(), identities))
    basic = {w for v in compile_policy(policy).values() for w in v["rules"]}
    forbidden, transcript_counts = _reference_words(root, cfg, identities)
    excluded_targets = {
        json.loads(line)["normalized"]
        for line in (reviewed_path.parent / "all_target_decisions.jsonl").read_text().splitlines()
        if line.strip()
    }
    pools, stats = {}, {}
    for lang in LANGUAGES:
        pools[lang], stats[lang] = _pool(
            root,
            lang,
            cfg["training_sources"][lang],
            cfg,
            vocab,
            excluded_targets,
            forbidden,
            basic,
            identities,
        )
    selected = balanced_select(pools, needed, cfg["seed"])
    table = _table(reviewed, selected)
    if len({_key(e["surface"]) for e in table["entries"]}) != cfg["target_size"]:
        raise ValueError("final unique surface count differs")
    protected = [reviewed_path.parent, (root / cfg["reference_run"]).resolve()]
    if any(output.resolve().is_relative_to(p) for p in protected) or any(
        Path(p).is_relative_to(output.resolve()) for p in identities
    ):
        raise ValueError("output must be separate from all input assets")
    for path, identity in identities.items():
        if _sha(Path(path)) != identity["sha256"]:
            raise ValueError(f"input changed while building: {path}")
    homophones: dict[tuple[int, ...], set[str]] = defaultdict(set)
    for entry in table["entries"]:
        for variant in entry["variants"]:
            homophones[tuple(variant["token_ids"])].add(entry["normalized"])
    report = {
        "status": "completed",
        "scope": SCOPE,
        "output_dir": str(output),
        "unique_surfaces": cfg["target_size"],
        "retained_target_surfaces": len(reviewed["entries"]),
        "fillers_added": needed,
        "fillers_by_language": dict(Counter(r["language"] for r in selected)),
        "target_review": review,
        "all_reviewed_targets_unchanged": True,
        "selected_filler_transcript_overlap": 0,
        "deleted_targets_reintroduced": 0,
        "transcript_counts": transcript_counts,
        "training_pools": stats,
        "inputs": identities,
        "seed": cfg["seed"],
        "predictions_used_for_selection": False,
        "model_loaded": False,
        "audio_read": False,
        "homophone_groups_retained": sum(len(v) > 1 for v in homophones.values()),
        "code_sha256": {
            _file: _sha(Path(__file__).parent / _file)
            for _file in (
                "joint_fillers.py",
                "reviewed_targets.py",
                "target_review.py",
                "text.py",
            )
        },
        "limitations": [
            "Transcript-conditioned distractor experiment, not a blind vocabulary benchmark.",
            "Absence means normalized whole-token absence in all references, not audio audit.",
            "Length/frequency/dictionary filters do not certify relevance or pronunciation.",
            "Natural near-homophones remain; no phonetic-distance selection or retrieval tuning.",
            "Approved targets unchanged, including short names; filters apply only to fillers.",
            "PT dictionary SHA is recorded on build; no prior independently pinned hash.",
        ],
        "return_files": [str(output / "report.json"), str(output / "sha256.txt")],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("config.json", cfg),
        ("report.json", report),
        ("keyword_table.json", table),
        ("reviewed_keyword_table.json", reviewed),
        ("basic_word_policy.json", policy),
        ("excluded_reference_words.json", sorted(forbidden)),
    ):
        external._write_json(output / name, value)
    external._write_jsonl(output / "selected_fillers.jsonl", selected)
    (output / "keywords.txt").write_text(
        "".join(e["surface"] + "\n" for e in table["entries"]), encoding="utf-8"
    )
    seal(output)
    return report


def load_joint_fillers(
    path: Path,
    reviewed_path: Path,
    bundles: dict[str, external.KeywordBundle],
    vocab: PhonemeVocab,
    identities: dict[str, Any],
) -> tuple[dict[str, external.KeywordBundle], dict[str, Any]]:
    """Validate the sealed expansion; targets are independently imported as deletion-only."""
    directory = path.parent
    _checked(directory / "sha256.txt", identities)
    table = _json(_verified(directory, path.name, identities))
    report = _json(_verified(directory, "report.json", identities))
    cfg = _json(_verified(directory, "config.json", identities))
    original = _json(_verified(directory, "reviewed_keyword_table.json", identities))
    policy = _json(_verified(directory, "basic_word_policy.json", identities))
    forbidden = set(_json(_verified(directory, "excluded_reference_words.json", identities)))
    selected = [
        json.loads(line)
        for line in _verified(directory, "selected_fillers.jsonl", identities)
        .read_text()
        .splitlines()
        if line.strip()
    ]
    if (
        report.get("status") != "completed"
        or report.get("scope") != SCOPE
        or cfg["reviewed_table_sha256"] != _sha(reviewed_path)
        or original != _json(reviewed_path)
        or table != _table(original, selected)
        or len(table["entries"]) != cfg["target_size"]
        or report["unique_surfaces"] != cfg["target_size"]
    ):
        raise ValueError("expanded table/review identity or schema differs")
    # Bind source identities too; a rebuild or changed transcript needs a new expansion.
    for filename, identity in report["inputs"].items():
        _checked(Path(filename), identities, identity["sha256"])
    basic = {w for v in compile_policy(policy).values() for w in v["rules"]}
    excluded = {
        json.loads(s)["normalized"]
        for s in (reviewed_path.parent / "all_target_decisions.jsonl").read_text().splitlines()
        if s.strip()
    }
    seen = set(excluded)
    entries = {lang: list(b.entries) for lang, b in bundles.items()}
    for row in selected:
        word, lang = row["surface"], row["language"]
        tokens = tokenize_ipa_to_vocab(row["pronunciation"], vocab)
        if (
            lang not in LANGUAGES
            or word in seen
            or row["normalized"] != word
            or lexical_rejection(word, forbidden, basic, cfg["min_letters"])
            or row["train_occurrences"] < cfg["min_train_occurrences"]
            or row["source_groups"] != []
            or row["role"] != "filler"
            or tokens.oov_units
            or len(tokens.token_ids) < cfg["min_phonemes"]
            or tokens.token_ids != row["token_ids"]
            or tokens.tokens != row["phoneme_tokens"]
        ):
            raise ValueError(f"invalid expanded filler: {word}")
        seen.add(word)
        entries[lang].append(
            HotwordEntry(
                hotword_id=f"filler_{lang}_" + hashlib.sha256(word.encode()).hexdigest()[:16],
                language="pt-BR" if lang == "pt" else lang,
                surface=word,
                normalized=word,
                words=normalize_match_words(word),
                pronunciation=row["pronunciation"],
                phoneme_tokens=tuple(tokens.tokens),
                token_ids=tuple(tokens.token_ids),
                source=str(path),
                validation_occurrences=0,
            )
        )
    return {
        lang: external.KeywordBundle(
            tuple(sorted(rows, key=lambda e: e.normalized)),
            {e.surface: e.pronunciation for e in rows},
            {},
        )
        for lang, rows in entries.items()
    }, report
