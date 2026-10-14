"""Frozen transcript-selected English targets replacing only old English targets."""

from __future__ import annotations

import copy
import hashlib
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import LANGUAGES, _checked, combine_variants
from qwen_hotword.evaluation.wave_keywords import _json, _key
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.hotword_prompt import normalize_match_words, strict_phrase_match
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, tokenize_ipa_to_vocab
from qwen_hotword.training.g2p_prep import normalize_training_text
from qwen_hotword.training.mfa_audit import load_mfa_dictionary
from qwen_hotword.training.spanish_capacity import _sha, _verified

NAME = "ziyan_en_noise10db"


def read_dataset(path: Path, expected: int) -> tuple[external.ExternalAudioRecord, ...]:
    data = _json(path)
    if not isinstance(data, list) or len(data) != expected:
        raise ValueError("JSON sample count/schema differs")
    records = []
    ids, paths = set(), set()
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("each JSON record must be an object")
        audios, text = row.get("audios"), row.get("response")
        if (
            not isinstance(audios, list)
            or len(audios) != 1
            or not isinstance(audios[0], str)
            or not Path(audios[0]).is_absolute()
            or not isinstance(text, str)
            or not text.strip()
            or "<asr_text>" in text
        ):
            raise ValueError("one absolute audio path and plain reference response required")
        audio = Path(audios[0]).resolve()
        if audio.suffix.lower() not in {".wav", ".flac"}:
            raise ValueError(f"unsupported audio format: {audio}")
        if audio.stem in ids or str(audio) in paths:
            raise ValueError("duplicate sample ID or audio path")
        ids.add(audio.stem)
        paths.add(str(audio))
        records.append(external.ExternalAudioRecord(audio.stem, NAME, audio, text, "en"))
    return tuple(sorted(records, key=lambda r: r.sample_id))


def read_targets(path: Path, records: tuple[external.ExternalAudioRecord, ...]) -> list[str]:
    words = path.read_text(encoding="utf-8").splitlines()
    if (
        not 300 <= len(words) <= 500
        or words != sorted(set(words))
        or any(not w.isascii() or not w.isalpha() or len(w) < 4 or w != w.lower() for w in words)
    ):
        raise ValueError("300–500 sorted distinct English content words required")
    absent = [
        w for w in words if not any(strict_phrase_match(r.reference_text, w) for r in records)
    ]
    if absent:
        raise ValueError(f"selected targets absent from references: {absent[:10]}")
    return words


def replace_targets(
    original: dict[str, Any], new: list[dict[str, Any]], expected_removed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    kept, removed = [], []
    for entry in original["entries"]:
        for variant in entry["variants"]:
            if variant["language"] == "en" and variant.get("role", "target") == "target":
                removed.append(copy.deepcopy(variant))
            else:
                kept.append(copy.deepcopy(variant))
    if len(removed) != expected_removed or len({v["normalized"] for v in removed}) != len(removed):
        raise ValueError("old English target count differs; no blanket English deletion")
    # Every old filler and every ES/PT variant survives, even if a new target shares its surface.
    table: dict[str, Any] = {
        "schema_version": 1,
        "scope": "ziyan_english_target_replacement",
        "entries": combine_variants(kept + new),
    }
    return table, {
        "removed_english_target_variants": len(removed),
        "retained_variants": len(kept),
        "retained_fillers": sum(v.get("role") == "filler" for v in kept),
        "new_english_targets": len(new),
        "unique_surfaces": len(table["entries"]),
        "new_targets_already_in_retained_table": sorted(
            {v["normalized"] for v in new} & {v["normalized"] for v in kept}
        ),
        "retained_variants_by_language": dict(Counter(v["language"] for v in kept)),
    }


def presence(
    records: tuple[external.ExternalAudioRecord, ...], targets: list[str]
) -> dict[str, Any]:
    counts = [sum(strict_phrase_match(r.reference_text, w) for w in targets) for r in records]
    denominator = sum(counts)
    return {
        "expected_target_sample_pairs": denominator,
        "samples_with_target": sum(c > 0 for c in counts),
        "samples_without_target": counts.count(0),
        "targets_per_audio": {str(k): v for k, v in sorted(Counter(counts).items())},
        "topk_capacity_upper_bound": {
            f"top{k}": sum(min(k, c) for c in counts) / denominator if denominator else None
            for k in (5, 7)
        },
    }


def build(root: Path, config: Path, output: Path, jobs: int, mfa: str) -> dict[str, Any]:
    if jobs <= 0 or output.exists():
        raise ValueError("positive jobs and a new assets directory required")
    identities: dict[str, Any] = {}
    cfg = _json(_checked(config, identities))
    dataset = _checked(Path(cfg["dataset"]), identities, cfg["dataset_sha256"])
    records = read_dataset(dataset, cfg["expected_samples"])
    target_path = _checked(root / cfg["targets"], identities, cfg["targets_sha256"])
    targets = read_targets(target_path, records)
    if len(targets) != cfg["expected_targets"]:
        raise ValueError("frozen target count differs")
    source = (root / cfg["source_run"]).resolve()
    _checked(source / "sha256.txt", identities)
    old_path = _checked(
        _verified(source, "reviewed_keyword_table.json", identities),
        identities,
        cfg["source_table_sha256"],
    )
    old_report = _json(_verified(source, "report.json", identities))
    if (
        old_report.get("status") != "completed"
        or old_report.get("candidate_scope") != "joint_multilingual"
        or old_report["review"].get("expanded_table_sha256") != cfg["source_table_sha256"]
        or old_report["review"].get("active_unique_surfaces_per_audio") != cfg["source_surfaces"]
    ):
        raise ValueError("completed original expanded 4000 table required")
    model = _checked(root / cfg["g2p_model"], identities, cfg["g2p_model_sha256"])
    vocab = load_phoneme_vocab(_checked(root / cfg["vocab"], identities, cfg["vocab_sha256"]))
    original = _json(old_path)
    if len(original["entries"]) != cfg["source_surfaces"]:
        raise ValueError("original table surface count differs")
    # Validate the old removal count before performing G2P.
    replace_targets(original, [], cfg["old_english_targets"])
    if output.is_relative_to(source) or any(
        output.is_relative_to(Path(p).parent) or Path(p).is_relative_to(output) for p in identities
    ):
        raise ValueError("assets output must be separate from all inputs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "words.txt").write_text("\n".join(targets) + "\n", encoding="utf-8")
    external._write_json(output / "config.json", cfg)
    external._write_jsonl(output / "dataset_manifest.jsonl", [r.to_dict() for r in records])
    command = [
        mfa,
        "g2p",
        "--num_jobs",
        str(jobs),
        "--num_pronunciations",
        "1",
        str(output / "words.txt"),
        str(model),
        str(output / "english_raw.dict"),
    ]
    report: dict[str, Any] = {
        "status": "running",
        "training_started": False,
        "qwen_model_loaded": False,
        "output_dir": str(output),
        "inputs": identities,
        "g2p_command": command,
        "targets": targets,
        "target_presence": presence(records, targets),
        "targets_selected_using_test_transcripts": True,
        "predictions_used_for_selection": False,
        "snr_verified": False,
    }
    external._write_json(output / "report.json", report)
    print(f"New assets: {output}; MFA G2P: {len(targets)} words", flush=True)
    try:
        with (output / "mfa.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "MFA_ROOT_DIR": str(output / "mfa_work")},
            )
        dictionary = {
            normalize_training_text(w): sorted(set(ps))
            for w, ps in load_mfa_dictionary(output / "english_raw.dict").items()
        }
        new, unresolved = [], []
        for word in targets:
            pronunciations = dictionary.get(word, [])
            if len(pronunciations) != 1:
                unresolved.append({"word": word, "reason": "missing_or_ambiguous_mfa"})
                continue
            phones = tokenize_ipa_to_vocab(pronunciations[0], vocab)
            if phones.oov_units or not phones.token_ids:
                unresolved.append({"word": word, "reason": "phoneme_oov_or_empty"})
                continue
            new.append(
                {
                    "language": "en",
                    "surface": word,
                    "normalized": word,
                    "pronunciation": pronunciations[0],
                    "phoneme_tokens": phones.tokens,
                    "token_ids": phones.token_ids,
                    "source_groups": [NAME],
                    "decision": "retained",
                    "role": "target",
                    "source": "fresh_english_us_mfa",
                }
            )
        report["unresolved"] = unresolved
        report["unexpected_mfa_words"] = sorted(set(dictionary) - set(targets))
        if unresolved:
            raise ValueError("MFA unresolved targets; no partial table or changed denominator")
        table, selection = replace_targets(original, new, cfg["old_english_targets"])
        bundle_from_table(table, output / "keyword_table.json", vocab)
        for path, identity in identities.items():
            if _sha(Path(path)) != identity["sha256"]:
                raise ValueError(f"input changed while building: {path}")
        filler_words = {
            v["normalized"]
            for e in table["entries"]
            for v in e["variants"]
            if v.get("role") == "filler"
        }
        reference_words = {w for r in records for w in normalize_match_words(r.reference_text)}
        report.update(
            status="completed",
            table_ready=True,
            selection=selection,
            retained_filler_reference_overlap=sorted(filler_words & reference_words),
        )
        external._write_json(output / "keyword_table.json", table)
        external._write_json(output / "targets.json", targets)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        report.update(status="failed", table_ready=False, error=str(error))
    report["code_sha256"] = {
        "ziyan_assets.py": _sha(Path(__file__)),
        "coverage.py": _sha(Path(__file__).parents[1] / "phonemes/coverage.py"),
    }
    external._write_json(output / "report.json", report)
    files = [p for p in output.rglob("*") if p.is_file() and "mfa_work" not in p.parts]
    (output / "sha256.txt").write_text(
        "".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in sorted(files)),
        encoding="utf-8",
    )
    return report


def bundle_from_table(table: dict[str, Any], path: Path, vocab: Any) -> tuple[Any, Any, Any]:
    from qwen_hotword.hotwords.registry import HotwordEntry
    from qwen_hotword.inference.reviewed_joint import joint_bundle

    by_lang: dict[str, list[HotwordEntry]] = {lang: [] for lang in LANGUAGES}
    for entry in table["entries"]:
        for n, v in enumerate(entry["variants"]):
            phones = tokenize_ipa_to_vocab(v["pronunciation"], vocab)
            if (
                v["normalized"] != _key(v["surface"])
                or v["normalized"] != entry["normalized"]
                or phones.token_ids != v["token_ids"]
                or phones.tokens != v["phoneme_tokens"]
                or not phones.token_ids
                or phones.oov_units
                or v["language"] not in by_lang
            ):
                raise ValueError("invalid replacement pronunciation variant")
            by_lang[v["language"]].append(
                HotwordEntry(
                    hotword_id="ziyan_"
                    + hashlib.sha256(
                        f"{entry['normalized']}:{v['language']}:{n}".encode()
                    ).hexdigest()[:20],
                    language="pt-BR" if v["language"] == "pt" else v["language"],
                    surface=v["surface"],
                    normalized=v["normalized"],
                    words=normalize_match_words(v["surface"]),
                    pronunciation=v["pronunciation"],
                    phoneme_tokens=tuple(v["phoneme_tokens"]),
                    token_ids=tuple(v["token_ids"]),
                    source=str(path),
                    validation_occurrences=0,
                )
            )
    return joint_bundle(
        {lang: external.KeywordBundle(tuple(es), {}, {}) for lang, es in by_lang.items()}
    )
