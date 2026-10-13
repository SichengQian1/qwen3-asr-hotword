"""Read sealed joint retrieval artifacts without reopening models or audio."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.target_review import LANGUAGES, _checked
from qwen_hotword.evaluation.wave_keywords import _json, _key
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.external_keyword_retrieval import KeywordBundle
from qwen_hotword.inference.external_keyword_topk_replay import (
    _validated_source_gate,
    replay_rows_top7,
)
from qwen_hotword.inference.hotword_prompt import normalize_match_words
from qwen_hotword.inference.reviewed_joint import collapse_variants, joint_bundle
from qwen_hotword.inference.wave_retrieval import _row_digest
from qwen_hotword.training.spanish_capacity import _verified


def load_run(directory: Path, identities: dict[str, Any]) -> dict[str, Any]:
    _checked(directory / "sha256.txt", identities)
    plan = _json(_verified(directory, "run_config.json", identities))
    report = _json(_verified(directory, "report.json", identities))
    table = _json(_verified(directory, "reviewed_keyword_table.json", identities))
    if report.get("status") != "completed" or report.get("candidate_scope") != "joint_multilingual":
        raise ValueError("completed joint multilingual runs required")
    indexed = _verified(directory, "indexed_pronunciations.jsonl", identities)
    rows = [json.loads(s) for s in indexed.read_text().splitlines() if s.strip()]
    entries = []
    table_variants = {
        (e["normalized"], tuple(v["token_ids"])) for e in table["entries"] for v in e["variants"]
    }
    for row in rows:
        row = {**row, **{k: tuple(row[k]) for k in ("words", "phoneme_tokens", "token_ids")}}
        entry = HotwordEntry(**row)
        if (
            entry.normalized != _key(entry.surface)
            or entry.words != normalize_match_words(entry.surface)
            or (entry.normalized, entry.token_ids) not in table_variants
        ):
            raise ValueError("indexed pronunciation differs from sealed table")
        entries.append(entry)
    if (
        len({e.hotword_id for e in entries}) != len(entries)
        or {(e.normalized, e.token_ids) for e in entries} != table_variants
    ):
        raise ValueError("incomplete/duplicate pronunciation index")
    bundles = {
        lang: KeywordBundle(
            tuple(e for e in entries if e.language == ("pt-BR" if lang == "pt" else lang)), {}, {}
        )
        for lang in LANGUAGES
    }
    bundle, variants, mapping = joint_bundle(bundles)
    if len(bundle.entries) != report["review"]["active_unique_surfaces_per_audio"]:
        raise ValueError("table unique count differs from report")
    phrase_ids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for entry in bundle.entries:
        phrase_ids[entry.words].add(entry.hotword_id)
    return {
        "directory": directory,
        "plan": plan,
        "report": report,
        "table": table,
        "bundle": bundle,
        "variants": variants,
        "mapping": mapping,
        "index": {e.hotword_id: e for e in variants},
        "canonical": {e.hotword_id: e for e in bundle.entries},
        "phrase_ids": phrase_ids,
        "widths": {len(w) for w in phrase_ids},
    }


def check_pair(old: dict[str, Any], new: dict[str, Any]) -> None:
    for key in ("checkpoint_sha256", "candidate_scope", "sample_count"):
        if old["report"][key] != new["report"][key]:
            raise ValueError(f"run identity differs: {key}")
    if old["report"]["review"]["filler_entries_added"] != 0:
        raise ValueError("baseline must be the reviewed target-only run")
    old_entries = {e["normalized"]: e for e in old["table"]["entries"]}
    new_entries = {e["normalized"]: e for e in new["table"]["entries"]}
    if not old_entries.keys() < new_entries.keys() or any(
        new_entries[k] != v for k, v in old_entries.items()
    ):
        raise ValueError("expansion must retain every baseline entry unchanged")
    added = new_entries.keys() - old_entries.keys()
    if any(v.get("role") != "filler" for k in added for v in new_entries[k]["variants"]):
        raise ValueError("new entries must be fillers")
    for filename, digest in old["plan"]["code_sha256"].items():
        if (
            filename
            in {
                "hotwords/anchor_index.py",
                "hotwords/scoring.py",
                "inference/external_keyword_retrieval.py",
                "inference/external_keyword_topk_replay.py",
                "inference/reviewed_joint.py",
                "inference/streaming_backends.py",
                "training/edit_distance.py",
                "phonemes/coverage.py",
                "inference/hotword_prompt.py",
            }
            and new["plan"]["code_sha256"].get(filename) != digest
        ):
            raise ValueError(f"retrieval/scoring code changed: {filename}")
    if set(old["plan"]["groups"]) != set(new["plan"]["groups"]):
        raise ValueError("dataset groups differ")
    ignored = {"git_commit", "keyword_bias", "table_filler_exclusion_used_transcripts"}
    for name, before in old["plan"]["groups"].items():
        after = new["plan"]["groups"][name]
        if any(before[k] != after[k] for k in ("records", "target_ids", "language")) or {
            k: v for k, v in before["config"].items() if k not in ignored
        } != {k: v for k, v in after["config"].items() if k not in ignored}:
            raise ValueError(f"samples/targets/settings differ: {name}")
        for record in before["records"]:
            audio = record["audio_path"]
            if old["plan"]["inputs"][audio]["sha256"] != new["plan"]["inputs"][audio]["sha256"]:
                raise ValueError(f"recorded audio SHA differs: {audio}")
    # Metadata identities are checked as recorded; do not load model assets for this audit.
    for key in ("model", "checkpoint", "vocab"):
        if old["plan"]["configuration"][key] != new["plan"]["configuration"][key]:
            raise ValueError(f"asset path differs: {key}")
    for filename in old["plan"]["inputs"].keys() & new["plan"]["inputs"].keys():
        if filename.endswith(("config.json", ".pt", "vocab.v0.2.json")) and (
            old["plan"]["inputs"][filename]["sha256"] != new["plan"]["inputs"][filename]["sha256"]
        ):
            raise ValueError(f"recorded metadata/Head/vocab identity differs: {filename}")


def expected_ids(run: dict[str, Any], reference: str) -> set[str]:
    words = normalize_match_words(reference)
    return {
        identifier
        for width in run["widths"]
        for i in range(len(words) - width + 1)
        for identifier in run["phrase_ids"].get(words[i : i + width], ())
    }


def validate_row(
    row: dict[str, Any], record: dict[str, Any], run: dict[str, Any], name: str
) -> dict[str, Any]:
    if any(row.get(k) != v for k, v in record.items()) or row.get("row_sha256") != _row_digest(row):
        raise ValueError(f"sample identity/checksum differs: {name}/{record['sample_id']}")
    if row.get("audio_sha256") != run["plan"]["inputs"][record["audio_path"]]["sha256"]:
        raise ValueError("sample recorded audio SHA differs")
    expected = expected_ids(run, row["reference_text"])
    if expected != set(row["expected_hotword_ids"]):
        raise ValueError("saved reference membership differs from normalized phrase matching")
    gate = _validated_source_gate(run["plan"]["groups"][name]["config"])
    if len(row["raw_ranked_matches"]) != row["shortlist_candidates"]:
        raise ValueError("incomplete saved shortlist; cannot diagnose missing candidates")
    if not 0 <= row["variant_shortlist_candidates"] <= 64:
        raise ValueError("unexpected variant shortlist capacity")
    collapsed = collapse_variants(
        {
            **row,
            "raw_ranked_matches": row["raw_variant_ranked_matches"],
            "shortlist_candidates": row["variant_shortlist_candidates"],
        },
        run["variants"],
        run["mapping"],
        gate,
    )
    if collapsed["raw_ranked_matches"] != row["raw_ranked_matches"]:
        raise ValueError("saved surface ranking does not reproduce variant collapse")
    decoded = [row.get(k) for k in ("decoded_token_ids", "decoded_tokens", "decoded_confidences")]
    lengths = [len(v) for v in decoded if isinstance(v, list)]
    if len(lengths) != 3 or len(set(lengths)) != 1:
        raise ValueError("missing or inconsistent saved CTC sequence")
    return dict(replay_rows_top7([row], gate=gate, bundle=run["bundle"])[0])
