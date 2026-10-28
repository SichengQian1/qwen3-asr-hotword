"""Freeze a positive case-hunting pool from the completed latest joint Top7 delivery."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.prompt_order import digest
from qwen_hotword.evaluation.recovered_audio_export import TABLE_SHA256, sha
from qwen_hotword.inference.hotword_prompt import normalize_match_words, strict_phrase_match
from qwen_hotword.inference.recovered_case_runtime import LANGUAGES
from qwen_hotword.inference.stream_context_contract import validate_contexts
from qwen_hotword.training.spanish_capacity import _verified

SOURCE_RUN = "outputs/multilingual_stream_contexts_ep27_noise10db_v1/full_same-step"
SOURCE_CONFIG_SHA = "47134f2d4489a260ea2bdeb8a7d120b0b4e0658ff687335952846ebe20224743"
HEAD_SHA = "872edc09460cc4a70c53e6868269ca8afcec111b2de82ef52dd34df354873466"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pool_rows(
    samples: list[dict[str, Any]],
    contexts: dict[str, Any],
    references: dict[tuple[str, str], str],
    table: dict[tuple[str, ...], str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reasons: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        key = (sample["key"], sample["sample_id"])
        if key in seen or sample["language"] not in LANGUAGES:
            raise ValueError("duplicate sample identity or unknown language")
        seen.add(key)
        reference = references[(sample["group"], sample["sample_id"])]
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("missing reference text; never infer it from a filename")
        candidates = contexts["RAG_top7_stream"][key[0]][key[1]][-1]
        words = [c["word"] for c in candidates]
        if any(normalize_match_words(word) not in table for word in words):
            raise ValueError("actual retrieved candidate absent from latest joint table")
        targets = [word for word in words if strict_phrase_match(reference, word)]
        row = {
            **sample,
            "reference_text": reference,
            "injected_hotwords": words,  # ALL actual Top7, including incorrect candidates.
            "retrieved_candidates": candidates,
            "potential_targets": targets,  # Selection/scoring only; not sent to the model.
            "uid": digest([sample["key"], sample["sample_id"], sample["audio_sha256"]]),
        }
        grouped[sample["audio_sha256"]].append(row)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for same_audio in grouped.values():
        identities = {
            (r["language"], normalize_match_words(r["reference_text"])) for r in same_audio
        }
        if len(identities) != 1:
            reasons["duplicate_audio_conflicting_reference_or_language"] += len(same_audio)
            continue
        eligible = [r for r in same_audio if r["potential_targets"]]
        if not eligible:
            reasons["no_retrieved_word_in_reference"] += len(same_audio)
            continue
        chosen = min(eligible, key=lambda r: digest([seed, r["uid"]]))
        reasons["duplicate_audio_not_selected"] += len(same_audio) - 1
        by_language[chosen["language"]].append(chosen)
    for values in by_language.values():
        values.sort(key=lambda r: digest([seed, r["uid"]]))
    pool = []
    for index in range(max((len(v) for v in by_language.values()), default=0)):
        for language in LANGUAGES:
            if index < len(by_language[language]):
                pool.append(by_language[language][index])
    return pool, dict(reasons)


def prepare_cases(
    root: Path, source: Path, table_path: Path, seed: int
) -> tuple[dict[str, Any], dict[tuple[str, ...], str]]:
    inputs: dict[str, Any] = {}

    def checked(directory: Path, name: str) -> Any:
        return read_json(_verified(directory, name, inputs))

    source_plan = checked(source, "run_config.json")
    if sha(source / "run_config.json") != SOURCE_CONFIG_SHA:
        raise ValueError("source is not the returned latest three-language full same-step run")
    report = checked(source, "report.json")
    if (
        report["status"] != "completed"
        or report["effect"] != "same-step"
        or report["table_sha256"] != TABLE_SHA256
        or report["checkpoint_sha256"] != HEAD_SHA
        or source_plan["checkpoint"]["sha256"] != HEAD_SHA
        or source_plan["source_configuration"]["table_sha256"] != TABLE_SHA256
    ):
        raise ValueError("saved retrieval completion/table/Head/effect identity differs")
    contexts = checked(source, "delivery/stream_contexts.json")
    durations = checked(source, "delivery/durations.json")
    manifest = checked(source, "delivery/audio_manifest.json")
    validate_contexts(contexts, durations)
    samples = source_plan["samples"]
    expected: dict[str, Any] = {}
    for sample in samples:
        bucket = expected.setdefault(sample["key"], {})
        if sample["sample_id"] in bucket:
            raise ValueError("duplicate sample in source plan")
        bucket[sample["sample_id"]] = sample
        if durations[sample["key"]][sample["sample_id"]] != sample["duration_seconds"]:
            raise ValueError("source duration differs from audio manifest")
    if manifest != expected or {k: set(v) for k, v in manifest.items()} != {
        k: set(v) for k, v in durations.items()
    }:
        raise ValueError("source plan/delivery audio inventory differs")
    if sha(table_path) != TABLE_SHA256:
        raise ValueError("current joint table SHA256 differs")
    inputs[str(table_path)] = {"sha256": TABLE_SHA256, "size_bytes": table_path.stat().st_size}
    table = {
        normalize_match_words(entry["surface"]): entry["surface"]
        for entry in read_json(table_path)["entries"]
    }
    baseline = (root / source_plan["source_configuration"]["baseline_run"]).resolve()
    old = checked(baseline, "run_config.json")
    references = {
        (group, record["sample_id"]): record["reference_text"]
        for group, data in old["groups"].items()
        for record in data["records"]
    }
    model = (root / old["configuration"]["model"]).resolve()
    for name in ("config.json", "model.safetensors.index.json", "preprocessor_config.json"):
        path = model / name
        recorded = source_plan["inputs"].get(str(path))
        if name != "preprocessor_config.json" and (
            recorded is None or sha(path) != recorded["sha256"]
        ):
            raise ValueError(f"model metadata identity differs: {path}")
        inputs[str(path)] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    if (
        "Qwen3ASRForConditionalGeneration" not in read_json(model / "config.json")["architectures"]
        or model.name != "Qwen3-ASR-1.7B"
    ):
        raise ValueError("requires the existing local Qwen3-ASR-1.7B snapshot")
    pool, reasons = pool_rows(samples, contexts, references, table, seed)
    return {
        "source_run": str(source),
        "inputs": inputs,
        "model": str(model),
        "table_sha256": TABLE_SHA256,
        "retrieval_head_sha256": HEAD_SHA,
        "source_samples": len(samples),
        "pool": pool,
        "pool_by_language": dict(Counter(r["language"] for r in pool)),
        "pool_exclusions": reasons,
        "scope": "new_full_audio_ASR_with_saved_last_prefix_Top7",
        "selection": "positive_case_hunt; reference never sent to ASR; not population evaluation",
        "reference_audio_audit": "not manually listened; reference agreement only",
    }, table
