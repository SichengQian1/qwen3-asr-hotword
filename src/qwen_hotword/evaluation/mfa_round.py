"""After training only: paired fixed-surface retrieval and unfiltered final ASR."""

from __future__ import annotations

import gc
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.recovered_case_run import prompt, validate_result
from qwen_hotword.evaluation.stream_retrieval_metrics import reference_matcher
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.english_head_preview import _index
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.recovered_case_runtime import LANGUAGES, RecoveredCaseRuntime
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.reviewed_retrieval import NAMES
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.phonemes.mfa_v03_review import rows
from qwen_hotword.training.mfa_round_common import digest, read
from qwen_hotword.training.noise_evaluation import aggregate, paired_delta
from qwen_hotword.training.stage_progress import stage_progress


def saved(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    row = read(path)
    if row.get("identity") != identity or row.get("sha256") != digest(
        {k: v for k, v in row.items() if k != "sha256"}
    ):
        raise ValueError(f"saved evaluation cell changed: {path}")
    return row


def save(path: Path, identity: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
    row = {"identity": identity, **value}
    row["sha256"] = digest(row)
    write_json(path, row)
    return row


def assets(root: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    completed = read(output / "training_completed.json")
    pin(output / "head/ctc_head_best.pt", completed["checkpoint_sha256"])
    pin(output / "head/report.json", completed["report_sha256"])
    if completed["head_initialization"] != "random" or completed["num_classes"] != 100:
        raise ValueError("evaluation requires the completed fresh 100-class Head")
    previous = root / cfg["noise_source"] / "evaluation"
    report = read(previous / "report.json")
    old = read(previous / "run_config.json")
    if (
        report["status"] != "completed"
        or report["rows"] != 40800
        or report["table_sha256"] != cfg["hotword_table_sha256"]
        or report["checkpoints"] != cfg["reference_checkpoint_sha256"]
        or old["checkpoints"] != report["checkpoints"]
    ):
        raise ValueError(
            "expected completed, returned 6800-audio baseline/30/50 retrieval experiment"
        )
    inventory = old["inventory"]
    expected_counts = {"en": 6000, "es": 400, "pt": 400}
    if (
        inventory["dataset_counts"] != expected_counts
        or dict(Counter(s["language"] for s in inventory["samples"])) != expected_counts
        or len({(s["group"], s["sample_id"]) for s in inventory["samples"]}) != 6800
    ):
        raise ValueError("evaluation inventory language counts differ")
    source = root / inventory["source_configuration"]["baseline_run"]
    inputs = inventory["inputs"]

    def verified(path: Path) -> Path:
        pin(path, inputs[str(path)]["sha256"])
        return path

    original = read(verified(source / "run_config.json"))
    old_vocab_path = root / cfg["old_vocab"]
    pin(old_vocab_path, cfg["old_vocab_sha256"])
    old_vocab = load_phoneme_vocab(old_vocab_path)
    bundle, variants, by_variant = _index(
        verified(source / "indexed_pronunciations.jsonl"), old_vocab
    )
    canonical = {e.normalized: e for e in bundle.entries}
    if len(canonical) != 4000:
        raise ValueError("the full 4000-surface denominator is required")
    new_vocab = load_phoneme_vocab(root / cfg["vocab"])
    new_variants = tuple(
        HotwordEntry(**{**r, **{k: tuple(r[k]) for k in ("words", "phoneme_tokens", "token_ids")}})
        for r in rows(output / "indexed_pronunciations.v03.jsonl")
    )
    if any(
        tuple(new_vocab.tokens[i] for i in e.token_ids) != e.phoneme_tokens for e in new_variants
    ):
        raise ValueError("new index phone IDs are incompatible")
    new_by_variant = {e.hotword_id: canonical[e.normalized] for e in new_variants}
    references = {
        (g, r["sample_id"]): r for g, data in original["groups"].items() for r in data["records"]
    }
    checkpoints = {
        "v1": root / cfg["source_run"] / "head/ctc_head_best.pt",
        "old50": root / cfg["noise_source"] / "head_50/ctc_head_best.pt",
        "new70": output / "head/ctc_head_best.pt",
    }
    for tag, key in (("v1", "baseline"), ("old50", "50")):
        pin(checkpoints[tag], cfg["reference_checkpoint_sha256"][key])
    write_json(
        output / "evaluation_inputs.json",
        {
            "historical_inventory_sha256": sha(previous / "run_config.json"),
            "historical_report_sha256": sha(previous / "report.json"),
            "new_index_sha256": sha(output / "indexed_pronunciations.v03.jsonl"),
            "checkpoints": {k: sha(v) for k, v in checkpoints.items()},
            "denominator": (
                "same declared 4000 surfaces; unsupported new pronunciations count as misses"
            ),
            "benchmark": "previously inspected regression set, not a new blind test",
        },
    )
    return {
        "inventory": inventory,
        "references": references,
        "bundle": bundle,
        "targets": {g: set(v["target_ids"]) for g, v in original["groups"].items()},
        "checkpoints": checkpoints,
        "old_index": (old_vocab, variants, by_variant),
        "new_index": (new_vocab, new_variants, new_by_variant),
    }


def evaluate(root: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    import torch

    data = assets(root, cfg, output)
    match = reference_matcher(data["bundle"].entries)
    inventory = data["inventory"]
    checkpoint_hashes = {tag: sha(path) for tag, path in data["checkpoints"].items()}
    new_index_hash = sha(output / "indexed_pronunciations.v03.jsonl")
    all_rows = []
    detector = None
    try:
        for tag, checkpoint in data["checkpoints"].items():
            vocab, variants, by_variant = data["new_index" if tag == "new70" else "old_index"]
            detector = None
            with stage_progress(
                f"Paired retrieval {tag}: original + noisy", total=13600, every=100
            ) as tick:
                done = 0
                for sample in inventory["samples"]:
                    group, sample_id, lang = (
                        sample["group"],
                        sample["sample_id"],
                        sample["language"],
                    )
                    original = data["references"][group, sample_id]
                    present = set(match(original["reference_text"]))
                    targets = data["targets"][group]
                    for condition in ("original", "noise10db"):
                        audio = Path(
                            original["audio_path"]
                            if condition == "original"
                            else sample["audio_path"]
                        )
                        checksum = (
                            sample["clean_sha256"]
                            if condition == "original"
                            else sample["audio_sha256"]
                        )
                        identity = {
                            "tag": tag,
                            "condition": condition,
                            "group": group,
                            "sample_id": sample_id,
                            "audio_sha256": checksum,
                            "checkpoint_sha256": checkpoint_hashes[tag],
                            "vocab_sha256": cfg["vocab_sha256"]
                            if tag == "new70"
                            else cfg["old_vocab_sha256"],
                            "index_sha256": new_index_hash
                            if tag == "new70"
                            else cfg["hotword_table_sha256"],
                            "gate": inventory["gate"][group],
                        }
                        path = output / "retrieval/cells" / (digest(identity) + ".json")
                        cell = saved(path, identity)
                        if cell is None:
                            pin(audio, checksum)
                            if detector is None:
                                detector = external._load_detector(
                                    model=Path(cfg["model"]),
                                    checkpoint=checkpoint,
                                    vocab=vocab,
                                    hotwords=variants,
                                    device="cuda:0",
                                    dtype="bfloat16",
                                    language=NAMES[lang],
                                )
                            detector.language = NAMES[lang]
                            waveform, _ = external._load_audio(audio)
                            retrieval = external._retrieve_waveform(
                                detector, waveform, vocab, saved_raw_rank_depth=64
                            )
                            retrieval = collapse_variants(
                                retrieval, variants, by_variant, inventory["gate"][group]
                            )
                            value = {
                                "language": lang,
                                "group": group,
                                "sample_id": sample_id,
                                "audio_path": str(audio),
                                "reference_text": original["reference_text"],
                                "tag": tag,
                                "condition": condition,
                                "pure_ms": retrieval["timing"]["pure_retrieval_seconds"] * 1000,
                            }
                            for k in (5, 7):
                                selected = _select(
                                    retrieval["raw_ranked_matches"],
                                    gate=inventory["gate"][group],
                                    top_k=k,
                                )
                                identifiers = {e["hotword_id"] for e in selected}
                                value[f"top{k}"] = {
                                    "expected": len(present & targets),
                                    "hits": len(identifiers & present & targets),
                                    "selected": len(identifiers),
                                    "outside_reference": len(identifiers - present),
                                    "no_reference_hotwords": not present,
                                    "selected_ids": sorted(identifiers),
                                    "words": [e["surface"] for e in selected],
                                }
                            cell = save(path, identity, value)
                        all_rows.append(cell)
                        done += 1
                        tick(done)
            detector = None
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        detector = None
        gc.collect()
        torch.cuda.empty_cache()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[f"{row['tag']}/{row['condition']}/{row['language']}"].append(row)
    comparisons = {}
    for baseline in ("v1", "old50"):
        for condition in ("original", "noise10db"):
            for lang in ("en", "es", "pt"):
                comparisons[f"new70_vs_{baseline}/{condition}/{lang}"] = {
                    f"top{k}": paired_delta(
                        grouped[f"{baseline}/{condition}/{lang}"],
                        grouped[f"new70/{condition}/{lang}"],
                        k,
                    )
                    for k in (5, 7)
                }
    report = {
        "status": "completed",
        "rows": len(all_rows),
        "by_language": {k: aggregate(v) for k, v in grouped.items()},
        "paired_comparisons": comparisons,
        "qwen_decoder_used": False,
        "parameter_tuning_on_this_set": False,
        "limitations": [
            "previously inspected regression set and transcript-conditioned hotword table",
            "source noise and SNR of business WAVs unverified",
            "audio-level bootstrap; speaker independence unverified",
            "combined new G2P/vocabulary/fresh training/70% experiment; not causal isolation",
        ],
    }
    write_json(output / "retrieval/report.json", report)
    return report


def wer_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(
        "".join(c if unicodedata.category(c)[0] in {"L", "M", "N"} else " " for c in text).split()
    )


def asr(root: Path, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    import jiwer
    import torch

    if read(output / "retrieval/report.json")["status"] != "completed":
        raise ValueError("complete retrieval before final ASR")
    data = assets(root, cfg, output)
    match = reference_matcher(data["bundle"].entries)
    index = {}
    for path in (output / "retrieval/cells").glob("*.json"):
        row = read(path)
        identity = row["identity"]
        if saved(path, identity) is None:
            raise ValueError("missing retrieval cell")
        index[identity["tag"], identity["condition"], identity["group"], identity["sample_id"]] = (
            row
        )
    engine = None
    totals: dict[str, Any] = {}
    latencies: dict[str, list[float]] = defaultdict(list)
    calls_made = reused_cells = 0
    model_identity = sha(Path(cfg["model"]) / "model.safetensors.index.json")
    contract_identity = sha(output / "contract.json")
    try:
        with stage_progress(
            "Final ASR: all audio, no-keyword / old50 / new70", total=40800, every=100
        ) as tick:
            number = 0
            for sample in data["inventory"]["samples"]:
                group, sample_id, lang = sample["group"], sample["sample_id"], sample["language"]
                for condition in ("original", "noise10db"):
                    comparison = [
                        index[tag, condition, group, sample_id] for tag in ("old50", "new70")
                    ]
                    prototype = comparison[0]
                    present = set(match(prototype["reference_text"]))
                    requests = [
                        ("no_hotwords", "", []),
                        *[
                            (
                                tag,
                                prompt(lang, r["top7"]["words"]) if r["top7"]["words"] else "",
                                r["top7"]["words"],
                            )
                            for tag, r in zip(("old50", "new70"), comparison, strict=True)
                        ],
                    ]
                    normalized_hash = None
                    baseline_recognized: set[str] = set()
                    for mode, context, _selected in requests:
                        request = {
                            "audio_path": prototype["audio_path"],
                            "audio_sha256": prototype["identity"]["audio_sha256"],
                            "language": LANGUAGES[lang],
                            "context": context,
                            "seed": cfg["asr"]["seed"],
                        }
                        identity = {
                            "request": request,
                            "model_weight_index_sha256": model_identity,
                            "max_new_tokens": cfg["asr"]["max_new_tokens"],
                            "contract_sha256": contract_identity,
                        }
                        path = output / "asr/cells" / (digest(identity) + ".json")
                        cell = saved(path, identity)
                        if cell is None:
                            pin(Path(request["audio_path"]), request["audio_sha256"])
                            if engine is None:
                                with stage_progress(
                                    "Load local Qwen3-ASR-1.7B decoder for paired final ASR"
                                ):
                                    engine = RecoveredCaseRuntime(Path(cfg["model"]), cfg["asr"])
                                write_json(output / "asr/runtime.json", engine.metadata)
                            with stage_progress(f"ASR {lang}/{condition}/{mode}/{sample_id}"):
                                started = time.perf_counter()
                                result = engine.transcribe(request)
                                calls_made += 1
                                result["wall_seconds"] = time.perf_counter() - started
                            validate_result(result, request)
                            cell = save(path, identity, {"result": result})
                        else:
                            reused_cells += 1
                        result = cell["result"]
                        validate_result(result, request)
                        if (
                            normalized_hash is not None
                            and result["normalized_audio_sha256"] != normalized_hash
                        ):
                            raise ValueError("paired ASR modes used different normalized audio")
                        normalized_hash = result["normalized_audio_sha256"]
                        prediction = result["prediction"]
                        score = jiwer.process_words(
                            wer_text(prototype["reference_text"]), wer_text(prediction)
                        )
                        stats = totals.setdefault(
                            f"{mode}/{condition}/{lang}",
                            {
                                "audio": 0,
                                "word_errors": 0,
                                "reference_words": 0,
                                "reference_hotwords": 0,
                                "recognized_reference_hotwords": 0,
                                "reference_external_hotwords_in_prediction": 0,
                                "truncated": 0,
                                "wrong_returned_language": 0,
                                "wall_seconds": 0.0,
                                "hotwords_recovered_vs_no_prompt": 0,
                                "hotwords_lost_vs_no_prompt": 0,
                            },
                        )
                        predicted = set(match(prediction))
                        recognized = predicted & present
                        if mode == "no_hotwords":
                            baseline_recognized = recognized
                        stats["hotwords_recovered_vs_no_prompt"] += len(
                            recognized - baseline_recognized
                        )
                        stats["hotwords_lost_vs_no_prompt"] += len(baseline_recognized - recognized)
                        stats["audio"] += 1
                        stats["word_errors"] += (
                            score.substitutions + score.deletions + score.insertions
                        )
                        stats["reference_words"] += (
                            score.hits + score.substitutions + score.deletions
                        )
                        stats["reference_hotwords"] += len(present)
                        stats["recognized_reference_hotwords"] += len(predicted & present)
                        stats["reference_external_hotwords_in_prediction"] += len(
                            predicted - present
                        )
                        stats["truncated"] += int(result["generation_limit_reached"])
                        stats["wrong_returned_language"] += int(
                            result["returned_language"] != LANGUAGES[lang]
                        )
                        stats["wall_seconds"] += result["wall_seconds"]
                        latencies[f"{mode}/{condition}/{lang}"].append(result["wall_seconds"])
                        number += 1
                        tick(number)
        for key, value in totals.items():
            value["wer"] = (
                value["word_errors"] / value["reference_words"]
                if value["reference_words"]
                else None
            )
            value["hotword_recall"] = (
                value["recognized_reference_hotwords"] / value["reference_hotwords"]
                if value["reference_hotwords"]
                else None
            )
            ordered = sorted(latencies[key])
            value["saved_request_wall_seconds"] = {
                "mean": sum(ordered) / len(ordered),
                "p50": ordered[len(ordered) // 2],
                "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            }
        report = {
            "status": "completed",
            "by_mode_condition_language": totals,
            "rows": number,
            "new_decoder_calls_this_invocation": calls_made,
            "reused_identical_request_cells_this_invocation": reused_cells,
            "qwen_decoder_used": True,
            "reference_sent_to_decoder": False,
            "selection": "all fixed 6800 audio x original/noisy; no successful-case filtering",
            "wer_normalization": "NFC casefold; punctuation to spaces; keep accents and digits",
            "all_reference_table_hotwords_metric": True,
            "outside_reference_is_text_based_not_audited_hallucination": True,
        }
        write_json(output / "asr/report.json", report)
        return report
    finally:
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
