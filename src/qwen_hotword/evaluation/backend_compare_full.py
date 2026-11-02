"""Complete saved regression inventory, resumable CPU preparation, compact return files."""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.backend_comparison import (
    CONDITIONS,
    LANGUAGES,
    block_legacy_qwen_vllm_registration,
    context,
    digest,
    read,
    saved_source_rows,
    sha,
    write_once,
)
from qwen_hotword.training.stage_progress import stage_progress


def counts(cfg: dict[str, Any]) -> dict[str, int]:
    value = cfg["expected_audio_counts"]
    if set(value) != set(LANGUAGES) or any(type(n) is not int or n <= 0 for n in value.values()):
        raise ValueError("full inventory needs positive per-language counts for en/es/pt")
    return dict(value)


def validate_inventory(items: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Retain every declared sample, including repeats; reject missing or duplicate identities."""
    expected = counts(cfg)
    observed: Counter[tuple[str, str]] = Counter()
    pairs: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in items:
        lang, condition = row["language"], row["condition"]
        if lang not in LANGUAGES or condition not in CONDITIONS:
            raise ValueError("unexpected language/condition in full inventory")
        identity = row["identity"]
        if any(identity.get(k) != row[k] for k in ("group", "sample_id", "condition")):
            raise ValueError("full inventory cell identity fields differ")
        if (
            identity.get("tag") != "new70"
            or identity.get("checkpoint_sha256") != cfg["source_head_sha256"]
            or identity.get("index_sha256") != cfg["source_index_sha256"]
            or identity.get("vocab_sha256") != cfg["source_vocab_sha256"]
        ):
            raise ValueError("full inventory Head/vocabulary/index identity differs")
        key = lang, row["group"], row["sample_id"]
        if condition in pairs[key]:
            raise ValueError("duplicate full inventory sample/condition identity")
        context(lang, row["top7"]["words"])
        pairs[key][condition] = row
        observed[lang, condition] += 1
    if dict(observed) != {(lang, c): n for lang, n in expected.items() for c in CONDITIONS}:
        raise ValueError(f"full inventory counts differ: {dict(observed)}; expected {expected}")
    for pair in pairs.values():
        if set(pair) != set(CONDITIONS):
            raise ValueError("original/noise10db samples are not paired by identity")
        if pair["original"]["reference_text"] != pair["noise10db"]["reference_text"]:
            raise ValueError("paired original/noisy references differ")
    return {
        "base_utterances": sum(expected.values()),
        "counts_by_language": expected,
        "audio_conditions": len(items),
        "backend_pairs": len(items) * 2,
        "planned_backend_calls": len(items) * 4,
        "inventory_sha256": digest([r["source_cell_sha256"] for r in items]),
        "selection": "all saved new70 identities; no duration, duplicate or success filtering",
        "counts_by_language_condition": {f"{a}/{b}": n for (a, b), n in observed.items()},
    }


def checked_cell(path: Path) -> dict[str, Any]:
    value = read(path)
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"modified preparation cell: {path}")
    return value


def prepare_full(root: Path, cfg: dict[str, Any], output: Path) -> None:
    import hashlib

    import librosa
    import numpy as np
    import soundfile as sf

    from qwen_hotword.evaluation.backend_compare_runtime import versions

    target = sum(counts(cfg).values()) * 2
    items = []
    with stage_progress("Audit all saved new70 audio identities", total=target, every=500) as tick:
        for number, row in enumerate(saved_source_rows(root / cfg["source_run"], cfg), 1):
            items.append(row)
            tick(number)
    inventory = validate_inventory(items, cfg)
    write_once(output / "full_inventory.json", inventory)
    runtime = versions()
    identity = {"inventory": inventory, "versions": runtime}
    write_once(output / "preparation_contract.json", identity)
    ready = output / "requests.json"
    if ready.exists():
        plan = read(ready)
        if plan["versions"] != runtime or plan["full_inventory"] != inventory:
            raise ValueError("full preparation identity changed")
        seen = set()
        with stage_progress(
            "Verify prepared full audio snapshots", total=target, every=500
        ) as tick:
            for number, row in enumerate(items, 1):
                cell = checked_cell(
                    output / "preparation/cells" / (digest(row["identity"]) + ".json")
                )
                if cell["source_cell_sha256"] != row["source_cell_sha256"]:
                    raise ValueError("prepared source identity changed")
                for request in cell["requests"]:
                    name = request["waveform_file"]
                    if name not in seen:
                        if sha(output / name) != request["waveform_file_sha256"]:
                            raise ValueError("prepared canonical waveform changed")
                        seen.add(name)
                tick(number)
        expected_requests = [
            r
            for row in items
            for r in checked_cell(
                output / "preparation/cells" / (digest(row["identity"]) + ".json")
            )["requests"]
        ]
        if plan["requests"] != expected_requests:
            raise ValueError("full requests differ from verified preparation cells")
        print(f"Reuse all {len(expected_requests)} full requests; no new selection", flush=True)
        return
    with block_legacy_qwen_vllm_registration():
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

    processor = Qwen3ASRProcessor.from_pretrained(
        cfg["model"], local_files_only=True, fix_mistral_regex=True
    )
    if processor.feature_extractor.hop_length != cfg["pad_waveform_to_samples"]:
        raise ValueError("shared waveform padding differs from processor hop length")
    requests = []
    file_hashes: dict[str, Counter[str]] = defaultdict(Counter)
    wave_hashes: dict[str, Counter[str]] = defaultdict(Counter)
    durations = []
    (output / "waveforms").mkdir(exist_ok=True)
    with stage_progress(
        "Prepare every audio and both Prompts (CPU)", total=target, every=100
    ) as tick:
        for number, row in enumerate(items, 1):
            cell_path = output / "preparation/cells" / (digest(row["identity"]) + ".json")
            if cell_path.exists():
                cell = checked_cell(cell_path)
                if cell["source_cell_sha256"] != row["source_cell_sha256"]:
                    raise ValueError("partial preparation source changed")
                first = cell["requests"][0]
                if sha(output / first["waveform_file"]) != first["waveform_file_sha256"]:
                    raise ValueError("partial canonical waveform changed")
            else:
                audio = Path(row["audio_path"])
                if sha(audio) != row["identity"]["audio_sha256"]:
                    raise ValueError(f"source audio changed: {audio}")
                waveform, rate = sf.read(audio, dtype="float32", always_2d=True)
                waveform = waveform.mean(axis=1)
                if rate != cfg["sample_rate"]:
                    waveform = librosa.resample(
                        waveform, orig_sr=rate, target_sr=cfg["sample_rate"]
                    )
                if not len(waveform) or not np.isfinite(waveform).all() or not np.any(waveform):
                    raise ValueError(f"empty/nonfinite/silent full audio: {audio}")
                samples = len(waveform)
                waveform = np.ascontiguousarray(
                    np.pad(waveform, (0, (-samples) % cfg["pad_waveform_to_samples"])),
                    dtype=np.float32,
                )
                wave_hash = hashlib.sha256(waveform.tobytes()).hexdigest()
                wave_file = f"waveforms/{wave_hash}.npy"
                destination = output / wave_file
                if destination.exists():
                    if not np.array_equal(np.load(destination, allow_pickle=False), waveform):
                        raise ValueError("canonical waveform snapshot differs")
                else:
                    temporary = destination.with_suffix(".pending")
                    with temporary.open("wb") as handle:
                        np.save(handle, waveform, allow_pickle=False)
                    temporary.replace(destination)
                wave_file_sha = sha(destination)
                prepared = []
                for mode, words in (("no_hotwords", []), ("new70_top7", row["top7"]["words"])):
                    prompt_context = context(row["language"], words)
                    messages = [
                        {"role": "system", "content": prompt_context},
                        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
                    ]
                    prompt = (
                        processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False
                        )
                        + f"language {LANGUAGES[row['language']]}<asr_text>"
                    )
                    batch = processor(
                        text=[prompt], audio=[waveform], return_tensors="pt", padding=True
                    )
                    ids = batch["input_ids"][0].tolist()
                    if len(ids) + cfg["max_new_tokens"] > cfg["max_model_len"]:
                        raise ValueError(
                            f"full context exceeds budget before inference: {audio}; "
                            f"input={len(ids)}, output={cfg['max_new_tokens']}, "
                            f"context={cfg['max_model_len']}; no sample was skipped"
                        )
                    prepared.append(
                        {
                            "request_id": digest([row["identity"], mode]),
                            **{
                                k: row[k]
                                for k in (
                                    "sample_id",
                                    "language",
                                    "condition",
                                    "audio_path",
                                    "reference_text",
                                )
                            },
                            "audio_sha256": row["identity"]["audio_sha256"],
                            "waveform_file": wave_file,
                            "waveform_file_sha256": wave_file_sha,
                            "waveform_sha256": wave_hash,
                            "original_normalized_samples": samples,
                            "shared_padded_samples": len(waveform),
                            "mode": mode,
                            "words": words,
                            "context": prompt_context,
                            "prompt": prompt,
                            "initial_prompt_ids": processor.tokenizer(prompt)["input_ids"],
                            "prepared_input_ids_sha256": digest(ids),
                            "prepared_input_token_count": len(ids),
                            "source_cell": row["source_cell"],
                            "source_cell_sha256": row["source_cell_sha256"],
                        }
                    )
                    del batch
                cell = {
                    "source_cell_sha256": row["source_cell_sha256"],
                    "requests": prepared,
                    "duration_seconds": samples / cfg["sample_rate"],
                }
                write_once(cell_path, {**cell, "sha256": digest(cell)})
            first = cell["requests"][0]
            bucket = f"{row['language']}/{row['condition']}"
            file_hashes[bucket][first["audio_sha256"]] += 1
            wave_hashes[bucket][first["waveform_sha256"]] += 1
            durations.append(cell["duration_seconds"])
            requests.extend(cell["requests"])
            tick(number)
    if len(requests) != target * 2 or len({r["request_id"] for r in requests}) != target * 2:
        raise ValueError("full request inventory is incomplete or contains duplicate identities")
    summary = {
        **inventory,
        "duration_min_seconds": min(durations),
        "duration_max_seconds": max(durations),
        "duration_sum_hours_original_plus_noisy": sum(durations) / 3600,
        "duplicate_file_records_retained": {
            k: sum(v.values()) - len(v) for k, v in file_hashes.items()
        },
        "duplicate_waveform_records_retained": {
            k: sum(v.values()) - len(v) for k, v in wave_hashes.items()
        },
        "duplicates_are_not_independent_evidence": True,
        "maximum_prepared_input_tokens": max(r["prepared_input_token_count"] for r in requests),
    }
    write_once(output / "preparation_summary.json", summary)
    write_once(ready, {"requests": requests, "versions": runtime, "full_inventory": inventory})


def full_report(plan: dict[str, Any]) -> dict[str, Any]:
    return {"scope": "full_H200_final_ASR_backend_comparison", **plan["full_inventory"]}


def preview(details: list[dict[str, Any]], limit: int = 24) -> dict[str, Any]:
    different = [r for r in details if r["classification"] != "exact"]
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in different:
        buckets[row["language"], row["condition"], row["mode"], row["classification"]].append(row)
    selected: list[dict[str, Any]] = []
    while buckets and len(selected) < limit:
        for key in list(buckets):
            selected.append(buckets[key].pop(0))
            if not buckets[key]:
                del buckets[key]
            if len(selected) == limit:
                break
    return {
        "total_different_pairs": len(different),
        "preview_only": True,
        "selection": (
            "round robin by language/condition/mode/difference type; not an error rate estimate"
        ),
        "rows": selected,
        "full_details": "comparison.tsv and details.json remain in the work-zone output",
    }


def compact_package(output: Path) -> Path:
    """Return under 100,000 bytes; full rows/cells/waveforms stay in the work zone."""
    names = (
        "report.json",
        "comparison.md",
        "run_contract.json",
        "model_metadata.json",
        "environment.json",
        "device.json",
        "gpu_probe.json",
        "decode_policy.json",
        "full_inventory.json",
        "preparation_summary.json",
        "transformers/runtime.json",
        "vllm/runtime.json",
        "vllm/implementation.json",
        "input_mismatch.json",
    )
    data = {name: (output / name).read_bytes() for name in names if (output / name).exists()}
    failures = sorted(output.glob("failure_*.json"))
    data.update({p.name: p.read_bytes() for p in failures[-3:]})
    for log in sorted((output / "logs").glob("*.log")):
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 4096))
            data[f"log_tails/{log.name}"] = handle.read()
    differences = output / "differences_preview.json"
    sample = read(differences) if differences.exists() else {"preview_only": True, "rows": []}
    while True:
        data["differences_preview.json"] = json.dumps(sample, ensure_ascii=False).encode()
        data["BUNDLE_README.md"] = (
            b"Compact return bundle (<100,000 bytes). Full comparison.tsv, details.json, "
            b"requests.json, preparation cells and inference cells stay in the work zone. "
            b"The differences preview is not the complete set of differences.\n"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sorted(data.items()):
                archive.writestr(name, content)
        if buffer.tell() < 100000:
            break
        if sample["rows"]:
            sample["rows"].pop()
        elif any(name.startswith("log_tails/") for name in data):
            data = {k: v for k, v in data.items() if not k.startswith("log_tails/")}
        elif "environment.json" in data:
            del data["environment.json"]
        else:
            raise ValueError("compact core reports exceed 100KB; inspect work-zone files")
    destination = output / "return_bundle.zip"
    temporary = output / "return_bundle.zip.pending"
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(destination)
    return destination
