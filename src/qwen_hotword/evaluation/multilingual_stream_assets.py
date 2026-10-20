"""Extend the frozen English stream inventory with the original ES/PT wave tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from qwen_hotword.evaluation.stream_context_assets import KEYS, prepare_stream
from qwen_hotword.evaluation.stream_retrieval_metrics import reference_matcher
from qwen_hotword.evaluation.target_review import _checked
from qwen_hotword.evaluation.wave_keywords import _json
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _validated_source_gate
from qwen_hotword.inference.stream_context_contract import boundaries
from qwen_hotword.training.spanish_capacity import _sha, _verified
from qwen_hotword.training.stage_progress import stage_progress


def prepare_multilingual_stream(root: Path, config: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import soundfile as sf

    cfg = _json(config)
    plan, runtime = prepare_stream(root, (root / cfg["english_stream_config"]).resolve(), 0)
    inputs = plan["inputs"]
    _checked(config, inputs)
    if plan["checkpoint"]["sha256"] != cfg["checkpoint_sha256"]:
        raise ValueError("final best Head SHA differs from the returned epoch-27 result")
    if cfg["full_run_authorization"] != "user_requested_direct_full_2026-10-20":
        raise ValueError("explicit direct-full configuration required")
    source = (root / plan["source_configuration"]["baseline_run"]).resolve()
    old = _json(_verified(source, "run_config.json", inputs))
    old_report = _json(_verified(source, "report.json", inputs))
    additional = cfg["additional_datasets"]
    wave_directories = cfg["wave_directories"]
    if set(wave_directories) != {f"wave{w}" for w in range(1, 5)} or any(
        not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name
        for name in wave_directories.values()
    ):
        raise ValueError("four explicit wave directory names required")
    if set(additional) != {f"wave{w}_{lang}" for lang in ("es", "pt") for w in range(1, 5)}:
        raise ValueError("all four original ES/PT waves required")
    if set(old["groups"]) != set(KEYS) | set(additional):
        raise ValueError("baseline dataset coverage differs")
    for sample in plan["samples"]:
        sample["language"] = "en"
    vocab_path = (root / old["configuration"]["vocab"]).resolve()
    for group, info in additional.items():
        lang, wave = info["language"], info["wave"]
        if lang not in ("es", "pt") or group != f"{wave}_{lang}":
            raise ValueError(f"invalid language/wave mapping: {group}")
        directory = wave_directories[wave]
        if info["key"] != f"{directory}@@{lang}":
            raise ValueError(f"downstream key differs from audio directory: {group}")
        previous = old["groups"][group]
        if previous["language"] != lang or len(previous["config"]["sources"]) != 1:
            raise ValueError(f"baseline language/source differs: {group}")
        original = previous["config"]["sources"][0]
        text = _checked(
            Path(original["transcripts"]["path"]), inputs, original["transcripts"]["sha256"]
        )
        audio = (root / cfg["wave_audio_root"] / directory / lang / "wav").resolve()
        spec = external.SourceSpec(group, audio, text)
        records, _ = external.build_external_dataset(
            (spec,), language="pt-BR" if lang == "pt" else lang
        )
        if len(records) != info["expected_samples"] or [
            {k: v for k, v in r.to_dict().items() if k != "audio_path"} for r in records
        ] != [{k: v for k, v in r.items() if k != "audio_path"} for r in previous["records"]]:
            raise ValueError(f"sample/reference coverage differs: {group}")
        current: dict[str, Any] = external._run_config(
            model=runtime["model"],
            checkpoint=runtime["checkpoint"],
            vocab_path=vocab_path,
            keyword_path=runtime["table"],
            sources=(spec,),
            keyword_set="all_keywords",
            device="cuda:0",
            dtype="bfloat16",
            language=lang,
            saved_raw_rank_depth=64,
        )
        current["sources"] = [{**s, "audio_dir": original["audio_dir"]} for s in current["sources"]]
        ignored = {"git_commit", "ctc_checkpoint"}
        if {k: v for k, v in current.items() if k not in ignored} != {
            k: v for k, v in previous["config"].items() if k not in ignored
        }:
            raise ValueError(f"retrieval settings differ: {group}")
        plan["gate"][group] = _validated_source_gate(previous["config"])
        with stage_progress(
            f"Verify noisy headers and SHA {directory}/{lang}", total=len(records), every=100
        ) as tick:
            for n, (record, clean) in enumerate(zip(records, previous["records"], strict=True), 1):
                path = _checked(record.audio_path, inputs)
                header = sf.info(str(path))
                ends = boundaries(header.frames, header.samplerate)
                plan["samples"].append(
                    {
                        "group": group,
                        "key": info["key"],
                        "language": lang,
                        "sample_id": record.sample_id,
                        "audio_path": str(path),
                        "frames": header.frames,
                        "sample_rate": header.samplerate,
                        "channels": header.channels,
                        "duration_seconds": header.frames / header.samplerate,
                        "step_count": len(ends),
                        "duration_bucket": (
                            "under_2s"
                            if header.frames < 2 * header.samplerate
                            else "exact_2s_multiple"
                            if header.frames % (2 * header.samplerate) == 0
                            else "tail"
                        ),
                        "audio_sha256": inputs[str(path)]["sha256"],
                        "clean_sha256": old["inputs"][clean["audio_path"]]["sha256"],
                    }
                )
                tick(n)
    entries = {e.hotword_id: e for e in runtime["bundle"].entries}
    match = reference_matcher(entries.values())
    evaluation = {}
    pairing = {}
    keys = {}
    for group, previous in old["groups"].items():
        samples = [s for s in plan["samples"] if s["group"] == group]
        key = samples[0]["key"]
        if key in keys:
            raise ValueError("duplicate downstream dataset key")
        keys[key] = group
        targets = set(previous["target_ids"])
        if not targets <= entries.keys():
            raise ValueError(f"target IDs missing from frozen joint index: {group}")
        denominator = sum(len(match(r["reference_text"]) & targets) for r in previous["records"])
        if any(
            old_report["groups"][group]["current"][k]["expected_target_sample_pairs"] != denominator
            for k in ("top5", "top7")
        ):
            raise ValueError(f"target denominator differs from sealed baseline: {group}")
        evaluation[group] = {
            "language": previous["language"],
            "target_ids": sorted(targets),
            "expected_target_sample_pairs": denominator,
            "references": {r["sample_id"]: r["reference_text"] for r in previous["records"]},
        }
        changed = sum(s["audio_sha256"] != s["clean_sha256"] for s in samples)
        if not changed:
            raise ValueError(f"all noisy audio equals clean audio: {group}")
        pairing[group] = {"changed_files": changed, "identical_files": len(samples) - changed}
    package = Path(__file__).resolve().parents[1]
    for relative in (
        "evaluation/multilingual_stream_assets.py",
        "evaluation/stream_retrieval_metrics.py",
        "inference/multilingual_stream_run.py",
    ):
        plan["code_sha256"][relative] = _sha(package / relative)
        _checked(package / relative, inputs)
    _checked(package.parents[1] / "scripts/run_multilingual_stream.py", inputs)
    plan.update(
        configuration=cfg,
        english_stream_configuration=plan["configuration"],
        inventory_samples=len(plan["samples"]),
        selected_samples=len(plan["samples"]),
        prefix_count=sum(s["step_count"] for s in plan["samples"]),
        dataset_counts=dict(Counter(s["language"] for s in plan["samples"])),
        inventory_duration_buckets=dict(Counter(s["duration_bucket"] for s in plan["samples"])),
        selected_duration_buckets=dict(Counter(s["duration_bucket"] for s in plan["samples"])),
        audio_pairing=pairing,
    )
    runtime["evaluation_groups"] = evaluation
    return plan, runtime
