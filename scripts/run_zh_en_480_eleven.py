"""Run the sealed 480h bilingual checkpoint on all eleven evaluation groups."""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation import zh_en_eleven as sets
from qwen_hotword.evaluation import zh_en_parallel as parallel
from qwen_hotword.evaluation.aishell_run import check_gpu, identity
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read

VIEWS = ("raw_top5_last", "raw_top7_last", "gated_top5_last", "gated_top7_last")


def cached(out: Path, sample: dict[str, Any]) -> Path:
    audio_hash = str(sample["audio_sha256"])
    return out / "audio_results" / str(sample["language"]) / audio_hash[:2] / (audio_hash + ".json")


def load_result(path: Path, run_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    record = read(path)
    body: dict[str, Any] = record["result"]
    if (record["run"], record["audio"], record["language"], record["digest"]) != (
        run_id,
        sample["audio_sha256"],
        sample["language"],
        digest(body),
    ):
        raise ValueError(f"corrupted or incompatible result: {path}")
    return body


def object_chunks(items: Iterable[tuple[str, bytes]]) -> Iterator[bytes]:
    yield b"{"
    for i, (key, value) in enumerate(items):
        yield (b"," if i else b"") + json.dumps(key, ensure_ascii=False).encode() + b":" + value
    yield b"}"


def compact(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def delivery_items(
    out: Path, run_id: str, records: list[dict[str, Any]], mode: str, kind: str
) -> Iterator[tuple[str, bytes]]:
    for sample in records:
        if mode:
            result = load_result(cached(out, sample), run_id, sample)
            value: Any = [step[: MODES[mode]] for step in result["steps"]]
        elif kind == "durations":
            value = sample["duration_seconds"]
        else:
            value = {k: v for k, v in sample.items() if k != "targets"}
        yield sample["sample_id"], compact(value)


def package(out: Path, groups: dict[str, list[dict[str, Any]]], run_id: str) -> None:
    # Keep at most one utterance's timelines in memory while building the delivery.
    temporary = out / "delivery.zip.pending"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for kind in ("durations", "audio_manifest", "stream_contexts"):
            with archive.open("delivery/" + kind + ".json", "w", force_zip64=True) as stream:
                modes = list(MODES) if kind == "stream_contexts" else [""]
                if kind == "stream_contexts":
                    stream.write(b"{")
                for j, mode in enumerate(modes):
                    if mode:
                        stream.write((b"," if j else b"") + compact(mode) + b":")
                    stream.write(b"{")
                    for i, group in enumerate(sets.DATASETS):
                        records = groups[group]
                        stream.write((b"," if i else b"") + compact(records[0]["key"]) + b":")

                        items = delivery_items(out, run_id, records, mode, kind)
                        for chunk in object_chunks(items):
                            stream.write(chunk)
                    stream.write(b"}")
                if kind == "stream_contexts":
                    stream.write(b"}")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("delivery archive CRC check failed")
    temporary.replace(out / "delivery.zip")


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def table(groups: dict[str, list[dict[str, Any]]], metrics: dict[str, Any]) -> str:
    lines = [
        "数据集\t音频数\t计分句数\t目标句词对\t门控前Top5\t门控前Top7\t门控后Top5\t门控后Top7\t无目标句数\t末次Top5误提示率\t末次Top7误提示率"
    ]
    for name in sets.DATASETS:
        m = metrics[name]
        last = m["gated_top7_last"]
        lines.append(
            "\t".join(
                map(
                    str,
                    [
                        name,
                        len(groups[name]),
                        last["utterances"],
                        last["target_pairs"],
                        *(pct(m[v]["target_recall"]) for v in VIEWS),
                        last["negative_utterances"],
                        *(pct(m[v]["negative_prompt_rate"]) for v in VIEWS[2:]),
                    ],
                )
            )
        )
    return "\n".join(lines) + "\n"


SERIAL_RUNNER_SHA = "cc744c3e8a40c25175f2d68980117b2f64022b08f207b50a996dc429cb13af13"


def resume_id(out: Path, plan: dict[str, Any]) -> str:
    # The original plan hashes this CLI, including its scheduling loop. Preserve the
    # original ID only for this explicitly reviewed predecessor, with every other
    # field identical. Keep the original plan on disk so its results remain readable.
    path = out / "run_config.json"
    if path.is_file():
        previous = read(path)
        if previous["runner_sha256"] == SERIAL_RUNNER_SHA:
            compatible = {**plan, "runner_sha256": SERIAL_RUNNER_SHA}
            if previous != compatible:
                raise ValueError("original evaluation inputs or scoring changed; cannot resume")
            plan = compatible
    external._write_or_verify_json(path, plan)
    run_id = digest(plan)
    external._write_or_verify_json(
        out / "retrieval_runtime.json",
        {
            "run": run_id,
            "runner_sha256": sha(Path(__file__)),
            "parallel_module_sha256": sha(Path(parallel.__file__)),
        },
    )
    return run_id


def run(out: Path, gpu: int, ner: Path | None, workers: int = 1) -> None:
    cfg = read(ROOT / "configs/zh_en_stream2351_probe.workzone.json")
    previous_cfg = read(ROOT / "configs/aishell_original_retrieval.workzone.json")
    with stage_progress("Verify original bilingual 480h run / epoch 25 and joint-2351 index"):
        model, _, _, vocab = identity(
            ROOT, previous_cfg, read(ROOT / previous_cfg["review_policy"])
        )
        if model["checkpoint_epoch"] != 25:
            raise ValueError("expected 480h best epoch 25")
        t = ROOT / "configs/zh_en_joint2351.v1.jsonl"
        pin(t, "33e73fa9df358475d766b7cd3c5f2fa07071a5ffc25ed8e87923e2effaae5e40")
        entries, owners = base.registry(t, vocab)
    common = base.inventory(cfg)
    with stage_progress("Resolve AISHELL-NER gold and English TTS index"):
        extra, audit = sets.extra_inventory(cfg, t, sets.official_file(out, ner))
    groups = {name: [s for s in common + extra if s["group"] == name] for name in sets.DATASETS}
    plan = {
        "model_identity": model,
        "gate": base.GATE,
        "table_sha256": sha(t),
        "common_samples_sha256": digest(common),
        "extra_samples_sha256": digest(extra),
        "audit": audit,
        "runner_sha256": sha(Path(__file__)),
        "additional_code": {
            n: sha(ROOT / "src/qwen_hotword" / (n + ".py"))
            for n in (
                "evaluation/zh_en_eleven",
                "evaluation/zh_en_delivery",
                "inference/stream_context_contract",
                "inference/reviewed_joint",
                "inference/external_keyword_topk_replay",
            )
        },
    }
    run_id = resume_id(out, plan)
    if workers > 1:
        parallel.prefill(
            out,
            groups,
            run_id,
            workers,
            gpu,
            (Path(model["model_path"]), Path(model["checkpoint_path"]), vocab, entries, owners),
            cached,
            load_result,
        )
    detector = None
    totals: dict[str, dict[str, Counter[str]]] = {}
    probe_metrics: dict[str, Counter[str]] = {v: Counter() for v in VIEWS}
    probe_targets: Counter[str] = Counter()
    for group, records in groups.items():
        with stage_progress(f"480h retrieval: {group}", total=len(records), every=50) as tick:
            for i, s in enumerate(records, 1):
                path = cached(out, s)
                if path.exists():
                    result = load_result(path, run_id, s)
                else:
                    if detector is None:
                        check_gpu(gpu, 35)
                        detector = external._load_detector(
                            model=Path(model["model_path"]),
                            checkpoint=Path(model["checkpoint_path"]),
                            vocab=vocab,
                            hotwords=entries,
                            device="cuda:0",
                            dtype="bfloat16",
                            language="English",
                        )
                    result = base.retrieve(detector, vocab, entries, owners, s)
                    write_json(
                        path,
                        {
                            "run": run_id,
                            "audio": s["audio_sha256"],
                            "language": s["language"],
                            "digest": digest(result),
                            "result": result,
                        },
                    )
                uid, key = s["sample_id"], s["key"]
                validate_contexts(
                    {
                        m: {key: {uid: [step[:k] for step in result["steps"]]}}
                        for m, k in MODES.items()
                    },
                    {key: {uid: s["duration_seconds"]}},
                )
                if group != sets.PROBE or s["targets"]:
                    base.score(result, s, totals)
                if group == sets.PROBE:
                    gold = set(s["targets"])
                    probe_targets.update(gold)
                    for k in (5, 7):
                        probe_metrics[f"raw_top{k}_last"].update(
                            gold & {base.canonical(w) for w in result["raw_last"][:k]}
                        )
                        probe_metrics[f"gated_top{k}_last"].update(
                            gold & {base.canonical(m["word"]) for m in result["steps"][-1][:k]}
                        )
                tick(i)
    metrics = {name: {v: base.rates(c) for v, c in totals[name].items()} for name in sets.DATASETS}
    summary = table(groups, metrics)
    print(summary, flush=True)
    (out / "by_dataset.tsv").write_text(summary, encoding="utf-8-sig")
    with stage_progress("Package validated same-step Top5/Top7 timelines"):
        package(out, groups, run_id)
    report = {
        "status": "completed",
        "checkpoint_sha256": model["checkpoint_sha256"],
        "checkpoint_epoch": 25,
        "num_classes": 183,
        "plan_sha256": run_id,
        "active_surfaces": 2351,
        "pronunciations": 2352,
        "effect": "same-step",
        "qwen_decoder_used": False,
        "by_dataset": metrics,
        "source_audit": audit,
        "en_tts_macro_word_recall": {
            v: sum(c[w] / n for w, n in probe_targets.items()) / len(probe_targets)
            for v, c in probe_metrics.items()
        },
        "note": (
            "AISHELL subset overlaps full test. "
            "TTS probe macro averages observed target words only."
        ),
        "delivery_zip": str(out / "delivery.zip"),
    }
    write_json(out / "report.json", report)
    temporary = out / "return_bundle.zip.pending"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for filename in ("report.json", "by_dataset.tsv"):
            z.write(out / filename, filename)
    temporary.replace(out / "return_bundle.zip")
    print(
        f"Completed. Results: {out / 'return_bundle.zip'}; downstream: {out / 'delivery.zip'}",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument(
        "--ner-file", type=Path, help="Optional local copy of the pinned official test transcript"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        choices=range(1, 17),
        help="Max independent retrieval workers on the selected card (1 = serial)",
    )
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = ROOT / "outputs/zh_en183_480_joint2351_eleven_v1"
    with exclusive_feature_cache_run(out):
        run(out, args.gpu, args.ner_file, args.workers)


if __name__ == "__main__":
    main()
