#!/usr/bin/env python3
"""One command: freeze audio -> mine actual terms -> EGY IPA -> whole-audio Top5/7."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import heapq
import json
import os
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from qwen_hotword.evaluation.egy_corpus_hotwords import (
    RATIOS,
    assert_disjoint,
    build_table,
    entity_surfaces,
    expected_ids,
    order_key,
    phrase_present,
    select_audio,
    summarize,
)
from qwen_hotword.evaluation.egy_hotword_review import verify_run
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.en_ar_source_audit import iter_jsonl
from qwen_hotword.training.stage_progress import stage_progress
from scripts.label_arabic_native import exclusive_job, verify_assets


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sealed(path: Path, value: Any) -> None:
    if path.exists():
        if read(path) != value:
            raise ValueError("sealed identity changed; use a new output directory: " + str(path))
    else:
        write_json(path, value)


def result_file(path: Path, identity: str) -> Any:
    if not path.exists():
        return None
    value = read(path)
    if value.get("identity") != identity:
        raise ValueError("cached result identity changed: " + str(path))
    return value


def acquire_ner(config: dict[str, Any]) -> Path:
    directory = ROOT / str(config["ner_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    for name, digest in config["ner_files"].items():
        path = directory / name
        with stage_progress("Acquire/verify pinned NER asset " + name) as tick:
            if not path.exists():
                url = (
                    f"https://huggingface.co/{config['ner_repository']}/resolve/"
                    f"{config['ner_revision']}/{name}"
                )
                # Explicit work-zone permission: scoped TLS bypass, pinned SHA before use.
                context = ssl._create_unverified_context()
                temporary = path.with_suffix(path.suffix + ".pending")
                with (
                    urllib.request.urlopen(url, context=context, timeout=120) as response,
                    temporary.open("wb") as stream,
                ):
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        size += len(chunk)
                        tick(size)
                pin(temporary, digest)
                temporary.replace(path)
            pin(path, digest)
    return directory


def mine(
    rows: list[dict[str, Any]], directory: Path, output: Path, identity: str
) -> dict[str, list[dict[str, Any]]]:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    with stage_progress("Load pinned Arabic NER (text only, temporary GPU use)"):
        tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, use_fast=True, model_max_length=512
        )
        model = AutoModelForTokenClassification.from_pretrained(directory, local_files_only=True)
        recognizer = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            stride=64,
            device=0,
        )
    collected = {}
    with stage_progress(
        "Extract actual entities from fixed transcripts", total=len(rows), every=50
    ) as tick:
        for index, row in enumerate(rows, 1):
            path = output / "ner_rows" / (order_key(row["id"]) + ".json")
            row_identity = fingerprint([identity, row["id"], row["text"]])
            saved = result_file(path, row_identity)
            if saved is None:
                predictions = recognizer(row["text"])
                saved = {
                    "identity": row_identity,
                    "entities": entity_surfaces(row["text"], predictions),
                }
                write_json(path, saved)
            collected[row["id"]] = saved["entities"]
            tick(index)
    del recognizer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return collected


def phonemize(output: Path, config: dict[str, Any]) -> None:
    from qwen_hotword.phonemes.arabic_pronunciation import PronunciationFrontend

    frontend_config = read(ROOT / config["frontend_config"])
    source = read(ROOT / config["source_config"])
    verify_assets(frontend_config)
    table = read(output / "candidates.json")
    vocab = read(ROOT / source["vocab"])["tokens"]
    identity = sha(output / "contract.json")
    with stage_progress("Load existing EGY pronunciation frontend"):
        frontend = PronunciationFrontend(
            ROOT / frontend_config["assets"],
            "ar_egy",
            vocab,
            read(ROOT / config["pronunciation_policy"]),
        )
    with stage_progress("Generate isolated EGY pronunciations", total=len(table), every=25) as tick:
        for index, entry in enumerate(table, 1):
            path = output / "pronunciations" / (entry["id"] + ".json")
            row_identity = fingerprint([identity, entry])
            saved = result_file(path, row_identity)
            if saved is None:
                try:
                    label = frontend.label(entry["surface"], mode="isolated")
                    if label["unreviewed_exception_words"]:
                        raise ValueError("unreviewed lexical pronunciation exception")
                    if len(label["ids"]) < 4 or any(i < 2 or i >= len(vocab) for i in label["ids"]):
                        raise ValueError("too short or invalid pronunciation IDs")
                    saved = {"identity": row_identity, "status": "active", "label": label}
                except ValueError as error:
                    saved = {"identity": row_identity, "status": "inactive", "reason": str(error)}
                write_json(path, saved)
            tick(index)
    sealed(
        output / "table.json",
        [
            dict(entry, pronunciation=read(output / "pronunciations" / (entry["id"] + ".json")))
            for entry in table
        ],
    )


def load_detector(
    model_path: Path, checkpoint: Path, table: list[dict[str, Any]], vocab: Any
) -> Any:
    import torch

    from qwen_hotword.config import ModelConfig
    from qwen_hotword.hotwords.anchor_index import AnchorIndexConfig
    from qwen_hotword.hotwords.registry import HotwordEntry
    from qwen_hotword.hotwords.scoring import HotwordScoringConfig
    from qwen_hotword.inference.streaming_backends import CumulativeAudioCtcDetector
    from qwen_hotword.modeling.ctc_head import build_ctc_head_from_checkpoint
    from qwen_hotword.modeling.qwen_backbone import load_asr_model
    from qwen_hotword.training.ctc_overfit import freeze_module

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if tuple(payload["vocab_tokens"]) != vocab.tokens:
        raise ValueError("Head vocabulary differs from fixed 167-class vocabulary")
    head = build_ctc_head_from_checkpoint(payload)
    if head.time_upsampling_factor != 3:
        raise ValueError("requires new EGY x3 Head")
    head.load_state_dict(payload["state_dict"], strict=True)
    head = head.to(device="cuda:0", dtype=torch.float32)
    freeze_module(head)
    hotwords = []
    for entry in table:
        pronunciation = entry["pronunciation"]
        if pronunciation["status"] != "active":
            continue
        label = pronunciation["label"]
        hotwords.append(
            HotwordEntry(
                hotword_id=entry["id"],
                language="ar",
                surface=entry["surface"],
                normalized=entry["surface"],
                words=tuple(label["words"]),
                pronunciation=" ".join(label["phones"]),
                phoneme_tokens=tuple(vocab.tokens[i] for i in label["ids"]),
                token_ids=tuple(label["ids"]),
                source="EGY_isolated_frontend",
                validation_occurrences=0,
            )
        )
    if not hotwords:
        raise ValueError("no usable pronunciations; cannot run retrieval")
    wrapper = load_asr_model(
        ModelConfig(
            path=model_path,
            expected_name="Qwen3-ASR-1.7B",
            dtype="bfloat16",
            device="cuda:0",
            local_files_only=True,
        )
    )
    freeze_module(wrapper.model.thinker.audio_tower)
    return CumulativeAudioCtcDetector(
        encoder_wrapper=wrapper,
        head=head,
        hotwords=tuple(hotwords),
        language="Arabic",
        scoring_config=HotwordScoringConfig(top_k=7, minimum_posterior_confidence=0.5),
        retrieval_mode="operating",
        retrieval_backend="anchor_guided",
        anchor_shortlist_size=64,
        anchor_start_radius=2,
        anchor_config=AnchorIndexConfig(),
        device="cuda:0",
    )


def retrieve(
    output: Path, model_path: Path, checkpoint: Path, vocab_path: Path
) -> list[dict[str, Any]]:
    from qwen_hotword.inference.external_keyword_retrieval import _load_audio, _retrieve_waveform
    from qwen_hotword.phonemes.coverage import load_phoneme_vocab

    selected, table = read(output / "selected_audio.json"), read(output / "table.json")
    vocab = load_phoneme_vocab(vocab_path)
    identity = fingerprint([sha(output / "contract.json"), sha(output / "table.json")])
    results = []
    detector = None
    with stage_progress(
        "Whole-audio Encoder + x3 CTC + Top5/7", total=len(selected), every=10
    ) as tick:
        for index, row in enumerate(selected, 1):
            path = output / "retrieval_rows" / (order_key(row["id"]) + ".json")
            row_identity = fingerprint([identity, row["id"], row["audio_sha256"]])
            saved = result_file(path, row_identity)
            if saved is None:
                if detector is None:
                    with stage_progress("Load r15 merged Encoder and sealed epoch29 x3 Head"):
                        detector = load_detector(model_path, checkpoint, table, vocab)
                audio_path = Path(row["audio_path"])
                pin(audio_path, row["audio_sha256"])
                waveform, metadata = _load_audio(audio_path)
                if abs(metadata["duration_seconds"] - row["duration_seconds"]) > 0.02:
                    raise ValueError("audio duration changed: " + str(audio_path))
                # Detector receives audio and frozen index ONLY, never reference or expected IDs.
                prediction = dict(
                    _retrieve_waveform(detector, waveform, vocab, saved_raw_rank_depth=64)
                )
                pin(audio_path, row["audio_sha256"])
                saved = {
                    "identity": row_identity,
                    "id": row["id"],
                    "source_group": row["source_group"],
                    "duration_seconds": row["duration_seconds"],
                    "expected_ids": sorted(expected_ids(row["text"], table)),
                    "reference_present_ids": [
                        t["id"] for t in table if phrase_present(row["text"], t["surface"])
                    ],
                    **prediction,
                }
                write_json(path, saved)
            results.append(saved)
            tick(index)
    return results


def package(output: Path) -> None:
    archive = output / "return_summary.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in ("report.json", "selection.json", "candidate_summary.json"):
            bundle.write(output / name, name)
    data = base64.b64encode(archive.read_bytes()).decode()
    size = 30 * 1024
    count = (len(data) + size - 1) // size
    for i in range(count):
        header = {
            "archive": archive.name,
            "archive_sha256": sha(archive),
            "encoding": "base64",
            "part": i + 1,
            "parts": count,
        }
        (output / f"return_part_{i + 1:03d}.txt").write_text(
            json.dumps(header) + "\n" + data[i * size : (i + 1) * size] + "\n", encoding="ascii"
        )


def run(config_path: Path, output: Path, gpu: int) -> dict[str, Any]:
    config = read(config_path)
    source = read(ROOT / config["source_config"])
    with stage_progress("Verify completed EGY epoch29, sealed datasets and actual r15 weights"):
        verified = verify_run(ROOT, source)
        train_path = Path(verified["train_manifest"])
        test_path, val_path = ROOT / config["test_manifest"], ROOT / config["validation_manifest"]
        pin(test_path, config["test_sha256"])
        pin(val_path, config["validation_sha256"])
        plan = read(ROOT / config["encoder_plan"])
        model_path = Path(plan["model"]["source"])
        pin(model_path / "model.safetensors", source["expected"]["model_weight_sha256"])
        for name, record in plan["model"]["files"].items():
            if name != "model.safetensors":
                pin(model_path / name, record["sha256"])
        frontend_config = read(ROOT / config["frontend_config"])
        verify_assets(frontend_config)
        identity_paths = [
            config_path,
            ROOT / config["source_config"],
            ROOT / config["encoder_plan"],
            ROOT / config["pronunciation_policy"],
            ROOT / config["frontend_config"],
            ROOT / source["candidate_file"],
            test_path,
            val_path,
            Path(__file__),
        ]
        identity_paths += list((ROOT / "src/qwen_hotword/evaluation").glob("egy_*.py"))
        identity_paths += [
            ROOT / ("src/qwen_hotword/" + name)
            for name in (
                "phonemes/arabic_pronunciation.py",
                "phonemes/arabic_native.py",
                "inference/external_keyword_retrieval.py",
                "hotwords/scoring.py",
                "hotwords/anchor_index.py",
                "modeling/ctc_head.py",
                "modeling/audio_encoder.py",
            )
        ]
        identity_paths += [
            p for p in model_path.iterdir() if p.suffix in {".json", ".jinja", ".txt"}
        ]
        contract = {
            "inputs": {
                **verified["files"],
                **{str(p): sha(p) for p in identity_paths},
                str(model_path / "model.safetensors"): source["expected"]["model_weight_sha256"],
            },
            "model_weight_sha256": source["expected"]["model_weight_sha256"],
            "policy": "corpus_conditioned_targets_plus_training_distractors",
            "mode": "whole_original_audio",
            "ratios_by_duration": RATIOS,
            "thresholds": {"score": 0.75, "posterior": 0.5, "edit_ratio": 0.35},
            "gpu_does_not_change_identity": True,
        }
        output.mkdir(parents=True, exist_ok=True)
        sealed(output / "contract.json", contract)
    with stage_progress("Freeze 8:1:1 audio BEFORE entity extraction; verify train/val isolation"):
        selected, selection = select_audio(list(iter_jsonl(test_path)))
        assert_disjoint(selected, iter_jsonl(val_path))
        pool: dict[str, list[dict[str, Any]]] = {g: [] for g in RATIOS}

        # Only compact text metadata is held; no audio/features are loaded in this stage.
        def protected() -> Any:
            for row in iter_jsonl(train_path):
                if row["language"] == "ar":
                    pool[row["source_group"]].append(
                        {k: row[k] for k in ("id", "source_group", "text", "source", "source_row")}
                    )
                yield row

        assert_disjoint(selected, protected())
        training = [
            r
            for rows in pool.values()
            for r in heapq.nsmallest(
                config["training_ner_samples_per_source"], rows, key=lambda r: order_key(r["id"])
            )
        ]
        pool.clear()
        sealed(output / "selected_audio.json", selected)
        sealed(output / "selection.json", selection)
    directory = acquire_ner(config)
    with stage_progress("Build corpus-backed candidate table"):
        extracted = mine(selected + training, directory, output, sha(output / "contract.json"))
        terms = [
            t
            for t in read(ROOT / source["candidate_file"])["candidates"]
            if t["category"] in {"医疗", "金融经济", "科技", "能源环境"}
        ]
        table = build_table(selected, training, extracted, terms)
        sealed(output / "candidates.json", table)
    with stage_progress("EGY pronunciation subprocess (reuse existing isolated runtime)"):
        subprocess.run(
            [
                str(ROOT / frontend_config["runtime"] / "bin/python"),
                "-u",
                "-B",
                str(Path(__file__).resolve()),
                "--phonemize-only",
                "--config",
                str(config_path),
                "--output-dir",
                str(output),
            ],
            check=True,
        )
    table = read(output / "table.json")
    counts = Counter(t["pronunciation"]["status"] for t in table)
    write_json(
        output / "candidate_summary.json",
        {
            "count": len(table),
            "roles": dict(Counter(t["role"] for t in table)),
            "pronunciation_status": dict(counts),
            "human_semantic_review_completed": False,
            "inactive_examples": [
                {k: t[k] for k in ("id", "surface", "pronunciation")}
                for t in table
                if t["pronunciation"]["status"] != "active"
            ][:20],
        },
    )
    with (output / "candidates.tsv").open("w", encoding="utf-8") as stream:
        stream.write("id\tsurface\tcategory\trole\tpronunciation_status\tIPA\n")
        for t in table:
            p = t["pronunciation"]
            stream.write(
                "\t".join(
                    [
                        t["id"],
                        t["surface"],
                        t["category"],
                        t["role"],
                        p["status"],
                        " ".join(p.get("label", {}).get("phones", [])),
                    ]
                )
                + "\n"
            )
    results = retrieve(
        output,
        model_path,
        ROOT / source["source_run"] / "head/ctc_head_best.pt",
        ROOT / source["vocab"],
    )
    active_ids = {t["id"] for t in table if t["pronunciation"]["status"] == "active"}
    all_pairs = sum(len(r["expected_ids"]) for r in results)
    active_pairs = sum(len(set(r["expected_ids"]) & active_ids) for r in results)
    with stage_progress("Recheck sealed inputs and summarize fixed Top5/7"):
        for name, digest in contract["inputs"].items():
            pin(Path(name), digest)
        report = {
            "status": "completed",
            "output_dir": str(output),
            "physical_gpu": gpu,
            **source["expected"],
            "table_sha256": sha(output / "table.json"),
            "selection": selection,
            "candidate_summary": read(output / "candidate_summary.json"),
            "all": summarize(results),
            "pronunciation_target_coverage": {
                "all_target_pairs": all_pairs,
                "active_target_pairs": active_pairs,
                "coverage": active_pairs / all_pairs if all_pairs else None,
                "inactive_targets_remain_in_main_recall_denominator": True,
            },
            "by_source": {
                g: summarize([r for r in results if r["source_group"] == g]) for g in RATIOS
            },
            "test_set_used": True,
            "qwen_decoder_used": False,
            "human_semantic_review_completed": False,
            "scope": (
                "corpus-conditioned automatic entity/term recall; "
                "not blind business-list evaluation"
            ),
            "gold_policy": (
                "conservative word boundaries; longest non-overlapping phrase; "
                "inactive targets retained"
            ),
            "ner_model": config["ner_repository"],
            "ner_revision": config["ner_revision"],
            "ner_domain": (
                "MSA-trained NER applied to declared EGY transcripts; automatic candidates"
            ),
        }
        write_json(output / "report.json", report)
        package(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/egy_corpus_retrieval.workzone.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/egy_ep29_corpus_hotwords_v1"
    )
    parser.add_argument("--phonemize-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = args.output_dir.resolve()
    try:
        if args.phonemize_only:
            phonemize(output, read(args.config))
        else:
            with exclusive_job(output):
                report = run(args.config.resolve(), output, args.gpu)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
