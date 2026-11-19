"""Evaluate the verified sentence-level Mandarin/English sets and export both stream modes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qwen_hotword.evaluation import zh_en_delivery as evaluation
from qwen_hotword.evaluation.aishell_run import check_checkpoint_metadata, check_gpu
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.stream_context_contract import MODES, validate_contexts
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read


def run(gpu: int, model: Path, output: Path) -> None:
    import torch

    cfg = read(ROOT / "configs/zh_en_stream2351_probe.workzone.json")
    train = ROOT / cfg["train"]
    checkpoint = train / "head/ctc_head_best.pt"
    table = ROOT / "configs/zh_en_joint2351.v1.jsonl"
    with stage_progress(
        "Verify epoch 16 Head, Qwen snapshot, joint pronunciations and 183 token IDs"
    ):
        marker = read(train / "training_completed.json")
        contract = read(train / "cache_contract.json")
        pin(train / "model_identity.json", contract["model_identity.json"])
        pin(train / "contract.json", contract["contract.json"])
        train_cfg = read(train / "contract.json")["config"]
        if (
            train_cfg["output"] != cfg["train"]
            or not train_cfg["scale_to_us"]
            or marker["best_epoch"] != 16
        ):
            raise ValueError("expected balanced530 uscap best Head, epoch 16")
        saved_model = read(train / "model_identity.json")
        required = {
            "config.json",
            "preprocessor_config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            *read(model / "model.safetensors.index.json")["weight_map"].values(),
        }
        if not required <= saved_model.keys():
            raise ValueError("incomplete training snapshot identity")
        for name, checksum in saved_model.items():
            pin(model / name, checksum)
        pin(checkpoint, marker["identity"]["checkpoint_sha256"])
        pin(train / "head/report.json", marker["identity"]["report_sha256"])
        foundation = read(train / "contract.json")["foundation"]
        vocab_path = ROOT / foundation["vocab"]
        pin(vocab_path, foundation["vocab_sha256"])
        pin(table, "33e73fa9df358475d766b7cd3c5f2fa07071a5ffc25ed8e87923e2effaae5e40")
        vocab = load_phoneme_vocab(vocab_path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check_checkpoint_metadata(payload, marker, read(train / "head/report.json"), vocab)
        del payload
        entries, owners = evaluation.registry(table, vocab)
    samples = evaluation.inventory(cfg)
    sources = [
        "evaluation/zh_en_delivery",
        "inference/stream_context_contract",
        "inference/external_keyword_retrieval",
        "inference/streaming_backends",
        "inference/reviewed_joint",
        "inference/external_keyword_topk_replay",
        "hotwords/anchor_index",
        "hotwords/scoring",
        "modeling/audio_encoder",
        "modeling/ctc_head",
        "training/ctc_overfit",
    ]
    plan = {
        "scope": "sentence-labeled groups; same-step cumulative 2s prefixes",
        "head_sha256": marker["identity"]["checkpoint_sha256"],
        "epoch": 16,
        "vocab_sha256": foundation["vocab_sha256"],
        "model": saved_model,
        "table_sha256": sha(table),
        "gate": evaluation.GATE,
        "samples_sha256": digest(samples),
        "script_sha256": sha(Path(__file__)),
        "source": {n: sha(ROOT / "src/qwen_hotword" / (n + ".py")) for n in sources},
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "qwen-asr", "soundfile", "numpy", "librosa")
        },
    }
    plan_id = digest(plan)
    external._write_or_verify_json(output / "run_config.json", plan)
    contexts: dict[str, Any] = {mode: {} for mode in MODES}
    durations: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    totals: dict[str, dict[str, Counter[str]]] = {}
    detector = None
    with stage_progress(
        "Retrieve cumulative audio prefixes (completed utterances)", total=len(samples), every=10
    ) as tick:
        for i, sample in enumerate(samples, 1):
            group, key, uid = sample["group"], sample["key"], sample["sample_id"]
            path = output / "utterances" / group / (uid + ".json")
            if path.is_file():
                saved = read(path)
                if (
                    saved["plan"] != plan_id
                    or saved["sample"] != [key, uid]
                    or digest(saved["result"]) != saved["result_sha256"]
                ):
                    raise ValueError(f"cached result identity differs: {path}")
                result = saved["result"]
            else:
                if detector is None:
                    check_gpu(gpu, 35)
                    detector = external._load_detector(
                        model=model,
                        checkpoint=checkpoint,
                        vocab=vocab,
                        hotwords=entries,
                        device="cuda:0",
                        dtype="bfloat16",
                        language="English",
                    )
                result = evaluation.retrieve(detector, vocab, entries, owners, sample)
                write_json(
                    path,
                    {
                        "plan": plan_id,
                        "sample": [key, uid],
                        "result": result,
                        "result_sha256": digest(result),
                    },
                )
            for mode, k in MODES.items():
                contexts[mode].setdefault(key, {})[uid] = [step[:k] for step in result["steps"]]
            evaluation.score(result, sample, totals)
            durations.setdefault(key, {})[uid] = sample["duration_seconds"]
            manifest.setdefault(key, {})[uid] = {k: v for k, v in sample.items() if k != "targets"}
            tick(i)
    del detector
    with stage_progress("Validate and package downstream delivery"):
        delivery = output / "delivery"
        delivery.mkdir(exist_ok=True)
        validation = validate_contexts(contexts, durations)
        for filename, data in (
            ("stream_contexts", contexts),
            ("durations", durations),
            ("audio_manifest", manifest),
        ):
            external._write_or_verify_json(delivery / (filename + ".json"), data)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tests/fixtures/stream_interface_v1_validator.py.txt"),
                "--ctx",
                str(delivery / "stream_contexts.json"),
                "--durations",
                str(delivery / "durations.json"),
            ],
            check=True,
        )
        report = {
            "status": "completed",
            "effect": "same-step",
            "schema_validation": validation,
            "checkpoint_sha256": plan["head_sha256"],
            "checkpoint_epoch": 16,
            "num_classes": 183,
            "plan_sha256": plan_id,
            "active_surfaces": 2351,
            "pronunciations": 2352,
            "sample_count": len(samples),
            "prefix_count": sum(s["step_count"] for s in samples),
            "qwen_decoder_used": False,
            "audio_condition": "as supplied; no added noise",
            "targets_outside_joint_table": sorted(
                {w for s in samples for w in s["targets"]} - {e.normalized for e in entries}
            ),
            "by_group": {
                group: {name: evaluation.rates(c) for name, c in metrics.items()}
                for group, metrics in totals.items()
            },
            "delivery_zip": str(output / "delivery.zip"),
        }
        bundle = output / "delivery.tmp"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("stream_contexts.json", "durations.json", "audio_manifest.json"):
                archive.write(delivery / name, "delivery/" + name)
        bundle.replace(output / "delivery.zip")
        write_json(output / "report.json", report)
        print(
            json.dumps(
                {**report, "by_group": report["by_group"]["overall"]}, ensure_ascii=False, indent=2
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model = args.model or Path(read(ROOT / "configs/zh_en_foundation.workzone.json")["base_model"])
    output = ROOT / "outputs/zh_en183_balanced530_joint2351_stream_v1"
    with exclusive_feature_cache_run(output):
        run(args.gpu, model, output)


if __name__ == "__main__":
    main()
