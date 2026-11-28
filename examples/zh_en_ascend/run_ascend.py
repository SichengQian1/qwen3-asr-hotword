"""Offline bilingual audio-tower / CTC retrieval on one Ascend or CUDA device."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import common_delivery as common  # type: ignore[import-not-found]

from qwen_hotword.hotwords.anchor_index import AnchorIndexConfig
from qwen_hotword.hotwords.scoring import HotwordScoringConfig
from qwen_hotword.inference import external_keyword_retrieval as ext
from qwen_hotword.inference.streaming_backends import CumulativeAudioCtcDetector
from qwen_hotword.phonemes.coverage import load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.stage_progress import stage_progress


def load_tower(model: Path, device: str, dtype: Any) -> Any:
    from kit_qwen.configuration_qwen3_asr import (  # type: ignore[import-not-found]
        Qwen3ASRAudioEncoderConfig,
    )
    from kit_qwen.modeling_qwen3_asr import Qwen3ASRAudioEncoder  # type: ignore[import-not-found]
    from safetensors import safe_open

    cfg = Qwen3ASRAudioEncoderConfig(
        **json.loads((model / "config.json").read_text())["thinker_config"]["audio_config"]
    )
    cfg._attn_implementation = "eager"
    tower = Qwen3ASRAudioEncoder(cfg)
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "thinker.audio_tower."
    keys = {k.removeprefix(prefix): shard for k, shard in index.items() if k.startswith(prefix)}
    if set(keys) != set(tower.state_dict()):
        raise ValueError("Qwen audio tower keys differ from local snapshot")
    state = {}
    for shard in sorted(set(keys.values())):
        with safe_open(model / shard, framework="pt", device="cpu") as f:
            for name, file in keys.items():
                if file == shard:
                    state[name] = f.get_tensor(prefix + name)
    tower.load_state_dict(state, strict=True)
    return tower.eval().requires_grad_(False).to(device=device, dtype=dtype)


def detector(model: Path, vocab: Any, entries: Any, device: str, dtype: Any) -> Any:
    import torch
    from kit_qwen.processing_qwen3_asr import Qwen3ASRProcessor  # type: ignore[import-not-found]

    from qwen_hotword.modeling.ctc_head import (
        TemporalUpsampleCtcHead,
        build_ctc_head_from_checkpoint,
    )

    tower = load_tower(model, device, dtype)
    from kit_qwen.configuration_qwen3_asr import Qwen3ASRConfig
    from transformers import AutoConfig

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
    proc = Qwen3ASRProcessor.from_pretrained(
        str(model), local_files_only=True, fix_mistral_regex=True
    )
    payload = torch.load(HERE / "assets/head.pt", map_location="cpu", weights_only=True)
    head = build_ctc_head_from_checkpoint(payload)
    if (
        tuple(payload["vocab_tokens"]) != vocab.tokens
        or not isinstance(head, TemporalUpsampleCtcHead)
        or head.time_upsampling_factor != 2
    ):
        raise ValueError("expected sealed temporal 2x, 183-class head")
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    wrapper = SimpleNamespace(
        model=SimpleNamespace(
            thinker=SimpleNamespace(audio_tower=tower), device=device, dtype=dtype
        ),
        processor=proc,
    )
    return CumulativeAudioCtcDetector(
        encoder_wrapper=wrapper,
        head=head,
        hotwords=entries,
        language="Chinese",
        scoring_config=HotwordScoringConfig(
            score_threshold=0.75,
            top_k=5,
            minimum_phonemes=1,
            maximum_edit_ratio=0.35,
            posterior_weight=0.25,
            minimum_posterior_confidence=0.5,
            minimum_top1_margin=0,
        ),
        retrieval_mode="operating",
        retrieval_backend="anchor_guided",
        anchor_shortlist_size=64,
        anchor_start_radius=2,
        anchor_config=AnchorIndexConfig(
            ngram_sizes=(2, 3, 4), anchors_per_entry=24, offset_tolerance=1
        ),
        device=device,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["npu", "cuda"], default="npu")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, default=Path("delivery.zip"))
    p.add_argument("--check", action="store_true", help="CPU-only file identity check")
    a = p.parse_args()
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    meta = json.loads((HERE / "manifest.json").read_text())
    receipt = a.output.with_suffix(a.output.suffix + ".run.json")
    if not a.check and (not a.manifest or a.output.exists() or receipt.exists()):
        p.error("provide --manifest and fresh --output")
    with stage_progress("Verify kit, snapshot, all 2696 indexed surfaces"):
        for name, h in meta["files"].items():
            pin(HERE / name, h)
        for name, h in meta["base_model_files"].items():
            pin(a.model / name, h)
        vocab = load_phoneme_vocab(HERE / "assets/vocab.json")
        entries, owners = common.registry(HERE / "assets/joint2696.jsonl", vocab)
        if (
            len(vocab.tokens),
            len(set(owners[e.hotword_id].surface for e in entries)),
            len(entries),
        ) != (183, 2696, 2697):
            raise ValueError("wrong assets")
    if a.check:
        print("Asset hashes passed (CPU only).")
        return
    import torch

    if metadata.version("transformers") != "4.57.6":
        raise SystemExit("Use transformers 4.57.6 inside the isolated test environment.")
    device = a.backend + ":0"
    if a.backend == "npu":
        import torch_npu  # type: ignore[import-not-found] # noqa: F401

        accel: Any = torch.npu  # type: ignore[attr-defined]
        if not accel.is_available() or accel.device_count() != 1:
            raise RuntimeError("Expose only the allocated test chip as logical npu:0.")

        # Synchronize NPU timings in this offline process only.
        def sync(torch: Any, device: str) -> None:
            torch.npu.synchronize(device)

        ext._sync_cuda = sync
    else:
        accel = torch.cuda
    accel.set_device(0)
    free, total = accel.mem_get_info(0)
    chip = accel.get_device_name(0)
    print(f"{chip}, {device}, {a.dtype}, free {free / 2**30:.1f} GiB", flush=True)
    samples = common.sample_rows(a.manifest)
    with stage_progress("Load eager Qwen audio tower only + FP32 CTC"):
        d = detector(a.model, vocab, entries, device, getattr(torch, a.dtype))
    common.deliver(samples, d, vocab, entries, owners, a.output)
    write_json(
        receipt,
        dict(
            delivery_sha256=sha(a.output),
            kit_sha256=sha(HERE / "manifest.json"),
            backend=a.backend,
            device_name=chip,
            dtype=a.dtype,
            attention="eager",
            torch=str(torch.__version__),
            torch_npu=metadata.version("torch-npu") if a.backend == "npu" else None,
            table_sha256=meta["files"]["assets/joint2696.jsonl"],
        ),
    )
    print(a.output.resolve(), receipt.resolve(), flush=True)


if __name__ == "__main__":
    main()
