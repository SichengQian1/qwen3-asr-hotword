#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qwen_hotword.config import ConfigError, load_workzone_config
from qwen_hotword.modeling.audio_encoder import extract_padded_ln_post
from qwen_hotword.modeling.ctc_head import LinearCtcHead, compute_ctc
from qwen_hotword.modeling.qwen_backbone import ModelValidationError, load_asr_model
from qwen_hotword.phonemes.coverage import load_phoneme_vocab, normalization_key

DEFAULT_VOCAB = REPO_ROOT / "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.2.json"
TARGET_PHONE_SEQUENCES = (("k", "æ", "t"), ("b", "o", "n", "dʒ", "u", "r"))


def synthetic_waveform(seconds: float, sample_rate: int = 16_000) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("numpy is required for the CTC training smoke test") from error

    sample_count = int(seconds * sample_rate)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    return (0.01 * np.sin(2.0 * math.pi * 440.0 * time)).astype(np.float32)


def build_audio_prompt(processor: Any, language: str) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    return prompt + f"language {language}<asr_text>"


def freeze_module(module: Any) -> int:
    parameter_count = 0
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter_count += parameter.numel()
    return parameter_count


def padded_targets(
    token_ids: list[list[int]],
    *,
    blank_id: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    target_lengths = torch.tensor(
        [len(sequence) for sequence in token_ids],
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (len(token_ids), int(target_lengths.max().item())),
        fill_value=blank_id,
        dtype=torch.long,
        device=device,
    )
    for row, sequence in enumerate(token_ids):
        targets[row, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
    return targets, target_lengths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen-Qwen plus linear phoneme CTC training step."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    args = parser.parse_args()

    try:
        import torch

        config = load_workzone_config(args.config, require_existing_model=True)
        vocab = load_phoneme_vocab(args.vocab)
        if not vocab.tokens or vocab.tokens[0] != "<blank>":
            raise ValueError("CTC vocabulary must place <blank> at token ID 0")
        blank_id = 0
        token_ids = [
            [vocab.token_to_id[normalization_key(phone)] for phone in sequence]
            for sequence in TARGET_PHONE_SEQUENCES
        ]

        wrapper = load_asr_model(config.model)
        audio_tower = wrapper.model.thinker.audio_tower
        frozen_parameter_count = freeze_module(audio_tower)

        seconds = [1.0, 2.0]
        waveforms = [synthetic_waveform(duration) for duration in seconds]
        prompts = [build_audio_prompt(wrapper.processor, "English") for _ in waveforms]
        processor_batch = wrapper.processor(
            text=prompts,
            audio=waveforms,
            return_tensors="pt",
            padding=True,
        )
        input_features = processor_batch["input_features"].to(
            device=wrapper.model.device,
            dtype=wrapper.model.dtype,
        )
        feature_attention_mask = processor_batch["feature_attention_mask"].to(
            device=wrapper.model.device
        )
        encoder_batch = extract_padded_ln_post(
            audio_tower,
            input_features,
            feature_attention_mask,
            no_grad=True,
        )

        ctc_head = LinearCtcHead(
            input_dimension=encoder_batch.hidden_states.shape[-1],
            num_classes=len(vocab.tokens),
        ).to(device=wrapper.model.device, dtype=wrapper.model.dtype)
        optimizer = torch.optim.AdamW(ctc_head.parameters(), lr=1e-3)
        targets, target_lengths = padded_targets(
            token_ids,
            blank_id=blank_id,
            device=wrapper.model.device,
        )

        optimizer.zero_grad(set_to_none=True)
        before_step = ctc_head.projection.weight.detach().clone()
        computation = compute_ctc(
            ctc_head,
            encoder_batch.hidden_states,
            encoder_batch.input_lengths,
            targets,
            target_lengths,
            blank_id=blank_id,
        )
        computation.loss.backward()
        head_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), max_norm=5.0).item()
        )
        backbone_gradients = sum(
            parameter.grad is not None for parameter in audio_tower.parameters()
        )
        optimizer.step()
        head_weight_changed = bool(
            torch.any(ctc_head.projection.weight.detach() != before_step).item()
        )

        loss_value = float(computation.loss.detach().cpu().item())
        errors: list[str] = []
        expected_logits_shape = [2, 26, len(vocab.tokens)]
        if list(computation.logits.shape) != expected_logits_shape:
            errors.append(
                f"logits shape differs: actual={list(computation.logits.shape)}, "
                f"expected={expected_logits_shape}"
            )
        if not math.isfinite(loss_value) or loss_value <= 0:
            errors.append(f"CTC loss must be positive and finite; got {loss_value}")
        if not math.isfinite(head_gradient_norm) or head_gradient_norm <= 0:
            errors.append(f"CTC head gradient norm is invalid: {head_gradient_norm}")
        if backbone_gradients != 0:
            errors.append(f"frozen audio tower received {backbone_gradients} gradients")
        if not head_weight_changed:
            errors.append("optimizer step did not change the CTC head weights")

        report = {
            "purpose": "minimal_ctc_training_smoke_test",
            "vocab": {
                "path": str(Path(args.vocab)),
                "num_classes": len(vocab.tokens),
                "blank_id": blank_id,
            },
            "encoder": {
                "hidden_states_shape": list(encoder_batch.hidden_states.shape),
                "input_lengths": encoder_batch.input_lengths.cpu().tolist(),
                "frozen_parameters": frozen_parameter_count,
                "parameters_with_gradients": backbone_gradients,
            },
            "targets": {
                "tokens": [list(sequence) for sequence in TARGET_PHONE_SEQUENCES],
                "token_ids": token_ids,
                "padded_shape": list(targets.shape),
                "target_lengths": target_lengths.cpu().tolist(),
            },
            "ctc": {
                "head_type": f"Linear(1024, {len(vocab.tokens)})",
                "trainable_parameters": sum(
                    parameter.numel() for parameter in ctc_head.parameters()
                ),
                "logits_shape": list(computation.logits.shape),
                "log_probs_shape": list(computation.log_probs.shape),
                "loss_float32": loss_value,
                "head_gradient_norm_before_clipping": head_gradient_norm,
                "head_weight_changed": head_weight_changed,
            },
            "errors": errors,
            "status": "pass" if not errors else "fail",
        }
    except (
        ConfigError,
        KeyError,
        ModelValidationError,
        RuntimeError,
        OSError,
        ValueError,
    ) as error:
        print(f"CTC TRAINING SMOKE TEST FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
