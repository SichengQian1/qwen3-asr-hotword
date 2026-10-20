"""Causal prefix audio and the downstream mode/key/utterance context contract."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

MODES = {"RAG_top5_stream": 5, "RAG_top7_stream": 7}
EFFECTS = ("same-step", "next-step")


def boundaries(frames: int, sample_rate: int) -> tuple[int, ...]:
    if frames <= 0 or sample_rate <= 0:
        raise ValueError("audio must contain positive frames and sample rate")
    stride = 2 * sample_rate
    return tuple(range(stride, frames, stride)) + (frames,)


def audio_prefixes(path: str, metadata: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(path) as handle:
        if (handle.frames, handle.samplerate, handle.channels) != (
            metadata["frames"],
            metadata["sample_rate"],
            metadata["channels"],
        ):
            raise ValueError(f"audio metadata changed: {path}")
        parts, previous = [], 0
        for step, end in enumerate(boundaries(handle.frames, handle.samplerate)):
            started = time.perf_counter()
            block = handle.read(end - previous, dtype="float32", always_2d=True)
            if len(block) != end - previous:
                raise ValueError(f"short audio read: {path}")
            parts.append(block.mean(axis=1, dtype=np.float32))
            prefix = np.concatenate(parts)
            # Resample ONLY the already received prefix, never the complete utterance.
            if handle.samplerate != 16000:
                import librosa

                prefix = librosa.resample(prefix, orig_sr=handle.samplerate, target_sr=16000)
            if not np.isfinite(prefix).all() or not len(prefix):
                raise ValueError(f"empty/non-finite audio prefix: {path}")
            yield {
                "step": step,
                "source_end_frame": end,
                "source_sample_rate": handle.samplerate,
                "audio_end_seconds": end / handle.samplerate,
                "model_input_samples": len(prefix),
                "model_sample_rate": 16000,
                "input_pcm_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
                "prefix_preparation_seconds": time.perf_counter() - started,
                "waveform": prefix,
            }
            previous = end


def candidates(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"word": m["surface"], "phoneme": m["matched_pronunciation"], "score": m["score"]}
        for m in matches
    ]


def contexts_for_steps(
    steps: Sequence[Mapping[str, Any]], effect: str, top_k: int
) -> list[list[dict[str, Any]]]:
    if effect not in EFFECTS or top_k not in MODES.values():
        raise ValueError("explicit same-step/next-step and Top5/Top7 required")
    current = [list(s["candidates"][:top_k]) for s in steps]
    return current if effect == "same-step" else [[]] + current[:-1]


def validate_contexts(data: Mapping[str, Any], durations: Mapping[str, Any]) -> dict[str, Any]:
    if set(data) != set(MODES) or not durations:
        raise ValueError("exactly Top5/Top7 modes and a nonempty duration map required")
    total_steps = 0
    for mode, top_k in MODES.items():
        if set(data[mode]) != set(durations):
            raise ValueError("dataset keys differ from duration manifest")
        for key, by_utt in durations.items():
            if set(data[mode][key]) != set(by_utt):
                raise ValueError(f"utterance coverage differs: {mode}/{key}")
            for utt, duration in by_utt.items():
                if (
                    not isinstance(duration, int | float)
                    or not math.isfinite(duration)
                    or duration <= 0
                ):
                    raise ValueError("invalid duration")
                steps = data[mode][key][utt]
                if not isinstance(steps, list) or len(steps) != math.ceil(duration / 2):
                    raise ValueError(f"wrong number of 2s boundaries: {key}/{utt}")
                for items in steps:
                    if not isinstance(items, list) or len(items) > top_k:
                        raise ValueError("boundary must be a list within mode TopK")
                    seen: set[str] = set()
                    previous = math.inf
                    for item in items:
                        if not isinstance(item, dict) or set(item) != {"word", "phoneme", "score"}:
                            raise ValueError("candidate requires word, phoneme and score")
                        if any(
                            not isinstance(item[k], str) or not item[k].strip()
                            for k in ("word", "phoneme")
                        ):
                            raise ValueError("empty/invalid word or pronunciation")
                        score = item["score"]
                        normalized = " ".join(item["word"].casefold().split())
                        if (
                            isinstance(score, bool)
                            or not isinstance(score, int | float)
                            or not math.isfinite(score)
                            or not 0 <= score <= 1
                            or score > previous
                            or normalized in seen
                        ):
                            raise ValueError("duplicate, unsorted or invalid candidate score")
                        seen.add(normalized)
                        previous = score
                total_steps += len(steps)
    for key, by_utt in durations.items():
        for utt in by_utt:
            if any(
                a != b[:5]
                for a, b in zip(
                    data["RAG_top5_stream"][key][utt],
                    data["RAG_top7_stream"][key][utt],
                    strict=True,
                )
            ):
                raise ValueError("Top5 and Top7 must share the same ranked retrieval")
    return {
        "status": "pass",
        "modes": len(MODES),
        "unique_utterances": sum(len(v) for v in durations.values()),
        "mode_boundary_count": total_steps,
    }


def digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in value.items() if k != "sha256"},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
