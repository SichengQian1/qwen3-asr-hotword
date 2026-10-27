"""Deterministic waveform-domain MUSAN mixing; no model or original audio writes."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from qwen_hotword.training.spanish_capacity import _sha
from qwen_hotword.training.stage_progress import stage_progress


def rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def read_audio(path: Path) -> Any:
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        import librosa

        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"empty or nonfinite waveform: {path}")
    return np.asarray(audio, dtype=np.float32)


def audit_noise(folder: Path, seed: int) -> dict[str, Any]:
    import numpy as np

    files = sorted(folder.rglob("*.wav"))
    if not files:
        raise ValueError(f"no noise WAV files found: {folder}")
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    with stage_progress("Decode/hash MUSAN noise", total=len(files), every=50) as tick:
        for index, path in enumerate(files, 1):
            try:
                audio = read_audio(path)
                power = float(np.mean(audio.astype(np.float64) ** 2))
                if power < 1e-12:
                    raise ValueError("near-silent noise")
                pcm = hashlib.sha256(audio.astype("<f4").tobytes()).hexdigest()
                if pcm in seen:
                    duplicates.append({"path": str(path), "same_pcm_as": seen[pcm]})
                else:
                    seen[pcm] = str(path)
                    entries.append(
                        {
                            "path": str(path),
                            "sha256": _sha(path),
                            "pcm_sha256": pcm,
                            "samples_16k": len(audio),
                            "seconds": len(audio) / 16000,
                            "source": path.parent.name,
                            "power": power,
                        }
                    )
            except (OSError, RuntimeError, ValueError) as error:
                failures.append({"path": str(path), "error": str(error)})
            tick(index)
    # Split canonical decoded recordings, not cuts from the same recording.
    for source in sorted({entry["source"] for entry in entries}):
        group = sorted(
            (e for e in entries if e["source"] == source),
            key=lambda e: rank(seed, e["pcm_sha256"]),
        )
        if len(group) < 2:
            raise ValueError(f"not enough distinct noise recordings for holdout: {source}")
        holdout = max(1, round(len(group) * 0.1))
        for index, entry in enumerate(group):
            entry["split"] = "validation" if index < holdout else "train"
    if not entries:
        raise ValueError("no usable noise recordings")
    metadata = {
        str(p): _sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name in {"README", "ANNOTATIONS", "LICENSE"}
    }
    return {
        "entries": entries,
        "excluded": failures,
        "duplicates": duplicates,
        "metadata_sha256": metadata,
        "wav_files_seen": len(files),
        "near_duplicate_detection": (
            "decoded exact equality only; acoustic near-duplicates unverified"
        ),
    }


def mix_10db(
    speech: Any, noise: Any, *, offset: int, snr_db: float = 10.0
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if speech.ndim != 1 or noise.ndim != 1 or not len(speech) or not len(noise):
        raise ValueError("mixing requires nonempty mono speech and noise")
    if not np.isfinite(speech).all() or not np.isfinite(noise).all():
        raise ValueError("mixing input has nonfinite values")
    # Explicit energy-based activity estimate on ORIGINAL audio; not a human VAD reference.
    energies = np.array(
        [np.mean(speech[start : start + 320] ** 2) for start in range(0, len(speech), 320)]
    )
    if float(energies.max()) < 1e-12:
        raise ValueError("near-silent source speech")
    active = np.repeat(energies >= energies.max() * 0.01, 320)[: len(speech)]
    indices = (np.arange(len(speech)) + offset) % len(noise)
    segment = noise[indices]
    speech_power = float(np.mean(speech[active] ** 2))
    noise_power = float(np.mean(segment[active] ** 2))
    if noise_power < 1e-12:
        raise ValueError("selected noise segment has insufficient active-region energy")
    alpha = math.sqrt(speech_power / noise_power / (10 ** (snr_db / 10)))
    mixture = speech + alpha * segment
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, 0.99 / max(peak, 1e-12))
    mixture *= gain
    measured = 10 * math.log10(speech_power / (alpha**2 * noise_power))
    if abs(measured - snr_db) > 1e-5 or not np.isfinite(mixture).all():
        raise ValueError("mixed SNR or numeric validation failed")
    return mixture.astype(np.float32), {
        "snr_db": measured,
        "noise_gain": alpha,
        "common_gain": gain,
        "peak": float(np.max(np.abs(mixture))),
        "active_samples": int(active.sum()),
        "source_samples": len(speech),
        "noise_looped": len(noise) < len(speech),
        "offset_16k": offset,
        "activity_rule": "20ms RMS power >= peak frame power -20dB",
        "snr_scope": "original recording versus added noise on estimated active frames",
    }
