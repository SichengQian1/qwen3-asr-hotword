# ruff: noqa: E402
"""Produce same-step Top5/Top7 for new audio using the sealed bilingual CTC/RAG kit."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from qwen_hotword.evaluation.zh_en_delivery import canonical, retrieve
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.external_keyword_retrieval import _load_detector
from qwen_hotword.inference.stream_context_contract import MODES, boundaries, validate_contexts
from qwen_hotword.phonemes.coverage import PhonemeVocab, load_phoneme_vocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress


def registry(
    path: Path, vocab: PhonemeVocab
) -> tuple[tuple[HotwordEntry, ...], dict[str, HotwordEntry]]:
    entries: list[HotwordEntry] = []
    owners = {}
    seen: set[str] = set()
    for line in path.read_text(encoding="utf8").splitlines():
        row = json.loads(line)
        word = row["surface"]
        norm = canonical(word)
        if not norm or norm in seen or not row["pronunciations"]:
            raise ValueError(f"invalid or duplicate surface: {word}")
        first = len(entries)
        i = len(seen)
        seen.add(norm)
        for j, p in enumerate(row["pronunciations"]):
            phones, ids = tuple(p["phoneme_tokens"]), tuple(p["token_ids"])
            if (
                not ids
                or any(type(t) is not int or not 2 <= t < len(vocab.tokens) for t in ids)
                or phones != tuple(vocab.tokens[t] for t in ids)
            ):
                raise ValueError(f"invalid pronunciation: {word}")
            entry = HotwordEntry(
                f"w{i}p{j}",
                "+".join(p["languages"]),
                word,
                norm,
                tuple(norm.split()),
                " ".join(phones),
                phones,
                ids,
                ",".join(p["sources"]),
                0,
            )
            entries.append(entry)
            owners[entry.hotword_id] = entries[first]
    return tuple(entries), owners


def sample_rows(manifest: Path) -> list[dict[str, Any]]:
    import soundfile as sf

    samples = []
    seen = set()
    for line in manifest.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uid, lang, group = r["sample_id"], r["language"], r.get("group", "custom")
        if (
            lang not in ("zh", "en")
            or not all(isinstance(s, str) and s.strip() for s in (uid, group))
            or "@@" in group
        ):
            raise ValueError(f"invalid sample/group/language: {r}")
        key = f"{group}@@{lang}"
        if (key, uid) in seen:
            raise ValueError(f"duplicate {key}/{uid}")
        seen.add((key, uid))
        wav = (manifest.parent / r["audio_path"]).resolve()
        info = sf.info(str(wav))
        samples.append(
            dict(
                sample_id=uid,
                group=group,
                key=key,
                language=lang,
                audio_path=str(wav),
                audio_sha256=sha(wav),
                frames=info.frames,
                channels=info.channels,
                sample_rate=info.samplerate,
                duration_seconds=info.frames / info.samplerate,
                step_count=len(boundaries(info.frames, info.samplerate)),
            )
        )
    if not samples:
        raise ValueError("empty input manifest")
    return samples


def deliver(
    samples: list[dict[str, Any]],
    detector: Any,
    vocab: PhonemeVocab,
    entries: tuple[HotwordEntry, ...],
    owners: dict[str, HotwordEntry],
    output: Path,
) -> None:
    durations: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    ctx: dict[str, Any] = {m: {} for m in MODES}
    with stage_progress("CTC/RAG: same-step audio prefixes", total=len(samples), every=1) as tick:
        for i, s in enumerate(samples, 1):
            key, uid = s["key"], s["sample_id"]
            result = retrieve(detector, vocab, entries, owners, s)
            for m, k in MODES.items():
                ctx[m].setdefault(key, {})[uid] = [step[:k] for step in result["steps"]]
            durations.setdefault(key, {})[uid] = s["duration_seconds"]
            manifest.setdefault(key, {})[uid] = s
            tick(i)
    print(validate_contexts(ctx, durations), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        pending = Path(tmp) / "delivery.zip"
        with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as z:
            for name, value in [
                ("stream_contexts", ctx),
                ("durations", durations),
                ("audio_manifest", manifest),
            ]:
                z.writestr(
                    "delivery/" + name + ".json",
                    json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                )
        with zipfile.ZipFile(pending) as z:
            if z.testzip() is not None:
                raise ValueError("delivery CRC failed")
        os.link(pending, output)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path, default=Path("delivery.zip"))
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if not a.check and (not a.manifest or a.output.exists()):
        p.error("provide --manifest and a fresh --output path")
    meta = json.loads((HERE / "manifest.json").read_text())
    with stage_progress("Verify kit and supplied Qwen snapshot"):
        for name, checksum in meta["files"].items():
            pin(HERE / name, checksum)
        for name, checksum in meta["base_model_files"].items():
            pin(a.model / name, checksum)
        vocab = load_phoneme_vocab(HERE / "assets/vocab.json")
        entries, owners = registry(HERE / "assets/joint2411.jsonl", vocab)
        if (len(vocab.tokens), len(entries), len({e.surface for e in entries})) != (
            183,
            2412,
            2411,
        ):
            raise ValueError("wrong vocabulary or hotword table")
    if a.check:
        print("Assets verified. No GPU inference was run.")
        return
    samples = sample_rows(a.manifest)
    with stage_progress("Load local Qwen and epoch-25 CTC; build Anchor index"):
        detector = _load_detector(
            model=a.model,
            checkpoint=HERE / "assets/ctc_head_best.pt",
            vocab=vocab,
            hotwords=entries,
            device="cuda:0",
            dtype="bfloat16",
            language="English",
        )
    deliver(samples, detector, vocab, entries, owners, a.output)
    print(a.output.resolve())


if __name__ == "__main__":
    main()
