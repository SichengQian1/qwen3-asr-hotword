"""Same-step retrieval for the joint Mandarin/English evaluation table."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.stream_context_contract import (
    MODES,
    audio_prefixes,
    boundaries,
    candidates,
)
from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import rows

AIS = "zh-aishell-kw_subset"
COUNTS = {
    AIS: 2135,
    **{
        f"{lang}-smarthome-val_{suffix}": count
        for lang, counts in (("zh", (258, 200, 60, 150)), ("en", (282, 200, 30, 150)))
        for suffix, count in zip(("v1", "v2", "n1", "n2"), counts, strict=True)
    },
}
GATE = {
    "threshold": 0.75,
    "maximum_edit_ratio": 0.35,
    "minimum_posterior_confidence": 0.5,
    "minimum_top1_margin": 0.0,
}


def canonical(word: str) -> str:
    return " ".join(word.casefold().split())


def registry(
    path: Path, vocab: PhonemeVocab
) -> tuple[tuple[HotwordEntry, ...], dict[str, HotwordEntry]]:
    entries: list[HotwordEntry] = []
    owners: dict[str, HotwordEntry] = {}
    seen: set[str] = set()
    for i, row in enumerate(rows(path)):
        word = row["surface"]
        key = canonical(word)
        if not key or key in seen or not row["pronunciations"]:
            raise ValueError(f"empty/duplicate joint surface or missing pronunciation: {word}")
        seen.add(key)
        first = len(entries)
        for j, pron in enumerate(row["pronunciations"]):
            phones, ids = tuple(pron["phoneme_tokens"]), tuple(pron["token_ids"])
            if (
                not ids
                or min(ids) < 2
                or max(ids) >= len(vocab.tokens)
                or phones != tuple(vocab.tokens[t] for t in ids)
            ):
                raise ValueError(f"pronunciation differs from 183-class IDs: {word}")
            entry = HotwordEntry(
                f"w{i}p{j}",
                "+".join(pron["languages"]),
                word,
                key,
                tuple(key.split()),
                " ".join(phones),
                phones,
                ids,
                ",".join(pron["sources"]),
                0,
            )
            entries.append(entry)
            owners[entry.hotword_id] = entries[first]
    if (len(seen), len(entries)) != (2351, 2352):
        raise ValueError("joint table count differs")
    return tuple(entries), owners


def inputs(cfg: dict[str, Any]) -> Iterable[tuple[str, str, Path, list[str]]]:
    path = Path(cfg["aishell"]) / "val/kw_subset/kw.jsonl"
    pin(path, "4d7efff4ff70e5a0f339dbc4049d0b2d380de525132563abc7387352440f26be")
    for row in rows(path):
        uid = row["utt_id"]
        yield (
            AIS,
            uid,
            Path(cfg["aishell_wav"]) / f"{uid}.wav",
            [k["word"] for k in row["keywords"]],
        )
    for row in rows(Path(cfg["home_meta"])):
        uid = row["id"]
        if not uid.startswith("val_"):
            continue
        key = re.fullmatch(r"(val_(?:v1|v2|n1|n2))_(zh|en)_\d+", uid)
        words = row["keywords"]
        if not key or row["status"] != "ok" or not isinstance(words, list):
            raise ValueError(f"invalid smart-home record: {uid}")
        if bool(words) != (key[1] in {"val_v1", "val_v2"}):
            raise ValueError(f"unexpected target polarity: {uid}")
        yield f"{key[2]}-smarthome-{key[1]}", uid, Path(row["wav"]), words


def inventory(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    import soundfile as sf

    samples = []
    seen: set[tuple[str, str]] = set()
    with stage_progress(
        "Verify evaluation WAVs and per-utterance targets", total=sum(COUNTS.values()), every=100
    ) as tick:
        for group, uid, path, words in inputs(cfg):
            if (group, uid) in seen or not re.fullmatch(r"[\w-]+", uid):
                raise ValueError(f"duplicate/invalid utterance: {group}/{uid}")
            seen.add((group, uid))
            if any(not isinstance(w, str) or not canonical(w) for w in words):
                raise ValueError(f"invalid target: {uid}")
            info = sf.info(str(path))
            lang = group[:2]
            sample = dict(
                group=group,
                key=f"{group}@@{lang}",
                language=lang,
                sample_id=uid,
                audio_path=str(path),
                audio_sha256=sha(path),
                frames=info.frames,
                sample_rate=info.samplerate,
                channels=info.channels,
                duration_seconds=info.frames / info.samplerate,
                step_count=len(boundaries(info.frames, info.samplerate)),
                targets=sorted({canonical(w) for w in words}),
            )
            samples.append(sample)
            tick(len(samples))
    if Counter(s["group"] for s in samples) != COUNTS:
        raise ValueError("dataset coverage differs from input inspection")
    targets = [s["targets"] for s in samples if s["group"] == AIS]
    if (len({w for t in targets for w in t}), sum(map(len, targets))) != (630, 2782):
        raise ValueError("AISHELL original target coverage differs")
    return samples


def retrieve(
    detector: Any,
    vocab: PhonemeVocab,
    entries: tuple[HotwordEntry, ...],
    owners: dict[str, HotwordEntry],
    sample: dict[str, Any],
    infer: Callable[..., Mapping[str, Any]] = external._retrieve_waveform,
) -> dict[str, Any]:
    detector.language = {"en": "English", "zh": "Chinese"}[sample["language"]]
    steps, raw_words = [], []
    for prefix in audio_prefixes(sample["audio_path"], sample):
        result = infer(detector, prefix["waveform"], vocab, saved_raw_rank_depth=64)
        ranked = collapse_variants(result, entries, owners, GATE)["raw_ranked_matches"]
        raw_words = [m["surface"] for m in ranked[:7]]
        steps.append(candidates(_select(ranked, gate=GATE, top_k=7)))
    if len(steps) != sample["step_count"]:
        raise ValueError("prefix count differs")
    return {"steps": steps, "raw_last": raw_words}


def update_metrics(counts: Counter[str], gold: set[str], selected: set[str]) -> None:
    counts["utterances"] += 1
    counts["target_pairs"] += len(gold)
    counts["target_hits"] += len(gold & selected)
    counts["selected_pairs"] += len(selected)
    counts["negative_utterances"] += int(not gold)
    counts["negative_with_candidate"] += int(not gold and bool(selected))


def score(
    result: dict[str, Any], sample: dict[str, Any], totals: dict[str, dict[str, Counter[str]]]
) -> None:
    gold = set(sample["targets"])
    for k in MODES.values():
        per_step = [{canonical(m["word"]) for m in step[:k]} for step in result["steps"]]
        for name, words in (
            (f"raw_top{k}_last", {canonical(w) for w in result["raw_last"][:k]}),
            (f"gated_top{k}_last", per_step[-1]),
            (f"gated_top{k}_ever", set().union(*per_step)),
        ):
            for group in (sample["group"], sample["language"], "overall"):
                update_metrics(
                    totals.setdefault(group, {}).setdefault(name, Counter()), gold, words
                )


def rates(counts: Counter[str]) -> dict[str, Any]:
    pairs, negatives = counts["target_pairs"], counts["negative_utterances"]
    return {
        **counts,
        "target_recall": counts["target_hits"] / pairs if pairs else None,
        "negative_prompt_rate": counts["negative_with_candidate"] / negatives
        if negatives
        else None,
    }
