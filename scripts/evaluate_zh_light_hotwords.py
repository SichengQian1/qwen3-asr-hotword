"""Score the seven Mandarin sets against the expanded bilingual table."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_zh_en_480_eleven import cached, load_result, pct

from qwen_hotword.evaluation import aishell_run as ar
from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation import zh_en_eleven as sets
from qwen_hotword.evaluation import zh_en_parallel as parallel
from qwen_hotword.evaluation import zh_light as light
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference.external_keyword_retrieval import _write_or_verify_json
from qwen_hotword.inference.stream_context_contract import boundaries
from qwen_hotword.phonemes.coverage import PhonemeVocab
from qwen_hotword.phonemes.mfa_v03_assets import pin, sha, write_json
from qwen_hotword.training.feature_cache import exclusive_feature_cache_run
from qwen_hotword.training.stage_progress import stage_progress
from qwen_hotword.training.zh_en_common import digest, read, rows, save_rows


def expand(
    cfg: dict[str, Any],
    out: Path,
    words: list[str],
    train: dict[str, Any],
    foundation: dict[str, Any],
    vocab: PhonemeVocab,
) -> tuple[tuple[HotwordEntry, ...], dict[str, HotwordEntry]]:
    path = ROOT / cfg["previous_table"]
    pin(path, cfg["previous_table_sha256"])
    entries, owners = base.registry(path, vocab)
    table = list(rows(path))
    added = light.additions(words, table)
    inputs = [light.SPOKEN.get(w, w) for w in added]
    converted = ar.pronunciations(
        ROOT,
        out,
        inputs,
        train,
        foundation,
        vocab,
        digest([inputs, foundation, train["frontend_packages"]]),
    )
    new_entries = []
    for word, spoken, entry in zip(added, inputs, converted, strict=True):
        if entry.surface != spoken:
            raise ValueError(f"G2P order differs for {word}")
        languages = ["zh", "us"] if word in {"芒果TV", "QQ音乐"} else ["zh"]
        pron = dict(
            phoneme_tokens=entry.phoneme_tokens,
            token_ids=entry.token_ids,
            languages=languages,
            sources=[light.LIGHT],
        )
        new = replace(
            entry,
            hotword_id=f"w{len(table)}p0",
            surface=word,
            normalized=base.canonical(word),
            words=(base.canonical(word),),
            language="+".join(languages),
            source=light.LIGHT,
        )
        owners[new.hotword_id] = new
        new_entries.append(new)
        table.append({"surface": word, "pronunciations": [pron]})
    save_rows(out / "joint2411.jsonl", table)
    return (*entries, *new_entries), owners


def inventory(
    cfg: dict[str, Any], previous: Path, ner: Path, extra: list[tuple[str, str, Path, list[str]]]
) -> dict[str, list[dict[str, Any]]]:
    import soundfile as sf

    full_words = {
        r["surface"]
        for r in rows(previous)
        if any(sets.FULL in p["sources"] for p in r["pronunciations"])
    }
    gold, _ = sets.full_gold(ner, Path(cfg["aishell"]) / "data/test/text", full_words)
    planned = [
        (sets.FULL, uid, Path(cfg["aishell_wav"]) / (uid + ".wav"), words)
        for uid, words in gold.items()
    ]
    planned.extend(row for row in base.inputs(cfg) if row[0].startswith("zh-"))
    planned.extend(extra)
    groups: dict[str, list[dict[str, Any]]] = {g: [] for g in light.GROUPS}
    seen: set[tuple[str, str]] = set()
    with stage_progress("核对七组音频与目标标注", total=len(planned), every=250) as tick:
        for i, (group, uid, path, words) in enumerate(planned, 1):
            if (group, uid) in seen or not all(isinstance(w, str) and w.strip() for w in words):
                raise ValueError(f"invalid row {group}/{uid}")
            seen.add((group, uid))
            info = sf.info(str(path))
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"empty or unreadable {path}")
            groups[group].append(
                dict(
                    group=group,
                    key=f"{group}@@zh",
                    language="zh",
                    sample_id=uid,
                    audio_path=str(path),
                    audio_sha256=sha(path),
                    frames=info.frames,
                    sample_rate=info.samplerate,
                    channels=info.channels,
                    step_count=len(boundaries(info.frames, info.samplerate)),
                    duration_seconds=info.frames / info.samplerate,
                    targets=sorted({base.canonical(w) for w in words}),
                )
            )
            tick(i)
    expected = ((7176, 3064), (2135, 2782), (258, 270), (200, 216), (60, 0), (150, 0))
    for group, counts in zip(light.GROUPS[:-1], expected, strict=True):
        data = groups[group]
        if (len(data), sum(len(s["targets"]) for s in data)) != counts:
            raise ValueError(f"previous gold changed: {group}")
    entire = {s["sample_id"]: s["audio_sha256"] for s in groups[sets.FULL]}
    if any(entire.get(s["sample_id"]) != s["audio_sha256"] for s in groups[base.AIS]):
        raise ValueError("AISHELL subset is not contained in full audio set")
    return groups


def run(cfg: dict[str, Any], out: Path, args: argparse.Namespace) -> None:
    previous_cfg = read(ROOT / cfg["checkpoint_config"])
    words = (ROOT / cfg["word_list"]).read_text(encoding="utf-8").splitlines()
    wavs = Path(cfg["audio_root"])
    files = sorted(p for p in wavs.rglob("*") if p.is_file() and p.suffix.casefold() == ".wav")
    write_json(
        out / "light_layout.json",
        {
            "audio_root": str(wavs),
            "wav_count": len(files),
            "examples": [str(p) for p in files[:8]],
            "nearby_files": sorted(
                str(p)
                for d in (wavs, wavs.parent)
                if d.is_dir()
                for p in d.iterdir()
                if p.suffix.lower() in {".tsv", ".jsonl", ".txt", ".json", ".csv"}
            )[:100],
        },
    )
    with stage_progress("核验 480h 中英模型并构建 2411 词联合词表"):
        model, train, foundation, vocab = ar.identity(
            ROOT, previous_cfg, read(ROOT / previous_cfg["review_policy"])
        )
        entries, owners = expand(cfg, out, words, train, foundation, vocab)
    print(
        f"联合词表已就绪：{out / 'joint2411.jsonl'}（2411 词面，{len(entries)} 条发音）", flush=True
    )
    try:
        labels = light.metadata(wavs, args.labels)
        extra = light.light_gold(wavs, files, labels, words)
    except (ValueError, OSError) as error:
        raise SystemExit(f"{error}\n请返回目录检查文件：{out / 'light_layout.json'}") from error
    observed = {w for _, _, _, ws in extra for w in ws}
    prior_ner = ROOT / cfg["cached_aishell_ner"]
    ner = sets.official_file(out, prior_ner if prior_ner.is_file() else None)
    groups = inventory(read(ROOT / cfg["sources_config"]), ROOT / cfg["previous_table"], ner, extra)
    vocabulary = {base.canonical(e.surface) for e in entries}
    if any(set(s["targets"]) - vocabulary for group in groups.values() for s in group):
        raise ValueError("annotated target outside joint vocabulary")
    plan = dict(
        model=model,
        table_sha256=sha(out / "joint2411.jsonl"),
        groups=digest(groups),
        label_sha256=sha(labels),
        gate=base.GATE,
        code_sha256=[
            sha(Path(p))
            for p in (__file__, light.__file__, base.__file__, sets.__file__, parallel.__file__)
        ],
    )
    _write_or_verify_json(out / "run_config.json", plan)
    run_id = digest(plan)
    parallel.prefill(
        out,
        groups,
        run_id,
        args.workers,
        args.gpu,
        (Path(model["model_path"]), Path(model["checkpoint_path"]), vocab, entries, owners),
        cached,
        load_result,
    )
    totals: dict[str, dict[str, Counter[str]]] = {}
    for records in groups.values():
        for s in records:
            base.score(load_result(cached(out, s), run_id, s), s, totals)
    totals[light.SUMMARY] = light.aggregate(totals)
    metrics = {
        g: {v: base.rates(c) for v, c in totals[g].items()} for g in (*light.GROUPS, light.SUMMARY)
    }
    lines = [
        "数据集\t句数\t目标句词对\tTop5前\tTop5后\tTop7前\tTop7后\t无目标句\t负句提示率Top5后\t负句提示率Top7后"
    ]
    for g, m in metrics.items():
        t = m["gated_top7_last"]
        rates = [
            pct(m[f"{mode}_top{k}_last"]["target_recall"])
            for k in (5, 7)
            for mode in ("raw", "gated")
        ]
        neg = [pct(m[f"gated_top{k}_last"]["negative_prompt_rate"]) for k in (5, 7)]
        lines.append(
            "\t".join(
                map(
                    str,
                    [g, t["utterances"], t["target_pairs"], *rates, t["negative_utterances"], *neg],
                )
            )
        )
    table = "\n".join(lines) + "\n"
    print(table, flush=True)
    (out / "by_dataset.tsv").write_text(table, encoding="utf-8-sig")
    write_json(
        out / "report.json",
        dict(
            status="completed",
            by_dataset=metrics,
            run_sha256=run_id,
            model=model,
            table_sha256=plan["table_sha256"],
            active_surfaces=len(vocabulary),
            pronunciations=len(entries),
            reused_words=sorted(light.REUSED),
            light_observed_words=sorted(observed),
            light_unobserved_words=sorted(set(words) - observed),
            light_labels=str(labels),
            light_label_sha256=sha(labels),
            spoken_inputs=light.SPOKEN,
            micro_summary_excludes=[base.AIS],
            audio_condition="as supplied",
            qwen_decoder_used=False,
        ),
    )
    pending = out / "return_bundle.zip.pending"
    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("report.json", "by_dataset.tsv", "g2p_report.json", "light_layout.json"):
            z.write(out / name, name)
    pending.replace(out / "return_bundle.zip")
    print(out / "return_bundle.zip", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 17))
    p.add_argument("--labels", type=Path)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = read(ROOT / "configs/zh_light_eval.workzone.json")
    out = ROOT / cfg["output"]
    with exclusive_feature_cache_run(out):
        run(cfg, out, args)


if __name__ == "__main__":
    main()
